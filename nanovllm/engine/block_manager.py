from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1  # In order to compare if two blocks are the same
        self.token_ids = []  # token_ids stored in the block

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # prefix is the hash of the previous block
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # allocate a new block with the given block_id, and return the block
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0  # a free block should have no reference
        block.reset()  # ref_count = 1, hash = -1, token_ids = []
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0  # only a block with no reference can be deallocated
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)  # append to the end

    # check if there are enough free blocks to allocate the sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # allocate blocks for the given sequence
    def allocate(self, seq: Sequence):
        assert not seq.block_table  # a sequence should be only allocated once
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):  # seq.num_blocks is the number of blocks needed to store the sequence
            token_ids = seq.block(i)  # get the token_ids of the i-th block
            # compute the hash of the block only if the block is full
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)  # check if the block is already in the cache
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:  # if cache miss, allocate a new block
                block_id = self.free_block_ids[0]  # get the first free block
                block = self._allocate_block(block_id)
            else:  # if cache hit, use the existing block
                seq.num_cached_tokens += self.block_size  # update the number of cached tokens
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:  # this block is deallocated (removed from the used_block_ids), but still in the hash_to_block_id
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):  # deallocate the blocks from end to start
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:  # no reference to the block, deallocate it
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # check if there is enough free blocks
    def can_append(self, seq: Sequence) -> bool:
        # https://github.com/GeeeekExplorer/nano-vllm/issues/30
        # len(self.free_block_ids) >= 1 or 0 (need one more or use existing block)
        # We need one more free block when the sequence length reaches exactly 1 element past a block boundary
        # (i.e., when len(seq) % block_size == 1). In all other cases, the sequence can continue using the currently allocated blocks
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:  # if the last block has only one token
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]  # get a new free block_id
            self._allocate_block(block_id)  # allocate a new block
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:  # the last block is full, update the hash and token_ids of the last block
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)  # compute the hash of the last block based on the previous hash
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
