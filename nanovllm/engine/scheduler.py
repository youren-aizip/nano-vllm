from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # finished if there are no waiting or running sequences
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        # scheduler prioritizes prefill over decode
        scheduled_seqs = []  # sequences scheduled to run in current step
        num_seqs = 0  # number of sequences in current batch
        num_batched_tokens = 0  # number of tokens in current batch
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # get the first sequence in the waiting queue
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                # len(seq) is the number of tokens in the sequence
                # break if the number of tokens current batch is greater than the maximum number of batched tokens
                # or there are no free blocks in the block manager
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # start to decode if there is no waiting sequences
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # get the first sequence in the running queue
            while not self.block_manager.can_append(seq):  # check if there are enough free blocks
                # if there are no enough free blocks, preempt the sequence
                if self.running:  # if there are more running sequences
                    self.preempt(self.running.pop())  # preempt the last running sequence
                else:
                    # keep the current sequence waiting
                    self.preempt(seq)
                    break
            else:
                # enough free blocks, allocate the sequence and append to the scheduled sequences
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # seq will ONLY be removed from running queue when it is finished (in postprocess)
        # so we need to put the scheduled sequences back to the running queue
        # why reversed? maybe to make the sequence that is decoded last in current step to be the first in the next step?
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # make the sequence waiting again, deallocate the blocks and put sequence back to waiting queue
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)  # put the sequence back to the front of the waiting queue

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # finish generating either when the EOS token is generated or the sequence reaches the maximum number of new tokens
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
