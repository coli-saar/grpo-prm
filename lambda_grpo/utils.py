
import re
import torch
from peft import PeftModel
from contextlib import contextmanager
from typing import Union, Generator, Any, TypeVar, Callable, List, Optional, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer, LogitsProcessor, LogitsProcessorList


Model = Union[PreTrainedModel, PeftModel]
AnyTypeVar = TypeVar('AnyTypeVar', bound=Any)
RewardFn = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


class InfPRMTree:
    def __init__(self, trajectories: List[torch.Tensor]):
        duplicates, trajectories = {}, [tuple(x.flatten().tolist()) for x in trajectories]

        for i, z in enumerate(trajectories):
            if z in duplicates.keys():
                duplicates[z].add(i)
            else:
                duplicates.update({z: {i}})

        trajectories = [(list(sorted(list(i))), list(z)) for z, i in duplicates.items()]
        max_len = max(len(z[1]) for z in trajectories)

        for z in trajectories:
            z[1].extend(None for _ in range(max_len - len(z)))

        def rec(in_list, idx):
            if len(in_list) == 1:
                return InfPRMNode(in_list[0][1][idx:], idx=in_list[0][0])

            prev_idx, same_char = idx, {None: None}
            idx -= 1

            while len(same_char.keys()) == 1:
                idx += 1
                same_char = {in_list[0][1][idx]: [in_list[0]]}

                for x in in_list[1:]:
                    if (x_pfx := x[1][idx]) in same_char.keys():
                        same_char[x_pfx].append(x)
                    else:
                        same_char.update({x_pfx: [x]})

            return InfPRMNode(in_list[0][1][prev_idx:idx], dtrs=[rec(x, idx) for x in same_char.values()])

        self.root = rec(trajectories, 0)

    def loss_scales(
            self,
            device: Union[torch.device, str, int] = 'cpu',
            dtype: Optional[torch.dtype] = None
    ) -> List[torch.Tensor]:
        if dtype is None:
            match device:
                case torch.device(index=None) | 'cpu':
                    dtype = torch.float32
                case _:
                    dtype = torch.bfloat16

        tensor_dict = self.root._loss_scale(torch.tensor((), dtype=dtype, device=device))

        return [tensor_dict[i] for i in sorted(list(tensor_dict.keys()))]


class InfPRMNode:
    def __init__(
            self,
            content: List[Optional[int]],
            idx: Optional[List[int]] = None,
            dtrs: Optional[List["InfPRMNode"]] = None
    ):
        self._len, self.dtrs = sum((1 for x in content if x is not None), start=0), dtrs

        if self.terminal:
            assert idx is not None
            self.idx = idx
        else:
            assert idx is None
            self.idx = []

            for d in self.dtrs:
                self.idx.extend(d.idx)

        self.idx.sort()

    @property
    def terminal(self) -> bool:
        return self.dtrs is None

    def __len__(self):
        return self._len

    def _loss_scale(self, pfx: torch.Tensor) -> Dict[int, torch.Tensor]:
        out_ten = torch.cat(
            (pfx, torch.full((len(self),), 1 / len(self.idx), dtype=pfx.dtype, device=pfx.device)), dim=0
        )

        if self.terminal:
            return {i: out_ten for i in self.idx}

        out_dict = {}

        for d in self.dtrs:
            out_dict.update(d._loss_scale(out_ten))

        return out_dict


class BoxedLP(LogitsProcessor):
    forced_eos: torch.Tensor
    batch_states: torch.Tensor
    batch_counts: List[Optional[int]]
    batch_idxs: List[int]

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(BoxedLP, self).__init__()
        self.forced_eos, self.batch_states, self.batch_counts, self.batch_idxs = None, None, None, None
        self.tokenizer = tokenizer

        if self.tokenizer.name_or_path.startswith('meta-llama/Llama-3.2'):
            self.split_str = '<|start_header_id|>assistant<|end_header_id|>'
        else:
            self.split_str = '<｜Assistant｜>'


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.forced_eos is None:
            self.forced_eos = torch.full((scores.size(1),), -float('inf'), device=scores.device, dtype=scores.dtype)
            self.forced_eos[self.tokenizer.eos_token_id] = 0

            batch_size = scores.size(0)
            self.batch_states = torch.full((batch_size,), False, device=scores.device)
            self.batch_counts, self.batch_idxs = [None for _ in range(batch_size)], [0 for _ in range(batch_size)]

        for i, ids in enumerate(input_ids):
            if not self.batch_states[i]:
                text = self.tokenizer.decode(ids).split(self.split_str, 1)[1]

                if self.batch_counts[i] is None and (re_match := re.search(r'\\boxed{', text)):
                    self.batch_counts[i], self.batch_idxs[i] = 1, re_match.span()[1]
                if self.batch_counts[i] is not None:
                    for c in text[self.batch_idxs[i]:]:
                        if c == '{':
                            self.batch_counts[i] += 1
                        elif c == '}':
                            self.batch_counts[i] -= 1
                        if self.batch_counts[i] == 0:
                            self.batch_states[i] = True
                            break

                    self.batch_idxs[i] = len(text)

        scores[self.batch_states] = self.forced_eos

        return scores

    @staticmethod
    def initialize(tokenizer: PreTrainedTokenizer) -> LogitsProcessorList:
        return LogitsProcessorList([BoxedLP(tokenizer)])


@contextmanager
def set_batch_size(model: Model, batch_size: int) -> Generator[None, None, None]:
    prev_bs = False

    try:
        if hasattr(model, 'config') and hasattr(model.config, 'batch_size'):
            prev_bs = model.config.batch_size
            model.config.batch_size = batch_size

        yield
    finally:
        if prev_bs is not False:  # False bc model.config.batch_size can be None
            model.config.batch_size = prev_bs


def identity(x: AnyTypeVar) -> AnyTypeVar:
    return x
