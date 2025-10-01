
import torch
import random
from tqdm import tqdm
from peft import PeftModel
from itertools import starmap
from contextlib import contextmanager
from typing import Optional, Union, Tuple, List, Any, Dict, Generator, Iterable, Literal
from lambda_grpo.utils import Model, BoxedLP, RewardFn, InfPRMTree, set_batch_size, identity
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorForLanguageModeling, BatchEncoding


_sample_rtn_type = Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[float],
    List[torch.Tensor],
    List[torch.Tensor],
    List[int]
]


class LambdaGRPOTrainer:
    def __init__(
            self,
            model: Model,
            tokenizer: PreTrainedTokenizer,
            optimizer: torch.optim.Optimizer,
            reward_function: RewardFn,
            ref_model: Optional[PreTrainedModel] = None,
            system_prompt: Optional[str] = None,
            group_size: int = 6,
            generation_kwargs: Optional[Dict[str, Any]] = None,
            generation_batch_size: int = 1,
            beta: float = 0.04,
            epsilon: float = 0.2,
            grad_acc_batch_size: int = 1,
            batch_size: Optional[int] = None,
            batch_steps: int = 1,
            loss_scale: bool = True,
            device: Optional[Union[str, int]] = None,
            loss_type: Literal['grpo', 'dapo'] = 'grpo'
    ):
        self.reward_fn, self.data_collator = reward_function, DataCollatorForLanguageModeling(tokenizer, mlm=False)
        self.model, self.ref_model, self.tokenizer, self.optimizer = model, ref_model, tokenizer, optimizer
        self.batch_size, self.batch_steps, self.system_prompt, self.beta = batch_size, batch_steps, system_prompt, beta
        self.grad_acc_batch_size, self.device, self.loss_type = grad_acc_batch_size, device, loss_type
        self.loss_scale, self._eps_min, self._eps_max = loss_scale, 1 - epsilon, 1 + epsilon
        self.generation_batch_size, self.group_size = generation_batch_size, group_size
        self.generation_kwargs = {
            **{'do_sample': True, 'top_k': 0, 'top_p': 1.0, 'temperature': 0.75}, **(generation_kwargs or {})
        }
        self.optimizer.zero_grad()


        if isinstance(self.model, PeftModel):
            assert self.ref_model is None
        elif self.ref_model is None:
            assert self.beta == 0.0
        else:
            assert self.beta > 0.0
            self.ref_model.eval()

    @torch.no_grad()
    def validate(
            self,
            prompts: List[str],
            targets: Optional[List[Any]] = None,
            batch_size: int = 1,
            use_tqdm: bool = True
    ) -> List[Tuple[str, float]]:
        self.model.eval()

        targets, qrr = (([None] * len(prompts)) if targets is None else targets), []
        gk = {**self.generation_kwargs, **{'do_sample': False}}
        tqdm_ = tqdm if use_tqdm else identity

        for i in tqdm_(range(-(-len(prompts) // batch_size))):
            batch_p = prompts[i * batch_size:(i + 1) * batch_size]

            with set_batch_size(self.model, len(batch_p)):
                out_toks = self.model.generate(
                    logits_processor=BoxedLP.initialize(self.tokenizer), **self._build_prompts(batch_p), **gk
                )
                qrr.extend(zip(
                    self.tokenizer.batch_decode(out_toks, skip_special_tokens=True),
                    starmap(self.reward_fn, zip(out_toks, targets[i * batch_size:(i + 1) * batch_size]))
                ))

        return [(x, y.item()) for x, y in qrr]

    def __call__(
            self,
            prompts: List[str],
            targets: Optional[List[Any]] = None,
            use_tqdm: bool = True
    ) -> Dict[str, Union[List[Tuple[str, float]], float]]:
        advs, input_ids, rewards, *step_args = self._sample(prompts, targets, use_tqdm)
        loss_scales = []

        for i in range(len(input_ids) // self.group_size):  # group size
            s, e = i * self.group_size, (i + 1) * self.group_size
            loss_scales.extend(
                x[-y.size(-1):].unsqueeze(0) for x, y in
                zip(InfPRMTree(input_ids[s:e]).loss_scales(device=advs[0].device, dtype=advs[0].dtype), advs[s:e])
            )

        train_idxs = list(range(len(input_ids)))
        step_args = [use_tqdm, train_idxs, advs, loss_scales, input_ids] + step_args

        self.model.train()
        random.shuffle(train_idxs)
        loss, base_logits = self._step(*step_args)

        for _ in range(self.batch_steps - 1):
            random.shuffle(train_idxs)
            loss += self._step(*step_args, base_logits=base_logits)

        return {
            # 'qrr': list(zip((self.tokenizer.decode(c.flatten()) for c in completions), rewards)),
            'qrr': list(zip((self.tokenizer.decode(c.flatten()) for c in input_ids), rewards)),
            'loss': loss / self.batch_steps
        }

    def _sample(
            self, prompts: List[str], targets: Optional[List[Any]], use_tqdm: bool
    ) -> _sample_rtn_type:
        advs, input_ids, attn_masks, n_logits, ref_logits = [], [], [], [], []
        rewards, k = [[] for _ in range(len(prompts))], 0
        prompt_targets = [
            x for x in (((p_, None) for p_ in prompts) if targets is None else zip(prompts, targets, strict=True))
            for _ in range(self.group_size)
        ]
        tqdm_ = tqdm if use_tqdm else (lambda x: x)
        self.model.eval()

        with torch.no_grad():
            for i in tqdm_(range(len(prompt_targets) // self.generation_batch_size)):
                batch_pt = prompt_targets[i * self.generation_batch_size:(i + 1) * self.generation_batch_size]
                p_input = self._build_prompts([p for p, _ in batch_pt])
                n_logits_step = [(x != self.tokenizer.pad_token_id).sum().item() for x in p_input['input_ids']]
                input_ids_step, attn_masks_step = [], []

                with set_batch_size(self.model, len(batch_pt)):
                    model_out = self.model.generate(
                        logits_processor=BoxedLP.initialize(self.tokenizer), **p_input, **self.generation_kwargs
                    )

                for (p, t), y in zip(batch_pt, model_out):
                    input_ids_step.append(y[y != self.tokenizer.pad_token_id].flatten())
                    attn_masks_step.append(torch.ones_like(input_ids_step[-1]))
                    rewards[k // self.group_size].append(self.reward_fn(input_ids_step[-1].unsqueeze(0), t).flatten())
                    k += 1

                max_len = max(x.size(-1) for x in input_ids_step)
                input_data_step = {
                    'input_ids': _pad_stack(input_ids_step, max_len, pad_val=self.tokenizer.pad_token_id),
                    'attention_masks': _pad_stack(attn_masks_step, max_len)
                }
                input_ids.extend(input_ids_step)
                attn_masks.extend(attn_masks_step)
                n_logits.extend(n_logits_step)

                with self._ref_logit_manager() as ref_model:
                    if ref_model is not None:
                        with set_batch_size(ref_model, len(input_ids_step)):
                            ref_logits.extend(self._strip_padding(
                                input_data_step['input_ids'], _forward_pass(ref_model, input_data_step), n_logits_step
                            ))

            k = 0

            for i, r_g in enumerate(rewards):
                r_g, rewards[i] = (torch.cat(r_g, dim=0),) * 2
                r_mean, r_std = r_g.mean(dim=0), r_g.std(dim=0) + 1e-4

                for r in r_g:
                    adv_k = torch.full(
                        (1, input_ids[k].size(-1) - n_logits[k]),
                        ((r - r_mean) / r_std).item(),
                        device=input_ids[k].device
                    )
                    advs.append(adv_k)
                    k += 1

        return advs, input_ids, torch.cat(rewards, dim=0).tolist(), attn_masks, ref_logits, n_logits

    def _step(
            self,
            use_tqdm: bool,
            indices: List[int],
            advantages: List[torch.Tensor],
            loss_scales: List[torch.Tensor],
            input_ids: List[torch.Tensor],
            attention_masks: List[torch.Tensor],
            ref_logits: List[torch.Tensor],
            n_logits: List[int],
            base_logits: Optional[List[torch.Tensor]] = None
    ) -> Union[float, Tuple[float, Union[List[torch.Tensor], bool]]]:
        batch_size = len(indices) if self.batch_size is None else self.batch_size
        step_loss, tqdm_ = 0.0, (tqdm if use_tqdm else (lambda z: z))

        if base_logits is None:
            if self.batch_steps > 1:
                base_logits_out = [None for _ in range(len(n_logits))]
            else:  # don't bother if we're not going to use
                base_logits_out, base_logits = True, False  # base_logits_out = True for consistent num of rtn vals
        else:
            base_logits_out = False

        for i in tqdm_(range(-(-len(indices) // batch_size))):
            batch_idxs = indices[i * batch_size:(i + 1) * batch_size]

            for k in range(-(-len(batch_idxs) // self.grad_acc_batch_size)):
                mini_batch_idxs = batch_idxs[k * self.grad_acc_batch_size:(k + 1) * self.grad_acc_batch_size]
                max_len = max(input_ids[j].size(-1) for j in mini_batch_idxs)
                logits_to_remove = min(n_logits[j] + (max_len - input_ids[j].size(-1)) for j in mini_batch_idxs)
                input_data_mb = {
                    'input_ids': _pad_stack(
                        (input_ids[j] for j in mini_batch_idxs), max_len, pad_val=self.tokenizer.pad_token_id
                    ),
                    'attention_masks': _pad_stack((attention_masks[j] for j in mini_batch_idxs), max_len)
                }

                with set_batch_size(self.model, len(mini_batch_idxs)):
                    policy_logits_mb = _forward_pass(
                        self.model, input_data_mb, n=(1 if base_logits is None else logits_to_remove)
                    )

                if base_logits:
                    base_logits_mb = _pad_stack((base_logits[j] for j in mini_batch_idxs), policy_logits_mb.size(-1))
                else:
                    base_logits_mb = policy_logits_mb.detach()

                    if base_logits is None:  # base_logits = False if we don't need to return them
                        bl_stripped = self._strip_padding(
                            input_data_mb['input_ids'], base_logits_mb, (n_logits[j] for j in mini_batch_idxs)
                        )
                        policy_logits_mb = policy_logits_mb[:, logits_to_remove - 1:]
                        base_logits_mb = base_logits_mb[:, logits_to_remove - 1:]

                        for j, x in enumerate(bl_stripped):
                            base_logits_out[mini_batch_idxs[j]] = x

                advs_mb = _pad_stack((advantages[j] for j in mini_batch_idxs), policy_logits_mb.size(-1))
                ls_mb = _pad_stack((loss_scales[j] for j in mini_batch_idxs), policy_logits_mb.size(-1))
                token_ratio = torch.exp(policy_logits_mb - base_logits_mb)
                loss = token_ratio * advs_mb

                if base_logits is not None:  # don't clip if we know token_ratio = 1
                    loss = torch.min(torch.clamp(token_ratio, min=self._eps_min, max=self._eps_max) * advs_mb, loss)
                if self.beta > 0.0:
                    log_ref_ratio = _pad_stack(
                        (ref_logits[j] for j in mini_batch_idxs),  policy_logits_mb.size(-1)
                    ) - policy_logits_mb
                    loss -= (torch.exp(log_ref_ratio) - log_ref_ratio - 1) * self.beta

                loss_mask = ls_mb > 0
                loss = -loss * (ls_mb if self.loss_scale else loss_mask)

                if self.loss_type == 'dapo':
                    loss = loss.sum() / loss_mask.sum()
                else:
                    loss = (loss.sum(dim=1) / loss_mask.sum(dim=1)).mean()

                loss *= advs_mb.size(0) / len(batch_idxs)
                step_loss += loss.item()
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        return (step_loss, base_logits_out) if base_logits_out else step_loss

    @contextmanager
    def _ref_logit_manager(self):
        try:
            if self.ref_model is None and self.beta > 0.0:
                with self.model.disable_adapter():
                    yield self.model
            else:
                yield self.ref_model
        finally:
            pass

    def _strip_padding(
            self,
            input_ids: torch.Tensor,
            logits: torch.Tensor,
            n_logits: Iterable[int]
    ) -> Generator[torch.Tensor, None, None]:
        offset = input_ids.size(-1) - logits.size(-1)

        for ids_i, logits_i, n in zip(torch.unbind(input_ids, dim=0), torch.unbind(logits, dim=0), n_logits):
            yield logits_i[(n + (ids_i == self.tokenizer.pad_token_id).sum().item()) - offset:].unsqueeze(0)

    def _build_prompts(self, prompts: List[str]) -> BatchEncoding:
        if self.system_prompt is None:
            template = [[{'role': 'user', 'content': p}] for p in prompts]
        else:
            template = [
                [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': p}] for p in prompts
            ]

        template = self.tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)

        if template[0].startswith('<｜begin▁of▁sentence｜>'):  # deepseek :(
            template = [x.split('<｜begin▁of▁sentence｜>', 1)[1] for x in template]
        elif template[0].startswith('<|begin_of_text|>'):  # llama :(
            template = [x.split('<|begin_of_text|>', 1)[1] for x in template]

        return self.tokenizer(template, padding=True, return_tensors='pt').to(device=self.device)


def _pad_stack(in_tensors: Iterable[torch.Tensor], max_len: int, pad_val: int = 0) -> torch.Tensor:
    return torch.cat(tuple(
        torch.cat((
            torch.full((1, max_len - t.size(-1)), pad_val, device=t.device, dtype=t.dtype), t.view(1, t.size(-1))
        ), dim=1) for t in in_tensors
    ), dim=0)


def _forward_pass(model: Model, batch_inputs: Dict[str, torch.Tensor], n: int = 1):
    return torch.stack(tuple(
        torch.gather(logits_row.log_softmax(dim=-1), dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        for logits_row, input_ids_row in zip(model(**batch_inputs).logits[:, :-1, :], batch_inputs['input_ids'][:, 1:])
    ))[:, (n - 1):]
