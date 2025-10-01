
import os
import json
import torch
import pickle
from peft import PeftModel
from argparse import ArgumentParser
from collections import OrderedDict
from lambda_grpo import LambdaGRPOTrainer
from open_rs_reward import accuracy_reward
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Optional, Union, Dict


class RewardFn:
    def __init__(self, pos_reward: float, neg_reward: float, tokenizer: PreTrainedTokenizer):
        self.tokenizer, self._rwd_map = tokenizer, {True: pos_reward, False: neg_reward}

    def __call__(self, pred: torch.Tensor, target: str) -> torch.Tensor:
        return torch.tensor(
            [self._rwd_map[accuracy_reward(self.tokenizer.decode(pred.flatten(), skip_special_tokens=True), target)]],
            device=pred.device
        )


def sd_to_cpu(
        state_dict: Union[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor]]
) -> OrderedDict[str, torch.Tensor]:
    sd_out = OrderedDict()

    for k, v in state_dict.items():
        sd_out.update({k: v.to('cpu')})

    return sd_out


def print_qrr(qrr: List[Tuple[str, float]]) -> None:
    print('\n\n----------------------------------')
    print('\n----------------------------------\n'.join(f'{x}\n({r})' for x, r in qrr))
    print('----------------------------------')
    print(f'\n\nMean reward: {round(sum(r for _, r in qrr) / len(qrr), 3)}')


def default_args() -> ArgumentParser:
    ap = ArgumentParser()
    ap.add_argument('-d', '--data_fp', type=str, default='')
    ap.add_argument('-f', '--save_fp', type=str, default=None)
    ap.add_argument('-i', '--model_id', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    ap.add_argument('-e', '--epochs', type=int, default=1)
    ap.add_argument('-v', '--val_rate', type=int, default=None)
    ap.add_argument('-s', '--step_size', type=int, default=4)
    ap.add_argument('-g', '--generation_batch_size', type=int, default=4)
    ap.add_argument('-a', '--grad_acc_batch_size', type=int, default=4)
    ap.add_argument('-t', '--train_batch_size', type=int, default=None)
    ap.add_argument('-u', '--update_steps', type=int, default=1)
    ap.add_argument('-l', '--learn_rate', type=float, default=1e-6)
    ap.add_argument('-w', '--weight_decay', type=float, default=0.0)
    ap.add_argument('-b', '--beta', type=float, default=0.0)
    ap.add_argument('-m', '--max_new_tokens', type=int, default=1200)
    ap.add_argument('--hf_access_token', type=str, default=None)
    ap.add_argument('--group_size', type=int, default=6)
    ap.add_argument('--no_loss_scale', action='store_false')
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--temperature', type=float, default=0.75)
    ap.add_argument('--no_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=64)
    ap.add_argument('--lora_alpha', type=int, default=128)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--lora_layers', type=str, default='all', help='\'all\', \'qv\', \'qvko\'')
    ap.add_argument('--loss_type', type=str, default='grpo', help='\'grpo\', \'dapo\'')
    ap.add_argument('--load', type=str, default=None)
    ap.add_argument(
        '--system_prompt',
        type=str,
        default='Let\'s think step by step and output the final answer within \\boxed{}.'
    )
    ap.add_argument('--initial_val', action='store_true')

    return ap


def validate_and_checkpoint(
        epoch: int,
        trainer: LambdaGRPOTrainer,
        model: PeftModel,
        optimizer: torch.optim.Optimizer,
        stats: dict,
        val_prompts: List[str],
        val_answers: List[str],
        generation_batch_size: int,
        max_new_tokens: int,
        save_fp: Optional[str],
        step: int = -1,
        idxs: Optional[List[str]] = None,
        del_opt: bool = False
) -> None:
    stats['epochs'][-1]['validation'].append(trainer.validate(
        val_prompts,
        targets=val_answers,
        batch_size=generation_batch_size,
        generation_kwargs={'max_new_tokens': max_new_tokens}
    ))
    print_qrr(stats['epochs'][-1]['validation'][-1])

    if save_fp is not None:
        if del_opt:
            del_opt_list = [fn for fn in os.listdir(save_fp) if fn.startswith('optimizer_') and fn.endswith('.chk')]
        else:
            del_opt_list = []

        with open(f'{save_fp}stats.json', 'w') as f:
            json.dump(stats, f)

        with open(f'{save_fp}optimizer_{epoch}_{step}.chk', 'wb') as f:
            pickle.dump(optimizer.state_dict(), f)

        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(f'{save_fp}model_{epoch}_{step}')
        else:
            with open(f'{save_fp}model_{epoch}_{step}.chk', 'wb') as f:
                pickle.dump(model.state_dict(), f)

        if idxs is not None:
            with open(f'{save_fp}idxs.json', 'w') as f:
                json.dump(idxs, f)

        for fn in del_opt_list:
            os.remove(save_fp + fn)


LORA_LAYER_MAP = {'all': 'all-linear', 'qv': None, 'qvko': ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
