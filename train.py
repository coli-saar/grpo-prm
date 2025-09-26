
import os
import json
import torch
import random
import pickle
from lambda_grpo import LambdaGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from train_utils import LORA_LAYER_MAP, RewardFn, print_qrr, default_args, validate_and_checkpoint, sd_to_cpu


args = default_args().parse_args()
load_fp, stats = None, None

if args.load is not None:
    assert args.save_fp is None
    load_fp = os.path.abspath(args.load).rstrip('/') + '/'

    with open(f'{load_fp}stats.json', 'r') as f:
        stats = json.load(f)

    for k, v in stats['args'].items():
        setattr(args, k, v)

    save_fp = load_fp
elif args.save_fp is not None:
    save_fp = os.path.abspath(args.save_fp).rstrip('/')
    l_save, r_save = save_fp.rsplit('/', 1)

    if r_save in os.listdir(l_save):
        raise ValueError(save_fp)

    os.mkdir(save_fp)
    save_fp += '/'
else:
    save_fp = None

train_prompts, train_answers, val_prompts, val_answers = [], [], [], []

for t, p, a in (('train', train_prompts, train_answers), ('val', val_prompts, val_answers)):
    with open(f'{args.data_fp}{t}.json', 'r') as f:
        data_raw = json.load(f)
        p.extend(x['problem'] for x in data_raw)
        a.extend(str(x['solution']) for x in data_raw)

model = AutoModelForCausalLM.from_pretrained(args.model_id, token=args.hf_access_token, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_access_token)
tokenizer.padding_side = 'left'
ref_model = None

if args.model_id.startswith('meta-llama/Llama-3.2'):
    tokenizer.pad_token_id, tokenizer.pad_token = 128001, '<|end_of_text|>'
elif tokenizer.pad_token is None:
    tokenizer.pad_token_id, tokenizer.pad_token = tokenizer.eos_token_id, tokenizer.eos_token

if not args.no_lora:
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=LORA_LAYER_MAP[args.lora_layers]
        )
    )
elif not args.beta == 0.0:
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto')

if args.weight_decay == 0.0:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)

grpo_trainer = LambdaGRPOTrainer(
    model,
    tokenizer,
    optimizer,
    RewardFn(1.0, 0.0, tokenizer),
    ref_model=ref_model,
    system_prompt=args.system_prompt,
    group_size=args.group_size,
    beta=args.beta,
    generation_kwargs={
        'max_new_tokens': args.max_new_tokens,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature
    },
    generation_batch_size=args.generation_batch_size,
    grad_acc_batch_size=args.grad_acc_batch_size,
    batch_size=args.train_batch_size,
    batch_steps=args.update_steps,
    device=0,
    loss_type=args.loss_type,
    loss_scale=args.no_loss_scale,
)
n_step = len(train_prompts) // args.step_size

if load_fp is None:
    if args.initial_val:
        init_val = grpo_trainer.validate(
            val_prompts,
            targets=val_answers,
            batch_size=args.generation_batch_size,
            generation_kwargs={'max_new_tokens': args.max_new_tokens}
        )
        print_qrr(init_val)
    else:
        init_val = None

    arg_dict = dict(args._get_kwargs())
    arg_dict.pop('load', None)
    arg_dict.pop('save', None)
    arg_dict.pop('save_fp', None)
    stats = {'args': arg_dict, 'init_val': init_val, 'epochs': []}

    train_indices = list(range(len(train_prompts)))
    random.shuffle(train_indices)
    max_ep, max_step = 0, 0

    if save_fp is not None:
        with open(f'{save_fp}idxs.json', 'w') as f:
            json.dump(train_indices, f)
else:
    max_step, max_ep = len(stats['epochs'][-1]['steps']), len(stats['epochs']) - 1

    if max_step == n_step:
        max_step = 0
        max_ep += 1

    if args.no_lora:  # TODO: hacky af
        base_model = AutoModelForCausalLM.from_pretrained(f'{load_fp}model_{max_ep}_{max_step - 1}')
        model.load_state_dict(base_model.state_dict())
        del base_model
    else:
        set_peft_model_state_dict(model, sd_to_cpu(load_peft_weights(f'{load_fp}model_{max_ep}_{max_step - 1}')))

    with open(f'{load_fp}optimizer_{max_ep}_{max_step - 1}.chk', 'rb') as f:
        optimizer.load_state_dict(pickle.load(f))

    with open(f'{load_fp}idxs.json', 'r') as f:
        train_indices = json.load(f)

for e in range(max_ep, args.epochs):  # training loop
    if max_step == 0:
        stats['epochs'].append({'steps': [], 'validation': []})

    for s in range(max_step, n_step):
        step_idxs = train_indices[s * args.step_size:(s + 1) * args.step_size]
        step_stats = grpo_trainer([train_prompts[i] for i in step_idxs], targets=[train_answers[i] for i in step_idxs])
        stats['epochs'][-1]['steps'].append({
            'instances': [
                {'idx': i, 'query_response': x, 'reward': r} for k, i in enumerate(step_idxs)
                for x, r in step_stats['qrr'][k * args.group_size:(k + 1) * args.group_size]
            ],
            'loss': step_stats['loss']
        })

        print_qrr(step_stats['qrr'])
        print(f'Mean loss: {round(step_stats["loss"], 4)}\n(Step {s + 1}/{n_step})')

        if (args.val_rate is not None) and ((s + 1) % args.val_rate == 0):
            validate_and_checkpoint(
                e,
                grpo_trainer,
                model,
                optimizer,
                stats,
                val_prompts,
                val_answers,
                args.generation_batch_size,
                args.max_new_tokens,
                save_fp,
                step=s,
                del_opt=True
            )

    max_step = 0
    random.shuffle(train_indices)

    if (args.val_rate is None) or not n_step % args.val_rate == 0:
        validate_and_checkpoint(
            e + 1,
            grpo_trainer,
            model,
            optimizer,
            stats,
            val_prompts,
            val_answers,
            args.generation_batch_size,
            args.max_new_tokens,
            save_fp,
            idxs=train_indices,
            del_opt=True
        )