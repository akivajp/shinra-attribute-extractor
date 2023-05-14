#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os

from collections import OrderedDict

import datasets
from datasets import (
    load_dataset,
)

import evaluate

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import (
    DataLoader,
)
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

import logzero
from logzero import logger

from preprocess import (
    preprocess_for_training,
    prepare_for_prediction,
)

from model import (
    TokenMultiClassificationModel,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description=
            "Finetune a transformers model on a text classification task (NER)"
            "with accelerate library",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    #parser.add_argument(
    #    "--validation_file", type=str, default=None,
    #    help="A csv or a json file containing the validation data.",
    #)
    parser.add_argument(
        "--window_size",
        type=int,
        #default=512,
        default=510,
        help=(
            "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be splitted into multiple windows."
        )
    )
    parser.add_argument(
        "--window_overlap_size",
        type=int,
        default=128,
        help=(
            "The overlap size of consecutive windows."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    #parser.add_argument(
    #    "--config_name",
    #    type=str,
    #    default=None,
    #    help="Pretrained config name or path if not the same as model_name",
    #)
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Common batch size (per device) for the training/evaluation/prediction.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        #default=8,
        default=None,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        #default=8,
        default=None,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_predict_batch_size",
        type=int,
        #default=8,
        default=None,
        help="Batch size (per device) for the prediction dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", type=int,
        #default=3,
        default=10,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
            "Total number of training steps to perform. "
            "If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        #default=None,
        required=True,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        #default=None,
        default=42,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        #default=None,
        default=500,
        #default=100,
        help=
            "The various states should be saved at the end of every n steps."
            
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether or enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help=(
            "Whether or not to enable to load a pretrained model "
            "whose head dimensions are different."
        ),
    )

    parser.add_argument(
        "--target_categories",
        type=str,
        help=(
            "Comma separated list of target categories to train on. "
        ),
    )

    parser.add_argument(
        "--split_eval_size",
        type=float,
        default=0.1,
        help=(
            "A float value between 0 and 1 indicating "
            "the ratio of the splitted eval dataset size"
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        #default=1,
        default=None,
        help=(
            "The number of workers to use for the dataset processing."
        ),
    )

    parser.add_argument(
        "--predict_input_jsonl",
        type=str,
        default=None,
        help=(
            "Path to the input jsonl file for prediction."
        ),
    )
    parser.add_argument(
        "--predict_html_dir",
        type=str,
        default=None,
        help=(
            "Path to the directory containing the html files for prediction."
        ),
    )
    parser.add_argument(
        "--predict_output_jsonl",
        type=str,
        default=None,
        help=(
            "Path to the output jsonl file for prediction."
        ),
    )

    args = parser.parse_args()

    if args.do_predict:
        if args.predict_input_jsonl is None:
            raise ValueError(
                "You must specify a `predict_input_jsonl` file for prediction."
            )
        if args.predict_html_dir is None:
            raise ValueError(
                "You must specify a `predict_html_dir` for prediction."
            )
        if args.predict_output_jsonl is None:
            raise ValueError(
                "You must specify a `predict_output_jsonl` file for prediction."
            )
        
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = args.per_device_batch_size
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_batch_size
    if args.per_device_predict_batch_size is None:
        args.per_device_predict_batch_size = args.per_device_batch_size

    if args.num_workers is None:
        args.num_workers = 2

    return args

def main():
    args = parse_args() 
    logger.debug('args: %s', args)

    # Initialize the accelerator.
    # We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here
    # and it will by default pick up all supported trackers in the environment
    accelerator = (
        #Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking
        Accelerator(log_with='all', logging_dir=args.output_dir) if args.with_tracking
        else Accelerator()
    )

    # Make one log on every process with the configuration for debugging.
    if accelerator.is_main_process:
        logger.info('accelerator.state: %s', accelerator.state)

    # ロギング設定
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.do_train and accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            logfile = os.path.join(args.output_dir, 'training.log')
            logzero.logfile(logfile)
    accelerator.wait_for_everyone()

    raw_dataset = load_dataset('./shinra_attribute_extraction_2022')
    logger.debug('raw_dataset: %s', raw_dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    preprocess_result = preprocess_for_training(
        raw_dataset,
        tokenizer,
        target_categories = args.target_categories,
        num_workers = args.num_workers,
        split_eval_size = args.split_eval_size,
        seed = args.seed,
        window_size = args.window_size,
        window_overlap_size = args.window_overlap_size,
    )
    dataset = preprocess_result.dataset
    mapping = preprocess_result.mapping

    if args.do_predict:
        #with open(args.predict_input_jsonl, 'r', encoding='utf-8') as f:
        #    predict_input = [json.loads(line) for line in f]

        predict_dataset = prepare_for_prediction(
            args.predict_input_jsonl,
            args.predict_html_dir,
            mapping,
            tokenizer,
            num_workers = args.num_workers,
            window_size = args.window_size,
            window_overlap_size = args.window_overlap_size,
        )
        logger.debug('predict_dataset: %s', predict_dataset)

        #sys.exit(1)

    num_attribute_names = len(mapping.attribute_names)
    tag_list = mapping.tag_list

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        #num_labels=num_labels,
        num_labels=len(tag_list)
        #num_labels=len(preprocess_result.tag_list),
    )

    model = TokenMultiClassificationModel(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config,
        num_attribute_names = num_attribute_names,
    )
    logger.debug('model: %s', model)

    model.config.id2label = mapping.tag_list
    model.config.label2id = mapping.map_tag_to_id

    logger.debug('new model.config: %s', model.config)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    #resuming = False
    skip_collation = False
    # ミニバッチ生成のための追加処理
    def my_collator(examples):
        # 全タグのtensor生成もそこそこコストが高いので、
        # resuming中に無駄なtensorを生成させない
        #if resuming:
        #    return
        if skip_collation:
            return
        input_ids = [
            torch.tensor(example['input_ids'], dtype=torch.int64)
            for example in examples
        ]
        if 'tag_ids' in examples[0]:
            # cuda の nll_loss のターゲットは long にしか対応していない模様
            tag_ids = [
                [
                    tag_id_list if tag_id_list is not None
                    else [-100] * len(example['input_ids'])
                    for tag_id_list in example['tag_ids']
                ]
                for example in examples
            ] # [batch_size, num_attribute_names, window_length]
            #tag_ids = [
            #    torch.tensor(example['tag_ids'], dtype=torch.int64).transpose(1, 0)
            #    for example in examples
            #]
            tag_ids = [
                torch.tensor(tag_id_list, dtype=torch.int64).transpose(1, 0) # [A, L] -> [L, A]
                for tag_id_list in tag_ids
            ] # [batch_size, window_length, num_attribute_names]
            tag_ids = pad_sequence(tag_ids, batch_first=True, padding_value=-100)
        else:
            tag_ids = None
        mask = [
            torch.ones(len(ids), dtype=torch.bool) for ids in input_ids
        ]
        # indices には int か long のみ対応
        token_type_ids = [
            torch.zeros(len(ids), dtype=torch.int32) for ids in input_ids
        ]
        return {
            'input_ids': pad_sequence(input_ids, batch_first=True),
            'tag_ids': tag_ids,
            'attention_mask': pad_sequence(mask, batch_first=True),
            'token_type_ids': pad_sequence(token_type_ids, batch_first=True),
            'page_id': [example['page_id'] for example in examples],
            'window_id': [example['window_id'] for example in examples],
            'ENE': [example['ENE'] for example in examples],
        }
    data_collator = my_collator
        
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    logger.debug('train_dataloader: %s', train_dataloader)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    logger.debug('optimizer: %s', optimizer)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader
    # may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    #if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #    checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("ner_no_trainer", experiment_config)

    # Metrics
    metric = evaluate.load("seqeval")

    def get_labels(predictions, references):
        # Transform predictions and references tensors to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        batch_size = y_pred.shape[0]
        seq_len = y_pred.shape[1]
        # [B, L, A] -> [B, A, L]
        #y_pred = y_pred.transpose(0, 2, 1)
        #y_true = y_true.transpose(0, 2, 1)
        # [B, L, A] -> [B, A, L] -> [B*A, L]
        y_pred = y_pred.transpose(0, 2, 1).reshape(batch_size*num_attribute_names, seq_len)
        y_true = y_true.transpose(0, 2, 1).reshape(batch_size*num_attribute_names, seq_len)
        # [B, L, A] -> [A, B, L]

        # Remove ignored index (special tokens)
        #true_predictions = [
        #    [tag_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        #    for pred, gold_label in zip(y_pred, y_true)
        #    if gold_label[1] != -100
        #]
        #true_labels = [
        #    [tag_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        #    for pred, gold_label in zip(y_pred, y_true)
        #    if gold_label[1] != -100
        #]
        #return true_predictions, true_labels
        true_pred = []
        true_labels = []
        for pred, gold_label in zip(y_pred, y_true):
            if gold_label[1] == -100:
                # 2番目のトークンが分類不要なら系列ごと分類不要
                continue
            seq_pred = []
            seq_labels = []
            for p, l in zip(pred, gold_label):
                if l == -100:
                    # 分類不要なら無視
                    continue
                seq_pred.append(tag_list[p])
                seq_labels.append(tag_list[l])
            true_pred.append(seq_pred)
            true_labels.append(seq_labels)
        return true_pred, true_labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    resume_path = None  
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        resume_path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = [
            os.path.join(args.output_dir, f.name)
            for f in os.scandir(args.output_dir) if f.is_dir()
        ]
        dirs = [
            d for d in dirs if os.path.isfile(os.path.join(d, "status.json"))
        ]
        if len(dirs) > 0:
            dirs.sort(key=os.path.getmtime)
            resume_path = dirs[-1] # Sorts folders by date modified, most recent checkpoint is the last
            accelerator.load_state(resume_path)
    logger.debug('resume_path: %s', resume_path)

    def save_best_status(eval_metric):
        status['last_epoch'] = epoch
        status['last_step'] = completed_steps
        for key in ['precision', 'recall', 'f1']:
            last_key = f'eval_{key}_last'
            best_key = f'eval_{key}_best'
            best_step_key = f'eval_{key}_best_step'
            status[last_key] = eval_metric[key]
            if best_key not in status or status[last_key] > status[best_key]:
                status[best_key] = status[last_key]
                status[best_step_key] = status['last_step']
                logger.debug('New best %s: %f', key, status[best_key])
                output_dir = f'best_eval_{key}'
                do_save(output_dir)
        with open(status_path, "w", encoding='utf-8') as f:
            json.dump(status, f, indent=2, sort_keys=True, ensure_ascii=False)

    def do_eval():
        model.eval()
        samples_seen = 0
        #for step, batch in enumerate(eval_dataloader):
        for step, batch in enumerate(tqdm(
            eval_dataloader,
            desc='Feeding for evaluation',
        )):
            with torch.no_grad():
                #outputs = model(**batch)
                #logger.debug('model.device: %s', model.device)
                #logger.debug('batch input_ids device: %s', batch['input_ids'].device)
                #logger.debug('batch tag_ids device: %s', batch['tag_ids'].device)
                outputs = model(
                    input_ids = batch['input_ids'],
                    tag_ids = batch['tag_ids'],
                    attention_mask = batch.get('attention_mask'),
                    token_type_ids = batch.get('token_type_ids'),
                )
            predictions = outputs.logits.argmax(dim=-1)
            tag_ids = batch["tag_ids"]
            predictions_gathered, tags_gathered = accelerator.gather([predictions, tag_ids])
            # If we are in a multi-process environment,
            # the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions_gathered = \
                        predictions_gathered[:, len(eval_dataloader.dataset) - samples_seen]
                    #labels_gathered = \
                        #labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
                    tags_gathered = \
                        tags_gathered[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    #samples_seen += labels_gathered.shape[0]
                    samples_seen += tags_gathered.shape[0]
            #preds, refs = get_labels(predictions_gathered, labels_gathered)
            preds, refs = get_labels(predictions_gathered, tags_gathered)
            # predictions and references are expected to be a nested list of labels, not label_ids
            metric.add_batch(
                predictions=preds,
                references=refs,
            )
        logger.debug('Computing metrics...')
        eval_metric = compute_metrics()
        logger.debug('Finished computing metrics')
        #accelerator.print(f"epoch {epoch}:", eval_metric)
        if args.do_train:
            if accelerator.is_main_process:
                logger.info(f'epoch {epoch} eval_metric: {eval_metric}')
            save_best_status(eval_metric)
        return eval_metric

    def do_save(output_dir):
        status['last_epoch'] = epoch
        status['last_step'] = completed_steps
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            status_path = os.path.join(output_dir, 'status.json')
            with open(status_path, "w", encoding='utf-8') as f:
                json.dump(status, f, indent=2, sort_keys=True, ensure_ascii=False)

    #def extract(f_out, html_lines, start_line, start_offset, end_line, end_offset): 
    #def extract(f_out, html_lines, html_offset):
    def extract(f_out, html_lines, offsets):
        text = ''
        start_line = offsets['start_line']
        start_offset = offsets['start_offset']
        end_line = offsets['end_line']
        end_offset = offsets['end_offset']
        for line_index in range(start_line, end_line+1):
            if start_line == end_line:
                line = html_lines[line_index][start_offset:end_offset]
            # (start_line < end_line)
            elif line_index == start_line:
                line = html_lines[line_index][start_offset:]
            elif line_index == end_line:
                line = html_lines[line_index][:end_offset]
            else:
                line = html_lines[line_index]
            text += line
        data = {}
        data['page_id'] = page_id
        data['title'] = title
        data['ENE'] = ene
        data['attribute'] = attribute_name
        data['html_offset'] = {
            #'start_line': start_line,
            #'start_offset': start_offset,
            #'end_line': end_line,
            #'end_offset': end_offset,
            'start': {
                'line_id': start_line,
                'offset': start_offset,
            },
            'end': {
                'line_id': end_line,
                'offset': end_offset,
            },
            'text': text,
        }
        #data['html_offset'] = html_offset
        #logger.debug('found: %s', data)
        f_out.write(json.dumps(data, ensure_ascii=False))
        f_out.write('\n')

    if args.do_train:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.max_train_steps)
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        resume_path = None

        starting_epoch = 0
        resume_step = -1
        eval_metric = None

        status_path = os.path.join(args.output_dir, "status.json")
        status = {}
        if resume_path:
            with open(os.path.join(resume_path, 'status.json'), "r", encoding='utf-8') as f:
                status = json.load(f)
                logger.debug('Loaded status: %s', status)
                starting_epoch = status['last_epoch']
                resume_step = status['last_step']
                resume_step -= starting_epoch * len(train_dataloader)
        elif os.path.isfile(status_path):
            with open(status_path, "r", encoding='utf-8') as f:
                status = json.load(f)
                logger.debug('Loaded status: %s', status)

        # Extract `epoch_{i}` or `step_{i}`
        #if resume_path:
        #    training_difference = os.path.splitext(resume_path)[0]
        #    logger.debug('training_difference: %s', training_difference)

        #    if "epoch_" in training_difference and "_step_" in training_difference:
        #        resume_step = int(training_difference.split("_")[-1])
        #        starting_epoch = resume_step // len(train_dataloader)
        #        resume_step -= starting_epoch * len(train_dataloader)
        #    else:
        #        raise ValueError(f'Failed to parse training_difference: {training_difference}')
        logger.debug('starting_epoch: %d', starting_epoch)
        logger.debug('resume_step: %d', resume_step)

        
        #for epoch in range(starting_epoch, args.num_train_epochs):
        for epoch in range(0, args.num_train_epochs):
            # 学習済みのステップはスキップ
            if epoch < starting_epoch:
                completed_steps += len(train_dataloader)
                progress_bar.update(len(train_dataloader))
                #if epoch + 1 == starting_epoch:
                #    resuming = False
                #if epoch + args.num_workers >= starting_epoch:
                #    skip_collation = False
                #else:
                #    skip_collation = True
                continue

            total_loss = 0
            #fed_samples = 0
            fed_iterations = 0
            zero_filled_epoch = str(epoch).zfill(len(str(args.num_train_epochs - 1)))
            for step, batch in enumerate(train_dataloader):
                if starting_epoch == epoch:
                    if step <= resume_step:
                        completed_steps += 1
                        progress_bar.update(1)
                        #if step == resume_step:
                        #    resuming = False
                        if step + args.num_workers >= resume_step:
                            skip_collation = False
                        else:
                            skip_collation = True
                        continue
                resuming = False
                model.train()
                #outputs = model(**batch)
                outputs = model(
                    input_ids = batch['input_ids'],
                    tag_ids = batch['tag_ids'],
                    attention_mask = batch.get('attention_mask'),
                    token_type_ids = batch.get('token_type_ids'),
                )
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                #fed_samples += len(batch["input_ids"])
                fed_iterations += 1
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        zero_filled_step = str(completed_steps).zfill(len(str(args.max_train_steps - 1)))
                        #output_dir = f"step_{completed_steps}"
                        #output_dir = f"epoch_{zero_filled_epoch}_step_{completed_steps}"
                        output_dir = f"epoch_{zero_filled_epoch}_step_{zero_filled_step}"
                        #if args.output_dir is not None:
                        #    output_dir = os.path.join(args.output_dir, output_dir)
                        #accelerator.save_state(output_dir)
                        do_save(output_dir)
                        if total_loss > 0:
                            train_log = {
                                #"train_loss": total_loss.item() / len(train_dataloader),
                                "train_loss": total_loss.item() / fed_iterations,
                                "epoch": epoch,
                                "step": completed_steps,
                            }
                            if accelerator.is_main_process:
                                logger.info(f'epoch {epoch} train_log: {train_log}')
                        eval_metric = do_eval()

                if completed_steps >= args.max_train_steps:
                    break

            if total_loss > 0:
                train_log = {
                    #"train_loss": total_loss.item() / len(train_dataloader),
                    "train_loss": total_loss.item() / fed_iterations,
                    "epoch": epoch,
                    "step": completed_steps,
                }
                if accelerator.is_main_process:
                #accelerator.print(f'epoch {epoch}:', train_log)
                    logger.info(f'epoch {epoch} train_log: {train_log}')
            eval_metric = do_eval()
            if args.with_tracking:
                train_log['seqeval'] = eval_metric
                accelerator.log(
                    train_log,
                    step=completed_steps,
                )

            # 各エポック毎のモデルは毎回保存
            #output_dir = f"epoch_{epoch}"
                zero_filled_step = str(completed_steps).zfill(len(str(args.max_train_steps - 1)))
            output_dir = f"epoch_{zero_filled_epoch}_step_{zero_filled_step}"
            #if args.output_dir is not None:
            #    output_dir = os.path.join(args.output_dir, output_dir)
            #accelerator.save_state(output_dir)
            do_save(output_dir)

        if args.with_tracking:
            accelerator.end_training()

        if args.output_dir is not None:
            if eval_metric is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

                    all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                    if args.with_tracking:
                        all_results.update({"train_loss": total_loss.item() / len(train_dataloader)})
                    all_results_path = os.path.join(args.output_dir, "all_results.json")
                    with open(all_results_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f)
                    with open(status_path, "w", encoding="utf-8") as f:
                        json.dump(status, f, ensure_ascii=False, sort_keys=True, indent=2)

    if not args.do_train and args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Instantaneous batch size per device = %d", args.per_device_eval_batch_size)
        eval_metric = do_eval()
        logger.info(f'eval_metric: {eval_metric}')

    if args.do_predict:
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(predict_dataset))
        logger.info("  Instantaneous batch size per device = %d", args.per_device_eval_batch_size)
        predict_dataloader = DataLoader(
            predict_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_predict_batch_size,
            num_workers=args.num_workers,
        )

        model, predict_dataloader = accelerator.prepare(
            model, predict_dataloader,
        )

        model.eval()
        map_page_id_and_window_id_to_prediction = OrderedDict()
        # 推論した結果を page_id と window_id で紐付ける
        for step, batch in enumerate(tqdm(
            predict_dataloader,
            desc='Feeding for prediction',
        )):
            #if step > 100:
            #    break
            page_ids = batch['page_id']
            window_ids = batch['window_id']
            enes = batch['ENE']
            with torch.no_grad():
                outputs = model(
                    input_ids = batch['input_ids'],
                    attention_mask = batch.get('attention_mask'),
                    token_type_ids = batch.get('token_type_ids'),
                )
            predictions = outputs.logits.argmax(dim=-1)
            page_ids_gathered, window_ids_gathered, predictions_gathered = \
                accelerator.gather([page_ids, window_ids, predictions])
            page_ids = page_ids_gathered
            window_ids = window_ids_gathered
            if device.type == "cpu":
                predictions = predictions_gathered.detach().clone().numpy()

            else:
                predictions = predictions_gathered.detach().cpu().clone().numpy()
            predictions = predictions.transpose(0, 2, 1) # [B, L, A] -> [B, A, L]
            predictions = predictions[:, :, 1:] # 1トークン目はCLSなので除去
            #logger.debug('new predictions shape: %s', predictions.shape)
            #logger.debug('predictions shape: %s', predictions.shape)
            #for page_id, window_id, pred in zip(page_ids, window_ids, predictions):
            #logger.debug('enes: %s', enes)
            for page_id, window_id, ene, pred in zip(
                page_ids, window_ids, enes, predictions
            ):
                #logger.debug('page_id: %s', page_id)
                #logger.debug('window_id: %s', window_id)
                #logger.debug('ene: %s', enes)
                #logger.debug('pred: %s', pred)
                #map_page_id_and_window_id_to_prediction[page_id, window_id] = pred.astype(np.int8)
                # 推論の必要の無い属性名は None にしてメモリと計算量の節約
                attribute_id_counter = mapping.map_ene_to_attribute_id_counter[ene]
                pred_list = [
                    None if attribute_id not in attribute_id_counter
                    else p.astype(np.int8)
                    for attribute_id, p in enumerate(pred)
                ]
                map_page_id_and_window_id_to_prediction[page_id, window_id] = pred_list

        # page_id と window_id と推論結果を元に
        # page_id に推論結果を集約
        map_page_id_to_predictions = OrderedDict()
        def merge_predictions(example):
            page_id, window_id = example['page_id'], example['window_id']
            prediction = map_page_id_and_window_id_to_prediction.get(
                (page_id, window_id)
            )
            if prediction is None:
                return
            example['prediction'] = prediction
            map_page_id_to_predictions.setdefault(page_id, []).append(example)
        predict_dataset.map(
            merge_predictions,
            desc='Merging windowed predictions into pages',
            #batched=True,
            #remove_columns=predict_dataset.column_names,
            load_from_cache_file=False,
        )
        #logger.debug('predicted page ids: %s', map_page_id_to_predicted_windows.keys())

        # ウィンドウをマージしつつ推論結果を出力
        with open(args.predict_output_jsonl, 'w', encoding='utf-8') as f:
            for page_id, windows in tqdm(
                map_page_id_to_predictions.items(),
                #desc='Merging windows',
                desc='Extracting predicted attributes',
            ):
                #logger.debug('page_id: %s', page_id)
                #logger.debug('windows: %s', windows)
                tokens_with_offsets = []
                #tags = []
                tag_ids = np.array([], dtype=np.int8).reshape(num_attribute_names, 0)
                #logger.debug('page_id: %s', page_id)
                #logger.debug('windows type: %s', type(windows))
                # リスト・配列を拡張しながらウィンドウをマージ
                #logger.debug('windows: %s', windows)
                for window in windows:
                    #logger.debug('window: %s', window)
                    title = window['title']
                    ene = window['ENE']
                    window_start = window['window_id'] * (args.window_size - args.window_overlap_size)
                    window_end = window_start + len(window['tokens'])
                    #if window_end > len(tags):
                    if window_end > len(tokens_with_offsets):
                        #extension_length = window_end - len(tags)
                        extension_length = window_end - len(tokens_with_offsets)
                        #tags.extend([mapping.map_tag_to_id['O']] * (window_end - len(tags)))
                        #tags.extend([None] * extension_length)
                        tokens_with_offsets.extend([None] * extension_length)
                        tag_ids_extension = np.zeros(
                            [num_attribute_names, extension_length],
                            dtype=np.int8
                        )
                        #logger.debug('tag_ids.shape: %s', tag_ids.shape)
                        #logger.debug('tag_ids_extension.shape: %s', tag_ids_extension.shape)
                        tag_ids = np.concatenate([tag_ids, tag_ids_extension], axis=1)
                        #logger.debug('new tag_ids.shape: %s', tag_ids.shape)
                    for i, token_with_offset in enumerate(window['tokens_with_offsets']):
                        tokens_with_offsets[window_start+i] = token_with_offset
                    #logger.debug('window prediction shape: %s', window['prediction'].shape)
                    #if window['prediction'] is None:
                    #    continue
                    #attribute_name_counter = mapping.map_ene_to_attribute_name_counter.get(ene)
                    #if attribute_name_counter is None:
                    #    break
                    for attribute_index, attribute_tag_ids in enumerate(window['prediction']):
                        if attribute_tag_ids is None:
                            continue
                        #attribute_name = mapping.attribute_names[attribute_index]
                        #if attribute_name not in attribute_name_counter:
                        #    continue
                        for i, tag_id in enumerate(attribute_tag_ids):
                            if i >= len(window['tokens']):
                                break
                            tag = mapping.tag_list[tag_id]
                            if tag in ['B', 'I']:
                                previous_tag_id = tag_ids[attribute_index, window_start+i]
                                previous_tag = mapping.tag_list[previous_tag_id]
                                if previous_tag == 'O':
                                    tag_ids[attribute_index, window_start+i] = tag_id
                html_path = os.path.join(args.predict_html_dir, f'{page_id}.html')
                with open(html_path, 'r', encoding='utf-8') as f_html:
                    html_lines = f_html.read().splitlines(keepends=True)
                #attribute_name_counter = mapping.map_ene_to_attribute_name_counter[ene]
                attribute_id_counter = mapping.map_ene_to_attribute_id_counter[ene]
                #attribute_name_counter = mapping.map_ene_to_attribute_name_counter.get(ene)
                #if attribute_name_counter is None:
                #    continue
                for attribute_index, attribute_tag_ids in enumerate(tag_ids):
                    attribute_name = mapping.attribute_names[attribute_index]
                    #if attribute_name not in attribute_name_counter:
                    #    continue
                    if attribute_index not in attribute_id_counter:
                        continue
                    found_b = False
                    offsets = {}
                    for i, tag_id in enumerate(attribute_tag_ids):
                        tag = mapping.tag_list[tag_id]
                        token_with_offsets = tokens_with_offsets[i]
                        if tag == 'B':
                            if found_b:
                                extract(f, html_lines, offsets)
                            #logger.debug('B tag')
                            found_b = True
                            offsets['start_line'] = token_with_offsets["start_line"]
                            offsets['start_offset'] = token_with_offsets["start_offset"]
                            offsets['end_line'] = token_with_offsets["end_line"]
                            offsets['end_offset'] = token_with_offsets["end_offset"]
                        if tag == 'I':
                            offsets['end_line'] = token_with_offsets["end_line"]
                            offsets['end_offset'] = token_with_offsets["end_offset"]
                        if tag == 'O':
                            if found_b:
                                found_b = False
                                extract(f, html_lines, offsets)
                    if found_b:
                        extract(f, html_lines, offsets)

if __name__ == '__main__':
    main()
