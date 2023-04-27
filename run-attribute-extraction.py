#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys

import datasets
from datasets import (
    ClassLabel,
    Dataset,
    load_dataset,
)

import evaluate

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
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

from tokenization import (
    tokenize_with_offsets,
)
from tagging import (
    tag_tokens_with_annotation_list,
)
from html_cleaning import (
    clean_up_html,
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
    #    "--dataset_name",
    #    type=str,
    #    default=None,
    #    help="The name of the dataset to use (via the datasets library).",
    #)
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
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
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
        default=None,
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
        #default=500,
        default=100,
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
        default=1,
        help=(
            "The number of workers to use for the dataset processing."
        ),
    )

    args = parser.parse_args()

    return args

class TokenMultiClassificationModel(transformers.PreTrainedModel):
    def __init__(self, config, pretrained_model_name_or_path, num_attribute_names):
        logger.debug('__init__')
        super().__init__(config)
        self.config = config
        self.num_attribute_names = num_attribute_names
        self.num_labels = config.num_labels
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path = pretrained_model_name_or_path,
            config = config,
            add_pooling_layer=False,
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(
            config.hidden_size,
            self.num_labels * self.num_attribute_names
        )

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        tag_ids = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        outputs = self.encoder(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )

        sequence_output = outputs[0] # [B, L, H]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # [B, L, C*A]
        logits = logits.reshape(
            batch_size, seq_length, self.num_attribute_names, self.num_labels
        ) # [B, L, A, C]

        loss = None
        if tag_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), tag_ids.view(-1))

        return transformers.modeling_outputs.MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
        )


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
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            logfile = os.path.join(args.output_dir, 'training.log')
            logzero.logfile(logfile)
    accelerator.wait_for_everyone()

    raw_dataset = load_dataset('./shinra_attribute_extraction_2022')
    logger.debug('raw_dataset: %s', raw_dataset)

    dataset = raw_dataset

    # 今回使わない特徴量は削除しておく
    dataset = dataset.flatten()
    dataset = dataset.remove_columns(['context_text', 'attributes.text_offset'])

    # --target_categories が与えられた場合は、そのカテゴリのみで絞り込む
    target_categories = None
    if args.target_categories is not None:
        target_categories = args.target_categories.split(',')
        logger.debug('target_categories: %s', target_categories)
        dataset = dataset.filter(
            lambda e: e['category_name'] in target_categories,
            desc='Filtering with target categories',
        )
    logger.debug('dataset: %s', dataset)

    # 全属性名のリストと、
    # ENE に対応する属性名のリストの辞書を作成
    set_attribute_names = set()
    map_ene_to_attribute_name_set = {}
    for example in dataset['train']:
        ene = example['ENE']
        for attribute_name in example['attributes.attribute']:
            set_attribute_names.add(attribute_name)
            if ene not in map_ene_to_attribute_name_set:
                map_ene_to_attribute_name_set[ene] = set()
            map_ene_to_attribute_name_set[ene].add(attribute_name)
    attribute_names = sorted(set_attribute_names)
    map_attribute_name_to_id = {name: i for i, name in enumerate(attribute_names)}
    num_attribute_names = len(attribute_names)
    #logger.debug('attribute_names: %s', attribute_names)

    # HTML をクリーニング
    # (属性値抽出の対象とならないであろう箇所を除去)
    # (HTMLタグは一般的なサブワード分割と相性が悪く)
    def clean_context_html(example):
        cleaned_html = clean_up_html(example['context_html'])
        return {'context_html': cleaned_html}
    #dataset = dataset.map(clean_context_html, desc='Cleaning HTML files')
    dataset = dataset.map(
        clean_context_html,
        desc='Cleaning HTML files',
        num_proc=args.num_workers,
    )
    logger.debug('dataset: %s', dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    if args.split_eval_size> 0:
        split_eval_size = args.split_eval_size
        if split_eval_size >= 1:
            split_eval_size = int(split_eval_size)
        # ラベル付きデータを訓練用と評価用に分割
        # ページ単位で分割することが好ましいので、ウィンドウ化よりも前に行う必要あり
        dataset = dataset['train'].train_test_split(
            test_size=split_eval_size,
            seed=args.seed
        )
        logger.debug('dataset: %s', dataset)

    # トークン化と IOB2 タグの付与
    def tokenize_and_tag(example):
        tokens_with_offsets = tokenize_with_offsets(tokenizer, example['context_html'])
        tokens = [token.text for token in tokens_with_offsets]
        tags, num_valids, num_skipped = tag_tokens_with_annotation_list(
            tokens_with_offsets,
            [
                {'attribute': attribute, 'html_offset': html_offset}
                for attribute, html_offset in zip(
                    example['attributes.attribute'], example['attributes.html_offset']
                )
            ]
        )

        extended_tags = [
            #[' '] * len(tokens)
            ' ' * len(tokens)
            for _ in range(len(attribute_names))
        ]
        ene = example['ENE']
        attribute_name_set = map_ene_to_attribute_name_set[ene]
        for attribute_name in attribute_name_set:
            attribute_id = map_attribute_name_to_id[attribute_name]
            if attribute_name in tags:
                extended_tags[attribute_id] = str.join('', tags[attribute_name])
                #extended_tags[attribute_id] = tags[attribute_name]
            else:
                extended_tags[attribute_id] = 'O' * len(tokens)
                #extended_tags[attribute_id] = ['O'] * len(tokens)
        #examples['tokens'].append(tokens)
        #examples['tags'].append(extended_tags)

        #logger.debug('tags: %s', tags)
        return {
            'tokens': tokens,
            #'tags': tags,
            'tags': extended_tags,
        }
    #def tokenize_and_tag(example):
    def batch_tokenize_and_tag(examples):
        #logger.debug('# examples: %d', len(examples))
        #logger.debug('examples type: %s', type(examples))
        #logger.debug('page_ids type: %s', type(examples['page_id']))
        #list_tokens = []
        examples['tokens'] = []
        examples['tags'] = []
        for context_html, attributes, html_offsets, ene in zip(
            examples['context_html'],
            examples['attributes.attribute'],
            examples['attributes.html_offset'],
            examples['ENE'],
        ):
        #for index, page_id in enumerate(examples['page_id']):
            #context_html = examples['context_html'][index]
            #attributes = examples['attributes.attribute'][index]
            #html_offsets = examples['attributes.html_offset'][index]
            tokens_with_offsets = tokenize_with_offsets(tokenizer, context_html)
            tokens = [token.text for token in tokens_with_offsets]
            #tags, num_valids, num_skipped = tag_tokens_with_annotation_list(tokens_with_offsets, [
            #    {'attribute': attribute, 'html_offset': html_offset}
            #    for attribute, html_offset in zip(
            #        example['attributes.attribute'], example['attributes.html_offset']
            #    )
            #])
            #tags = {name: str.join('', chars) for name, chars in tags.items()}
            tags, num_valids, num_skipped = tag_tokens_with_annotation_list(
                tokens_with_offsets, [
                    {'attribute': attribute, 'html_offset': html_offset}
                    for attribute, html_offset in zip(attributes, html_offsets)
                ]
            )

            extended_tags = [
                #[' '] * len(tokens)
                ' ' * len(tokens)
                for _ in range(len(attribute_names))
            ]
            #ene = example['ENE']
            #ene = examples['ENE'][index]
            attribute_name_set = map_ene_to_attribute_name_set[ene]
            #for attribute_name, tags in example['tags'].items():
            #    attribute_id = map_attribute_name_to_id[attribute_name]
            #    if ene not in attribute_name_set:
            #        extended_tags[attribute_id] = [' '] * len(tokens)
            #logger.debug('# extended_tags: %s', len(extended_tags))
            for attribute_name in attribute_name_set:
                attribute_id = map_attribute_name_to_id[attribute_name]
                #logger.debug('attribute_id: %s', attribute_id)
                if attribute_name in tags:
                    extended_tags[attribute_id] = str.join('', tags[attribute_name])
                    #extended_tags[attribute_id] = tags[attribute_name]
                else:
                    extended_tags[attribute_id] = 'O' * len(tokens)
                    #extended_tags[attribute_id] = ['O'] * len(tokens)
            examples['tokens'].append(tokens)
            examples['tags'].append(extended_tags)

            #extended_tags = np.array(extended_tags) 
            #logger.debug('extended_tags: %s', extended_tags)
            #logger.debug('extended_tags type: %s', type(extended_tags))
        
        #logger.debug('tags: %s', tags)
        #return {
        #    'tokens': tokens,
        #    #'tags': tags,
        #    'tags': extended_tags,
        #}
        return examples
    #dataset = dataset.map(tokenize_and_tag, desc='Tokenizing and tagging')

    # ページ毎にトークン長 x 属性名数とデータセット内部で配列サイズ上限をオーバーフローするらしく
    # 後述のウィンドウ化と同時に行う必要がありそう
    dataset = dataset.map(
        tokenize_and_tag,
        #batched=True,
        desc='Tokenizing and tagging',
        writer_batch_size=1,
        num_proc=args.num_workers,
    )
    #logger.debug('dataset: %s', dataset)

    # 通常、512トークンを超える系列の処理はそのままできないため、
    # スライドウィンドウを用いて分割する
    # (データセットのチャンク分割はバッチ処理でのみ可能)
    def batch_split_into_windows(examples):
        windows = {
            field: []
            for field in [
                'page_id', 'window_id', 'title', 'category_name', 'ENE', 'tokens', 'tags'
            ]
        }
        #del examples['tokens']
        #del examples['tags']
        #return examples
        for page_id, title, category_name, ene, tokens, tags in zip(
            examples['page_id'],
            examples['title'],
            examples['category_name'],
            examples['ENE'],
            examples['tokens'],
            examples['tags'],
        ):
        #for index, page_id in enumerate(examples['page_id']):
            #tokens = examples['tokens'][index]
            #tags = examples['tags'][index]
            #for i in range(0, len(tokens), args.window_size - args.window_overlap_size):
            for window_id, i in enumerate(
                range(0, len(tokens), args.window_size - args.window_overlap_size)
            ):
                window_tokens = tokens[i:i+args.window_size]
                #window_tags = {
                #    attribute_name: tag_list[i:i+args.window_size]
                #    for attribute_name, tag_list in tags.items() if tag_list is not None
                #}
                window_tags = [tag_list[i:i+args.window_size] for tag_list in tags]
                #logger.debug('window_tags: %s', window_tags)
                windows['page_id'].append(page_id)
                windows['window_id'].append(window_id)
                #windows['title'].append(examples['title'][index])
                windows['title'].append(title)
                #windows['category_name'].append(examples['category_name'][index])
                windows['category_name'].append(category_name)
                #windows['ENE'].append(examples['ENE'][index])
                windows['ENE'].append(ene)
                windows['tokens'].append(window_tokens)
                windows['tags'].append(window_tags)
        #del windows['tags']
        #logger.debug('# examples: %s', len(examples['page_id']))
        #logger.debug('# windows: %s', len(windows['page_id']))
        return windows
    #dataset['train'] = dataset['train'].map(
    dataset = dataset.map(
        batch_split_into_windows,
        desc='Splitting into slide windows',
        #lambda examples: batch_split_into_windows(batch_tokenize_and_tag(examples)),
        #desc='Tokenizing, tagging and splitting into slide windows',
        batched=True,
        remove_columns=dataset['train'].column_names,
        batch_size=1,
        writer_batch_size=1,
        num_proc=args.num_workers,
    )
    logger.debug('dataset: %s', dataset)
    #logger.debug('dataset["train"][0]: %s', dataset['train'][0])

    tag_list = ['O', 'B', 'I']
    map_tag_to_id = {tag: i for i, tag in enumerate(tag_list)}
    num_labels = len(tag_list)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )

    model = TokenMultiClassificationModel(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config,
        num_attribute_names = num_attribute_names,
    )
    logger.debug('model: %s', model)

    model.config.label2id = map_tag_to_id
    model.config.id2label = tag_list

    logger.debug('new model.config: %s', model.config)

    #logger.debug('attribute_names: %s', attribute_names)
    # トークン ID とタグ ID の付与
    def prepare_ids(example):
        token_ids = tokenizer.encode(example['tokens'])
        #tag_ids = [
        #    [map_tag_to_id['O']] * len(token_ids)
        #    for _ in range(len(attribute_names))
        #]
        # 先頭 (CLS), 末尾 (SEP) はタグの予測不要
        # BIO以外の文字(空白)はタグの予測不要
        tag_ids = [
            [-100] + [map_tag_to_id.get(char, -100) for char in tag_str] + [-100]
            for tag_str in example['tags']
        ]
        #logger.debug('tag_ids: %s', tag_ids)
        #for attribute_name, tags in example['tags'].items():
        #    attribute_id = map_attribute_name_to_id[attribute_name]
        #    ene = example['ENE']
        #    attirbute_name_set = map_ene_to_attribute_name_set[ene]
        #    if ene not in attirbute_name_set:
        #        tag_ids[attribute_id] = [-100] * len(token_ids)
        #    if tags is None:
        #        continue
        #    tag_ids[attribute_id][0] = -100
        #    for index, tag in enumerate(tags):
        #        tag_ids[attribute_id][index+1] = map_tag_to_id[tag]
        #    tag_ids[attribute_id][-1] = -100
        #logger.debug('token_ids: %s', token_ids)
        #logger.debug('tag_ids: %s', tag_ids)
        return {
            #'token_ids': token_ids,
            'input_ids': token_ids,
            'tag_ids': tag_ids,
            #'labels': tag_ids,
            #'labels': tag_ids[0],
        }
    #dataset['train'] = dataset['train'].map(prepare_ids, remove_columns=dataset['train'].column_names)
    dataset = dataset.map(
        prepare_ids,
        remove_columns=dataset['train'].column_names,
        desc='Converting tokens and tags into ids',
        writer_batch_size=1,
        num_proc=args.num_workers,
    )
    logger.debug('dataset: %s', dataset)
    #logger.debug('dataset["train"][0]: %s', dataset['train'][0])

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # ミニバッチ生成のための追加処理
    def my_collator(examples):
        input_ids = [
            torch.tensor(example['input_ids'], dtype=torch.int64)
            for example in examples
        ]
        # cuda の nll_loss のターゲットは long にしか対応していない模様
        tag_ids = [
            torch.tensor(example['tag_ids'], dtype=torch.int64).transpose(1, 0)
            for example in examples
        ]
        mask = [
            torch.ones(len(ids), dtype=torch.bool) for ids in input_ids
        ]
        # indices には int か long のみ対応
        token_type_ids = [
            torch.zeros(len(ids), dtype=torch.int32) for ids in input_ids
        ]
        return {
            'input_ids': pad_sequence(input_ids, batch_first=True),
            'tag_ids': pad_sequence(tag_ids, batch_first=True, padding_value=-100),
            'attention_mask': pad_sequence(mask, batch_first=True),
            'token_type_ids': pad_sequence(token_type_ids, batch_first=True),
        }
    data_collator = my_collator
        
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
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
        # [B, N, A] -> [B, A, N] -> [B*A, N]
        y_pred = y_pred.transpose(0, 2, 1).reshape(batch_size*num_attribute_names, seq_len)
        y_true = y_true.transpose(0, 2, 1).reshape(batch_size*num_attribute_names, seq_len)

        # Remove ignored index (special tokens)
        true_predictions = [
            #[label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            [tag_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            #[label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            [tag_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

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

    status_path = os.path.join(args.output_dir, "status.json")
    status = {}
    if os.path.isfile(status_path):
        with open(status_path, "r", encoding='utf-8') as f:
            status = json.load(f)
            logger.debug('Loaded status: %s', status)
 
    if args.resume_from_checkpoint:
    #if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
    #if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
        accelerator.print(f"Resumed from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        #path = os.path.basename(args.resume_from_checkpoint)
        resume_path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = [
            os.path.join(args.output_dir, f.name)
            for f in os.scandir(args.output_dir) if f.is_dir()
        ]
        if len(dirs) > 0:
            #dirs.sort(key=os.path.getctime)
            dirs.sort(key=os.path.getmtime)
            #path = dirs[-1] # Sorts folders by date modified, most recent checkpoint is the last
            resume_path = dirs[-1] # Sorts folders by date modified, most recent checkpoint is the last
            #resume_path = os.path.join(args.output_dir, dirs[-1])
            #logger.debug('path: %s', path)
            #resume_path = get_last_checkpoint(args.output_dir)
            #logger.debug('resume_path: %s', resume_path)
            accelerator.load_state(resume_path)
    logger.debug('resume_path: %s', resume_path)

    starting_epoch = 0
    resume_step = -1
    eval_metric = None

    # Extract `epoch_{i}` or `step_{i}`
    if resume_path:
        training_difference = os.path.splitext(resume_path)[0]
        logger.debug('training_difference: %s', training_difference)

        if "epoch_" in training_difference and "_step_" in training_difference:
            resume_step = int(training_difference.split("_")[-1])
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
        else:
            raise ValueError(f'Failed to parse training_difference: {training_difference}')
    logger.debug('starting_epoch: %d', starting_epoch)
    logger.debug('resume_step: %d', resume_step)

    def save_best_status(eval_metric):
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
        with open(status_path, "w", encoding='utf-8') as f:
            json.dump(status, f, indent=2, sort_keys=True, ensure_ascii=False)

    def do_eval():
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
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
        eval_metric = compute_metrics()
        #accelerator.print(f"epoch {epoch}:", eval_metric)
        if accelerator.is_main_process:
            logger.info(f'epoch {epoch} eval_metric: {eval_metric}')
        save_best_status(eval_metric)
        return eval_metric


    #for epoch in range(starting_epoch, args.num_train_epochs):
    for epoch in range(0, args.num_train_epochs):
        # 学習済みのステップはスキップ
        if starting_epoch > epoch:
            completed_steps += len(train_dataloader)
            progress_bar.update(len(train_dataloader))
            continue

        total_loss = 0
        zfilled_epoch = str(epoch).zfill(len(str(args.num_train_epochs - 1)))
        for step, batch in enumerate(train_dataloader):
            if starting_epoch == epoch:
                if step <= resume_step:
                    completed_steps += 1
                    progress_bar.update(1)
                    continue
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    #output_dir = f"step_{completed_steps}"
                    output_dir = f"epoch_{zfilled_epoch}_step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    if total_loss > 0:
                        train_log = {
                            "train_loss": total_loss.item() / len(train_dataloader),
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
                "train_loss": total_loss.item() / len(train_dataloader),
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
        output_dir = f"epoch_{zfilled_epoch}_step_{completed_steps}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)

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

if __name__ == '__main__':
    main()
