#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
#import random
#from pathlib import Path

import sqliteshelve as shelve

import datasets
from datasets import (
    ClassLabel,
    Dataset,
    load_dataset,
)

import evaluate

import torch
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    #CONFIG_MAPPING,
    #MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from logzero import logger

#from converter import load_train_dataset

from tokenization import (
    tokenize_with_offsets,
)
from tagging import (
    tag_tokens_with_annotation_list,
)
from html_cleaning import (
    clean_up_html,
)

#MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
#MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
#logger.debug('MODEL_CONFIG_CLASSES: %s', MODEL_CONFIG_CLASSES)
#logger.debug('MODEL_TYPES: %s', MODEL_TYPES)

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
    #    "--dataset_config_name",
    #    type=str,
    #    default=None,
    #    help="The configuration name of the dataset to use (via the datasets library).",
    #)
    #parser.add_argument(
    #    "--train_file", type=str, default=None,
    #    #help="A csv or a json file containing the training data.",
    #    help="A json lines (.jsonl) or sqlite-shelve (.db) file containing the training data.",
    #)
    #parser.add_argument(
    #    "--train_html_dir", type=str, default=None,
    #    help="A directory path containing the training html files.",
    #)
    #parser.add_argument(
    #    "--validation_file", type=str, default=None,
    #    help="A csv or a json file containing the validation data.",
    #)
    #parser.add_argument(
    #    "--text_column_name",
    #    type=str,
    #    default=None,
    #    help="The column name of text to input in the file (a csv or JSON file).",
    #)
    #parser.add_argument(
    #    "--label_column_name",
    #    type=str,
    #    default=None,
    #    help="The column name of label to input in the file (a csv or JSON file).",
    #)
    #parser.add_argument(
    #    "--max_length",
    #    type=int,
    #    default=128,
    #    help=(
    #        "The maximum total input sequence length after tokenization. "
    #        "Sequences longer than this will be truncated, "
    #        "sequences shorter will be padded if `--pad_to_max_length` is passed."
    #    )
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
    #parser.add_argument(
    #    "--model_type",
    #    type=str,
    #    default=None,
    #    help="Model type to use if training from scratch.",
    #    choices=MODEL_TYPES,
    #)
    #parser.add_argument(
    #    "--label_all_tokens",
    #    action="store_true",
    #    help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    #)
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    #parser.add_argument(
    #    "--task_name",
    #    type=str,
    #    default="ner",
    #    choices=["ner", "pos", "chunk"],
    #    help="The name of the tasks.",
    #)
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
    #parser.add_argument(
    #    "--report_to",
    #    type=str,
    #    default="all",
    #    help=(
    #        "The integration to report the results and logs to. "
    #        'Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"`, and `"clearml"`. '
    #        'Use `"all"` (default) to report to all integrations. '
    #        "Only applicable when `--with_tracking` is passed."
    #    ),
    #)
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
        "--split_eval_ratio",
        type=float,
        default=0.1,
        help=(
            "A float value between 0 and 1 indicating "
            "the ratio of the splitted eval dataset size"
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    #if args.task_name is None and args.train_file is None and args.validation_file is None:
    #if args.train_file is None and args.validation_file is None:
    #    raise ValueError("Need either a task name or a training/validation file.")
    #if args.train_file is not None:
    #    extension = args.train_file.split(".")[-1]
    #    assert extension in ["db", "jsonl"], "`train_file` should be a db or jsonl file."
    #if args.validation_file is not None:
    #    extension = args.validation_file.split(".")[-1]
    #    #assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #    assert extension in ["jsonl"], "`validation_file` should be a json file."

    return args

#class TokenMultiClassificationModel(torch.nn.Module):
class TokenMultiClassificationModel(transformers.PreTrainedModel):
    #def __init__(self, config, num_attribute_names, **hparams):
    #def __init__(self, config, num_attribute_names):
    def __init__(self, config, pretrained_model_name_or_path, num_attribute_names):
    #def __init__(self, config, num_attribute_names):
    #def __init__(self, config):
        logger.debug('__init__')
        super().__init__(config)
        #self.base_model = torch.nn.Linear(50, 50)
        #self.base_model2 = torch.nn.Linear(50, 50)
        self.config = config
        #self.num_attribute_names = config.num_attribute_names
        self.num_attribute_names = num_attribute_names
        self.num_labels = config.num_labels
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        #if hparams.get('num_labels') is not None:
        #    self.num_labels = hparams['num_labels']
        #self.base_classifier = AutoModelForTokenClassification.from_pretrained(
        #    config = config,
        #    #num_labels = self.num_labels * self.num_attribute_names,
        #    **hparams
        #)
        self.encoder = AutoModel.from_pretrained(
        #self.base_model = AutoModel.from_pretrained(
        #self.base_model = AutoModel.from_config(
        #self.base_model = transformers.BertModel(
        #self.base_model = transformers.AlbertModel(
            pretrained_model_name_or_path = pretrained_model_name_or_path,
            config = config,
            add_pooling_layer=False,
        )
        #logger.debug('base_model: %s', base_model)
        #self.base_model = torch.nn.Linear(50, 50)
        #logger.debug('self.base_model: %s', self.base_model)
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
        #logger.debug('input_ids: %s', input_ids)
        #logger.debug('tag_ids: %s', tag_ids)
        #logger.debug('kwargs: %s', kwargs)
        #if attention_mask is not None:
        #    logger.debug('attention_mask.shape: %s', attention_mask.shape)

        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        #logger.debug('self.encoder: %s', self.encoder)
        #logger.debug('self.base_mode: %s', self.base_model)
        #logger.debug('self: %s', self)
        #return
        #outputs = self.base_model(
        outputs = self.encoder(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )

        #logger.debug('outputs: %s', outputs)
        #logger.debug('outputs last hidden: %s', outputs.last_hidden_state)
        #logger.debug('outputs shape: %s', outputs.shape)

        #last_hidden_state = outputs.last_hidden_state
        #logger.debug('last_hidden_state: %s', last_hidden_state)
        #logger.debug('last_hidden_state.shape: %s', last_hidden_state.shape)
        #logger.debug('last_hidden_state[0].shape: %s', last_hidden_state[0].shape)
        ##sequence_outputs = outputs[0]
        #logger.debug('outputs[0]: %s', outputs[0])
        #logger.debug('outputs[0].shape: %s', outputs[0].shape)

        sequence_output = outputs[0] # [B, L, H]
        #logger.debug('sequence_output: %s', sequence_output)
        #logger.debug('sequence_output.shape: %s', sequence_output.shape)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # [B, L, C*A]
        logits = logits.reshape(
            batch_size, seq_length, self.num_attribute_names, self.num_labels
        ) # [B, L, A, C]
        #logger.debug('logits: %s', logits)
        #logger.debug('logits.shape: %s', logits.shape)

        loss = None
        if tag_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            #logger.debug('logits view shape: %s', logits.view(-1, self.num_labels).shape)
            #logger.debug('tag_ids view shape: %s', tag_ids.view(-1).shape)
            loss = loss_fct(logits.view(-1, self.num_labels), tag_ids.view(-1))
            #logger.debug('loss: %s', loss)

        #return logits
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
    #import logging
    #logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    level=logging.INFO,
    #)
    #logger.info(accelerator.state, main_process_only=False)
    logger.info('accelerator.state: %s', accelerator.state)

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
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provider your own CSV/JSON/TXT training and evaluation files
    # (see below) or just provide the name of one of the public datasets available on the hub at
    # https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or
    # the first column if no column called 'tokens' is found.
    # You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee
    # that only one local process can concurrently download the dataset.

    #def gen_dataset(file_path):
    #    extension = file_path.split(".")[-1]
    #    if extension == 'db':
    #        pass
    #    elif extension == 'jsonl':
    #        pass
    #    else:
    #        raise ValueError(f'"{file_path}" with not supported extension: "{extension}"')

    #data_files = {}
    #if args.train_file is not None:
    #    data_files["train"] = args.train_file
    #if args.validation_file is not None:
    #    data_files["validation"] = args.validation_file
    #extension = args.train_file.split(".")[-1]
    #raw_datasets = load_dataset(extension, data_files=data_files)
    #raw_datasets = load_dataset('json', data_files=data_files)

    raw_dataset = load_dataset('./shinra_attribute_extraction_2022')
    logger.debug('raw_dataset: %s', raw_dataset)

    dataset = raw_dataset

    # 今回使わない特徴量は削除しておく
    dataset = dataset.flatten()
    dataset = dataset.remove_columns(['context_text', 'attributes.text_offset'])

    target_categories = None
    if args.target_categories is not None:
        target_categories = args.target_categories.split(',')
        logger.debug('target_categories: %s', target_categories)
        dataset = dataset.filter(
            lambda e: e['category_name'] in target_categories,
            desc='Filtering with target categories',
        )
    logger.debug('dataset: %s', dataset)

    set_attribute_names = set()
    map_ene_to_attribute_name_set = {}
    #for attribute_names_in_annotation in dataset['train']['attributes.attribute']:
    #    for attribute_name in attribute_names_in_annotation:
    #        set_attribute_names.add(attribute_name)
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


    def clean_context_html(example):
        cleaned_html = clean_up_html(example['context_html'])
        return {'context_html': cleaned_html}
    dataset = dataset.map(clean_context_html)
    logger.debug('dataset: %s', dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    #dataset['train'] = dataset['train'].select(range(5))
    def tokenize_and_tag(example):
        #logger.debug('example: %s', example)
        #logger.debug('example["offsets"]: %s', example["offsets"])
        tokens_with_offsets = tokenize_with_offsets(tokenizer, example['context_html'])
        tokens = [token.text for token in tokens_with_offsets]
        #logger.debug('# tokens: %s', len(tokens))
        #tags = tag_tokens_with_annotation_list(tokens_with_offsets, example['attributes'])
        #tags = tag_tokens_with_annotation_list(tokens_with_offsets, [
        tags, num_valids, num_skipped = tag_tokens_with_annotation_list(tokens_with_offsets, [
            {'attribute': attribute, 'html_offset': html_offset}
            for attribute, html_offset in zip(
                example['attributes.attribute'], example['attributes.html_offset']
            )
        ])
        tags = {name: str.join('', chars) for name, chars in tags.items()}
        #logger.debug('tags: %s', tags)
        return {
            'tokens': tokens,
            'tags': tags,
        }
    dataset = dataset.map(tokenize_and_tag)
    logger.debug('dataset: %s', dataset)

    if args.split_eval_ratio > 0:
        dataset = dataset['train'].train_test_split(
            test_size=args.split_eval_ratio, seed=args.seed
        )

    def split_into_windows(examples):
        windows = {
            field: []
            for field in [
                'page_id', 'window_id', 'title', 'category_name', 'ENE', 'tokens', 'tags'
            ]
        }
        for index, page_id in enumerate(examples['page_id']):
            tokens = examples['tokens'][index]
            tags = examples['tags'][index]
            #for i in range(0, len(tokens), args.window_size - args.window_overlap_size):
            for window_id, i in enumerate(
                range(0, len(tokens), args.window_size - args.window_overlap_size)
            ):
                window_tokens = tokens[i:i+args.window_size]
                window_tags = {
                    attribute_name: tag_list[i:i+args.window_size]
                    for attribute_name, tag_list in tags.items() if tag_list is not None
                }
                windows['page_id'].append(page_id)
                windows['window_id'].append(window_id)
                windows['title'].append(examples['title'][index])
                windows['category_name'].append(examples['category_name'][index])
                windows['ENE'].append(examples['ENE'][index])
                windows['tokens'].append(window_tokens)
                windows['tags'].append(window_tags)
        return windows
    #dataset['train'] = dataset['train'].map(
    dataset = dataset.map(
        split_into_windows,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    logger.debug('dataset: %s', dataset)
    #logger.debug('dataset["train"][0]: %s', dataset['train'][0])

    tag_list = ['O', 'B', 'I']
    map_tag_to_id = {tag: i for i, tag in enumerate(tag_list)}
    num_labels = len(tag_list)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        #num_labels=num_labels * num_attribute_names,
        #num_attribute_names=num_attribute_names,
        #finetuning_task=args.task_name
    )

    #model = AutoModelForTokenClassification.from_pretrained(
    #    args.model_name_or_path,
    #    from_tf=bool(".ckpt" in args.model_name_or_path),
    #    config=config,
    #    ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    #)
    model = TokenMultiClassificationModel(
    #model = TokenMultiClassificationModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config,
        num_attribute_names = num_attribute_names,
        #num_labels=num_labels,
        #pretrained_model_name_or_path=args.model_name_or_path,
        #ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    logger.debug('model: %s', model)

    # # We resize the embeddings only when necessary to avoid index errors.
    # # If you are creating a model from scratch on a small vocab and want
    # # a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # #if len(tokenizer) > embedding_size:
    # #    embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))
    # logger.debug('embedding_size: %s', embedding_size)

    #label_list = ['O', 'B', 'I']

    #model.config.label2id = label_to_id
    #model.config.label2id = map_tag_to_id
    #model.config.label2id = {l: i for i, l in enumerate(label_list)}
    #model.config.id2label = label_list
    model.config.label2id = map_tag_to_id
    model.config.id2label = tag_list

    logger.debug('new model.config: %s', model.config)

    # Preprocessing the datasets.
    # First we tokenizer all the texts.
    padding = "max_length" if args.pad_to_max_length else False
    logger.debug('padding: %s', padding)

    logger.debug('attribute_names: %s', attribute_names)
    def prepare_ids(example):
        token_ids = tokenizer.encode(example['tokens'])
        tag_ids = [
            [map_tag_to_id['O']] * len(token_ids)
            for _ in range(len(attribute_names))
        ]
        #tag_ids = [map_tag_to_id[tag] for tag in example['tags']]
        for attribute_name, tags in example['tags'].items():
            attribute_id = attribute_names.index(attribute_name)
            ene = example['ENE']
            attirbute_name_set = map_ene_to_attribute_name_set[ene]
            if ene not in attirbute_name_set:
                tag_ids[attribute_id] = [-100] * len(token_ids)
            if tags is None:
                continue
            tag_ids[attribute_id][0] = -100
            for index, tag in enumerate(tags):
                tag_ids[attribute_id][index+1] = map_tag_to_id[tag]
            tag_ids[attribute_id][-1] = -100
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
    dataset = dataset.map(prepare_ids, remove_columns=dataset['train'].column_names)
    logger.debug('dataset: %s', dataset)
    #logger.debug('dataset["train"][0]: %s', dataset['train'][0])

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    def my_collator(examples):
        #logger.debug('# examples: %s', len(examples))
        #logger.debug('examples: %s', examples)
        #logger.debug('examples["input_ids"]: %s', examples["input_ids"])
        input_ids = [
            #torch.tensor(ids, dtype=torch.int32) for ids in examples['input_ids']
            #torch.tensor(example['input_ids'], dtype=torch.int32)
            torch.tensor(example['input_ids'], dtype=torch.int64)
            for example in examples
        ]
        # cuda の nll_loss のターゲットは long にしか対応していない模様
        tag_ids = [
            #torch.tensor(ids, dtype=torch.int8) for ids in examples['labels']
            #torch.tensor(example['labels'], dtype=torch.int8)
            #torch.tensor(example['labels'], dtype=torch.int8).transpose(1, 0)
            #'tag_ids': pad_sequence(tag_ids, batch_first=True),
            #torch.tensor(example['tag_ids'], dtype=torch.int8).transpose(1, 0)
            #torch.tensor(example['tag_ids'], dtype=torch.int32).transpose(1, 0)
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
        #logger.debug('input_ids shapes: %s', [input_id.shape for input_id in input_ids])
        #logger.debug('label shapes: %s', [label.shape for label in labels])
        #logger.debug('mask: %s', mask)
        #logger.debug('mask.shape: %s', [m.shape for m in mask])
        return {
            'input_ids': pad_sequence(input_ids, batch_first=True),
            #'labels': pad_sequence(labels, batch_first=True),
            #'tag_ids': pad_sequence(tag_ids, batch_first=True),
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

        #logger.debug('y_pred: %s', y_pred)
        #logger.debug('y_true: %s', y_true)
        #logger.debug('y_pred shape: %s', y_pred.shape)
        #logger.debug('y_true shape: %s', y_true.shape)

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
            logger.debug('resume_path: %s', resume_path)
            accelerator.load_state(resume_path)
    logger.debug('resume_path: %s', resume_path)

    starting_epoch = 0
    resume_step = -1
    # Extract `epoch_{i}` or `step_{i}`
    if resume_path:
        training_difference = os.path.splitext(resume_path)[0]
        logger.debug('training_difference: %s', training_difference)

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.split("_")[1])
            #resume_step = None
        else:
            resume_step = int(training_difference.split("_")[1])
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    logger.debug('starting_epoch: %d', starting_epoch)
    logger.debug('resume_step: %d', resume_step)

    def do_eval():
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            #logger.debug('logits shape: %s', outputs.logits.shape)
            predictions = outputs.logits.argmax(dim=-1)
            #logger.debug('predictions shape: %s', predictions.shape)
            #labels = batch["labels"]
            tag_ids = batch["tag_ids"]
            #if not args.pad_to_max_length:
            #    # necessary to pad predictions and labels for being gathered
            #    predictions = accelerator.pad_across_processes(
            #        predictions, dim=1, pad_index=-100,
            #    )
            #    #labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            #    labels = accelerator.pad_across_processes(tag_ids, dim=1, pad_index=-100)
            #predictions_gathered, labels_gathered = accelerator.gather([predictions, labels])
            #predictions_gathered, labels_gathered = accelerator.gather([predictions, tag_ids])
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
        return eval_metric


    #for epoch in range(starting_epoch, args.num_train_epochs):
    for epoch in range(0, args.num_train_epochs):

        if starting_epoch > epoch:
            completed_steps += len(train_dataloader)
            progress_bar.update(len(train_dataloader))
            continue

        model.train()
        #if args.with_tracking:
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            #logger.debug('step: %d', step)
            #logger.debug('batch: %s', batch)
            #logger.debug('batch input_ids shape: %s', batch['input_ids'].shape)
            #logger.debug('batch attention_mask shape: %s', batch['attention_mask'].shape)
            #logger.debug('batch labels shape: %s', batch['labels'].shape)
            # We need to skip steps until we reach the resumed step
            #if args.resume_from_checkpoint and epoch == starting_epoch:
            #    if resume_step is not None and step <= resume_step:
            #        completed_steps += 1
            #        continue
            if starting_epoch == epoch:
                #if resume_step is not None and step <= resume_step:
                #if step <= resume_step:
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
                    output_dir = f"step_{completed_steps}"
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
                #{
                #    "seqeval": eval_metric,
                #    "train_loss": total_loss.item() / len(train_dataloader),
                #    "epoch": epoch,
                #    "step": completed_steps,
                #},
                step=completed_steps,
            )

        #if args.checkpointing_steps == "epoch":
        if True:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
        #accelerator.save(
            #model,
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

if __name__ == '__main__':
    main()
