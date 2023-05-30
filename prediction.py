'''
    推論関係の処理
'''

import json
import os

from accelerate import Accelerator

from collections import OrderedDict

from datasets import (
    Dataset,
)

import numpy as np

import torch
from torch.utils.data import (
    DataLoader,
)
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm

import transformers

from logzero import logger

# local packages

from preprocess import (
    prepare_for_prediction,
)

from model import (
    TokenMultiClassificationModel,
)

def extract(
    base_record,
    html_lines,
    offsets,
    entity_scores,
):
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
    record = dict(base_record)
    record['html_offset'] = {
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
    # 単純平均でスコアを算出
    record['score'] = sum(entity_scores) / len(entity_scores)
    return record

def predict(
    predict_dataset,
    html_dir,
    mapping,
    model: TokenMultiClassificationModel,
    tokenizer: transformers.PreTrainedTokenizer,
    window_size,
    window_overlap_size,
    data_collator,
    per_device_predict_batch_size,
    load_from_cache_file = False,
    #num_workers = 1,
    num_workers = 2,
):
    predict_dataset = prepare_for_prediction(
        predict_dataset,
        html_dir,
        mapping,
        tokenizer,
        num_workers = num_workers,
        window_size = window_size,
        window_overlap_size = window_overlap_size,
        load_from_cache_file = load_from_cache_file,
    )
    #logger.debug('predict_dataset: %s', predict_dataset)

    extracted = []

    if len(predict_dataset) == 0:
        # NOTE: データサイズが0の場合は何もしない
        # (例: 入力のENEs内のカテゴリが全て訓練データに存在せず推論不要と判断された場合)
        #return predict_dataset
        return extracted

    num_attribute_names = len(mapping.attribute_names)
    
    accelerator = (
        Accelerator()
    )
    #logger.debug('hash accelerator: %s', Hasher.hash(accelerator))

    predict_dataloader = DataLoader(
        predict_dataset,
        collate_fn=data_collator,
        batch_size=per_device_predict_batch_size,
        num_workers=num_workers,
    )

    #logger.debug('hash model: %s', Hasher.hash(model))
    model_acc, predict_dataloader = accelerator.prepare(
        model, predict_dataloader,
    )
    #logger.debug('hash model: %s', Hasher.hash(model))

    #model.eval()
    #model_acc.eval()
    map_window_key_to_prediction = OrderedDict()
    # 推論した結果を page_id と ENE と window_id で紐付ける
    for step, batch in enumerate(tqdm(
        predict_dataloader,
        desc='Feeding for prediction',
        leave=False,
    )):
        page_ids = batch['page_id']
        window_ids = batch['window_id']
        enes = batch['ENE']
        with torch.no_grad():
            #outputs = model(
            outputs = model_acc(
                input_ids = batch['input_ids'],
                attention_mask = batch.get('attention_mask'),
                token_type_ids = batch.get('token_type_ids'),
                decode=True,
            )
        #predictions = outputs.logits.argmax(dim=-1)
        predictions = outputs['decoded']
        extended_scores = outputs['scores']
        [
            page_ids_gathered,
            window_ids_gathered,
            predictions_gathered,
            scores_gathered
        ] = \
            accelerator.gather([page_ids, window_ids, predictions, extended_scores])
        page_ids = page_ids_gathered
        window_ids = window_ids_gathered
        predictions = predictions_gathered.detach().cpu().clone().numpy()
        predictions = predictions.transpose(0, 2, 1) # [B, L, A] -> [B, A, L]
        predictions = predictions[:, :, 1:] # 1トークン目はCLSなので除去
        extended_scores = scores_gathered.detach().cpu().clone().numpy()
        extended_scores = extended_scores.transpose(0, 2, 1) # [B, L, A] -> [B, A, L]
        extended_scores = extended_scores[:, :, 1:] # 1トークン目はCLSなので除去
        for page_id, window_id, ene, pred, extended_scores in zip(
            page_ids, window_ids, enes, predictions, extended_scores,
        ):
            # 推論の必要の無い属性名は None にしてメモリと計算量の節約
            attribute_id_counter = mapping.map_ene_to_attribute_id_counter[ene]
            pred_list = [
                None if attribute_id not in attribute_id_counter
                else p.astype(np.int8)
                for attribute_id, p in enumerate(pred)
            ]
            key = (page_id, ene, window_id)
            #map_window_key_to_prediction[key] = pred_list
            map_window_key_to_prediction[key] = [pred_list, extended_scores]

    # page_id と ENE と window_id と推論結果を元に
    # page_id と ENE に推論結果を集約
    map_page_key_to_predictions = OrderedDict()
    for example in tqdm(
        predict_dataset,
        desc = 'Merging windowed predictions into pages',
        leave=False,
    ):
        page_id = example['page_id']
        ene = example['ENE']
        title = example['title']
        window_id = example['window_id']
        window_key = (page_id, ene, window_id)
        prediction = map_window_key_to_prediction.get(window_key)
        if prediction is None:
            continue
        #example['prediction'] = prediction
        example['prediction'] = prediction[0]
        example['scores'] = prediction[1]
        #window_pred = {}
        #window_pred['window_id'] = window_id
        #window_pred['prediction'] = prediction[0]
        #window_pred['scores'] = prediction[1]
        #window_pred['tokens'] = example['tokens']
        #window_pred['tokens_with_offsets'] = example['tokens_with_offsets']
        page_key = (page_id, title, ene)
        map_page_key_to_predictions.setdefault(page_key, []).append(example)
        #map_page_key_to_predictions.setdefault(page_key, []).append(window_pred)

    def gen_page_key_predictions():
        #for (page_id, ene), window_preds in map_page_key_to_predictions.items():
        for (page_id, title, ene), windows in map_page_key_to_predictions.items():
            example = {}
            example['page_id'] = page_id
            example['title'] = title
            example['ENE'] = ene
            example['windows'] = windows
            yield example
    # NOTE: from_generatorの方がキャッシュが効いて都合がよいが
    # ローディングのログを無効化できないため from_list を使う
    #page_key_predictions = Dataset.from_generator(gen_page_key_predictions)
    #page_key_predictions = Dataset.from_list(list(gen_page_key_predictions()))
    #logger.debug('page_key_predictions: %s', page_key_predictions)
    page_key_predictions = list(gen_page_key_predictions())

    # ウィンドウをマージしつつ推論結果を出力
    def extract_from_example(example):
        extracted = []
        tokens_with_offsets = []
        tag_ids = np.array([], dtype=np.int8).reshape(num_attribute_names, 0)
        extended_scores = np.array([], dtype=np.float32).reshape(num_attribute_names, 0)
        windows = example['windows']
        page_id = example['page_id']
        title = example['title']
        ene = example['ENE']
        # リスト・配列を拡張しながらウィンドウをマージ
        for window in windows:
            window_start = window['window_id'] * (window_size - window_overlap_size)
            window_end = window_start + len(window['tokens'])
            if window_end > len(tokens_with_offsets):
                extension_length = window_end - len(tokens_with_offsets)
                tokens_with_offsets.extend([None] * extension_length)
                tag_ids_extension = np.zeros(
                    [num_attribute_names, extension_length],
                    dtype=np.int8
                )
                scores_extension = np.zeros(
                    [num_attribute_names, extension_length],
                    dtype=np.float32
                )
                tag_ids = np.concatenate([tag_ids, tag_ids_extension], axis=1)
                extended_scores = np.concatenate([extended_scores, scores_extension], axis=1)
            for i, token_with_offset in enumerate(window['tokens_with_offsets']):
                tokens_with_offsets[window_start+i] = token_with_offset
            for attribute_index, attribute_tag_ids in enumerate(window['prediction']):
                #scores = window['scores'][attribute_index]
                if attribute_tag_ids is None:
                    continue
                for i, tag_id in enumerate(attribute_tag_ids):
                    if i >= len(window['tokens']):
                        break
                    tag = mapping.tag_list[tag_id]
                    score = window['scores'][attribute_index][i]
                    if tag in ['B', 'I']:
                        previous_tag_id = tag_ids[attribute_index, window_start+i]
                        previous_tag = mapping.tag_list[previous_tag_id]
                        if previous_tag == 'O':
                            tag_ids[attribute_index, window_start+i] = tag_id
                            extended_scores[attribute_index, window_start+i] = score
        html_path = os.path.join(html_dir, f'{page_id}.html')
        with open(html_path, 'r', encoding='utf-8') as f_html:
            html_lines = f_html.read().splitlines(keepends=True)
        attribute_id_counter = mapping.map_ene_to_attribute_id_counter[ene]
        for attribute_index, attribute_tag_ids in enumerate(tag_ids):
            attribute_name = mapping.attribute_names[attribute_index]
            if attribute_index not in attribute_id_counter:
                continue
            scores = extended_scores[attribute_index]
            entity_scores = []
            found_b = False
            offsets = {}
            base_record = {
                'page_id': page_id,
                'title': title,
                'ENE': ene,
                'attribute': attribute_name,
            }
            for i, tag_id in enumerate(attribute_tag_ids):
                score = scores[i]
                tag = mapping.tag_list[tag_id]
                token_with_offsets = tokens_with_offsets[i]
                if tag == 'B':
                    if found_b:
                        #extract(f, html_lines, offsets, entity_scores)
                        extracted.append(extract(base_record, html_lines, offsets, entity_scores))
                    #logger.debug('B tag')
                    found_b = True
                    offsets['start_line'] = token_with_offsets["start_line"]
                    offsets['start_offset'] = token_with_offsets["start_offset"]
                    offsets['end_line'] = token_with_offsets["end_line"]
                    offsets['end_offset'] = token_with_offsets["end_offset"]
                    entity_scores = [score]
                if tag == 'I':
                    offsets['end_line'] = token_with_offsets["end_line"]
                    offsets['end_offset'] = token_with_offsets["end_offset"]
                    entity_scores.append(score)
                if tag == 'O':
                    if found_b:
                        found_b = False
                        #extracted.append(extract(base_record, html_lines, offsets, entity_scores))
                        #try:
                        #    extracted.append(extract(base_record, html_lines, offsets, entity_scores))
                        #except Exception as e:
                        #    logger.error('base_record: %s', base_record)
                        #    logger.error('offsets: %s', offsets)
                        #    logger.error('entity_scores: %s', entity_scores)
                        #    raise e
                    entity_scores = []
            if found_b:
                #extract(f, html_lines, offsets, entity_scores)
                extracted.append(extract(base_record, html_lines, offsets, entity_scores))
        return extracted
    def extract_from_batch(batch):
        dataset = Dataset.from_dict(batch)
        extracted = []
        for example in dataset:
            extracted.extend(extract_from_example(example))
        return Dataset.from_list(extracted).to_dict()
    #extracted = page_key_predictions.map(
    #    extract_from_batch,
    #    batched=True,
    #    remove_columns=page_key_predictions.column_names,
    #    desc = 'Extracting predicted attributes',
    #    #load_from_cache_file=load_from_cache_file,
    #    load_from_cache_file=False,
    #    batch_size = 1,
    #    num_proc=num_workers,
    #)

    extracted = []
    for example in tqdm(
        page_key_predictions,
        desc = 'Extracting predicted attributes',
        leave=False,
    ):
        #logger.debug('example: %s', example)
        extracted.extend(extract_from_example(example))

    #logger.debug('extracted: %s', extracted)
    return extracted
    #return extracted.to_list()
