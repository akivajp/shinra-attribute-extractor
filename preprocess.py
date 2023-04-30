'''
データセットの前処理
'''

import dataclasses

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
)

from logzero import logger

from datasets import Dataset
from transformers import PreTrainedTokenizer

from html_cleaning import clean_up_html
from tokenization import tokenize_with_offsets
from tagging import tag_tokens_with_annotation_list

@dataclasses.dataclass
class PreprocessResult:
    '''
    前処理結果
    '''
    dataset: Dataset
    attribute_names: List[str]
    map_attribute_name_to_id: Dict[str, int]
    map_ene_to_attribute_name_counter: Dict[str, Dict[str, int]]
    tag_list: List[str]
    map_tag_to_id: Dict[str, int]
    map_page_id_to_information: Dict[str, Dict[str, Any]]

def slice_batch_from_keys(
    batch: Mapping[str, Iterable],
    keys: Iterable[str],
):
    '''
    keys (取得したい特徴量名一覧) を元に
    Dict[str, Iterable] (厳密にはLazyDict) から
    Iterable[Dict[str, Any]] に変換する
    '''
    if not keys:
        raise ValueError('keys must not be empty')
    map_key_to_index = {key: i for i, key in enumerate(keys)}
    for example in zip(*(batch[key] for key in keys)):
        yield {
            key: example[map_key_to_index[key]]
            for key in keys
        }

def convert_example_mapper_to_batch_mapper(
    example_mapper: Callable[[Dict[str, Any]], Dict[str,Any]|List[Dict[str,Any]]],
    keys: Iterable[str],
):
    '''
    example_mapper (Iterable[Dict[str, Any]]) を
    batch_mapper (Iterable[Dict[str, Any]]) に変換する
    '''
    def batch_mapper(batch):
        map_key_to_list_features = None
        #for sample in slice_batch_from_keys(batch, keys):
        #for mapped in slice_batch_from_keys(batch, keys):
        for example in slice_batch_from_keys(batch, keys):
            #mapped_sample = sample_mapper(sample)
            mapped = example_mapper(example)
            if isinstance(mapped, dict):
                # 1つのサンプルで1つのマップ結果が返された場合
                # 単一要素のリストとして扱う
                mapped_examples = [mapped]
            elif isinstance(mapped, list):
                # 複数のサンプルが返された場合
                # (e.g. テキストを複数の文に分割した場合)
                mapped_examples = mapped
            else:
                raise ValueError(f'mapped is not dict or list: {mapped}')
            for mapped_example in mapped_examples:
                if map_key_to_list_features is None:
                    # 最初に取得したサンプルの特徴量名一覧を元に
                    # Dict[str, list] を用意する
                    map_key_to_list_features = {
                        key: [] for key in mapped_example.keys()
                    }
                for key, feature in mapped_example.items():
                    map_key_to_list_features[key].append(feature)
        return map_key_to_list_features
    return batch_mapper

# トークン化と IOB2 タグの付与
#def tokenize_and_tag(example):
def tokenize_and_tag(
    tokenizer: PreTrainedTokenizer,
    map_attribute_name_to_id: Dict[str, int],
    map_ene_to_attribute_name_counter: Dict[str, Dict[str, int]],
):
    def _tokenize_and_tag(example):
        tokens_with_offsets = tokenize_with_offsets(tokenizer, example['context_html'])
        tokens = [token.text for token in tokens_with_offsets]
        tags, num_tagged, num_skipped = tag_tokens_with_annotation_list(
            tokens_with_offsets,
            [
                {'attribute': attribute, 'html_offset': html_offset}
                for attribute, html_offset in zip(
                    example['attributes.attribute'], example['attributes.html_offset']
                )
            ]
        )
        #extended_tags = [
        #    ' ' * len(tokens)
        #    for _ in range(len(attribute_names))
        #]
        #extended_tags = [None] * num_attribute_names
        extended_tags = [None] * len(map_attribute_name_to_id)
        ene = example['ENE']
        attribute_name_counter = map_ene_to_attribute_name_counter[ene]
        for attribute_name in attribute_name_counter:
            attribute_id = map_attribute_name_to_id[attribute_name]
            if attribute_name in tags:
                extended_tags[attribute_id] = str.join('', tags[attribute_name])
            else:
                extended_tags[attribute_id] = 'O' * len(tokens)
        #logger.debug('tags: %s', tags)
        return {
            'tokens': tokens,
            'tags': extended_tags,
        }
    return _tokenize_and_tag


# 通常、512トークンを超える系列の処理はそのままできないため、
# スライドウィンドウを用いて分割する
# (データセットのチャンク分割はバッチ処理でのみ可能)
#def split_into_windows(example):
def split_into_windows(
    window_size = 510,
    window_overlap_size = 128,
):
    def _split_into_windows(example):
        windows = []
        tokens = example['tokens']
        tags = example['tags']
        for window_id, i in enumerate(
            range(0, len(tokens), window_size - window_overlap_size)
        ):
            window_tokens = tokens[i : i+window_size]
            #window_tags = [tag_list[i:i+args.window_size] for tag_list in tags]
            window_tags = [
                tag_list[i : i+window_size] if tag_list is not None
                else None
                for tag_list in tags
            ]
            window = {}
            window['page_id'] = example['page_id']
            window['title'] = example['title']
            window['category_name'] = example['category_name']
            window['ENE'] = example['ENE']
            window['window_id'] = window_id
            window['tokens'] = window_tokens
            window['tags'] = window_tags
            windows.append(window)
        return windows
    return _split_into_windows

#logger.debug('attribute_names: %s', attribute_names)
# トークン ID とタグ ID の付与
#def prepare_ids(example):
def prepare_ids(
    tokenizer: PreTrainedTokenizer,
    map_tag_to_id: Dict[str, int],
):
    def _prepare_ids(example):
        token_ids = tokenizer.encode(example['tokens'])
        # 先頭 (CLS), 末尾 (SEP) はタグの予測不要
        # BIO以外の文字(空白)はタグの予測不要
        tag_ids = [
            [-100] + [map_tag_to_id.get(char, -100) for char in tag_str] + [-100]
            if tag_str is not None
            else None
            for tag_str in example['tags']
        ]
        return {
            'input_ids': token_ids,
            'tag_ids': tag_ids,
        }
    return _prepare_ids

def preprocess_dataset(
    raw_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    target_categories = None,
    num_workers = None,
    seed = None,
    split_eval_size = 0,
    window_size = 510,
    window_overlap_size = 128,
):
    dataset = raw_dataset

    # 今回使わない特徴量は削除しておく
    dataset = dataset.flatten()
    dataset = dataset.remove_columns(['context_text', 'attributes.text_offset'])
    
    # --target_categories が与えられた場合は、そのカテゴリのみで絞り込む
    target_categories = None
    if target_categories is not None:
        target_categories = target_categories.split(',')
        logger.debug('target_categories: %s', target_categories)
        dataset = dataset.filter(
            lambda e: e['category_name'] in target_categories,
            desc='Filtering with target categories',
        )
    logger.debug('dataset: %s', dataset)

    # 全属性名のリストと、
    # ENE に対応する属性名のリストの辞書を作成
    set_attribute_names = set()
    # NOTE: 超重要
    #   Dict[str,Set] で事足りることだけど、
    #   何故か Dataset.map()の適用関数内でsetオブジェクトに依存すると
    #   関数が同一視できなくなるらしくキャッシュロードできなくなるので(不具合?)
    #   計算量的に Dict[str,Dict] で代用する
    #   ついでに属性の出現回数でもカウントしておく
    #map_ene_to_attribute_name_set: dict[str,set] = {}
    map_ene_to_attribute_name_counter: dict[str,dict[str,int]] = {}
    for example in dataset['train']:
        ene = example['ENE']
        for attribute_name in example['attributes.attribute']:
            set_attribute_names.add(attribute_name)
            # NOTE: setの代わりにdictを使う必要あり(前述の理由)
            # if ene not in map_ene_to_attribute_name_set:
            #     map_ene_to_attribute_name_set[ene] = set()
            # map_ene_to_attribute_name_set[ene].add(attribute_name)
            if ene not in map_ene_to_attribute_name_counter:
                map_ene_to_attribute_name_counter[ene] = {}
            count = map_ene_to_attribute_name_counter[ene].get(attribute_name, 0)
            map_ene_to_attribute_name_counter[ene][attribute_name] = count + 1
    attribute_names = sorted(set_attribute_names)
    #mappers = Mappers()
    map_attribute_name_to_id = {name: i for i, name in enumerate(attribute_names)}
    num_attribute_names = len(attribute_names)
    #logger.debug('attribute_names: %s', attribute_names)

    # HTML をクリーニング
    # 属性値抽出の対象とならないであろう箇所を除去
    # HTMLタグは一般的なサブワード分割と相性が悪く
    # 無駄にトークン数が増えてしまうのは防ぎたい
    def clean_context_html(example):
        cleaned_html = clean_up_html(example['context_html'])
        return {'context_html': cleaned_html}
    # デフォルトの batch_size=1000 だと処理1回の負荷が大きすぎる
    dataset = dataset.map(
        convert_example_mapper_to_batch_mapper(clean_context_html, ['context_html']),
        batched=True,
        desc='Cleaning HTML files',
        batch_size=10,
        writer_batch_size=10,
        num_proc=num_workers,
    )
    logger.debug('dataset: %s', dataset)

    # デフォルトの writer_batch_size は 1000 で、
    # 1サンプルのサイズが大きい場合に結合に失敗することがあるので制限する必要がある
    # writer_batch_size=100, batch_size=100 でもちょっと怪しい
    dataset = dataset.map(
        #tokenize_and_tag,
        convert_example_mapper_to_batch_mapper(
            tokenize_and_tag(
                tokenizer,
                map_attribute_name_to_id,
                map_ene_to_attribute_name_counter,
            ),
            [
                'context_html',
                'attributes.attribute',
                'attributes.html_offset',
                'ENE',
            ]
        ),
        batched=True,
        desc='Tokenizing and tagging',
        batch_size=10,
        writer_batch_size=10,
        #num_proc=args.num_workers,
        num_proc=num_workers,
    )
    #logger.debug('dataset: %s', dataset)

    if split_eval_size> 0:
        if split_eval_size >= 1:
            split_eval_size = int(split_eval_size)
        # ラベル付きデータを訓練用と評価用に分割
        # ページ単位で分割することが好ましいので、ウィンドウ化よりも前に行う必要あり
        dataset = dataset['train'].train_test_split(
            test_size=split_eval_size,
            seed=seed,
        )
        logger.debug('dataset: %s', dataset)
    #logger.debug('dataset test page_id: %s', dataset['test']['page_id'])

    dataset = dataset.map(
        #split_into_windows,
        convert_example_mapper_to_batch_mapper(
            split_into_windows(
                window_size,
                window_overlap_size,
            ), [
                'page_id',
                'title',
                'category_name',
                'ENE',
                'tokens',
                'tags',
                'tokens',
            ]
        ),
        desc='Splitting into slide windows',
        batched=True,
        remove_columns=dataset['train'].column_names,
        batch_size=10,
        writer_batch_size=10,
        num_proc=num_workers,
    )
    logger.debug('dataset: %s', dataset)

    tag_list = ['O', 'B', 'I']
    map_tag_to_id = {tag: i for i, tag in enumerate(tag_list)}
    num_labels = len(tag_list)

    dataset = dataset.map(
        #prepare_ids,
        convert_example_mapper_to_batch_mapper(
            prepare_ids(
                tokenizer,
                map_tag_to_id,
            ),
            [
                'tokens',
                'tags',
            ]
        ),
        batched=True,
        desc='Converting tokens and tags into ids',
        batch_size=10,
        writer_batch_size=10,
        num_proc=num_workers,
    )
    logger.debug('dataset: %s', dataset)
    #logger.debug('dataset["train"][0]: %s', dataset['train'][0])

    map_page_id_to_information = {}
    def record_information(example):
        keys = [
            'page_id',
            'title',
            'category_name',
            'ENE',
        ]
        page_id = example['page_id']
        info = {
            key: example[key] for key in keys
        }
        map_page_id_to_information[page_id] = info
    raw_dataset.map(
        record_information,
        desc='Recording page information',
    )

    #return dataset
    return PreprocessResult(
        dataset,
        attribute_names,
        map_attribute_name_to_id,
        map_ene_to_attribute_name_counter,
        tag_list,
        map_tag_to_id,
        map_page_id_to_information,
    )
