#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import unicodedata

from html.parser import HTMLParser

from dataclasses import (
    dataclass,
)

from logzero import logger
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
)

DEBUG_MODE = False

OVERLAP_STRATEGIES = [
    'front-longest',
    'longest-front',
]
DEFAULT_OVERLAP_STRATEGY = OVERLAP_STRATEGIES[0]
#DEFAULT_OVERLAP_STRATEGY = OVERLAP_STRATEGIES[1]

@dataclass
class TraceableToken:
    '''
    トークンとその出現位置を保持する
    '''
    text: str
    start_line: int
    start_offset: int
    end_line: int
    end_offset: int

def map_substr(lines, start_line, start_offset, end_line, end_offset, map_func):
    '''
    文字列の指定した範囲を関数に渡して置き換える
    '''
    assert start_line <= end_line
    for line_index in range(start_line, end_line+1):
        if line_index >= len(lines):
            # 末尾の空行を超えた場合
            # (HTMLParser.feedで最後に加えられた改行の可能性)
            break
        line = lines[line_index]
        if line_index == start_line:
            if line_index == end_line:
                # タグが同じ行にある場合
                lines[line_index] = \
                    line[:start_offset] + \
                    map_func(line[start_offset:end_offset]) + \
                    line[end_offset:]
            else:
                assert line_index < end_line
                # タグが複数行にまたがる場合
                lines[line_index] = \
                    line[:start_offset] + \
                    map_func(line[start_offset:])
        else:
            # line_index > start_line
            if line_index == end_line:
                lines[line_index] = \
                    map_func(line[:end_offset]) + \
                    line[end_offset:]
                #break
            else:
                # line_index < end_line
                lines[line_index] = map_func(line)
    return lines

def replace_with_spaces(substr):
    def character_to_space(c):
        if c == '\n':
            return c
        else:
            return ' '
    return str.join('', map(character_to_space, substr))

#def remove_ranges(html, remove_positions):
def remove_ranges(lines, remove_positions):
    '''
    指定した範囲を削除する
    '''
    #def replace_with_spaces(substr):
    #    return ' ' * len(substr)
    for start_pos, end_pos in remove_positions:
        start_line = start_pos[0]
        start_offset = start_pos[1]
        end_line = end_pos[0]
        end_offset = end_pos[1]
        lines = map_substr(lines, start_line, start_offset, end_line, end_offset, replace_with_spaces)
    return lines

class ExtendedHTMLParser(HTMLParser):
    '''
    1オリジンの行番号を0オリジンに修正する
    '''
    def reset(self):
        super().reset()
        self.lineno = 0

    def get_new_pos(self, offset, forward):
        lineno, offset = self.getpos()
        buffer = self.rawdata[offset:offset+forward]
        num_newlines = buffer.count('\n')
        if num_newlines:
            last_line = buffer[buffer.rfind('\n')+1:]
            lineno += num_newlines
            offset = len(last_line)
        else:
            offset += len(buffer)
        #self.start_pos = (lineno, offset)
        return lineno, offset

def pickup_html_tags(lines, target_tag, target_attrs=None):
    '''
    HTMLタグを抽出する
    一致しない部分は半角スペースで置き換える
    '''
    remove_positions = []
    class Parser(ExtendedHTMLParser):
        '''
        該当するタグ以外の場所を記録する
        '''
        def __init__(self):
            super().__init__()
            #self.start_pos = None
            self.start_pos = self.getpos()
            #self.start_pos = self.getpos0()
            self.count_tags = 0
            self.found_end = False
            self.last_closed_tag = None

        def handle_starttag(self, tag, attrs):
            if tag != target_tag:
                return
            if self.count_tags > 0:
                # 既にタグが見つかっている場合は同名タグでカウントアップ
                self.count_tags += 1
                return
            if target_attrs is not None:
                for attr in attrs:
                    # 属性名が一致しない場合は無視
                    if attr[0] not in target_attrs:
                        return
                    # 属性値が一致しない場合は無視
                    if attr[1] != target_attrs[attr[0]]:
                        return
            self.count_tags += 1
            # タグ直前までの内容を削除する
            remove_positions.append([self.start_pos, self.getpos()])
            return
        def handle_endtag(self, tag):
            self.last_closed_tag = tag
            if tag != target_tag:
                return
            if self.count_tags > 0:
                self.count_tags -= 1
                if self.count_tags == 0:
                    self.found_end = True
        def parse_endtag(self, i: int):
            gtpos = super().parse_endtag(i)
            # この時点で self.handle_endtag は実行されている
            if self.count_tags > 0 and self.last_closed_tag == target_tag:
                self.start_pos = self.get_new_pos(i, gtpos-i)
                return gtpos
            if self.found_end:
                self.start_pos = self.get_new_pos(i, gtpos-i)
                self.found_end = False
            return gtpos
    parser = Parser()
    for line in lines:
        parser.feed(line + "\n")
    if not remove_positions:
        # 1つもタグが見つからなかった場合はエラー
        raise ValueError(f'tag not found: {target_tag}, {target_attrs}')
    # 最後に見つかったタグ直後から最後までも削除する
    remove_positions.append([parser.start_pos, parser.getpos()])
    # 指定された位置を取り除く
    return remove_ranges(lines, remove_positions)

#re_tag_attribute = re.compile(r'(\w+)=["\']([^"\']+)["\']')
re_tag_attribute = re.compile(r'[a-zA-Z0-9_]+="[^"]*"')
def remove_html_attributes(lines, keep_attributes=None):
    '''
    HTMLタグの属性を削除する
    '''
    if keep_attributes is None:
        keep_attributes = set()
    else:
        keep_attributes = set(keep_attributes)
    remove_positions = []
    class Parser(ExtendedHTMLParser):
        '''
        属性値を検出して削除位置を登録する
        '''
        def parse_starttag(self, i: int):
            endpos = super().parse_starttag(i)
            # この時点で self.handle_starttag は実行されている
            buffer = self.rawdata[i:endpos]
            offset = 0
            while True:
                found = re.search(re_tag_attribute, buffer[offset:])
                if not found:
                    break
                attr = found.group(0)
                attr_name, attr_value = attr.split('=', 1)
                if attr_name in keep_attributes:
                    #start = start + found.end()
                    offset += found.end()
                    continue
                remove_positions.append([
                    self.get_new_pos(i, offset+found.start()),
                    self.get_new_pos(i, offset+found.end()),
                ])
                #logger.debug('parse_startag appended: %s', remove_positions[-1])
                offset += found.end()
            return endpos
    parser = Parser()
    for line in lines:
        parser.feed(line + "\n")
    return remove_ranges(lines, remove_positions)


comment_tag = re.compile(r'<!--.*?-->', re.DOTALL)
def remove_comments(html):
    '''
    コメントを削除する
    '''
    #return comment_tag.sub(' ', html)
    while True:
        found = comment_tag.search(html)
        if found:
            substr = replace_with_spaces(found.group())
            html = html[:found.start()] + substr + html[found.end():]
        else:
            break
    return html

def clean_up_html(html):
    html = remove_comments(html)
    lines = html.splitlines()
    # Wikipediaのバージョンによっては <div class="mw-parser-output"> がない場合がある
    #lines = pickup_html_tags(lines, 'div', {'class': 'mw-parser-output'})
    # Wikipediaのバージョンによっては <div id="content"> がない場合がある
    #lines = pickup_html_tags(lines, 'div', {'id': 'content'})
    # Wikipediaのバージョンによっては <main id="content"> がない場合がある
    #lines = pickup_html_tags(lines, 'main', {'id': 'content'})
    lines = pickup_html_tags(lines, 'div')
    lines = remove_html_attributes(lines)
    return str.join("\n", lines)
    
def normalize(text):
    # Unicode正規化 
    return unicodedata.normalize('NFKC', text)

def tokenize_with_positions(tokenizer, text) -> list[TraceableToken]:
    # トークナイズした上で元のポジション位置を突き合わせる
    lines = text.splitlines()
    traceable_tokens = []
    if tokenizer.subword_tokenizer_type != 'wordpiece':
        raise ValueError(f'unsupported subword tokenizer: {tokenizer.subword_tokenizer_type}')
    
    def decompose_line(line, line_index):
        # ライン文字列を構成文字単位でトークナイズする
        # (TRADE_MARK_SIGNのような合字をサブワード化された '##T' などとマッチさせるための対策)
        traceable_chars: list[TraceableToken] = []
        for i, c in enumerate(line):
            assert len(c) == 1
            if c in [' ', '\t', '　']:
                continue
            sub_chars = normalize(c)
            for sub_char in sub_chars:
                # 合字の場合は1文字ずつ、元の合字の開始位置と終了位置を記録する
                # 合字でなければそのままトークン化する
                if sub_char in [' ']:
                    # 稀にノーマライズすると空白文字が入っている場合がある
                    # 例: ACUTE ACCENT -> SPACE + COMBINING ACUTE ACCENT
                    continue
                traceable_char = TraceableToken(
                    text=sub_char,
                    start_line=line_index,
                    start_offset=i,
                    end_line=line_index,
                    end_offset=i+1,
                )
                traceable_chars.append(traceable_char)
        return traceable_chars
    
    def merge_tokens(tokens: list[TraceableToken]):
        # トークンを結合して文字列を生成する
        #return str.join('', [token.text for token in tokens])
        return normalize( str.join('', [token.text for token in tokens]) )
    
    for line_index, line in enumerate(lines):
        #if DEBUG_MODE and line_index > 5:
        #    break
        line_chars = decompose_line(line, line_index)
        words = tokenizer.word_tokenizer.tokenize(line)
        offset = 0
        for word in words:
            #logger.debug('word: %s', word)
            #for token in tokens:
            tokens = tokenizer.subword_tokenizer.tokenize(word)
            if tokens == [tokenizer.unk_token]:
                if len(word) >= 2:
                    # 複数文字で構成される単語が未知語となった場合は
                    # 1文字ずつトークン化する
                    #logger.debug('tokens: %s', tokens)
                    #tokens = [tokenizer.subword_tokenizer.tokenize(c) for c in word]
                    tokens = []
                    for c in word:
                        tokens.extend(tokenizer.subword_tokenizer.tokenize(c))
                    #logger.debug('new tokens: %s', tokens)
            #for token in tokens:
            for token_index, token in enumerate(tokens):
                if token == tokenizer.unk_token:
                    # 未知語トークンは元の1文字に戻す
                    #token = line[offset]
                    token = line_chars[offset].text
                #logger.debug('token: %s', token)
                actual_token = token
                if token_index >= 1:
                    # サブワードの '##' は2番目以降のサブワードにしか付与されない
                    if token.startswith('##'):
                        actual_token = token[2:]
                #logger.debug('actual token: %s', actual_token)
                normal_token = normalize(actual_token)
                #logger.debug('normal token: %s', normal_token)
                #start_offset = line.find(actual_token, offset)
                end_offset = offset + 1
                matched = False
                #for end_offset in range(offset+1, len(line)+1):
                #    if normalize(line[offset:end_offset]) == normal_token:
                #        matched = True
                #        break
                merged = None
                #for end_offset in range(offset+1, len(line_chars)+1):
                for end_offset in range(offset, len(line_chars)):
                    #merged = merge_tokens(line_chars[offset:end_offset])
                    merged = merge_tokens(line_chars[offset:end_offset+1])
                    #if DEBUG_MODE:
                    #    logger.debug('token: %s', token)
                    #    logger.debug('merged: %s', merged)
                    if normal_token == merged:
                        matched = True
                        break
                if not matched:
                    logger.error('token not found: %s', token)
                    logger.error('line: %s', line)
                    logger.error('offset: %s', offset)
                    logger.error('line digest: %s', line[:50])
                    logger.error('word: %s', word)
                    logger.error('tokens digest: %s', tokens[:20])
                    logger.error('line characters digest: %s', line_chars[:20])
                    logger.error('merged: %s', merged)
                    raise
                start_token = line_chars[offset]
                end_token = line_chars[end_offset]
                traceable_token = TraceableToken(
                    text=token,
                    start_line=start_token.start_line,
                    start_offset=start_token.start_offset,
                    end_line=end_token.end_line,
                    end_offset=end_token.end_offset,
                )
                traceable_tokens.append(traceable_token)
                #offset = end_offset
                offset = end_offset + 1
    return traceable_tokens

def tag_tokens_with_annotation(
    traceable_tokens: list[TraceableToken],
    records,
    overlap_strategy: str = DEFAULT_OVERLAP_STRATEGY,
):
    map_attribute_name_to_tags = {}
    def get_sort_key_func(strategy):
        if strategy == 'front-longest':
            def sort_key(rec):
                return (
                    rec['html_offset']['start']['line_id'], # 上にある方を優先
                    rec['html_offset']['start']['offset'], # 先に出現するものを優先
                    -len(rec['html_offset']['text']), # 長い文字列を優先
                )
        elif strategy == 'longest-front':
            def sort_key(rec):
                return (
                    -len(rec['html_offset']['text']), # 長い文字列を優先
                    rec['html_offset']['start']['line_id'], # 上にある方を優先
                    rec['html_offset']['start']['offset'], # 先に出現するものを優先
                )
        else:
            raise ValueError(f'unknown overlap strategy: {strategy}')
        return sort_key

    num_precessed_records = 0
    num_skipped_records = 0

    records = sorted(records, key=get_sort_key_func(overlap_strategy))
    #for i, record in enumerate(records):
    for record_index, record in enumerate(records):
        #if record_index > 2:
        if DEBUG_MODE and record_index > 2:
            break
        attribute_name = record['attribute']
        if attribute_name not in map_attribute_name_to_tags:
            map_attribute_name_to_tags[attribute_name] = ['O'] * len(traceable_tokens)
        tags = map_attribute_name_to_tags[attribute_name]
        #logger.debug('record: %s', record)
        start_line = record['html_offset']['start']['line_id']
        start_offset = record['html_offset']['start']['offset']
        end_line = record['html_offset']['end']['line_id']
        end_offset = record['html_offset']['end']['offset']
        attribute_text = record['html_offset']['text']
        last_token_index = None
        # Bタグを付けるべきトークンを探す
        tagged_b = False
        start_token_index = None
        token_index = -1
        token = None
        for token_index, token in enumerate(traceable_tokens):
            if token.start_line < start_line:
                #last_token = token
                last_token_index = token_index
                continue
            assert token.start_line == start_line, f'{token.start_line} != {start_line}'
            if token.start_offset < start_offset:
                last_token_index = token_index
                continue
            assert token.start_offset >= start_offset
            if token.start_offset == start_offset:
                # ちょうどアノテーション位置から始まるトークン
                start_token_index = token_index
            else:
                # アノテーション位置が始まる直前のトークン
                start_token_index = last_token_index
            if tags[start_token_index] == 'O':
                #tags[start_token_index] = 'B'
                #logger.debug('start_token: %s', token)
                tagged_b = True
            else:
                #raise ValueError(f'invalid tag: {tags[start_token_index]}')
                pass
            break
        if not tagged_b:
            if DEBUG_MODE:
                if record_index > 0:
                    prev_record = records[record_index-1]
                    logger.debug('prev_record: %s', prev_record)
                logger.debug('record: %s', record)
                logger.debug('record_index: %s', record_index)
                logger.debug('page_id: %s', record['page_id'])
                logger.debug('title: %s', record['title'])
                logger.debug('ENE_name: %s', record['ENE_name'])
                logger.debug('attribute name: %s', record['attribute'])
                logger.debug('attribute text: %s', attribute_text)
                if start_token_index:
                    logger.debug('invalid tag: %s', tags[start_token_index])
            if not start_token_index:
                logger.error('token_index: %s', token_index)
                logger.error('token: %s', token)
                raise
            num_skipped_records += 1
            continue
        # Iタグを付けるべきトークンがあるか探す
        end_token_index = start_token_index
        for token_index in range(start_token_index+1, len(traceable_tokens)):
            token = traceable_tokens[token_index]
            if token.start_line > end_line:
                break
            if token.start_line == end_line:
                if token.start_offset >= end_offset:
                    break
            if tags[token_index] == 'O':
                #tags[token_index] = 'I'
                #logger.debug('inter-token: %s', token)
                end_token_index = token_index
            else:
                if DEBUG_MODE:
                    # 既に別の位置で属性が付与されている
                    logger.debug('invalid tag: %s', tags[token_index])
                end_token_index = None
                num_skipped_records += 1
                break
        if end_token_index is not None:
            for token_index in range(start_token_index, end_token_index+1):
                if token_index == start_token_index:
                    tags[token_index] = 'B'
                else:
                    tags[token_index] = 'I'
            num_precessed_records += 1
    return (
        map_attribute_name_to_tags,
        num_precessed_records,
        num_skipped_records,
    )

def pre_process_train_data(
    input_dir,
    output_file,
    tokenizer,
    *,
    output_html_dir = None,
    target_categories = None,
    overlap_strategy = DEFAULT_OVERLAP_STRATEGY,
):
    logger.info('Pre-processing train data')
    logger.info('input_dir: %s', input_dir)
    logger.info('output_file: %s', output_file)

    if output_html_dir:
        os.makedirs(output_html_dir, exist_ok=True)

    annotation_dir = os.path.join(input_dir, 'annotation')
    html_dir = os.path.join(input_dir, 'html')
    if not os.path.isdir(annotation_dir):
        raise ValueError(f'directory not found: {annotation_dir}')
    if not os.path.isdir(html_dir):
        raise ValueError(f'directory not found: {html_dir}')
    entries = os.scandir(annotation_dir)
    categories = []
    for entry in tqdm(entries, desc='scanning annotation directory'):
        #logger.debug('entry: %s', entry)
        #logger.debug('entry.name: %s', entry.name)
        if entry.name.endswith('_dist.jsonl'):
            category = entry.name.replace('_dist.jsonl', '')
            if target_categories and category not in target_categories:
                # target_categories が与えられた場合は
                # 対象外のカテゴリは無視する
                continue
            categories.append(category)
    #logger.debug('categories: %s', categories)
    if not categories:
        raise ValueError('no categories found')

    categories.sort()
    with open(output_file, 'w', encoding='utf-8') as f_output_json:
        desc = 'loading annotation for each category'
        set_attribute_names = set()
        map_category_page_id_to_records = {}
        for i, category in enumerate(tqdm(categories, desc=desc)):
            if DEBUG_MODE and i > 1:
                break
            annotation_file = os.path.join(annotation_dir, f'{category}_dist.jsonl')
            map_page_id_to_records = {}
            with open(annotation_file, 'r', encoding='utf-8') as f_annotation:
                #for line in tqdm(f_annotation):
                for line in f_annotation:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec['ENE_name'] = category
                    page_id = rec['page_id']
                    set_attribute_names.add(rec['attribute'])
                    if not page_id in map_page_id_to_records:
                        map_page_id_to_records[page_id] = []
                    map_page_id_to_records[page_id].append(rec)
            map_category_page_id_to_records[category] = map_page_id_to_records
        list_attribute_names = sorted(set_attribute_names)
        logger.debug('# list_attribute_names: %s', len(list_attribute_names))
        processed_records_counter = tqdm(desc='pre-processed records')
        skipped_records_counter = tqdm(desc='skipped records')
        for category in tqdm(categories, desc='pre-processing categories'):
            if category not in map_category_page_id_to_records:
                continue
            map_page_id_to_records = map_category_page_id_to_records[category]
            desc = 'pre-processing category pages'
            for page_id, records in tqdm(map_page_id_to_records.items(), desc=desc, leave=False):
                output = {}
                first = records[0]
                output['page_id'] = first['page_id']
                output['title'] = first['title']
                output['ENE'] = first['ENE']
                output['ENE_name'] = first['ENE_name']
                if DEBUG_MODE:
                    logger.debug('page_id: %s', page_id)
                    logger.debug('# records: %s', len(records))
                html_file = os.path.join(html_dir, category, f'{page_id}.html')
                with open(html_file, 'r', encoding='utf-8') as f_html:
                    html = f_html.read()
                    try:
                        html = clean_up_html(html)
                        if output_html_dir:
                            output_html_file = os.path.join(output_html_dir, f'{page_id}.html')
                            with open(output_html_file, 'w', encoding='utf-8') as f_output_html:
                                f_output_html.write(html)
                        traceable_tokens = tokenize_with_positions(tokenizer, html)
                        (
                            mapped_tags,
                            num_precessed_records,
                            num_skipped_records,
                        ) = tag_tokens_with_annotation(traceable_tokens, records)
                        processed_records_counter.update(num_precessed_records)
                        skipped_records_counter.update(num_skipped_records)
                        tokens = [token.text for token in traceable_tokens]
                        output['tokens'] = tokens
                        output['tags'] = mapped_tags
                    except Exception as e:
                        logger.error('category: %s', category)
                        logger.error('title: %s', output['title'])
                        logger.error('page_id: %s', page_id)
                        raise e
                    #if DEBUG_MODE:
                    #    logger.debug('page_id: %s', page_id)
                    #    logger.debug('# tokens: %s', len(tokens))
                    #    sys.exit(1)
                    json.dump(output, f_output_json, ensure_ascii=False)

def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser('Pre-process annotation data')
    parser.add_argument(
        '--train_data_dir',type=str, required=True,
        help='Directory containing training data',
    )
    parser.add_argument(
        '--output_json_file', type=str, required=True,
        help='Output file to store pre-processed data (json lines)'
    )
    parser.add_argument(
        '--model_name_or_path', type=str, default='cl-tohoku/bert-base-japanese-v2',
        help='Pre-trained model name or path'
    )
    parser.add_argument(
        '--output_html_dir', type=str, default=None,
        help='Path to output directory to store pre-processed html files'
    )
    parser.add_argument(
        '--target_categories', type=str, default=None,
        help='Comma separated list of target categories',
    )
    parser.add_argument(
        '--overlap_strategy', type=str, default='longest-front', choices=OVERLAP_STRATEGIES,
        help='Strategy to handle overlapping annotations'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()
    if args.debug:
        DEBUG_MODE = True
    logger.debug('args: %s', args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    target_categories = None
    if args.target_categories:
        target_categories = args.target_categories.split(',')

    pre_process_train_data(
        args.train_data_dir,
        args.output_json_file,
        tokenizer,
        output_html_dir=args.output_html_dir,
        target_categories=target_categories,
        overlap_strategy=args.overlap_strategy,
    )

if __name__ == "__main__":
    main()
