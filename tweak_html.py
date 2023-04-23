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
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

DEBUG_MODE = False

#head_tag = re.compile(r'<head[^>]*>.*</head>', re.DOTALL)

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
    #assert start_line <= end_line
    #if start_line == end_line:
    #    return lines[start_line][start_offset:end_offset]
    #else:
    #    assert start_line < end_line
    #    substr = lines[start_line][start_offset:]
    #    for line_index in range(start_line+1, end_line):
    #        substr += lines[line_index]
    #    substr += lines[end_line][:end_offset]
    #    return substr
    assert start_line <= end_line
    #for line_index, line in enumerate(lines):
    #logger.debug('# lines: %d', len(lines))
    #logger.debug('start_line: %d', start_line)
    #logger.debug('start_offset: %d', start_offset)
    #logger.debug('end_line: %d', end_line)
    #logger.debug('end_offset: %d', end_offset)
    for line_index in range(start_line, end_line+1):
        if line_index >= len(lines):
            # 末尾の空行を超えた場合
            # (HTMLParser.feedで最後に加えられた改行の可能性)
            break
        #logger.debug('line_index: %d', line_index)
        line = lines[line_index]
        #if line_index < start_line:
        #    continue
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
        #elif line_index == end_line:
        #    assert line_index > start_line
        #    lines[line_index] = \
        #        map_func(line[:end_offset]) + \
        #        line[end_offset:]
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

    #def getpos0(self, adjust_line=-1):
    #    return (self.lineno+adjust_line, self.offset)

#def pickup_html_tags(html, target_tag, target_attrs=None):
def pickup_html_tags(lines, target_tag, target_attrs=None):
    '''
    HTMLタグを抽出する
    一致しない部分は半角スペースで置き換える
    '''
    #tag_positions = []
    remove_positions = []
    #class Parser(HTMLParser):
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

        def handle_starttag(self, tag, attrs):
            if tag != target_tag:
                return
            if self.count_tags > 0:
                # 既にタグが見つかっている場合は同名タグでカウントアップ
                self.count_tags += 1
                #logger.debug('parse_endtag getpos: %s',self.getpos0())
                #logger.debug('parse_endtag getpos: %s',self.getpos())
                return
            #logger.debug('tag: %s', tag)
            #logger.debug('attrs: %s', attrs)
            if target_attrs is not None:
                for attr in attrs:
                    # 属性名が一致しない場合は無視
                    if attr[0] not in target_attrs:
                        return
                    # 属性値が一致しない場合は無視
                    if attr[1] != target_attrs[attr[0]]:
                        return
            #self.start_pos = self.getpos()
            self.count_tags += 1
            # タグ直前までの内容を削除する
            remove_positions.append([self.start_pos, self.getpos()])
            #remove_positions.append([self.start_pos, self.getpos0()])
            return
        def handle_endtag(self, tag):
            #endpos = super().handle_endtag(tag)
            if tag != target_tag:
                return
            if self.count_tags > 0:
                self.count_tags -= 1
                if self.count_tags == 0:
                    #logger.debug('endtag tag: %s', tag)
                    #logger.debug('endtag getpos: %s', self.getpos())
                    self.found_end = True
        def parse_endtag(self, i: int):
            #logger.debug('parse_endtag i: %s', i)
            gtpos = super().parse_endtag(i)
            # この時点で self.handle_endtag は実行されている
            if self.found_end:
                self.start_pos = self.get_new_pos(i, gtpos-i)
                #logger.debug('parse_endtag i: %s', i)
                #logger.debug('parse_endtag pos: %s', gtpos)
                #logger.debug('parse_endtag start_pos: %s', self.start_pos)
                self.found_end = False
            return gtpos
    parser = Parser()
    #parser.feed(html)
    for line in lines:
        parser.feed(line + "\n")
    #if not tag_positions:
    if not remove_positions:
        # 1つもタグが見つからなかった場合はエラー
        raise ValueError(f'tag not found: {target_tag}, {target_attrs}')
    # 最後に見つかったタグ直後から最後までも削除する
    remove_positions.append([parser.start_pos, parser.getpos()])
    #remove_positions.append([parser.start_pos, parser.getpos0()])
    # 指定された位置を取り除く
    #return remove_ranges(html, remove_positions)
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
            #logger.debug('parse_starttag buffer: %s',buffer)
            #logger.debug('parse_starttag i: %s', i)
            #logger.debug('parse_starttag endpos: %s', endpos)
            #logger.debug('parse_starttag pos: %s', self.getpos())
            offset = 0
            while True:
                #logger.debug('offset: %s', offset)
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
            #html = html[:found.start()] + ' ' * (found.end() - found.start()) + html[found.end():]
            #substr = replace_with_spaces(html[found.start():found.end()])
            substr = replace_with_spaces(found.group())
            html = html[:found.start()] + substr + html[found.end():]
        else:
            break
    return html

def clean_up_html(html):
    html = remove_comments(html)
    lines = html.splitlines()
    lines = pickup_html_tags(lines, 'div', {'class': 'mw-parser-output'})
    lines = remove_html_attributes(lines)
    return str.join("\n", lines)
    
def normalize(text):
    # Unicode正規化 
    return unicodedata.normalize('NFKC', text)

def tokenize_with_positions(tokenizer, text) -> list[TraceableToken]:
    # トークナイズした上で元のポジション位置を突き合わせる
    lines = text.splitlines()
    #line_index = 0
    #offset = 0
    traceable_tokens = []
    #logger.debug('DEBUG_MODE: %s', DEBUG_MODE)
    #logger.debug('tokenizer: %s', tokenizer)
    if tokenizer.subword_tokenizer_type != 'wordpiece':
        raise ValueError(f'unsupported subword tokenizer: {tokenizer.subword_tokenizer_type}')
    for line_index, line in enumerate(lines):
        #if DEBUG_MODE and line_index > 5:
        #    break
        #logger.debug('line: %s', line)
        #logger.debug('line digest: %s', line[:50])
        #tokens = tokenizer.tokenize(line)
        words = tokenizer.word_tokenizer.tokenize(line)
        offset = 0
        #logger.debug('words digest: %s', words[:20])
        #logger.debug('tokens digest: %s', tokens[:20])
        #for token in tokens:
        for word in words:
            #logger.debug('word: %s', word)
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
            for token in tokens:
                while line[offset] in [' ', '\t', '　']:
                    # 空白文字を読み飛ばす
                    offset += 1
                if token == tokenizer.unk_token:
                    # 未知語トークンは1文字になっているはず
                    token = line[offset]
                #logger.debug('token: %s', token)
                actual_token = token
                if token.startswith('##'):
                    actual_token = token[2:]
                #logger.debug('actual token: %s', actual_token)
                normal_token = normalize(actual_token)
                #logger.debug('normal token: %s', normal_token)
                #start_offset = line.find(actual_token, offset)
                end_offset = offset + 1
                matched = False
                for end_offset in range(offset+1, len(line)+1):
                    if normalize(line[offset:end_offset]) == normal_token:
                        matched = True
                        break
                #logger.debug('start_offset: %s', start_offset)
                #logger.debug('offset: %s', offset)
                #logger.debug('matched: %s', matched)
                #if start_offset == -1:
                if not matched:
                    logger.error('token not found: %s', token)
                    logger.error('line: %s', line)
                    logger.error('offset: %s', offset)
                    logger.error('line digest: %s', line[:50])
                    logger.error('tokens digest: %s', tokens[:20])
                    sys.exit(1)
                #end_offset = start_offset + len(actual_token)
                traceable_token = TraceableToken(
                    text=token,
                    start_line=line_index,
                    #start_offset=start_offset,
                    start_offset=offset,
                    end_line=line_index,
                    end_offset=end_offset,
                )
                traceable_tokens.append(traceable_token)
                offset = end_offset
                #logger.debug('traceable_token: %s', traceable_token)
    #logger.debug('traceable_tokens: %s', traceable_tokens)
    #logger.debug('traceable_tokens: %s', traceable_tokens[:5])
    return traceable_tokens

def tag_tokens_with_annotation(
    traceable_tokens: list[TraceableToken],
    records,
    #list_attribute_names
):
    #tags = [['O'] * len(tokens) for _ in range(len(list_attribute_names))]
    #logger.debug('tags: %s', tags)
    map_attribute_name_to_tags = {}
    #for i, attribute_name in enumerate(list_attribute_names):
    def sort_key(rec):
        return (
            rec['html_offset']['start']['line_id'],
            rec['html_offset']['start']['offset'],
        )
    records = sorted(records, key=sort_key)
    #for i, record in enumerate(records):
    for record_index, record in enumerate(records):
        if record_index > 2:
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
        last_token_index = None
        # Bタグを付けるべきトークンを探す
        for token_index, token in enumerate(traceable_tokens):
            if token.start_line < start_line:
                #last_token = token
                last_token_index = token_index
                continue
            assert token.start_line == start_line
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
                tags[start_token_index] = 'B'
                #logger.debug('start_token: %s', token)
            else:
                raise ValueError(f'invalid tag: {tags[start_token_index]}')
            break
        # Iタグを付けるべきトークンがあるか探す
        for token_index in range(start_token_index+1, len(traceable_tokens)):
            token = traceable_tokens[token_index]
            if token.start_line > end_line:
                break
            assert token.start_line == start_line
            if token.start_offset >= end_offset:
                break
            if tags[token_index] == 'O':
                tags[token_index] = 'I'
                #logger.debug('inter-token: %s', token)
    return map_attribute_name_to_tags

def pre_process_train_data(input_dir, output_file, tokenizer):
    logger.info('Pre-processing train data')
    logger.info('input_dir: %s', input_dir)
    logger.info('output_file: %s', output_file)

    annotation_dir = os.path.join(input_dir, 'annotation')
    html_dir = os.path.join(input_dir, 'html')
    if not os.path.isdir(annotation_dir):
        raise ValueError(f'directory not found: {annotation_dir}')
    if not os.path.isdir(html_dir):
        raise ValueError(f'directory not found: {html_dir}')
    entries = os.scandir(annotation_dir)
    categories = []
    for entry in tqdm(entries):
        #logger.debug('entry: %s', entry)
        #logger.debug('entry.name: %s', entry.name)
        if entry.name.endswith('_dist.jsonl'):
            category = entry.name.replace('_dist.jsonl', '')
            categories.append(category)
    #logger.debug('categories: %s', categories)
    if not categories:
        raise ValueError('no categories found')

    #with open(output_file, 'w', encoding='utf-8') as f:
    with open(output_file, 'w', encoding='utf-8') as f_output:
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
                    page_id = rec['page_id']
                    set_attribute_names.add(rec['attribute'])
                    if not page_id in map_page_id_to_records:
                        map_page_id_to_records[page_id] = []
                    map_page_id_to_records[page_id].append(rec)
            map_category_page_id_to_records[category] = map_page_id_to_records
        list_attribute_names = sorted(set_attribute_names)
        #map_attribute_name_to_id = {name: i for i, name in enumerate(list_attribute_names)}
        #logger.debug('list_attribute_names: %s', list_attribute_names)
        logger.debug('# list_attribute_names: %s', len(list_attribute_names))
        #for page_id in tqdm(map_page_id_to_records):
        for category in tqdm(categories):
            map_page_id_to_records = map_category_page_id_to_records[category]
            for page_id, records in tqdm(map_page_id_to_records.items()):
                output = {}
                first = records[0]
                output['page_id'] = first['page_id']
                output['title'] = first['title']
                output['ENE'] = first['ENE']
                #logger.debug('page_id: %s', page_id)
                #logger.debug('# records: %s', len(records))
                html_file = os.path.join(html_dir, category, f'{page_id}.html')
                with open(html_file, 'r', encoding='utf-8') as f_html:
                    html = f_html.read()
                    html = clean_up_html(html)
                    traceable_tokens = tokenize_with_positions(tokenizer, html)
                    #tag_tokens_with_annotation(tokens, records, list_attribute_names)
                    mapped_tags = tag_tokens_with_annotation(traceable_tokens, records)
                    tokens = [token.text for token in traceable_tokens]
                    output['tokens'] = tokens
                    output['tags'] = mapped_tags
                    #logger.debug('# tokens: %s', len(tokens))
                    json.dump(output, f_output, ensure_ascii=False)
                    #f_output.write(html)
                    #sys.exit(1)

def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser('Pre-process annotation data')
    parser.add_argument(
        '--train_data_dir',type=str, required=True,
        help='Directory containing training data',
    )
    parser.add_argument(
        '--output_file', type=str, required=True,
        help='Output file to store pre-processed data (json lines)'
    )
    parser.add_argument(
        '--model_name_or_path', type=str, default='cl-tohoku/bert-base-japanese-v2',
        help='Pre-trained model name or path'
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

    pre_process_train_data(args.train_data_dir, args.output_file, tokenizer)

if __name__ == "__main__":
    main()
