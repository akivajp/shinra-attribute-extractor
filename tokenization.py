#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
オフセット付きでトークナイズする
'''

import sys
import unicodedata

from dataclasses import (
    asdict,
    dataclass,
)
import pandas as pd
import transformers
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from logzero import logger

DEBUG_MODE = False

@dataclass
class TokenWithOffset:
    '''
    トークンとその出現位置を保持する
    '''
    text: str
    start_line: int
    start_offset: int
    end_line: int
    end_offset: int

def normalize(text):
    # Unicode正規化 
    return unicodedata.normalize('NFKC', text)

def merge_tokens(tokens: list[TokenWithOffset]):
    # トークンを結合して文字列を生成する
    #return normalize( str.join('', [token.text for token in tokens]) )
    return str.join('', [token.text for token in tokens])

def decompose_line(line, line_index):
    # ライン文字列を構成文字単位でトークナイズする
    # (TRADE_MARK_SIGNのような合字をサブワード化された '##T' などとマッチさせるための対策)
    traceable_chars: list[TokenWithOffset] = []
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
            traceable_char = TokenWithOffset(
                text=sub_char,
                start_line=line_index,
                start_offset=i,
                end_line=line_index,
                end_offset=i+1,
            )
            traceable_chars.append(traceable_char)
    return traceable_chars

def log_error_info(
    char_index = None,
    line = None,
    line_chars = None,
    line_index = None,
    merged = None,
    tokens = None,
    token_index = None,
    traceable_tokens = None,
    word_index = None,
    words = None,
):
    def get_index(l, index):
        if l is None or index is None:
            return None
        return l[index]
    word = get_index(words, word_index)
    token = get_index(tokens, token_index)
    logger.error('line_index: %s', line_index)
    #logger.error('line: %s', line)
    logger.error('char_index: %s', char_index)
    logger.error('word: %s', word)
    if word is not None:
        logger.error('word_index: %s', word_index)
        for i, c in enumerate(word):
            logger.error('word[%s]: %s', i, c)
            logger.error('word[%s] name: %s', i, unicodedata.name(c))
        logger.error('words digest: %s', words[word_index-5:word_index+6])
    if token is not None:
        for i, c in enumerate(token):
            logger.error('token[%s]: %s', i, c)
            logger.error('token[%s] name: %s', i, unicodedata.name(c))
        normal = normalize(token)
        for i, c in enumerate(normal):
            logger.error('normal token[%s]: %s', i, c)
            logger.error('normal token[%s] name: %s', i, unicodedata.name(c))
        logger.error('words digest: %s', words[word_index-5:word_index+5])
    logger.error('tokens digest: %s', tokens[token_index-5:token_index+5])
    if line_chars is not None and char_index is not None:
        for o in range(char_index-2, char_index+3):
            if o >= 0 and o < len(line_chars):
                logger.error('line character %s: %s', o, line_chars[o])
                name = unicodedata.name(line_chars[o].text)
                logger.error('line character %s name: %s', o, name)
    if merged is not None:
        logger.error('merged: %s', merged)
    if traceable_tokens is not None:
        for i in range(-5, 0):
            logger.error('traceable token %s: %s', i, traceable_tokens[i])

def tokenize_with_offsets(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text,
) -> list[TokenWithOffset]:
    # トークナイズした上で元のポジション位置を突き合わせる
    lines = text.splitlines()
    traceable_tokens = []
    if tokenizer.subword_tokenizer_type != 'wordpiece':
        raise ValueError(f'unsupported subword tokenizer: {tokenizer.subword_tokenizer_type}')
    
    for line_index, line in enumerate(lines):
        #if DEBUG_MODE and line_index > 5:
        #    break
        line_chars = decompose_line(line, line_index)
        words = tokenizer.word_tokenizer.tokenize(line)
        char_index = 0
        last_token = None
        unk_start_index = None
        for word_index, word in enumerate(words):
            #logger.debug('word %s: %s', word_index, word)
            tokens = tokenizer.subword_tokenizer.tokenize(word)
            if tokens == [tokenizer.unk_token]:
                if len(word) >= 2:
                    # 複数文字で構成される単語が未知語となった場合は
                    # 1文字ずつトークン化する
                    #logger.debug('tokens: %s', tokens)
                    tokens = []
                    for c in word:
                        tokens.extend(tokenizer.subword_tokenizer.tokenize(c))
                        #tokens.extend(c)
                    #logger.debug('new tokens: %s', tokens)
            for token_index, token in enumerate(tokens):
                #logger.debug('token %s: %s', token_index, token)
                if token is tokenizer.unk_token:
                    if last_token is tokenizer.unk_token:
                        # 連続する UNK は1つの UNK とみなす
                        continue
                    #unk_start_offset = offset
                    unk_start_index = char_index
                    # UNK トークンは実際には元の文内の文字に対応していない場合もある
                    # 次の有効トークンまでスキップ
                    last_token = token
                    continue
                #logger.debug('token: %s', token)
                actual_token = token
                if token_index >= 1:
                    # サブワードの '##' は2番目以降のサブワードにしか付与されない
                    if token.startswith('##'):
                        actual_token = token[2:]
                #logger.debug('actual token: %s', actual_token)
                normal_token = normalize(actual_token)
                first_char = normal_token[0]
                found_first_char = False
                #for first_char_offset in range(offset, len(line_chars)):
                # 最初の1文字を探す
                #for first_char_index in range(char_index, len(line_chars)):
                for test_end in range(char_index, len(line_chars)):
                    # NOTE: 入力文は normalize されていない場合が多々ある
                    # line_chars の各文字には normalize で結合される前の文字が入っている場合がある
                    # 例: 'ザ' の代わりに
                    # 'KATAKANA LETTER SA' + 'COMBINING KATAKANA=HIRAGANA VOICED SOUND MARK'
                    # の 2 文字が入っているなど
                    # 対策: まずは終了位置を広げながら結合して first_char が含まれる位置を探す
                    # 次に、開始位置を狭めながら first_char が含まれなくなる位置を探す
                    # first_char が含まれない開始位置の1つ前が first_char の開始位置として適切
                    merged = merge_tokens(line_chars[char_index:test_end+1])
                    normal_merged = normalize(merged)
                    #if first_char == unicodedata.lookup('COMBINING ACUTE ACCENT'):
                    #    logger.warning('merged: %s', merged)
                    if first_char in merged or first_char in normal_merged:
                        #logger.debug('found first char %s in merged: %s', first_char, merged)
                        # 開始位置の探索
                        for test_start in range(
                            char_index+1,
                            test_end+2,
                        ):
                            merged = merge_tokens(
                                line_chars[test_start:test_end+1]
                            )
                            normal_merged = normalize(merged)
                            #if first_char not in merged:
                            if first_char not in merged and first_char not in normal_merged:
                                first_char_start_index = test_start - 1
                                found_first_char = True
                                break
                        break
                if not found_first_char:
                    #logger.debug('short_merged: %s', memo_short_merged)
                    # 1文字目が見つからない場合はエラー
                    logger.error('first_char not found: %s', first_char)
                    log_error_info(
                        char_index=char_index,
                        line_chars=line_chars,
                        line=line,
                        line_index=line_index,
                        tokens=tokens,
                        token_index=token_index,
                        traceable_tokens=traceable_tokens,
                        words=words,
                        word_index=word_index,
                    )
                    raise ValueError(f'first char not found: {first_char}')
                if last_token is tokenizer.unk_token:
                    unk_start_token = line_chars[unk_start_index]
                    first_char_token = line_chars[first_char_start_index]
                    if unk_start_token.start_offset == first_char_token.start_offset:
                        # NOTE: UNK は元の文字に対応していない
                        # 空文字列のトークンは不要であり問題が起こるため無視
                        #logger.debug('empty unk token before: %s', first_char_token)
                        pass
                    else:
                        traceable_token = TokenWithOffset(
                            text=merge_tokens(line_chars[unk_start_index:first_char_start_index]),
                            start_line=unk_start_token.start_line,
                            start_offset=unk_start_token.start_offset,
                            end_line=first_char_token.end_line,
                            # 最初の有効文字の start_offset が UNK の end_offset になる
                            end_offset=first_char_token.start_offset,
                        )
                        traceable_tokens.append(traceable_token)
                        #logger.debug('unk token: %s', traceable_token)
                char_index = first_char_start_index
                matched = False
                merged = None
                for test_end in range(char_index, len(line_chars)):
                    merged = merge_tokens(line_chars[char_index:test_end+1])
                    normal_merged = normalize(merged)
                    #if DEBUG_MODE:
                    #    logger.debug('char_index: %s', char_index)
                    #    logger.debug('end_char_index: %s', end_char_index)
                    #    logger.debug('token: %s', token)
                    #    logger.debug('normal_token: %s', normal_token)
                    #    logger.debug('merged: %s', merged)
                    #if normal_token == merged:
                    if normal_token == normal_merged:
                        matched = True
                        end_char_index = test_end
                        break
                if not matched:
                    logger.error('token not found: %s', token)
                    log_error_info(
                        char_index=char_index,
                        line_chars=line_chars,
                        line=line,
                        line_index=line_index,
                        merged=merged,
                        tokens=tokens,
                        token_index=token_index,
                        traceable_tokens=traceable_tokens,
                        words=words,
                        word_index=word_index,
                    )
                    raise ValueError('token not found')
                start_token = line_chars[char_index]
                end_token = line_chars[end_char_index]
                traceable_token = TokenWithOffset(
                    text=token,
                    start_line=start_token.start_line,
                    start_offset=start_token.start_offset,
                    end_line=end_token.end_line,
                    end_offset=end_token.end_offset,
                )
                traceable_tokens.append(traceable_token)
                char_index = end_char_index + 1
                last_token = token
        if last_token is tokenizer.unk_token:
            # 行末の UNK トークン
            unk_start_token = line_chars[unk_start_index]
            end_token = line_chars[-1]
            if unk_start_token.start_offset == end_token.start_offset:
                # NOTE: UNK は元の文字に対応していない
                # 空文字列のトークンは不要であり問題が起こるため無視
                #logger.debug('empty unk token at line end')
                pass
            else:
                #start_token = line_chars[unk_start_offset]
                #end_token = line_chars[offset]
                traceable_token = TokenWithOffset(
                    text=merge_tokens(line_chars[unk_start_token.start_offset:]),
                    start_line=start_token.start_line,
                    start_offset=start_token.start_offset,
                    end_line=end_token.end_line,
                    end_offset=end_token.end_offset,
                )
                traceable_tokens.append(traceable_token)
                #logger.debug('line end unk token: %s', traceable_token)
    return traceable_tokens

if __name__ == '__main__':
    DEBUG_MODE = True
    with open(sys.argv[1], encoding='utf-8') as f:
        html = f.read()
        #config_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        config_name = 'cl-tohoku/bert-base-japanese-v2'
        if len(sys.argv) >= 3:
            config_name = sys.argv[2]
        test_tokenizer = AutoTokenizer.from_pretrained(config_name)
        tokenized = tokenize_with_offsets(test_tokenizer, html)
        #logger.debug('tokenized: %s', tokenized)
