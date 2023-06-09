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
from transformers import (
    AutoTokenizer,
    BertJapaneseTokenizer,
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

def log_error_info(
    char_index = None,
    line = None,
    line_chars = None,
    line_index = None,
    tokenizer: BertJapaneseTokenizer = None,
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
    logger.error('char_index: %s', char_index)
    logger.error('token_index: %s', token_index)
    logger.error('word: %s', word)
    if word is not None:
        logger.error('word_index: %s', word_index)
        for i, c in enumerate(word):
            logger.error('word[%s]: %s', i, c)
            logger.error('word[%s] name: %s', i, unicodedata.name(c))
        logger.error('words digest: %s', words[word_index-5:word_index+6])
    if token is not None:
        logger.error('token: %s', token)
        if tokenizer is not None:
            logger.error('token is unk: %s', token not in tokenizer.subword_tokenizer.vocab)
        for i, c in enumerate(token):
            logger.error('token[%s]: %s', i, c)
            logger.error('token[%s] name: %s', i, unicodedata.name(c))
        kd_token = unicodedata.normalize('NFKD', token)
        for i, c in enumerate(kd_token):
            logger.error('kd token[%s]: %s', i, c)
            logger.error('kd token[%s] name: %s', i, unicodedata.name(c))
        logger.error('words digest: %s', words[word_index-5:word_index+5])
    logger.error('tokens digest: %s', tokens[token_index-5:token_index+5])
    if line is not None and char_index is not None:
        logger.error('offset character: %s', line[char_index])
        name = unicodedata.name(line[char_index])
        logger.error('offset character name: %s', name)
        #logger.error('line digest: %s', line[char_index-5:char_index+5])
        for i in range(-2, 3):
            logger.error('char %s: %s', i, line[char_index+i])
            logger.error('char %s name: %s', i, unicodedata.name(line[char_index+i]))
    if line_chars is not None and char_index is not None:
        for i in range(char_index-2, char_index+3):
            if i >= 0 and i < len(line_chars):
                logger.error('line character %s: %s', i, line_chars[i])
                name = unicodedata.name(line_chars[i].text)
                logger.error('line character %s name: %s', i, name)
    if traceable_tokens is not None:
        for i in range(-5, 0):
            if i >= 0 and i < len(line_chars):
                logger.error('traceable token %s: %s', i, traceable_tokens[i])

def merge_tokens(
    tokens: list[TokenWithOffset],
):
    traceable_token = TokenWithOffset(
        text=''.join([t.text for t in tokens]),
        start_line=tokens[0].start_line,
        start_offset=tokens[0].start_offset,
        end_line=tokens[-1].end_line,
        end_offset=tokens[-1].end_offset,
    )
    return traceable_token

def decompose_line_to_characters_with_offset(line, line_index):
    # ライン文字列を構成文字単位でトークナイズする
    # (TRADE_MARK_SIGNのような合字をサブワード化された '##T' などとマッチさせるための対策)
    offset_start = 0
    offset_end = 0
    kd_chars = []
    for i, c in enumerate(unicodedata.normalize('NFKD', line)):
        assert len(c) == 1
        while True:
            # NOTE: 正規化すると順序が変わる場合がある (結合文字などで順序の正規化が起こる)
            # 例 (順序正規化, canonical ordering):
            # normalize('NFD', COMBINING DIAERESIS' + 'COMBINING BREVE BELOW' + ...)
            # -> 'COMBINING BREVE BELOW' + 'COMBINING DIAERESIS' + ...
            # 対策: kd_chars に溜めておいて、有効な幅を記録する
            # 全ての kd_chars が一度消費された段階で次の offset_start に進む
            if c in kd_chars:
                kd_chars.remove(c)
                traceable_char = TokenWithOffset(
                    text=c,
                    start_line=line_index,
                    start_offset=offset_start,
                    end_line=line_index,
                    end_offset=offset_end,
                )
                yield traceable_char
                break
            else:
                kd_chars.extend(unicodedata.normalize('NFKD', line[offset_end]))
                offset_end += 1
        if len(kd_chars) == 0:
            offset_start = offset_end
    #logger.warning('kd_chars: %s', kd_chars)

def tokenize_with_offsets(
    tokenizer: BertJapaneseTokenizer,
    text,
) -> list[TokenWithOffset]:
    # トークナイズした上で元のポジション位置を突き合わせる
    lines = text.splitlines()
    traceable_tokens = []

    # 取り急ぎ、サブワードトークナイザに wordpiece を想定
    # 他のサブワードトークナイザに対応させるのも比較的用意だが、
    # サブワードの仕様に対して個別に対応する必用あり
    if tokenizer.subword_tokenizer_type != 'wordpiece':
        raise ValueError(f'unsupported subword tokenizer: {tokenizer.subword_tokenizer_type}')
    
    for line_index, line in enumerate(lines):
        #if DEBUG_MODE and line_index > 5:
        #    break
        words = tokenizer.word_tokenizer.tokenize(line)
        line_chars = list(decompose_line_to_characters_with_offset(line, line_index))
        char_index = 0
        for word_index, word in enumerate(words):
            tokens = tokenizer.subword_tokenizer.tokenize(word)
            if tokens == [tokenizer.unk_token]:
                # サブワードトークナイザで単一の未知語になった場合は、
                # word 単体が未知語であったため、文字単位に分解する
                tokens = list(word)
            for token_index, token in enumerate(tokens):
                # NOTE: この分解方法の場合、 token が UNK になることはないはず
                actual_token = token.strip()
                if token_index >= 1:
                    # サブワードの '##' は2番目以降のサブワードにしか付与されない
                    if token.startswith('##'):
                        actual_token = actual_token[2:]
                kd_token = unicodedata.normalize('NFKD', actual_token)
                first_char = kd_token[0]
                found_first_char = False
                for test_index in range(char_index, len(line_chars)):
                    # NOTE: 最初の1文字を探す
                    # ここでスキップされるのは空白文字くらいのはず
                    if first_char == line_chars[test_index].text:
                        found_first_char = True
                        char_index = test_index
                        break
                if not found_first_char:
                    # NOTE: 1文字目が見つからない場合はエラー
                    logger.error('first_char not found: %s', first_char)
                    logger.error('first char name: %s', unicodedata.name(first_char))
                    log_error_info(
                        char_index=char_index,
                        line=line,
                        line_chars=line_chars,
                        line_index=line_index,
                        tokenizer=tokenizer,
                        tokens=tokens,
                        token_index=token_index,
                        traceable_tokens=traceable_tokens,
                        words=words,
                        word_index=word_index,
                    )
                    raise ValueError(f'first char not found: {first_char}')
                merged = merge_tokens(line_chars[char_index:char_index+len(kd_token)])
                if kd_token != merged.text:
                    logger.error('token not found: %s', token)
                    logger.error('merged: %s', merged)
                    logger.error('kd token: %s', kd_token)
                    logger.error('word tokens: %s', tokens)
                    logger.error('word -1 tokens: %s', list(words[word_index-1]))
                    log_error_info(
                        #char_index=first_char_start_index,
                        char_index=char_index,
                        line=line,
                        line_index=line_index,
                        tokenizer=tokenizer,
                        tokens=tokens,
                        token_index=token_index,
                        traceable_tokens=traceable_tokens,
                        words=words,
                        word_index=word_index,
                    )
                    raise ValueError('token not found')
                traceable_tokens.append(merged)
                char_index += len(kd_token)
    return traceable_tokens

if __name__ == '__main__':
    DEBUG_MODE = True
    #config_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    config_name = 'cl-tohoku/bert-base-japanese-v2'
    test_tokenizer = AutoTokenizer.from_pretrained(config_name)
    for path in sys.argv[1:]:
        logger.debug('testing: %s', path)
        with open(path, encoding='utf-8') as f:
            html = f.read()
            if len(sys.argv) >= 3:
                config_name = sys.argv[2]
            tokenized = tokenize_with_offsets(test_tokenizer, html)
            #logger.debug('tokenized: %s', tokenized)
