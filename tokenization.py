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
    line_index = None,
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
    if traceable_tokens is not None:
        for i in range(-5, 0):
            logger.error('traceable token %s: %s', i, traceable_tokens[i])

def merge_tokens(
    tokens: list[TokenWithOffset],
):
    
    first_unk = tokens[0]
    last_unk = tokens[-1]
    traceable_token = TokenWithOffset(
        text=''.join([t.text for t in tokens]),
        start_line=first_unk.start_line,
        start_offset=first_unk.start_offset,
        end_line=last_unk.end_line,
        end_offset=last_unk.end_offset,
    )
    return traceable_token

def process_pended(
    tokenizer: BertJapaneseTokenizer,
    pending: list[str],
    line_index: int,
    start_offset: int,
    end_offset: int,
):
    # NOTE: オフセット合わせできていなかったトークンの辻褄を合わせる
    unk_list: list[TokenWithOffset] = []
    traceable_tokens: list[TokenWithOffset] = []
    for p in pending:
        traceable_token = TokenWithOffset(
            text=p,
            start_line=line_index,
            start_offset=start_offset,
            end_line=line_index,
            end_offset=end_offset,
        )
        if p not in tokenizer.subword_tokenizer.vocab:
            # 連続する未知語は溜めておいて後で1つにまとめる
            unk_list.append(traceable_token)
            continue
        # 既知のトークン
        if len(unk_list) > 0:
            # 先に溜めておいた未知語を1つのトークンにまとめる
            traceable_tokens.append(merge_tokens(unk_list))
            unk_list.clear()
        # 既知のトークンを出力
        traceable_tokens.append(traceable_token)
    if len(unk_list) > 0:
        # 残りの溜まっていた未知語を1つのトークンにまとめる
        traceable_tokens.append(merge_tokens(unk_list))
        unk_list.clear()
    pending.clear()
    return traceable_tokens

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
        char_index = 0
        pending: list[str] = []
        for word_index, word in enumerate(words):
            #logger.debug('word %s: %s', word_index, word)
            tokens = tokenizer.subword_tokenizer.tokenize(word)
            if tokens == [tokenizer.unk_token]:
                # サブワードトークナイザで単一の未知語になった場合は、
                # word 単体が未知語であったため、文字単位に分解する
                tokens = list(word)
            for token_index, token in enumerate(tokens):
                # NOTE: この分解方法の場合、 token が UNK になることはないはず
                #logger.debug('token %s: %s', token_index, token)
                actual_token = token.strip()
                if token_index >= 1:
                    # サブワードの '##' は2番目以降のサブワードにしか付与されない
                    if token.startswith('##'):
                        actual_token = actual_token[2:]
                #logger.debug('actual token: %s', actual_token)
                kd_token = unicodedata.normalize('NFKD', actual_token)
                first_char = kd_token[0]
                first_char_category = unicodedata.category(first_char)
                if first_char_category == 'Mn':
                    if len(token) == 1:
                        # NOTE: 幅なし結合文字単体のトークンは非常に厄介
                        # 正規化すると順序が変わる場合があるため、オフセット合わせが困難
                        # 例 (順序正規化, canonical ordering):
                        # normalize('NFD', COMBINING DIAERESIS' + 'COMBINING BREVE BELOW' + ...)
                        # -> 'COMBINING BREVE BELOW' + 'COMBINING DIAERESIS' + ...
                        # 対策: pending に溜めておいて、有効な文字が出てきたら一気に処理する
                        pending.append(token)
                        continue
                found_first_char = -1
                # 最初の1文字を探す
                for test_index in range(char_index, len(line)):
                    # NOTE: 入力文は normalize されていない場合が多々ある
                    # line の各文字には正規化で結合される前の文字が入っている場合がある
                    # 例: 'ザ' の代わりに
                    # 'KATAKANA LETTER SA' + 'COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK'
                    # の 2 文字が入っているなど
                    # 対策: まずは first_char を構成文字に含む文字を探す
                    kd_char = unicodedata.normalize('NFKD', line[test_index])
                    found_first_char = kd_char.find(first_char)
                    if found_first_char >= 0:
                        first_char_start_index = test_index
                        break
                if found_first_char < 0:
                    # NOTE: 1文字目が見つからない場合はエラー
                    logger.error('first_char not found: %s', first_char)
                    logger.error('first char name: %s', unicodedata.name(first_char))
                    log_error_info(
                        char_index=char_index,
                        line=line,
                        line_index=line_index,
                        tokens=tokens,
                        token_index=token_index,
                        traceable_tokens=traceable_tokens,
                        words=words,
                        word_index=word_index,
                    )
                    raise ValueError(f'first char not found: {first_char}')
                if len(pending) > 0:
                    # NOTE: オフセット合わせできていなかったトークンの辻褄を合わせる
                    traceable_tokens.extend(process_pended(
                        tokenizer,
                        pending,
                        line_index,
                        char_index,
                        first_char_start_index,
                    ))
                    pending.clear()
                matched = False
                for test_end in range(first_char_start_index+1, len(line)+1):
                    span = line[first_char_start_index:test_end]
                    kd_span = unicodedata.normalize('NFKD', span)
                    kd_span = kd_span[found_first_char:]
                    if kd_token == kd_span:
                        # NOTE: 完全一致
                        # 次の文字にオフセットを移して問題無い
                        matched = True
                        end_char_index = test_end
                        next_char_index = test_end
                        break
                    if kd_span.startswith(kd_token):
                        # NOTE: 前方部分一致
                        # 次に来るトークンが最後の文字の構成要素の可能性もある
                        matched = True
                        end_char_index = test_end
                        next_char_index = test_end - 1
                        break
                if not matched:
                    logger.warning('word tokens: %s', tokens)
                    logger.warning('word -1 tokens: %s', list(words[word_index-1]))

                    logger.error('token not found: %s', token)
                    log_error_info(
                        char_index=first_char_start_index,
                        line=line,
                        line_index=line_index,
                        tokens=tokens,
                        token_index=token_index,
                        traceable_tokens=traceable_tokens,
                        words=words,
                        word_index=word_index,
                    )
                    raise ValueError('token not found')
                traceable_token = TokenWithOffset(
                    text=token,
                    start_line=line_index,
                    start_offset=first_char_start_index,
                    end_line=line_index,
                    end_offset=end_char_index,
                )
                traceable_tokens.append(traceable_token)
                char_index = next_char_index
        if len(pending) > 0:
            # NOTE: オフセット合わせできていなかったトークンの辻褄を合わせる
            if len(pending) > 0:
                # NOTE: オフセット合わせできていなかったトークンの辻褄を合わせる
                traceable_tokens.extend(process_pended(
                    tokenizer,
                    pending,
                    line_index,
                    char_index,
                    len(line),
                ))
            pending.clear()
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
