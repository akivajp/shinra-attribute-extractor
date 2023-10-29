#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
オフセット付きでトークナイズする
'''

import sys
import unicodedata

from dataclasses import (
    dataclass,
)
from transformers import (
    AutoTokenizer,
    BertJapaneseTokenizer,
)

from logzero import logger

DEBUG_MODE = False
DEBUG_LINES = 5

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

SPACE_CHARS = [' ', '\t', '　']

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
        if DEBUG_MODE and line_index >= DEBUG_LINES:
            break
        words = tokenizer.word_tokenizer.tokenize(line)
        start_offset = 0
        end_offset = 0
        kd_char_counter: dict[str,int] = {}
        for word_index, word in enumerate(words):
            tokens = tokenizer.subword_tokenizer.tokenize(word)
            if tokens == [tokenizer.unk_token]:
                # サブワードトークナイザで単一の未知語になった場合は、
                # word 単体が未知語であったため、文字単位に分解する
                tokens = list(word)
            for token_index, token in enumerate(tokens):
                assert token is not tokenizer.unk_token
                actual_token = token
                if token_index >= 1:
                    # サブワードの '##' は2番目以降のサブワードにしか付与されない
                    if token.startswith('##'):
                        actual_token = actual_token[2:]
                # NOTE: 正規化すると順序が変わる場合がある (結合文字などで順序の正規化が起こる)
                # 例1
                #   [COMBINING DIAERESIS] + [COMBINING BREVE BELOW] + ...
                #   -> [COMBINING BREVE BELOW] + [COMBINING DIAERESIS] + ...
                # 例2
                #   [ARABIC LETTER ALEF WITH HAMZA ABOVE] + [ARABIC FATHA] + ...
                #   -> [ARABIC LETTER ALEF] + [ARABIC FATHA] + [ARABIC LETTER HAMZA ABOVE] + ...
                #   この場合、トークナイザはNFKCした状態で
                #   [ARABIC LETTER ALEF WITH HAMZA ABOVE] 単体で切ったりしてきて厄介
                # 対策: バッファから読み取った構成要素のカウントを kd_char_counter に溜めておいて
                # トークン承認時点で kd_char_counter が空になった段階で次の start_offset に進む
                kd_token = unicodedata.normalize('NFKD', actual_token)
                found_char = True
                kd_char = None
                for kd_char in kd_token:
                    # トークン内の1文字ずつカウンターを照合する
                    #logger.debug('kd_char: %s', kd_char)
                    found_char = False
                    while end_offset <= len(line):
                    #while True:
                        count = kd_char_counter.get(kd_char, 0)
                        #logger.debug('count: %s', count)
                        if count > 0:
                            # 存在するならカウントを減らす
                            if count == 1:
                                kd_char_counter.pop(kd_char)
                            else:
                                kd_char_counter[kd_char] = count - 1
                            found_char = True
                            break
                        # 存在しないなら次の文字の構成文字をカウンターに追加する
                        if end_offset < len(line):
                            for c in unicodedata.normalize('NFKD', line[end_offset]):
                                if c in SPACE_CHARS:
                                    # 空白文字はスキップする
                                    continue
                                kd_char_counter[c] = kd_char_counter.get(c, 0) + 1
                        end_offset += 1
                    if not found_char:
                        # 行内の文字を読み切ってもトークン内の全構成文字を見つけられなかった
                        not_found_char = kd_char
                        break
                if not found_char:
                    logger.error('token not found: %s', token)
                    logger.error('kd token: %s', kd_token)
                    logger.error('not found char: %s', not_found_char)
                    logger.error('counter: %s', kd_char_counter)
                    logger.error('start_offset: %s', start_offset)
                    logger.error('end_offset: %s', end_offset)
                    logger.error('line length: %s', len(line))
                    raise ValueError('token not found')
                while line[start_offset] in SPACE_CHARS:
                    # 空白文字はスキップする
                    start_offset += 1
                token_with_offset = TokenWithOffset(
                    text=token,
                    start_line=line_index,
                    start_offset=start_offset,
                    end_line=line_index,
                    end_offset=end_offset,
                )
                #logger.debug('token_with_offset: %s', token_with_offset)
                traceable_tokens.append(token_with_offset)
                if not kd_char_counter:
                    # カウンタが空なら開始オフセットを進める
                    start_offset = end_offset
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
    #tokenized = tokenize_with_offsets(test_tokenizer, '<td> 　銀')
    #for i, token in enumerate(tokenized):
    #    logger.debug('token[%s]: %s', i, token)
