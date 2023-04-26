'''
オフセット付きでトークナイズする
'''

import unicodedata

from dataclasses import (
    dataclass,
)

from logzero import logger

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

def tokenize_with_offsets(tokenizer, text) -> list[TokenWithOffset]:
    # トークナイズした上で元のポジション位置を突き合わせる
    lines = text.splitlines()
    traceable_tokens = []
    if tokenizer.subword_tokenizer_type != 'wordpiece':
        raise ValueError(f'unsupported subword tokenizer: {tokenizer.subword_tokenizer_type}')
    
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
    
    def merge_tokens(tokens: list[TokenWithOffset]):
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
                    raise ValueError('token not found')
                start_token = line_chars[offset]
                end_token = line_chars[end_offset]
                traceable_token = TokenWithOffset(
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
