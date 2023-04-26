from logzero import logger

from tokenization import (
    TokenWithOffset,
)

OVERLAP_STRATEGIES = [
    'front-longest',
    'longest-front',
]
DEFAULT_OVERLAP_STRATEGY = OVERLAP_STRATEGIES[0]
#DEFAULT_OVERLAP_STRATEGY = OVERLAP_STRATEGIES[1]

DEBUG_MODE = False

def tag_tokens_with_annotation_list(
    traceable_tokens: list[TokenWithOffset],
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
                raise ValueError('start_token_index is None')
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
