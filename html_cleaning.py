import re
from html.parser import HTMLParser

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
    