# -*- ecoding: utf-8 -*-
# @ModuleName: json_utils
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/8/11 12:32 PM

import json
from pygments import highlight, formatters, lexers


def beautify_json(ori_dict: dict) -> str:
    """beautify the original dictionary by converting it to formatted JSON string with syntax highlighting.

    Args:
        ori_dict: A dictionary to be beautified

    Returns:
        A string of formatted JSON with syntax highlighting
    """
    formatted_json = json.dumps(ori_dict, indent=4, ensure_ascii=False, sort_keys=True)
    return highlight(formatted_json, lexers.get_lexer_by_name('json'), formatters.TerminalFormatter())
