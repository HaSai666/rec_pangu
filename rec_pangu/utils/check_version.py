# -*- ecoding: utf-8 -*-
# @ModuleName: check_version
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/8/11 2:24 PM

import json
from threading import Thread
from loguru import logger

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.org/pypi/rec_pangu/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logger.warning('\nRec_Pangu version {0} detected. Your version is {1}.\nUse `pip install -U rec_pangu` to upgrade.'.format(latest_version,version))
        except:
            print("Please check the latest version manually on https://pypi.org/project/rec_pangu/#history")
            return

    Thread(target=check, args=(version,)).start()