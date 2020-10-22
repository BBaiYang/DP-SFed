# -*- coding: utf-8 -*-
"""
提供一些工具函数
"""

import numpy as np
import torch
from configs import DEBUG as debug

"""打印详细日志"""


def FED_LOG(*kwargs):
    if debug:
        print(*kwargs)


"""输出关键信息"""


def FED_WRITE(*kwargs):
    # TODO: 向文件输入信息流
    if debug:
        print(*kwargs)
