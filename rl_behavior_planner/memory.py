# -- coding: utf-8 --
# @Time : 2021/11/14 下午6:28
# @Author : fujiawei0724
# @File : memory.py
# @Software: PyCharm

"""
Memory buffer for storing data.
"""

import random


# 定义记忆重放
class MemoryReplay:
    def __init__(self, max_size):
        # 得到记忆区最大长度
        self.capability_ = max_size
        # 定义记忆区数据
        self.memory_buffer_ = []
        # 当前下标
        self.index_ = -1

    # 获得当前记忆区长度
    def size(self):
        return len(self.memory_buffer_)

    # 更新记忆区
    def update(self, data):
        self.index_ += 1
        self.index_ = self.index_ % self.capability_
        if self.size() < self.capability_:
            self.memory_buffer_.append(data)
        else:
            self.memory_buffer_[self.index_] = data

    # 获得记忆数据
    def getBatch(self, batch_size):
        return random.choices(self.memory_buffer_, k=batch_size)

    # 判断记忆区是否填满
    def isFull(self):
        if self.size() == self.capability_:
            return True
        else:
            return False