#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-31 10:48
# @Description: In User Settings Edit
# @Software : PyCharm
import re


text = '文本的无监督学习文本是你的'
text = re.sub('文本', '知识', text, count=1)
print(text)
