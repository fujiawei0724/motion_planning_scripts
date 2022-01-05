#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/11 下午9:50
# @Author  : fjw
# @Email   : fujiawei0724@gmail.com
# @File    : converter.py
# @Software: PyCharm

"""
Convert the formation of the trained model to be deployed in C++.
"""

import torch
import torchvision
from Double_DQN_net import DQN

if __name__ == '__main__':
    # Official model
    # model = torchvision.models.resnet18()
    # traced_script_module = torch.jit.script(model)
    # traced_script_module.save('model.pt')

    model = DQN(94, 231)
    model.load_state_dict(torch.load('./model/checkpoint2.pt', map_location='cpu'))
    model.eval()
    dummy_example = torch.rand(1, 94)
    traced_module = torch.jit.trace(model, dummy_example)
    traced_module.save('./model/model2.pt')

    test_input = torch.ones(1, 94)
    test_res = model.forward(test_input)
    print('Test res: {}'.format(test_res))



