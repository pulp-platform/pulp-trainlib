'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini
'''

"""
LAYER TEMPLATES
"""

def linear_template(layer_number, chin, chout, bias):
    template = "\t\tself.l"+str(layer_number)+" = nn.Linear(in_features=l"+str(layer_number)+"_in_ch, out_features=l"+str(layer_number)+"_out_ch, bias="+str(bias)+")\n"
    return template


def conv2d_template(layer_number, chin, chout, hk, wk, hstr, wstr, hpad, wpad, bias):
    template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_out_ch, kernel_size=(l"+str(layer_number)+"_hk, l"+str(layer_number)+"_wk), padding=(l"+str(layer_number)+"_hpad, l"+str(layer_number)+"_wpad), stride=(l"+str(layer_number)+"_hstr, l"+str(layer_number)+"_wstr), bias="+str(bias)+")\n"
    return template


def DW_template(layer_number, ch_io, hk, wk, hstr, wstr, hpad, wpad, bias):
    template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_in_ch, kernel_size=(l"+str(layer_number)+"_hk, l"+str(layer_number)+"_wk), stride = 1, groups=l"+str(layer_number)+"_in_ch, bias="+str(bias)+")\n"
    return template


def PW_template(layer_number, chin, chout, bias):
    template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_out_ch, kernel_size=1, stride=1, bias="+str(bias)+")\n"
    return template



"""
ACTIVATIONS TEMPLATES
"""

def ReLU_template(layer):
    template = "\t\tself.l"+str(layer)+" = nn.ReLU()\n"
    return template



""""
POOLING TEMPLATES
"""

def MaxPool_template(layer, hk, wk, hstr, wstr):
    template ="\t\tself.l"+str(layer)+" = nn.MaxPool2d(kernel_size=(l"+str(layer)+"_hk, l"+str(layer)+"_wk), stride=(l"+str(layer)+"_hstr, l"+str(layer)+"_wstr))\n"
    return template

def AvgPool_template(layer, hk, wk, hstr, wstr):
    template ="\t\tself.l"+str(layer)+" = nn.AvgPool2d(kernel_size=(l"+str(layer)+"_hk, l"+str(layer)+"_wk), stride=(l"+str(layer)+"_hstr, l"+str(layer)+"_wstr))\n"
    return template
