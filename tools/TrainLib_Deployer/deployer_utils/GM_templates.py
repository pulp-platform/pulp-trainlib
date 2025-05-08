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

def linear_template(layer_number, chin, chout, bias, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer_number)+" = nn.Linear(in_features=l"+str(layer_number)+"_in_ch, out_features=l"+str(layer_number)+"_out_ch, bias="+str(bias)+")\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer_number)+" = nn.Linear(in_features=l"+str(layer_number)+"_in_ch, out_features=l"+str(layer_number)+"_out_ch, bias="+str(bias)+").half()\n"
    else:
        print("[GM_templates.linear_template] Invalid data type!!")
        exit()
    return template


def conv2d_template(layer_number, chin, chout, hk, wk, hstr, wstr, hpad, wpad, bias, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_out_ch, kernel_size=(l"+str(layer_number)+"_hk, l"+str(layer_number)+"_wk), padding=(l"+str(layer_number)+"_hpad, l"+str(layer_number)+"_wpad), stride=(l"+str(layer_number)+"_hstr, l"+str(layer_number)+"_wstr), bias="+str(bias)+")\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_out_ch, kernel_size=(l"+str(layer_number)+"_hk, l"+str(layer_number)+"_wk), padding=(l"+str(layer_number)+"_hpad, l"+str(layer_number)+"_wpad), stride=(l"+str(layer_number)+"_hstr, l"+str(layer_number)+"_wstr), bias="+str(bias)+").half()\n"
    else:
        print("[GM_templates.conv2d_template] Invalid data type!!")
        exit()
    return template


def DW_template(layer_number, ch_io, hk, wk, hstr, wstr, hpad, wpad, bias, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_in_ch, kernel_size=(l"+str(layer_number)+"_hk, l"+str(layer_number)+"_wk), stride = 1, groups=l"+str(layer_number)+"_in_ch, bias="+str(bias)+")\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_in_ch, kernel_size=(l"+str(layer_number)+"_hk, l"+str(layer_number)+"_wk), stride = 1, groups=l"+str(layer_number)+"_in_ch, bias="+str(bias)+").half()\n"
    else:
        print("[GM_templates.DW_template] Invalid data type!!")
        exit()
    return template


def PW_template(layer_number, chin, chout, bias, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_out_ch, kernel_size=1, stride=1, bias="+str(bias)+")\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer_number)+" = nn.Conv2d(in_channels=l"+str(layer_number)+"_in_ch, out_channels=l"+str(layer_number)+"_out_ch, kernel_size=1, stride=1, bias="+str(bias)+").half()\n"
    else:
        print("[GM_templates.PW_template] Invalid data type!!")
        exit()        
    return template



"""
ACTIVATIONS TEMPLATES
"""

def ReLU_template(layer, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer)+" = nn.ReLU()\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer)+" = nn.ReLU()\n"
    else:
        print("[GM_templates.ReLU_template] Invalid data type!!")
        exit() 
    return template

def LeakyReLU_template(layer, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer)+" = nn.LeakyReLU()\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer)+" = nn.LeakyReLU()\n"
    else:
        print("[GM_templates.LeakyReLU_template] Invalid data type!!")
        exit() 
    return template

def Sigmoid_template(layer, data_type):
    if data_type == 'FP32':
        template = "\t\tself.l"+str(layer)+" = nn.Sigmoid()\n"
    elif data_type == 'FP16':
        template = "\t\tself.l"+str(layer)+" = nn.Sigmoid()\n"
    else:
        print("[GM_templates.LeakyReLU_template] Invalid data type!!")
        exit() 
    return template



""""
POOLING TEMPLATES
"""

def MaxPool_template(layer, hk, wk, hstr, wstr, data_type):
    if data_type == 'FP32':
        template ="\t\tself.l"+str(layer)+" = nn.MaxPool2d(kernel_size=(l"+str(layer)+"_hk, l"+str(layer)+"_wk), stride=(l"+str(layer)+"_hstr, l"+str(layer)+"_wstr))\n"
    elif data_type == 'FP16':
        template ="\t\tself.l"+str(layer)+" = nn.MaxPool2d(kernel_size=(l"+str(layer)+"_hk, l"+str(layer)+"_wk), stride=(l"+str(layer)+"_hstr, l"+str(layer)+"_wstr)).half()\n"
    else:
        print("[GM_templates.MaxPool_template] Invalid data type!!")
        exit()
    return template

def AvgPool_template(layer, hk, wk, hstr, wstr, data_type):
    if data_type == 'FP32':
        template ="\t\tself.l"+str(layer)+" = nn.AvgPool2d(kernel_size=(l"+str(layer)+"_hk, l"+str(layer)+"_wk), stride=(l"+str(layer)+"_hstr, l"+str(layer)+"_wstr))\n"
    elif data_type == 'FP16':
        template ="\t\tself.l"+str(layer)+" = nn.AvgPool2d(kernel_size=(l"+str(layer)+"_hk, l"+str(layer)+"_wk), stride=(l"+str(layer)+"_hstr, l"+str(layer)+"_wstr)).half()\n"
    else:
        print("[GM_templates.AvgPool_template] Invalid data type!!")
        exit()
    return template

'''
SKIPCONN TEMPLATES
'''

def Skipnode_template(layer):                                       
    template= "\t\tself.l"+str(layer) +" =Skipnode() #Skip layer\n"
    return template                                             

def Sumnode_template(layer, ls):                                        
    template= "\t\tself.l"+str(layer) +f"= Sumnode({ls}) #Sumnode layer\n"
    return template                                             


'''
NORMALIZATION TEMPLATE
'''

def InstNorm_template(layer, ch, data_type):
    if data_type == 'FP32':
        template = f"\t\tself.l{layer}= nn.InstanceNorm2d(num_features={ch}, eps=1e-10, momentum=0, affine=True)\n"
    else:
        template = f"\t\tself.l{layer}= nn.InstanceNorm2d(num_features={ch}, eps=1e-10, momentum=0, affine=True).half()\n"
    return template