import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dump_utils as dump
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Define hyperparameters
learning_rate = 0.01
batch_size = 1
epochs = 50


labels_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

classes = [4, 9]
num_train = 5
num_test = 2
new_labels_map = []
for c in classes:
  new_labels_map.append(labels_map[c])

train_data = datasets.CIFAR10(root='./', train=True, download='True', transform=ToTensor())
test_data = datasets.CIFAR10(root='./', train=False, download='True', transform=ToTensor())

new_train_data = []
new_test_data = []
new_label = 0
label_list = []

for lbl in classes:
  count = 0
  idx = 0
  while(count<num_train):
    if train_data[idx][1] == lbl:
      temp = [train_data[idx][0], new_label]
      new_train_data.append(temp)
      label_list.append(torch.nn.functional.one_hot(torch.tensor(new_label), num_classes=len(classes)).float().tolist())
      count += 1
    idx += 1
  new_label += 1


'''
print("TRAIN DATA")
for i in new_train_data:
  img, label = i
  plt.axis("off")
  plt.imshow(img.permute(1, 2, 0))
  plt.show()
  print(new_labels_map[int(label)])
'''
new_label = 0
for lbl in classes:
  count = 0
  idx = 0
  while(count<num_test):
    if test_data[idx][1] == lbl:
      temp = [test_data[idx][0], new_label]
      new_test_data.append(temp)
      label_list.append(torch.nn.functional.one_hot(torch.tensor(new_label), num_classes=len(classes)).float().tolist())
      count += 1
    idx += 1
  new_label += 1

  
'''
print("TEST DATA")
for i in new_test_data:
  img, label = i
  plt.axis("off")
  plt.imshow(img.permute(1, 2, 0))
  plt.show()
  print(new_labels_map[int(label)])
'''
# LAYER 0 SIZES
l0_in_ch = 3
l0_out_ch = 3
l0_hk = 7
l0_wk = 7
l0_hin = 32
l0_win = 32
l0_hstr = 1
l0_wstr = 1
l0_hpad = 0
l0_wpad = 0
# LAYER 1 SIZES
l1_in_ch = 3
l1_out_ch = 8
l1_hk = 1
l1_wk = 1
l1_hin = 26
l1_win = 26
l1_hstr = 1
l1_wstr = 1
l1_hpad = 0
l1_wpad = 0
# LAYER 2 SIZES
l2_in_ch = 8
l2_out_ch = 8
l2_hk = 1
l2_wk = 1
l2_hin = 26
l2_win = 26
l2_hstr = 1
l2_wstr = 1
l2_hpad = 0
l2_wpad = 0
# LAYER 3 SIZES
l3_in_ch = 8
l3_out_ch = 8
l3_hk = 1
l3_wk = 1
l3_hin = 26
l3_win = 26
l3_hstr = 1
l3_wstr = 1
l3_hpad = 0
l3_wpad = 0
# LAYER 4 SIZES
l4_in_ch = 8
l4_out_ch = 8
l4_hk = 7
l4_wk = 7
l4_hin = 26
l4_win = 26
l4_hstr = 1
l4_wstr = 1
l4_hpad = 0
l4_wpad = 0
# LAYER 5 SIZES
l5_in_ch = 8
l5_out_ch = 16
l5_hk = 1
l5_wk = 1
l5_hin = 20
l5_win = 20
l5_hstr = 1
l5_wstr = 1
l5_hpad = 0
l5_wpad = 0
# LAYER 6 SIZES
l6_in_ch = 16
l6_out_ch = 16
l6_hk = 1
l6_wk = 1
l6_hin = 20
l6_win = 20
l6_hstr = 1
l6_wstr = 1
l6_hpad = 0
l6_wpad = 0
# LAYER 7 SIZES
l7_in_ch = 16
l7_out_ch = 16
l7_hk = 1
l7_wk = 1
l7_hin = 20
l7_win = 20
l7_hstr = 1
l7_wstr = 1
l7_hpad = 0
l7_wpad = 0
# LAYER 8_1 SIZES
l8_1_in_ch = 16
l8_1_out_ch = 16
l8_1_hk = 11
l8_1_wk = 11
l8_1_hin = 20
l8_1_win = 20
l8_1_hstr = 1
l8_1_wstr = 1
l8_1_hpad = 0
l8_1_wpad = 0
# LAYER 8_2 SIZES
l8_2_in_ch = 16
l8_2_out_ch = 32
l8_2_hk = 1
l8_2_wk = 1
l8_2_hin = 10
l8_2_win = 10
l8_2_hstr = 1
l8_2_wstr = 1
l8_2_hpad = 0
l8_2_wpad = 0
# LAYER 9 SIZES
l9_in_ch = 16
l9_out_ch = 16
l9_hk = 7
l9_wk = 7
l9_hin = 20
l9_win = 20
l9_hstr = 1
l9_wstr = 1
l9_hpad = 0
l9_wpad = 0
# LAYER 10 SIZES
l10_in_ch = 16
l10_out_ch = 24
l10_hk = 1
l10_wk = 1
l10_hin = 14
l10_win = 14
l10_hstr = 1
l10_wstr = 1
l10_hpad = 0
l10_wpad = 0
# LAYER 11 SIZES
l11_in_ch = 24
l11_out_ch = 24
l11_hk = 1
l11_wk = 1
l11_hin = 14
l11_win = 14
l11_hstr = 1
l11_wstr = 1
l11_hpad = 0
l11_wpad = 0
# LAYER 12 SIZES
l12_in_ch = 24
l12_out_ch = 24
l12_hk = 1
l12_wk = 1
l12_hin = 14
l12_win = 14
l12_hstr = 1
l12_wstr = 1
l12_hpad = 0
l12_wpad = 0
# LAYER 13 SIZES
l13_in_ch = 24
l13_out_ch = 24
l13_hk = 5
l13_wk = 5
l13_hin = 14
l13_win = 14
l13_hstr = 1
l13_wstr = 1
l13_hpad = 0
l13_wpad = 0
# LAYER 14 SIZES
l14_in_ch = 24
l14_out_ch = 32
l14_hk = 1
l14_wk = 1
l14_hin = 10
l14_win = 10
l14_hstr = 1
l14_wstr = 1
l14_hpad = 0
l14_wpad = 0
# LAYER 15 SIZES
l15_in_ch = 32
l15_out_ch = 32
l15_hk = 1
l15_wk = 1
l15_hin = 10
l15_win = 10
l15_hstr = 1
l15_wstr = 1
l15_hpad = 0
l15_wpad = 0
# LAYER 16 SIZES
l16_in_ch = 32
l16_out_ch = 32
l16_hk = 1
l16_wk = 1
l16_hin = 10
l16_win = 10
l16_hstr = 1
l16_wstr = 1
l16_hpad = 0
l16_wpad = 0
# LAYER 17 SIZES
l17_in_ch = 32
l17_out_ch = 32
l17_hk = 1
l17_wk = 1
l17_hin = 10
l17_win = 10
l17_hstr = 1
l17_wstr = 1
l17_hpad = 0
l17_wpad = 0
# LAYER 18 SIZES
l18_in_ch = 3200
l18_out_ch = 2
l18_hk = 1
l18_wk = 1
l18_hin = 1
l18_win = 1
l18_hstr = 1
l18_wstr = 1
l18_hpad = 0
l18_wpad = 0

f = open('init-defines.h', 'w')
f.write('// Layer0\n')
f.write('#define Tin_C_l0 '+str(l0_in_ch)+'\n')
f.write('#define Tout_C_l0 '+str(l0_out_ch)+'\n')
f.write('#define Tker_H_l0 '+str(l0_hk)+'\n')
f.write('#define Tker_W_l0 '+str(l0_wk)+'\n')
f.write('#define Tin_H_l0 '+str(l0_hin)+'\n')
f.write('#define Tin_W_l0 '+str(l0_win)+'\n')
f.write('#define Tout_H_l0 '+str(math.floor((l0_hin-l0_hk+2*l0_hpad+l0_hstr)/l0_hstr))+'\n')
f.write('#define Tout_W_l0 '+str(math.floor((l0_win-l0_wk+2*l0_wpad+l0_wstr)/l0_wstr))+'\n')
f.write('#define Tstr_H_l0 '+str(l0_hstr)+'\n')
f.write('#define Tstr_W_l0 '+str(l0_wstr)+'\n')
f.write('#define Tpad_H_l0 '+str(l0_hpad)+'\n')
f.write('#define Tpad_W_l0 '+str(l0_wpad)+'\n')
f.write('// Layer1\n')
f.write('#define Tin_C_l1 '+str(l1_in_ch)+'\n')
f.write('#define Tout_C_l1 '+str(l1_out_ch)+'\n')
f.write('#define Tker_H_l1 '+str(l1_hk)+'\n')
f.write('#define Tker_W_l1 '+str(l1_wk)+'\n')
f.write('#define Tin_H_l1 '+str(l1_hin)+'\n')
f.write('#define Tin_W_l1 '+str(l1_win)+'\n')
f.write('#define Tout_H_l1 '+str(math.floor((l1_hin-l1_hk+2*l1_hpad+l1_hstr)/l1_hstr))+'\n')
f.write('#define Tout_W_l1 '+str(math.floor((l1_win-l1_wk+2*l1_wpad+l1_wstr)/l1_wstr))+'\n')
f.write('#define Tstr_H_l1 '+str(l1_hstr)+'\n')
f.write('#define Tstr_W_l1 '+str(l1_wstr)+'\n')
f.write('#define Tpad_H_l1 '+str(l1_hpad)+'\n')
f.write('#define Tpad_W_l1 '+str(l1_wpad)+'\n')
f.write('// Layer2\n')
f.write('#define Tin_C_l2 '+str(l2_in_ch)+'\n')
f.write('#define Tout_C_l2 '+str(l2_out_ch)+'\n')
f.write('#define Tker_H_l2 '+str(l2_hk)+'\n')
f.write('#define Tker_W_l2 '+str(l2_wk)+'\n')
f.write('#define Tin_H_l2 '+str(l2_hin)+'\n')
f.write('#define Tin_W_l2 '+str(l2_win)+'\n')
f.write('#define Tout_H_l2 '+str(math.floor((l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr))+'\n')
f.write('#define Tout_W_l2 '+str(math.floor((l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr))+'\n')
f.write('#define Tstr_H_l2 '+str(l2_hstr)+'\n')
f.write('#define Tstr_W_l2 '+str(l2_wstr)+'\n')
f.write('#define Tpad_H_l2 '+str(l2_hpad)+'\n')
f.write('#define Tpad_W_l2 '+str(l2_wpad)+'\n')
f.write('// Layer3\n')
f.write('#define Tin_C_l3 '+str(l3_in_ch)+'\n')
f.write('#define Tout_C_l3 '+str(l3_out_ch)+'\n')
f.write('#define Tker_H_l3 '+str(l3_hk)+'\n')
f.write('#define Tker_W_l3 '+str(l3_wk)+'\n')
f.write('#define Tin_H_l3 '+str(l3_hin)+'\n')
f.write('#define Tin_W_l3 '+str(l3_win)+'\n')
f.write('#define Tout_H_l3 '+str(math.floor((l3_hin-l3_hk+2*l3_hpad+l3_hstr)/l3_hstr))+'\n')
f.write('#define Tout_W_l3 '+str(math.floor((l3_win-l3_wk+2*l3_wpad+l3_wstr)/l3_wstr))+'\n')
f.write('#define Tstr_H_l3 '+str(l3_hstr)+'\n')
f.write('#define Tstr_W_l3 '+str(l3_wstr)+'\n')
f.write('#define Tpad_H_l3 '+str(l3_hpad)+'\n')
f.write('#define Tpad_W_l3 '+str(l3_wpad)+'\n')
f.write('// Layer4\n')
f.write('#define Tin_C_l4 '+str(l4_in_ch)+'\n')
f.write('#define Tout_C_l4 '+str(l4_out_ch)+'\n')
f.write('#define Tker_H_l4 '+str(l4_hk)+'\n')
f.write('#define Tker_W_l4 '+str(l4_wk)+'\n')
f.write('#define Tin_H_l4 '+str(l4_hin)+'\n')
f.write('#define Tin_W_l4 '+str(l4_win)+'\n')
f.write('#define Tout_H_l4 '+str(math.floor((l4_hin-l4_hk+2*l4_hpad+l4_hstr)/l4_hstr))+'\n')
f.write('#define Tout_W_l4 '+str(math.floor((l4_win-l4_wk+2*l4_wpad+l4_wstr)/l4_wstr))+'\n')
f.write('#define Tstr_H_l4 '+str(l4_hstr)+'\n')
f.write('#define Tstr_W_l4 '+str(l4_wstr)+'\n')
f.write('#define Tpad_H_l4 '+str(l4_hpad)+'\n')
f.write('#define Tpad_W_l4 '+str(l4_wpad)+'\n')
f.write('// Layer5\n')
f.write('#define Tin_C_l5 '+str(l5_in_ch)+'\n')
f.write('#define Tout_C_l5 '+str(l5_out_ch)+'\n')
f.write('#define Tker_H_l5 '+str(l5_hk)+'\n')
f.write('#define Tker_W_l5 '+str(l5_wk)+'\n')
f.write('#define Tin_H_l5 '+str(l5_hin)+'\n')
f.write('#define Tin_W_l5 '+str(l5_win)+'\n')
f.write('#define Tout_H_l5 '+str(math.floor((l5_hin-l5_hk+2*l5_hpad+l5_hstr)/l5_hstr))+'\n')
f.write('#define Tout_W_l5 '+str(math.floor((l5_win-l5_wk+2*l5_wpad+l5_wstr)/l5_wstr))+'\n')
f.write('#define Tstr_H_l5 '+str(l5_hstr)+'\n')
f.write('#define Tstr_W_l5 '+str(l5_wstr)+'\n')
f.write('#define Tpad_H_l5 '+str(l5_hpad)+'\n')
f.write('#define Tpad_W_l5 '+str(l5_wpad)+'\n')
f.write('// Layer6\n')
f.write('#define Tin_C_l6 '+str(l6_in_ch)+'\n')
f.write('#define Tout_C_l6 '+str(l6_out_ch)+'\n')
f.write('#define Tker_H_l6 '+str(l6_hk)+'\n')
f.write('#define Tker_W_l6 '+str(l6_wk)+'\n')
f.write('#define Tin_H_l6 '+str(l6_hin)+'\n')
f.write('#define Tin_W_l6 '+str(l6_win)+'\n')
f.write('#define Tout_H_l6 '+str(math.floor((l6_hin-l6_hk+2*l6_hpad+l6_hstr)/l6_hstr))+'\n')
f.write('#define Tout_W_l6 '+str(math.floor((l6_win-l6_wk+2*l6_wpad+l6_wstr)/l6_wstr))+'\n')
f.write('#define Tstr_H_l6 '+str(l6_hstr)+'\n')
f.write('#define Tstr_W_l6 '+str(l6_wstr)+'\n')
f.write('#define Tpad_H_l6 '+str(l6_hpad)+'\n')
f.write('#define Tpad_W_l6 '+str(l6_wpad)+'\n')
f.write('// Layer7\n')
f.write('#define Tin_C_l7 '+str(l7_in_ch)+'\n')
f.write('#define Tout_C_l7 '+str(l7_out_ch)+'\n')
f.write('#define Tker_H_l7 '+str(l7_hk)+'\n')
f.write('#define Tker_W_l7 '+str(l7_wk)+'\n')
f.write('#define Tin_H_l7 '+str(l7_hin)+'\n')
f.write('#define Tin_W_l7 '+str(l7_win)+'\n')
f.write('#define Tout_H_l7 '+str(math.floor((l7_hin-l7_hk+2*l7_hpad+l7_hstr)/l7_hstr))+'\n')
f.write('#define Tout_W_l7 '+str(math.floor((l7_win-l7_wk+2*l7_wpad+l7_wstr)/l7_wstr))+'\n')
f.write('#define Tstr_H_l7 '+str(l7_hstr)+'\n')
f.write('#define Tstr_W_l7 '+str(l7_wstr)+'\n')
f.write('#define Tpad_H_l7 '+str(l7_hpad)+'\n')
f.write('#define Tpad_W_l7 '+str(l7_wpad)+'\n')
f.write('// Layer8_1\n')
f.write('#define Tin_C_l8_1 '+str(l8_1_in_ch)+'\n')
f.write('#define Tout_C_l8_1 '+str(l8_1_out_ch)+'\n')
f.write('#define Tker_H_l8_1 '+str(l8_1_hk)+'\n')
f.write('#define Tker_W_l8_1 '+str(l8_1_wk)+'\n')
f.write('#define Tin_H_l8_1 '+str(l8_1_hin)+'\n')
f.write('#define Tin_W_l8_1 '+str(l8_1_win)+'\n')
f.write('#define Tout_H_l8_1 '+str(math.floor((l8_1_hin-l8_1_hk+2*l8_1_hpad+l8_1_hstr)/l8_1_hstr))+'\n')
f.write('#define Tout_W_l8_1 '+str(math.floor((l8_1_win-l8_1_wk+2*l8_1_wpad+l8_1_wstr)/l8_1_wstr))+'\n')
f.write('#define Tstr_H_l8_1 '+str(l8_1_hstr)+'\n')
f.write('#define Tstr_W_l8_1 '+str(l8_1_wstr)+'\n')
f.write('#define Tpad_H_l8_1 '+str(l8_1_hpad)+'\n')
f.write('#define Tpad_W_l8_1 '+str(l8_1_wpad)+'\n')
f.write('// Layer8_2\n')
f.write('#define Tin_C_l8_2 '+str(l8_2_in_ch)+'\n')
f.write('#define Tout_C_l8_2 '+str(l8_2_out_ch)+'\n')
f.write('#define Tker_H_l8_2 '+str(l8_2_hk)+'\n')
f.write('#define Tker_W_l8_2 '+str(l8_2_wk)+'\n')
f.write('#define Tin_H_l8_2 '+str(l8_2_hin)+'\n')
f.write('#define Tin_W_l8_2 '+str(l8_2_win)+'\n')
f.write('#define Tout_H_l8_2 '+str(math.floor((l8_2_hin-l8_2_hk+2*l8_2_hpad+l8_2_hstr)/l8_2_hstr))+'\n')
f.write('#define Tout_W_l8_2 '+str(math.floor((l8_2_win-l8_2_wk+2*l8_2_wpad+l8_2_wstr)/l8_2_wstr))+'\n')
f.write('#define Tstr_H_l8_2 '+str(l8_2_hstr)+'\n')
f.write('#define Tstr_W_l8_2 '+str(l8_2_wstr)+'\n')
f.write('#define Tpad_H_l8_2 '+str(l8_2_hpad)+'\n')
f.write('#define Tpad_W_l8_2 '+str(l8_2_wpad)+'\n')
f.write('// Layer9\n')
f.write('#define Tin_C_l9 '+str(l9_in_ch)+'\n')
f.write('#define Tout_C_l9 '+str(l9_out_ch)+'\n')
f.write('#define Tker_H_l9 '+str(l9_hk)+'\n')
f.write('#define Tker_W_l9 '+str(l9_wk)+'\n')
f.write('#define Tin_H_l9 '+str(l9_hin)+'\n')
f.write('#define Tin_W_l9 '+str(l9_win)+'\n')
f.write('#define Tout_H_l9 '+str(math.floor((l9_hin-l9_hk+2*l9_hpad+l9_hstr)/l9_hstr))+'\n')
f.write('#define Tout_W_l9 '+str(math.floor((l9_win-l9_wk+2*l9_wpad+l9_wstr)/l9_wstr))+'\n')
f.write('#define Tstr_H_l9 '+str(l9_hstr)+'\n')
f.write('#define Tstr_W_l9 '+str(l9_wstr)+'\n')
f.write('#define Tpad_H_l9 '+str(l9_hpad)+'\n')
f.write('#define Tpad_W_l9 '+str(l9_wpad)+'\n')
f.write('// Layer10\n')
f.write('#define Tin_C_l10 '+str(l10_in_ch)+'\n')
f.write('#define Tout_C_l10 '+str(l10_out_ch)+'\n')
f.write('#define Tker_H_l10 '+str(l10_hk)+'\n')
f.write('#define Tker_W_l10 '+str(l10_wk)+'\n')
f.write('#define Tin_H_l10 '+str(l10_hin)+'\n')
f.write('#define Tin_W_l10 '+str(l10_win)+'\n')
f.write('#define Tout_H_l10 '+str(math.floor((l10_hin-l10_hk+2*l10_hpad+l10_hstr)/l10_hstr))+'\n')
f.write('#define Tout_W_l10 '+str(math.floor((l10_win-l10_wk+2*l10_wpad+l10_wstr)/l10_wstr))+'\n')
f.write('#define Tstr_H_l10 '+str(l10_hstr)+'\n')
f.write('#define Tstr_W_l10 '+str(l10_wstr)+'\n')
f.write('#define Tpad_H_l10 '+str(l10_hpad)+'\n')
f.write('#define Tpad_W_l10 '+str(l10_wpad)+'\n')
f.write('// Layer11\n')
f.write('#define Tin_C_l11 '+str(l11_in_ch)+'\n')
f.write('#define Tout_C_l11 '+str(l11_out_ch)+'\n')
f.write('#define Tker_H_l11 '+str(l11_hk)+'\n')
f.write('#define Tker_W_l11 '+str(l11_wk)+'\n')
f.write('#define Tin_H_l11 '+str(l11_hin)+'\n')
f.write('#define Tin_W_l11 '+str(l11_win)+'\n')
f.write('#define Tout_H_l11 '+str(math.floor((l11_hin-l11_hk+2*l11_hpad+l11_hstr)/l11_hstr))+'\n')
f.write('#define Tout_W_l11 '+str(math.floor((l11_win-l11_wk+2*l11_wpad+l11_wstr)/l11_wstr))+'\n')
f.write('#define Tstr_H_l11 '+str(l11_hstr)+'\n')
f.write('#define Tstr_W_l11 '+str(l11_wstr)+'\n')
f.write('#define Tpad_H_l11 '+str(l11_hpad)+'\n')
f.write('#define Tpad_W_l11 '+str(l11_wpad)+'\n')
f.write('// Layer12\n')
f.write('#define Tin_C_l12 '+str(l12_in_ch)+'\n')
f.write('#define Tout_C_l12 '+str(l12_out_ch)+'\n')
f.write('#define Tker_H_l12 '+str(l12_hk)+'\n')
f.write('#define Tker_W_l12 '+str(l12_wk)+'\n')
f.write('#define Tin_H_l12 '+str(l12_hin)+'\n')
f.write('#define Tin_W_l12 '+str(l12_win)+'\n')
f.write('#define Tout_H_l12 '+str(math.floor((l12_hin-l12_hk+2*l12_hpad+l12_hstr)/l12_hstr))+'\n')
f.write('#define Tout_W_l12 '+str(math.floor((l12_win-l12_wk+2*l12_wpad+l12_wstr)/l12_wstr))+'\n')
f.write('#define Tstr_H_l12 '+str(l12_hstr)+'\n')
f.write('#define Tstr_W_l12 '+str(l12_wstr)+'\n')
f.write('#define Tpad_H_l12 '+str(l12_hpad)+'\n')
f.write('#define Tpad_W_l12 '+str(l12_wpad)+'\n')
f.write('// Layer13\n')
f.write('#define Tin_C_l13 '+str(l13_in_ch)+'\n')
f.write('#define Tout_C_l13 '+str(l13_out_ch)+'\n')
f.write('#define Tker_H_l13 '+str(l13_hk)+'\n')
f.write('#define Tker_W_l13 '+str(l13_wk)+'\n')
f.write('#define Tin_H_l13 '+str(l13_hin)+'\n')
f.write('#define Tin_W_l13 '+str(l13_win)+'\n')
f.write('#define Tout_H_l13 '+str(math.floor((l13_hin-l13_hk+2*l13_hpad+l13_hstr)/l13_hstr))+'\n')
f.write('#define Tout_W_l13 '+str(math.floor((l13_win-l13_wk+2*l13_wpad+l13_wstr)/l13_wstr))+'\n')
f.write('#define Tstr_H_l13 '+str(l13_hstr)+'\n')
f.write('#define Tstr_W_l13 '+str(l13_wstr)+'\n')
f.write('#define Tpad_H_l13 '+str(l13_hpad)+'\n')
f.write('#define Tpad_W_l13 '+str(l13_wpad)+'\n')
f.write('// Layer14\n')
f.write('#define Tin_C_l14 '+str(l14_in_ch)+'\n')
f.write('#define Tout_C_l14 '+str(l14_out_ch)+'\n')
f.write('#define Tker_H_l14 '+str(l14_hk)+'\n')
f.write('#define Tker_W_l14 '+str(l14_wk)+'\n')
f.write('#define Tin_H_l14 '+str(l14_hin)+'\n')
f.write('#define Tin_W_l14 '+str(l14_win)+'\n')
f.write('#define Tout_H_l14 '+str(math.floor((l14_hin-l14_hk+2*l14_hpad+l14_hstr)/l14_hstr))+'\n')
f.write('#define Tout_W_l14 '+str(math.floor((l14_win-l14_wk+2*l14_wpad+l14_wstr)/l14_wstr))+'\n')
f.write('#define Tstr_H_l14 '+str(l14_hstr)+'\n')
f.write('#define Tstr_W_l14 '+str(l14_wstr)+'\n')
f.write('#define Tpad_H_l14 '+str(l14_hpad)+'\n')
f.write('#define Tpad_W_l14 '+str(l14_wpad)+'\n')
f.write('// Layer15\n')
f.write('#define Tin_C_l15 '+str(l15_in_ch)+'\n')
f.write('#define Tout_C_l15 '+str(l15_out_ch)+'\n')
f.write('#define Tker_H_l15 '+str(l15_hk)+'\n')
f.write('#define Tker_W_l15 '+str(l15_wk)+'\n')
f.write('#define Tin_H_l15 '+str(l15_hin)+'\n')
f.write('#define Tin_W_l15 '+str(l15_win)+'\n')
f.write('#define Tout_H_l15 '+str(math.floor((l15_hin-l15_hk+2*l15_hpad+l15_hstr)/l15_hstr))+'\n')
f.write('#define Tout_W_l15 '+str(math.floor((l15_win-l15_wk+2*l15_wpad+l15_wstr)/l15_wstr))+'\n')
f.write('#define Tstr_H_l15 '+str(l15_hstr)+'\n')
f.write('#define Tstr_W_l15 '+str(l15_wstr)+'\n')
f.write('#define Tpad_H_l15 '+str(l15_hpad)+'\n')
f.write('#define Tpad_W_l15 '+str(l15_wpad)+'\n')
f.write('// Layer16\n')
f.write('#define Tin_C_l16 '+str(l16_in_ch)+'\n')
f.write('#define Tout_C_l16 '+str(l16_out_ch)+'\n')
f.write('#define Tker_H_l16 '+str(l16_hk)+'\n')
f.write('#define Tker_W_l16 '+str(l16_wk)+'\n')
f.write('#define Tin_H_l16 '+str(l16_hin)+'\n')
f.write('#define Tin_W_l16 '+str(l16_win)+'\n')
f.write('#define Tout_H_l16 '+str(math.floor((l16_hin-l16_hk+2*l16_hpad+l16_hstr)/l16_hstr))+'\n')
f.write('#define Tout_W_l16 '+str(math.floor((l16_win-l16_wk+2*l16_wpad+l16_wstr)/l16_wstr))+'\n')
f.write('#define Tstr_H_l16 '+str(l16_hstr)+'\n')
f.write('#define Tstr_W_l16 '+str(l16_wstr)+'\n')
f.write('#define Tpad_H_l16 '+str(l16_hpad)+'\n')
f.write('#define Tpad_W_l16 '+str(l16_wpad)+'\n')
f.write('// Layer17\n')
f.write('#define Tin_C_l17 '+str(l17_in_ch)+'\n')
f.write('#define Tout_C_l17 '+str(l17_out_ch)+'\n')
f.write('#define Tin_H_l17 '+str(l17_hin)+'\n')
f.write('#define Tin_W_l17 '+str(l17_win)+'\n')
f.write('#define Tout_H_l17 '+str(math.floor((l17_hin-l17_hk+2*l17_hpad+l17_hstr)/l17_hstr))+'\n')
f.write('#define Tout_W_l17 '+str(math.floor((l17_win-l17_wk+2*l17_wpad+l17_wstr)/l17_wstr))+'\n')
f.write('// Layer18\n')
f.write('#define Tin_C_l18 '+str(l18_in_ch)+'\n')
f.write('#define Tout_C_l18 '+str(l18_out_ch)+'\n')
f.write('#define Tker_H_l18 '+str(l18_hk)+'\n')
f.write('#define Tker_W_l18 '+str(l18_wk)+'\n')
f.write('#define Tin_H_l18 '+str(l18_hin)+'\n')
f.write('#define Tin_W_l18 '+str(l18_win)+'\n')
f.write('#define Tout_H_l18 '+str(math.floor((l18_hin-l18_hk+2*l18_hpad+l18_hstr)/l18_hstr))+'\n')
f.write('#define Tout_W_l18 '+str(math.floor((l18_win-l18_wk+2*l18_wpad+l18_wstr)/l18_wstr))+'\n')
f.write('#define Tstr_H_l18 '+str(l18_hstr)+'\n')
f.write('#define Tstr_W_l18 '+str(l18_wstr)+'\n')
f.write('#define Tpad_H_l18 '+str(l18_hpad)+'\n')
f.write('#define Tpad_W_l18 '+str(l18_wpad)+'\n')
f.close()

f = open('init-defines.h', 'a')
f.write('\n// HYPERPARAMETERS\n')
f.write('#define LEARNING_RATE '+str(learning_rate)+'\n')
f.write('#define EPOCHS '+str(epochs)+'\n')
f.write('#define NUM_TRAIN '+str(len(classes)*num_train)+'\n')
f.write('#define NUM_TEST '+str(len(classes)*num_test)+'\n')
f.write('#define NUM_CLASSES '+str(len(classes))+'\n')
f.write('#define IMG_SIZE '+str(3*32*32)+'\n')
f.close()

sftmx = nn.Softmax(dim = 0)
class Sumnode():
	def __init__(self, ls):
		self.MySkipNode = ls

class Skipnode():
	def __init__(self):
		self.data = 0

	def __call__(self, x):
		self.data = x
		return self.data

class DNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.l0 = nn.Conv2d(in_channels=l0_in_ch, out_channels=l0_in_ch, kernel_size=(l0_hk, l0_wk), stride = 1, groups=l0_in_ch, bias=False)
		self.l1 = nn.Conv2d(in_channels=l1_in_ch, out_channels=l1_out_ch, kernel_size=1, stride=1, bias=False)
		self.l2= nn.InstanceNorm2d(num_features=8, eps=1e-10, momentum=0, affine=True)
		self.l3 = nn.ReLU()
		self.l4 = nn.Conv2d(in_channels=l4_in_ch, out_channels=l4_in_ch, kernel_size=(l4_hk, l4_wk), stride = 1, groups=l4_in_ch, bias=False)
		self.l5 = nn.Conv2d(in_channels=l5_in_ch, out_channels=l5_out_ch, kernel_size=1, stride=1, bias=False)
		self.l6= nn.InstanceNorm2d(num_features=16, eps=1e-10, momentum=0, affine=True)
		self.l7 = nn.ReLU()
		self.l8_1 = nn.Conv2d(in_channels=l8_1_in_ch, out_channels=l8_1_out_ch, kernel_size=(l8_1_hk, l8_1_wk), stride = 1, groups=l8_1_in_ch, bias=False)
		self.l8_2 = nn.Conv2d(in_channels=l8_2_in_ch, out_channels=l8_2_out_ch, kernel_size=1, stride=1, bias=False)
		self.l9 = nn.Conv2d(in_channels=l9_in_ch, out_channels=l9_in_ch, kernel_size=(l9_hk, l9_wk), stride = 1, groups=l9_in_ch, bias=False)
		self.l10 = nn.Conv2d(in_channels=l10_in_ch, out_channels=l10_out_ch, kernel_size=1, stride=1, bias=False)
		self.l11= nn.InstanceNorm2d(num_features=24, eps=1e-10, momentum=0, affine=True)
		self.l12 = nn.ReLU()
		self.l13 = nn.Conv2d(in_channels=l13_in_ch, out_channels=l13_in_ch, kernel_size=(l13_hk, l13_wk), stride = 1, groups=l13_in_ch, bias=False)
		self.l14 = nn.Conv2d(in_channels=l14_in_ch, out_channels=l14_out_ch, kernel_size=1, stride=1, bias=False)
		self.l15= nn.InstanceNorm2d(num_features=32, eps=1e-10, momentum=0, affine=True)
		self.l16 = nn.ReLU()
		self.l17= Sumnode(8) #Sumnode layer
		self.l18 = nn.Linear(in_features=l18_in_ch, out_features=l18_out_ch, bias=False)
		
	def forward(self, x):
		x = self.l0(x)
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = self.l6(x)
		x = self.l7(x)
		y17 = self.l8_1(x)
		y17 = self.l8_2(y17)
		x = self.l9(x)
		x = self.l10(x)
		x = self.l11(x)
		x = self.l12(x)
		x = self.l13(x)
		x = self.l14(x)
		x = self.l15(x)
		x = self.l16(x)
		x = y17 + x	# Sumnode
		x = torch.reshape(x, (-1,))
		x = self.l18(x).float()
		#print(f"{x}, {sftmx(x)}")
		return x


# Initialize network
net = DNN()


for p in net.parameters():
    nn.init.normal_(p, mean=0.0, std=0.01)

net.zero_grad()
loss_fn = nn.CrossEntropyLoss()

f = open('io_data.h', 'w')
f.write('// Init weights\n')
f.write('#define WGT_SIZE_L0 '+str(l0_in_ch*l0_out_ch*l0_hk*l0_wk)+'\n')
f.write('PI_L2 float init_WGT_l0[WGT_SIZE_L0] = {'+dump.tensor_to_string(net.l0.weight.data)+'};\n')
f.write('#define WGT_SIZE_L1 '+str(l1_in_ch*l1_out_ch*l1_hk*l1_wk)+'\n')
f.write('PI_L2 float init_WGT_l1[WGT_SIZE_L1] = {'+dump.tensor_to_string(net.l1.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L2  2*{l2_in_ch}\n')
f.write('PI_L2 float init_WGT_l2[WGT_SIZE_L2] = {'+dump.tensor_to_string(net.l2.weight.data)+dump.tensor_to_string(net.l2.bias.data)+'};\n')
f.write('#define WGT_SIZE_L3 '+str(l3_in_ch*l3_out_ch*l3_hk*l3_wk)+'\n')
f.write('PI_L2 float init_WGT_l3[WGT_SIZE_L3];\n')
f.write('#define WGT_SIZE_L4 '+str(l4_in_ch*l4_out_ch*l4_hk*l4_wk)+'\n')
f.write('PI_L2 float init_WGT_l4[WGT_SIZE_L4] = {'+dump.tensor_to_string(net.l4.weight.data)+'};\n')
f.write('#define WGT_SIZE_L5 '+str(l5_in_ch*l5_out_ch*l5_hk*l5_wk)+'\n')
f.write('PI_L2 float init_WGT_l5[WGT_SIZE_L5] = {'+dump.tensor_to_string(net.l5.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L6  2*{l6_in_ch}\n')
f.write('PI_L2 float init_WGT_l6[WGT_SIZE_L6] = {'+dump.tensor_to_string(net.l6.weight.data)+dump.tensor_to_string(net.l6.bias.data)+'};\n')
f.write('#define WGT_SIZE_L7 '+str(l7_in_ch*l7_out_ch*l7_hk*l7_wk)+'\n')
f.write('PI_L2 float init_WGT_l7[WGT_SIZE_L7];\n')
f.write('#define WGT_SIZE_L8_1 '+str(l8_1_in_ch*l8_1_out_ch*l8_1_hk*l8_1_wk)+'\n')
f.write('PI_L2 float init_WGT_l8_1[WGT_SIZE_L8_1] = {'+dump.tensor_to_string(net.l8_1.weight.data)+'};\n')
f.write('#define WGT_SIZE_L8_2 '+str(l8_2_in_ch*l8_2_out_ch*l8_2_hk*l8_2_wk)+'\n')
f.write('PI_L2 float init_WGT_l8_2[WGT_SIZE_L8_2] = {'+dump.tensor_to_string(net.l8_2.weight.data)+'};\n')
f.write('#define WGT_SIZE_L9 '+str(l9_in_ch*l9_out_ch*l9_hk*l9_wk)+'\n')
f.write('PI_L2 float init_WGT_l9[WGT_SIZE_L9] = {'+dump.tensor_to_string(net.l9.weight.data)+'};\n')
f.write('#define WGT_SIZE_L10 '+str(l10_in_ch*l10_out_ch*l10_hk*l10_wk)+'\n')
f.write('PI_L2 float init_WGT_l10[WGT_SIZE_L10] = {'+dump.tensor_to_string(net.l10.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L11  2*{l11_in_ch}\n')
f.write('PI_L2 float init_WGT_l11[WGT_SIZE_L11] = {'+dump.tensor_to_string(net.l11.weight.data)+dump.tensor_to_string(net.l11.bias.data)+'};\n')
f.write('#define WGT_SIZE_L12 '+str(l12_in_ch*l12_out_ch*l12_hk*l12_wk)+'\n')
f.write('PI_L2 float init_WGT_l12[WGT_SIZE_L12];\n')
f.write('#define WGT_SIZE_L13 '+str(l13_in_ch*l13_out_ch*l13_hk*l13_wk)+'\n')
f.write('PI_L2 float init_WGT_l13[WGT_SIZE_L13] = {'+dump.tensor_to_string(net.l13.weight.data)+'};\n')
f.write('#define WGT_SIZE_L14 '+str(l14_in_ch*l14_out_ch*l14_hk*l14_wk)+'\n')
f.write('PI_L2 float init_WGT_l14[WGT_SIZE_L14] = {'+dump.tensor_to_string(net.l14.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L15  2*{l15_in_ch}\n')
f.write('PI_L2 float init_WGT_l15[WGT_SIZE_L15] = {'+dump.tensor_to_string(net.l15.weight.data)+dump.tensor_to_string(net.l15.bias.data)+'};\n')
f.write('#define WGT_SIZE_L16 '+str(l16_in_ch*l16_out_ch*l16_hk*l16_wk)+'\n')
f.write('PI_L2 float init_WGT_l16[WGT_SIZE_L16];\n')
f.write('#define WGT_SIZE_L17 '+str(l17_in_ch*l17_out_ch*l17_hk*l17_wk)+'\n')
f.write('PI_L2 float init_WGT_l17[WGT_SIZE_L17];\n')
f.write('#define WGT_SIZE_L18 '+str(l18_in_ch*l18_out_ch*l18_hk*l18_wk)+'\n')
f.write('PI_L2 float init_WGT_l18[WGT_SIZE_L18] = {'+dump.tensor_to_string(net.l18.weight.data)+'};\n')
f.close()

INPUT_DATA = []
for i in new_train_data:
  INPUT_DATA.append(i[0].tolist())
for i in new_test_data:
  INPUT_DATA.append(i[0].tolist())

def train_epoch():
	current_loss = 0
	count = 0
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)

	for t in new_train_data:
		optimizer.zero_grad()
		inputs, labels = t
		outputs = net(inputs)
		#print(outputs, labels)
		loss = loss_fn(outputs, torch.tensor(labels))
		loss.backward()
		optimizer.step()
		current_loss += loss.item()
		count += 1
		#print(count)
	return current_loss / count

test_loss = 0
correct_predictions = 0
log_list = []
voutputs = 0
for e in range(epochs):

    #learning_rate = 0.99*learning_rate
    train_loss = train_epoch()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        test_loss = 0
        correct_predictions = 0
        for test in new_test_data:
            vinputs, vlabels = test
            #print(vlabels)
            voutputs = net(vinputs)
            #print(voutputs)
            test_loss += loss_fn(voutputs, torch.tensor(vlabels)).item()
            predicted_out = torch.argmax(voutputs, dim=0)
            #print(predicted_out)
            #hotone_output = torch.nn.functional.one_hot(torch.argmax(voutputs, dim=1), num_classes=2)

            if vlabels == predicted_out:
                correct_predictions += 1
        test_loss = test_loss / len(new_test_data)
    print(f"epoch: {e}, train_loss: {train_loss}, test_loss: {test_loss}, accuracy:{correct_predictions} ({100 *correct_predictions / len(new_test_data)}%), lr: {learning_rate}")
    log_list.append([e, train_loss, float(test_loss), correct_predictions, 100 *correct_predictions / len(new_test_data), learning_rate])
    if (e % 10) == 9:
        df = pd.DataFrame(log_list)
        df.to_csv('log.csv', header=False, index=False)



f = open('io_data.h', 'a')
f.write('// Input and Output data\n')
f.write(f'#define IN_SIZE {(num_train + num_test)*len(classes)*3*32*32}\n')
f.write('PI_L2 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(torch.tensor(INPUT_DATA))+'};\n')
f.write(f'#define OUT_SIZE {len(classes)}\n')
f.write(f'PI_L2 float LABEL[{(num_train + num_test)*len(classes)}*OUT_SIZE] = '+'{'+dump.tensor_to_string(torch.tensor(label_list))+'};\n')
f.close()
