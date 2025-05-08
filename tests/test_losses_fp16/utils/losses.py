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

import torch
import torch.nn as nn

# berHu loss computed for each element and used for this loss
# SOURCE: https://github.com/abduallahmohamed/reversehuberloss/blob/master/rhuloss.py

# MODULE VERSION
class berHuLoss_fp16(nn.Module):
    
    def __init__(self, alpha=0.2, invalid_value=-1, device='cpu'):
        super(berHuLoss_fp16, self).__init__()
        self.alpha = alpha
        self.invalid_value = invalid_value
        self.device = device

    def forward(self, output, target):
        # First, find the mask of valid values
        valid_mask = (target != self.invalid_value) # (target > invalid_value).float()
        N_valid = torch.sum(valid_mask).half()

        # Then, normalize prediction and target according to the max value of target disparity
        norm_value = torch.max(target)
        target     = target / norm_value
        output     = output / norm_value

        if output.size() != target.size():
            raise Exception("[ProxySuperVisionLoss] Differently sized output and target!")

        # Compute the berHu loss on the normalized and masked values
        absdiff   = torch.abs(output - target)
        C         = 0.2 * torch.max(absdiff).item()
        berhu_map = torch.where(absdiff < C, absdiff, (absdiff*absdiff - C*C) / (2*C))
        valid_map = torch.zeros_like(output).to(self.device)
        valid_map[valid_mask] = berhu_map[valid_mask]
        loss      = torch.sum(valid_map) / N_valid

        return loss
    
class berHuLoss_bfloat16(nn.Module):
    
    def __init__(self, alpha=0.2, invalid_value=-1, device='cpu'):
        super(berHuLoss_bfloat16, self).__init__()
        self.alpha = alpha
        self.invalid_value = invalid_value
        self.device = device

    def forward(self, output, target):
        # First, find the mask of valid values
        valid_mask = (target != self.invalid_value) # (target > invalid_value).float()
        N_valid = torch.sum(valid_mask).bfloat16()

        # Then, normalize prediction and target according to the max value of target disparity
        norm_value = torch.max(target)
        target     = target / norm_value
        output     = output / norm_value

        if output.size() != target.size():
            raise Exception("[ProxySuperVisionLoss] Differently sized output and target!")

        # Compute the berHu loss on the normalized and masked values
        absdiff   = torch.abs(output - target)
        C         = 0.2 * torch.max(absdiff).item()
        berhu_map = torch.where(absdiff < C, absdiff, (absdiff*absdiff - C*C) / (2*C))
        valid_map = torch.zeros_like(output).to(self.device)
        valid_map[valid_mask] = berhu_map[valid_mask]
        loss      = torch.sum(valid_map) / N_valid

        return loss

