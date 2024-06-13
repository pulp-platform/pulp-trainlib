/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pulp_train_utils_fp32.h"
#include "pulp_pooling_fp32.h"

void pulp_avgpool_fp32_fw_cl(void * pool_args) {

  struct pool_args * args = (struct pool_args *) pool_args;
  float * inData = args->input->data;
  float * outData = args->output->data;
  uint16_t W = args->input->W;
  uint16_t H = args->input->H;
  uint16_t C = args->input->C;
  uint16_t Ho = args->output->H;
  uint16_t Wo = args->output->W;
  uint16_t Hker = args->Hker;
  uint16_t Wker = args->Wker;
  uint16_t Hstr = args->Hstride;
  uint16_t Wstr = args->Wstride;

  // Internal variables
  uint32_t Hact = (H-Hker+Hstr)/Hstr; //H / Hker;
  uint32_t Wact = (W-Wker+Wstr)/Wstr; //W / Wker;
  if (Hact!=Ho || Wact!=Wo)   {printf("\n[pulp_avgpool_fp32_fw_cl] Invalid pooling kernel size or output size!\n"); return;}
  int HW = H*W;
  int HWk = Hker*Wker;
  int HWo = Ho*Wo;

  const int blockSize = (C+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > C ? C : start+blockSize;

  for (int k=start; k<stop; k++) {
    for (int ha=0; ha<Hact; ha++) {
      for (int wa=0; wa<Wact; wa++) {
        float avgpool = 0;
        for (int hk=0; hk<Hker; hk++) {
          for (int wk=0; wk<Wker; wk++) {
            uint32_t  in_idx = wk + wa*Wstr + (hk + ha*Hstr)*W + k*HW;
            avgpool += inData[in_idx];
          }
        }
        avgpool = avgpool / HWk;
        uint32_t out_idx = wa + ha*Wo + k*HWo;
        outData[out_idx] = avgpool;
      }
    }
  }
}

void pulp_avgpool_fp32_bw_cl(void * pool_args){

  struct pool_args * args = (struct pool_args *) pool_args;
  float * inDiff = args->input->diff;
  float * outDiff = args->output->diff;
  uint16_t W = args->input->W;
  uint16_t H = args->input->H;
  uint16_t C = args->input->C;
  uint16_t Ho = args->output->H;
  uint16_t Wo = args->output->W;
  uint16_t Hker = args->Hker;
  uint16_t Wker = args->Wker;
  uint16_t Hstr = args->Hstride;
  uint16_t Wstr = args->Wstride;

  // Internal variables
  uint32_t Hact = (H-Hker+Hstr)/Hstr; //H / Hker;
  uint32_t Wact = (W-Wker+Wstr)/Wstr; //W / Wker;
  if (Hact!=Ho || Wact!=Wo)   {printf("\n[pulp_avgpool_fp32_bw_cl] Invalid pooling kernel size or output size!\n"); return;}
  int HW = H*W;
  int HWk = Hker*Wker;
  int HWo = Ho*Wo;

  const int blockSize = (C+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > C ? C : start+blockSize;

  // Initialize gradient
  for (int k=start; k<stop; k++) {
    for (int hw=0; hw<HW; hw++) {
      inDiff[k*HW+hw] = 0;
    }
  }

  // Compute input gradient
  for (int k=start; k<stop; k++) {
    for (int ho=0; ho<Ho; ho++) {
      for (int wo=0; wo<Wo; wo++) {
        for (int hact=0; hact<Hker; hact++) {
          for (int wact=0; wact<Wker; wact++) {
            int in_idx = (wact+wo*Wstr) + (hact+ho*Hstr)*W + k*HW;
            int out_idx = wo + ho*Wo + k*HWo;
            inDiff[in_idx] += outDiff[out_idx] / HWk;
          }
        }
      }
    }
  }
}





void pulp_maxpool_fp32_fw_cl(void * pool_args) {

  struct pool_args * args = (struct pool_args *) pool_args;
  float * inData = args->input->data;
  float * outData = args->output->data;
  uint16_t W = args->input->W;
  uint16_t H = args->input->H;
  uint16_t C = args->input->C;
  uint16_t Ho = args->output->H;
  uint16_t Wo = args->output->W;
  uint16_t Hker = args->Hker;
  uint16_t Wker = args->Wker;
  uint16_t Hstr = args->Hstride;
  uint16_t Wstr = args->Wstride;

  // Internal variables
  uint32_t Hact = (H-Hker+Hstr)/Hstr; //H / Hker;
  uint32_t Wact = (W-Wker+Hstr)/Wstr; //W / Wker;
  if (Hact!=Ho || Wact!=Wo)   {printf("\n[pulp_maxpool_fp32_fw_cl] Invalid pooling kernel size or output size!\n"); return;}
  int HW = H*W;
  int HWk = Hker*Wker;
  int HWo = Ho*Wo;

  const int blockSize = (C+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > C ? C : start+blockSize;

  for (int k=start; k<stop; k++) {
    for (int ha=0; ha<Hact; ha++) {
      for (int wa=0; wa<Wact; wa++) {
        float maxpool = inData[wa*Wstr + ha*Hstr*W + k*HW];
        for (int hk=0; hk<Hker; hk++) {
          for (int wk=0; wk<Wker; wk++) {
            uint32_t  in_idx = wk + wa*Wstr + (hk + ha*Hstr)*W + k*HW;
            float newData = inData[in_idx];
            if (newData > maxpool)  {maxpool = newData;}
          }
        }
        uint32_t out_idx = wa + ha*Wo + k*HWo;
        outData[out_idx] = maxpool;
      }
    }
  }
}

void pulp_maxpool_fp32_bw_cl(void * pool_args){

  struct pool_args * args = (struct pool_args *) pool_args;
  float * inData = args->input->data;
  float * inDiff = args->input->diff;
  float * outDiff = args->output->diff;
  uint16_t W = args->input->W;
  uint16_t H = args->input->H;
  uint16_t C = args->input->C;
  uint16_t Ho = args->output->H;
  uint16_t Wo = args->output->W;
  uint16_t Hker = args->Hker;
  uint16_t Wker = args->Wker;
  uint16_t Hstr = args->Hstride;
  uint16_t Wstr = args->Wstride;

  // Internal variables
  uint32_t Hact = (H-Hker+Hstr)/Hstr;
  uint32_t Wact = (W-Wker+Hstr)/Wstr;
  if (Hact!=Ho || Wact!=Wo)   {printf("\n[pulp_maxpool_fp32_bw_cl] Invalid pooling kernel size or output size!\n"); return;}
  int HW = H*W;
  int HWk = Hker*Wker;
  int HWo = Ho*Wo;

  const int blockSize = (C+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > C ? C : start+blockSize;

  // Initialize gradient and mask
  for (int k=start; k<stop; k++) {
    for (int hw=0; hw<HW; hw++) {
      inDiff[k*HW+hw] = 0;
    }
  }

  // Mask for the update
  uint8_t maxmask[HWk];
  uint32_t maxidx = 0;

  // Compute input gradient
  for (int k=start; k<stop; k++) {
    for (int ho=0; ho<Ho; ho++) {
      for (int wo=0; wo<Wo; wo++) {
        // Create local mask
        float max = inData[wo*Wstr + ho*Hstr*W + k*HW];
        maxidx = 0;
        // Find maximum location and set mask
        for (int hact=0; hact<Hker; hact++) {
          for (int wact=0; wact<Wker; wact++) {
            maxmask[wact+hact*Wker] = 0;
            uint32_t  in_idx = wact + wo*Wstr + (hact + ho*Hstr)*W + k*HW;
            float newData = inData[in_idx];
            if (newData > max)  {
              max = newData;
              maxidx = wact + hact*Wker;
            }          
          }
        }        
        maxmask[maxidx] = 1;
        // Update gradients
        for (int hact=0; hact<Hker; hact++) {
          for (int wact=0; wact<Wker; wact++) {
            int in_idx = (wact+wo*Wstr) + (hact+ho*Hstr)*W + k*HW;
            int out_idx = wo + ho*Wo + k*HWo;
            inDiff[in_idx] += maxmask[wact+hact*Wker]*outDiff[out_idx];
          }
        }
      }
    }
  }
}