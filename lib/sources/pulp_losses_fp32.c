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

#include "math.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_losses_fp32.h"


void pulp_CrossEntropyLoss ( void * loss_args )
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;

  float loss = 0.0;
  for(int i=0; i<size; i++){
    loss += -target[i]*logf(outData[i]);
    
    #ifdef DEBUG
      printf("target: %f, out_data:%f\n", target[i], outData[i]);
      printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);  
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;
}


void pulp_CrossEntropyLoss_backward ( void * loss_args )
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;

  for(int i=0; i<size; i++){
    outDiff[i] = (-target[i] / outData[i]);
    
    #ifdef DEBUG
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }
}




void pulp_L1Loss ( void * loss_args ) 
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;
  int off = 0;

  float loss = 0.0f;
  float meanval = 1.0f / size;
  
  #ifdef DEBUG
  printf("loss meanval is: %f\n", meanval);
  #endif
  
  for(int i=0; i<size; i++){
    loss += meanval * fabsf(target[i] - outData[i]);

    #ifdef DEBUG
    printf("target: %f, out_data:%f\n", target[i], outData[i]);
    printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;
}


void pulp_L1Loss_backward ( void * loss_args ) 
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;
  int off = 0;

  float meanval = 1.0f / size;

  for(int i=0; i<size; i++){
    if ((outData[i] - target[i]) > 0) {
      outDiff[i] = meanval;
    }
    else if ((outData[i] - target[i]) == 0) {
      outDiff[i] = 0;
    }
    else {
      outDiff[i] = -meanval;
    }

    #ifdef DEBUG
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }

}





void pulp_MSELoss ( void * loss_args ) 
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;
  int off = 0;

  float loss = 0.0f;
  float meanval = 1.0f / size;
  
  #ifdef DEBUG
  printf("loss meanval is: %f\n", meanval);
  #endif
  
  for(int i=0; i<size; i++){
    loss += meanval * (target[i] - outData[i]) * (target[i] - outData[i]);

    #ifdef DEBUG
    printf("target: %f, out_data:%f\n", target[i], outData[i]);
    printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;
}


void pulp_MSELoss_backward ( void * loss_args ) 
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;
  int off = 0;

  float meanval = 1.0f / size;

  for(int i=0; i<size; i++){
    outDiff[i] = meanval * 2.0f *(outData[i] - target[i]);

    #ifdef DEBUG
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }

}






void pulp_berHuLoss ( void * berHu_loss_args ) 
{
  struct berHu_loss_args * args = (struct berHu_loss_args *) berHu_loss_args;
  float * outData = args->output->data;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  float alpha = args->alpha;
  int size = args->output->dim;
  int off = 0;

  float loss = 0.0f;
  float meanval = 1.0f / size;
  
  // Compute c constant
  float c = 0;
  for (int i=0; i<size; i++) {
    float value = fabsf(outData[i] - target[i]);
    if (value > c) {c = value;}
  }
  c = alpha * c;

  // Compute loss
  for(int i=0; i<size; i++){
    float value = fabsf(outData[i] - target[i]);
    if (value < c) {
      loss += value;
    }
    else {
      float num = value*value - c*c;
      float den = 2*c;
      loss += num / den; 
    }
  }
  loss = loss * meanval;

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;
}


void pulp_berHuLoss_backward ( void * berHu_loss_args ) 
{
  struct berHu_loss_args * args = (struct berHu_loss_args *) berHu_loss_args;
  float * outData = args->output->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  float alpha = args->alpha;
  int size = args->output->dim;
  int off = 0;

  float meanval = 1.0f / size;

  // Compute c constant
  float c = 0;
  for (int i=0; i<size; i++) {
    float value = fabsf(outData[i] - target[i]);
    if (value > c) {c = value;}
  }
  c = alpha * c;

  // Compute output grad
  for(int i=0; i<size; i++){
    float value  = fabsf(outData[i] - target[i]);
    float derval = outData[i] - target[i];
    if (value < c) {
      if      (derval > 0)  outDiff[i] = meanval;
      else if (derval == 0) outDiff[i] = 0;
      else                  outDiff[i] = -meanval;
    }
    else {
      outDiff[i] = meanval * derval / c;
    }
  }

}
