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

#include "stdio.h"
#include "math.h"
#include "stm32_train_utils.h"
#include "stm32_losses.h"


void stm32_CrossEntropyLoss (struct blob * output, float * target, float * wr_loss)
{
  float * outData = output->data;
  float * outDiff = output->diff;
  int size = output->dim;


  float loss = 0.0;
  for(int i=0; i<size; i++){
    loss += -target[i]*logf(outData[i]);
    
    #ifdef DEBUG_APP
      printf("target: %f, out_diff: %f, out_data:%f\n", target[i], outDiff[i], outData[i]);
      printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG_APP
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);  
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;

  for(int i=0; i<size; i++){
    outDiff[i] = (-target[i]+outData[i]);
    
    #ifdef DEBUG_APP
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }
}


void stm32_MSELoss (struct blob * output, float * target, float * wr_loss) 
{
  float * outData = output->data;
  float * outDiff = output->diff;
  int size = output->dim;
  int off = 0;

  float loss = 0.0;
  float meanval = 1.0f / size;
  
  #ifdef DEBUG_APP
  printf("loss meanval is: %f\n", meanval);
  #endif
  
  for(int i=0; i<size; i++){
    loss += meanval * (target[i] - outData[i]) * (target[i] - outData[i]);

    #ifdef DEBUG_APP
    printf("target: %f, out_diff: %f, out_data:%f\n", target[i], outDiff[i], outData[i]);
    printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG_APP
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;

  for(int i=0; i<size; i++){
    outDiff[i] = meanval * 2.0f *(outData[i] - target[i]);

    #ifdef DEBUG_APP
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }

}
