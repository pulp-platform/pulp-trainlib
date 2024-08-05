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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino
*/

#include "pulp_train_utils_fp32.h"
#include "pulp_act_fp32.h"
#include "math.h"


void pulp_sigmoid_fp32_fw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, sigmoid_core_fw_fp32, act_args);
}

void pulp_sigmoid_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, sigmoid_core_bw_fp32, act_args);
}

void sigmoid_core_fw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i=start; i<stop; i++) {
    float sigma = 0.0f;
    sigma = 1 + expf(-inData[i]);
    sigma = 1 / sigma;
    outData[i] = sigma;
  }
}

void sigmoid_core_bw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outData = args->output->data;
  float* outDiff = args->output->diff;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i=start; i<stop; i++) {
    float sigma = 0.0f;
    float sigma_prime = 0.0f;
    //sigma = 1 + expf(-inData[i]);
    //sigma = 1 / sigma;
    sigma = outData[i];
    sigma_prime = sigma * (1.0f - sigma);
    inDiff[i] = outDiff[i] * sigma_prime;
  }
}




void pulp_relu_fp32_fw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, act_args);
}

void pulp_relu_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, relu_core_bw_fp32, act_args);
}

void relu_core_fw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : 0;
  }

}

void relu_core_bw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outDiff = args->output->diff;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : 0;
  }
}




void pulp_leakyrelu_fp32_fw_cl( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  pi_cl_team_fork(NUM_CORES, leakyrelu_core_fw_fp32, args);
}

void pulp_leakyrelu_fp32_bw_cl( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  pi_cl_team_fork(NUM_CORES, leakyrelu_core_bw_fp32, args);
}

void leakyrelu_core_fw_fp32( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;
  float neg_slope = args->negative_slope;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : neg_slope*inData[i];
  }

}

void leakyrelu_core_bw_fp32( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outDiff = args->output->diff;
  float neg_slope = args->negative_slope;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : neg_slope*outDiff[i];
  }
}





void pulp_softmax_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  float* maxes = args->maxes;
  float* sums = args->sums;

  struct max_args m_args;
  m_args.input = inData;
  m_args.maxes = maxes;
  m_args.dim = dim;

  pi_cl_team_fork(NUM_CORES, pulp_row_max_fp32_cl, &m_args);
  
  struct exp_sum_args e_s_args;
  e_s_args.input = inData;
  e_s_args.sums = sums;
  e_s_args.output = outData;
  e_s_args.dim = dim;
  e_s_args.maxes = maxes;
  
  pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp32_cl, &e_s_args);


  struct row_div_args r_d_args;
  r_d_args.input = outData;
  r_d_args.sums = sums;
  r_d_args.dim = dim;

  pi_cl_team_fork(NUM_CORES, pulp_row_div_fp32_cl, &r_d_args);

  #ifdef DEBUG
    if(pi_core_id()==0){
        int L = dim;
        printf("\nCurrent softmax output: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%((int)L))) printf("\n");
            printf("%.8f ", outData[j]);
        }
    }
    printf("\n");
    #endif
}

/*
void pulp_softmax_fp32_fw_cl( void * act_args ){
  struct softmax_args * args = (struct softmax_args *) act_args;

  int dim = args->input->dim;
  int i, j;
  float* inData = args->input->data;
  float* outData = args->output->data;

  float* maxes = args->maxes;
  float* sums = args->sums;

  const int blockSize=(dim + NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  float* input = inData + start * dim;
  float* output = outData + start * dim;

  for(i=start; i<stop; i++){
    for(j=0; j<dim; j++){
      if(maxes[i] < *input || j==0)
        maxes[i] = *input;
      input++;    
    }
  }

  input = inData + start * dim;
  float o;

  for(i=start; i<stop; i++){
    sums[i] = 0;
    for(j=0; j<dim; j++){
      o = fastexp_gist(*input - maxes[i]);
      //float o = expf(*input - maxes[i]);
      *output = o;
      sums[i] += o;
      input++;
      output++;    
    }   
  }

  output = outData + start * dim;

  
  for(int i=start; i<stop; i++){
    for(int j=0; j<dim; j++){
      *output = *output/sums[i];
      output++;    
    }
  }
}
*/

void pulp_partial_softmax_simple_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int dim = args->input->dim; // L
  int dim2 = dim * dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  float* maxes = args->maxes;
  float* sums = args->sums;

  struct max_args m_args;
  m_args.input = inData;
  m_args.maxes = maxes;
  m_args.dim = dim;
  m_args.dim2 = dim2;

  pi_cl_team_fork(NUM_CORES, pulp_row_max_fp32_cl, &m_args);

  struct shift_sum_args ss_args;
  ss_args.input = inData;
  ss_args.sums = sums;
  ss_args.output = outData;
  ss_args.dim = dim;
  ss_args.dim2 = dim2;
  ss_args.maxes = maxes;

  pi_cl_team_fork(NUM_CORES, pulp_shift_sum_fp32_cl, &ss_args);


  struct row_div_args r_d_args;
  r_d_args.input = outData;
  r_d_args.sums = sums;
  r_d_args.dim = dim;
  r_d_args.dim2 = dim2;

  pi_cl_team_fork(NUM_CORES, pulp_row_div_fp32_cl, &r_d_args);
}

void pulp_softmax_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  int i = args->output->dim;
  float* inDiff = args->input->diff;
  float* outData = args->output->data;
  float* outDiff = args->output->diff;
  //float sum = 0.0f;

  for(int j = 0; j < dim; j++){ // Cycle over the elements of the i-th head buffer
      float sum = 0.0f;
      const float neg_sft_j  =  -(outData)[j]; 
      for(int z = 0; z < dim; ++z){ // Softmax involves all the elements of the i-th head buffer
          float mul =  (outDiff)[z] * (outData)[z] * neg_sft_j;
          sum +=  mul; // adding to the total sum of this row.
      }
      inDiff[j] = sum;
  }

  for(int j=0; j<dim; j++){
      inDiff[j] += (outData)[j] * (outDiff)[j]; // Gradient of pre-softmax head buffer: (L x L)
  }
}


void pulp_partial_softmax_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int L = args->L;
  int n_heads = args->n_heads;
  float * inData = args->input->data;
  float * outData = args->output->data;

  float * partial_exp_sum = args->partial_exp_sum + pi_core_id()*L;
  float * global_max = args->global_max + pi_core_id()*L;

  const float eps_max = 0.03125f;
  
  float zerox = 0.0f;
  float minfloat = -340282346638528859811704183484516925440.0f;

  const int blockSize=(n_heads+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > n_heads ? n_heads : start+blockSize;

  //Cycle over the heads
  for(int i = start; i < stop; i++){
    //STAGE 1: Calculate the denominator
    if(i != start){
      for(int l=0; l < L; l++){
        partial_exp_sum[l] = zerox;
        global_max[l] = minfloat;
      }
    }
    float* pointer = inData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float max_shift = zerox;
        if(global_max[j] < (*pointer)){
          max_shift = ((*pointer) - global_max[j]) * eps_max; 
          global_max[j] = *pointer;
        }
        float shift = (global_max[j] - (*pointer)) * eps_max;
        float exp_sum = 1.0f / powf(2.0f, shift); 
        partial_exp_sum[j] = (partial_exp_sum[j]) * (1.0f / powf(2.0f, max_shift)) + exp_sum;
        pointer++;
      }
    }
    //STAGE 2: Calculate activation value
    pointer = inData + i*L*L;
    float* out_pointer = outData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float shift = (global_max[j] - (*pointer)) * eps_max;
        (*out_pointer) = (1.0f / powf(2.0f, shift))/partial_exp_sum[j];
        pointer++;
        out_pointer++;
      }
    }
  }
}

void pulp_partial_softmax_shift_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int L = args->L;
  int n_heads = args->n_heads;
  float * inData = args->input->data;
  float * outData = args->output->data;

  float * partial_exp_sum = args->partial_exp_sum + pi_core_id()*L;
  float * global_max = args->global_max + pi_core_id()*L;

  float eps_max = 0.03125f;

  float zerox = 0.0f;
  float minfloat = -340282346638528859811704183484516925440.0f;

  const int blockSize=(n_heads+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > n_heads ? n_heads : start+blockSize;

  //Cycle over the heads
  for(int i = start; i < stop; i++){
    //STAGE 1: Calculate the denominator
    if(i != start){
      for(int l=0; l < L; l++){
        partial_exp_sum[l] = zerox;
        global_max[l] = minfloat;
      }
    }
    float* pointer = inData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float max_shift = zerox;
        if(global_max[j] < (*pointer)){
          max_shift = ((*pointer) - global_max[j]) * eps_max; 
          global_max[j] = *pointer;
        }
        float shift = (global_max[j] - (*pointer)) * eps_max;
        float exp_sum = 1  >> (int)(ceilf(shift)); 
        partial_exp_sum[j] = ((int)(partial_exp_sum[j]) >> (int)ceilf(max_shift)) + exp_sum;
        pointer++;
      }
    }
    //STAGE 2: Calculate activation value
    pointer = inData + i*L*L;
    float* out_pointer = outData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float shift = (global_max[j] - (*pointer)) * eps_max;
        (*out_pointer) = (1 >> (int) ceilf(shift))/partial_exp_sum[j];
        pointer++;
        out_pointer++;
      }
    }
  }
}

void pulp_partial_softmax_approximate_fp32_fw_cl(void * act_args){
  struct softmax_args * args = (struct softmax_args *) act_args;

  int L = args->L;
  int n_heads = args->n_heads;
  float * inData = args->input->data;
  float * outData = args->output->data;

  float * partial_exp_sum = args->partial_exp_sum + pi_core_id()*L;
  float * global_max = args->global_max + pi_core_id()*L;

  float eps_max = 0.03125f;

  float zero = 0.0f;
  float minfloat = -340282346638528859811704183484516925440.0f;

  const int blockSize=(n_heads+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > n_heads ? n_heads : start+blockSize;

  //Cycle over the heads
  for(int i = start; i < stop; i++){
    //STAGE 1: Calculate the denominator
    if(i != start){
      for(int l=0; l < L; l++){
        partial_exp_sum[l] = zero;
        global_max[l] = minfloat;
      }
    }
    float* pointer = inData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float max_shift = zero;
        if(global_max[j] < (*pointer)){
          max_shift = ((*pointer) - global_max[j]) * eps_max; 
          global_max[j] = *pointer;
        }
        float shift = (global_max[j] - (*pointer)) * eps_max;
        float exp_sum = threshold(shift); 
        partial_exp_sum[j] = (partial_exp_sum[j]) * (threshold(max_shift)) + exp_sum;
        pointer++;
      }
    }
    //STAGE 2: Calculate activation value
    pointer = inData + i*L*L;
    float* out_pointer = outData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float shift = (global_max[j] - (*pointer)) * eps_max;
        (*out_pointer) = (threshold(shift))/partial_exp_sum[j];
        pointer++;
        out_pointer++;
      }
    }
  }
}


void tanh_prll(void * args){

  struct tanh_args* args_tanh=(struct tanh_args *) args;

  const int blockSize=(args_tanh->dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > args_tanh->dim ? args_tanh->dim : start+blockSize;

  for(int i=start;i<stop;i++){
    args_tanh->output[i]=fasttanh(args_tanh->input[i]);
  }
}

static inline float
fastexp (float p)
{
  return fastpow2 (1.442695040f * p);
}

static inline float
fasttanh (float p)
{
  return -1.0f + 2.0f / (1.0f + fastexp (-2.0f * p));
}

static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { (uint32_t) ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

void pulp_vector_softmax_fp32(float* out, float* in, float* buffer_n_cores, unsigned int size){ 
  struct max_args ma;
  ma.input = in;
  ma.maxes = buffer_n_cores;
  ma.dim = size;

  pi_cl_team_fork(NUM_CORES, pulp_max_fp32_cl, &ma);

  float max = ma.maxes[0];

  for(int i=1;i<NUM_CORES; i++)
    if(ma.maxes[i] > max)
      max = ma.maxes[i];
  
  struct vector_exp_sum_args vesa;
  vesa.input = in;
  vesa.output = out;
  vesa.max = max;
  vesa.sums = buffer_n_cores;
  vesa.dim = size;
  
  pi_cl_team_fork(NUM_CORES, vector_exp_sum_fp32_cl, &vesa);

  float sum = 0;

  for(int i=0; i<NUM_CORES; i++)
    sum += vesa.sums[i];

  struct div_args da;
  da.input = out;
  da.n = sum;
  da.dim = size;

  pi_cl_team_fork(NUM_CORES, pulp_div_fp32_cl, &da); 
}


void pulp_swiglu_fp32_cl(void *swiglu_args){
  struct swiglu_args* args = (struct swiglu_args*) swiglu_args;
  float* in1 = args->in1;
  float* in2 = args->in2;
  float* out = args->out;
  int size = args->dim;

  const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > size ? size : start+blockSize;

  for(int i=start; i<stop; i++){
    float val = in1[i];

    #ifdef FASTEXPF
    val *= (1.0f / (1.0f + fastexp_gist(-val)));
    #else
    val *= (1.0f / (1.0f + expf(-val)));
    #endif

    val *= in2[i];
    out[i] = val;
  }
}