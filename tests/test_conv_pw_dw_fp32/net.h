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

#include "step-check.h"

// User profiling flags

#if defined(DW_FORWARD) && !defined(DEBUG) 
#define PROF_DW_FWD
#endif

#if (defined(DW_BACKWARD_ERROR) || defined(DW_BACKWARD_GRAD)) && !defined(DEBUG)
#define PROF_DW_BKWD
#endif

#if defined(PW_FORWARD) && !defined(DEBUG)
#define PROF_PW_FWD
#endif

#if (defined(PW_BACKWARD_ERROR) || defined(PW_BACKWARD_GRAD)) && !defined(DEBUG)
#define PROF_PW_BKWD
#endif

// Net sizes

// DEPTHWISE CONV
#define stride_l1   (1)
#define stride_l1_steps (Tin_H_l1-Tker_H_l1+1)

// POINTWISE CONV
#define stride_l2   (1)
#define stride_l2_steps (Tin_H_l1-Tker_H_l1+1)

// Tensor checksum definition
#define CHECK_TOLERANCE 1e-7
#define ERROR_TOLERANCE 1e-7

// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

// Support functions
static inline void forward();
static inline void compare_tensors(float *A, float *B, int length);
int check_tensor(float * tensor_out, float * tensor_ref, int size);
static inline void train();
// Main function
void net_step ();
