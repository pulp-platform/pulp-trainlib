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

// SEPARABLE CONV
#define Tout_H_l1   (Tin_H_l1-Tker_H_l1+1)
#define Tout_W_l1   (Tin_W_l1-Tker_W_l1+1)
#define Tout_C_l1   (Tin_C_l1)
#define Tpad_l1     (0)
#define stride_l1   (1)
#define stride_l1_steps (Tin_H_l1-Tker_H_l1+1)

// POINTWISE CONV
#define Tin_H_l2    (Tin_H_l1-Tker_H_l1+1)
#define Tin_W_l2    (Tin_W_l1-Tker_W_l1+1)
#define Tin_C_l2    (Tout_C_l1)
#define Tker_H_l2   (1)
#define Tker_W_l2   (1)
#define Tout_H_l2   (Tin_H_l1-Tker_H_l1+1)
#define Tout_W_l2   (Tin_W_l1-Tker_W_l1+1)
#define Tpad_l2     (0)
#define stride_l2   (1)
#define stride_l2_steps (Tin_H_l1-Tker_H_l1+1)

// Tensor checksum definition
#define ABS(x) ((x)>0?(x):(-(x)))
#define CHECK_TOLERANCE 1e-3
#define ERROR_TOLERANCE 0.01

// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

void net_step ();
