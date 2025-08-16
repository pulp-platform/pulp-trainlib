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

#include "pulp_train_defines.h"
#include "step-check.h"

// User profiling flags

//#define DEBUG

#if defined(FORWARD) && !defined(DEBUG) 
#define PROF_FWD
#endif

#if (defined(BACKWARD_ERROR) || defined(BACKWARD_GRAD) || defined(BACKWARD)) && !defined(DEBUG)
#define PROF_BCKWD
#endif

// Net sizes

#define Tker_l0     (Tin_l0*Tout_l0)

// Tensor checksum definition
#define CHECK_TOLERANCE 0x00000021
#define ERROR_TOLERANCE 0x00000001

// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

// Support functions
static inline void forward();
static inline void compare_tensors(fp16 *A, fp16 *B, int length);
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size);
static inline void train();
// Main function
void net_step ();

