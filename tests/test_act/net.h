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

// User profiling flags
#define FP32 32
#define FP16 16
// Tensor checksum definition
#if DATA_TYPE == FP32
    #define CHECK_TOLERANCE 1e-9
    #define ERROR_TOLERANCE 1e-9

    #define GELU_TANH_APPROX_CHECK_TOLERANCE 1e-4
    #define GELU_TANH_APPROX_ERROR_TOLERANCE 1e-4

    #define TANH_CHECK_TOLERANCE 1e-4
    #define TANH_ERROR_TOLERANCE 1e-4
#elif DATA_TYPE == FP16
    #define CHECK_TOLERANCE 1e-2
    #define ERROR_TOLERANCE 1e-2
#endif


// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

void net_step();
