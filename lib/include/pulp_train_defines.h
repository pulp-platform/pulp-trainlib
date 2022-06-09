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


/**
 * =====> GLOBAL DEFINES <=====
 */

/**
 * @defgroup Data formats
 * @{
 */
typedef float16alt fp16;                                    // Only bfloat16 present on VEGA
typedef fp16 v2f16 __attribute__((vector_size (4)));        // Vectorized fp16 for SIMD
/**
 * @}
 */

/**
 * @defgroup General macro 
 * @{
 */
#define ABS(x) ((x)>0?(x):(-(x)))
/**
 * @}
 */

/**
 * @defgroup Selects the kind of layer which is the target of "mm_manager" function.
 * @{
 */
#define LAYER_CONV2D 0
#define LAYER_DW_CONV 1
#define LAYER_PW_CONV 2
#define LAYER_LINEAR 3
/**
 * @}
 */

/**
 * @defgroup Selects the target step inside "mm_manager" function.
 * @{
 */
#define STEP_FW 0
#define STEP_WGT_GRAD 1
#define STEP_IN_GRAD 2
/**
 * @}
 */

    