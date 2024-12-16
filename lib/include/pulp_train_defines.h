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

#ifndef GLOBAL_TRAINLIB_DEFINES
#define GLOBAL_TRAINLIB_DEFINES

/**
 * =====> GLOBAL DEFINES <=====
 */

/**
 * @defgroup Data formats
 * @{
 */
typedef float16alt fp16;                                    // FP16 format (float16 or float16alt)
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

/**
 * Constants for Taylor's propagation of 1/2^x 
 */

#define LOG2    0.6931471805599453f
#define LOG2_2  0.4804530139182014f
#define LOG2_3  0.3330246519889294f
#define LOG2_4  0.2308350985830834f
#define LOG2_5  0.1600026977571413f
#define T1      1.0f
#define T2      0.5f
#define T3      0.16f
#define T4      0.0416f
#define T5      0.008f 

#define GIST_A  12102203.17133801f
#define GIST_B  1064986823.010288f
#define GIST_C  8388608
#define GIST_D  2139095040

#define GIST_A_fp16  184.665f
#define GIST_B_fp16  16.256f
#define GIST_C_fp16  128
#define GIST_D_fp16  32640



/**
 * Constants for Normalization Layers
*/

/**
 * @brief small number used to avoid division by zero 
 */ 
#define EPSILON 1e-10


#endif
    