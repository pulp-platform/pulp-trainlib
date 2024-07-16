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
 * Authors: Davide Nadalini
*/ 


/**
 * SOURCE: https://github.com/ESultanik/mtwister?tab=readme-ov-file (Evan Sultanik)
*/ 
#ifndef __MTWISTER_H
#define __MTWISTER_H

#include <stdint.h>

#define STATE_VECTOR_LENGTH 16 //624
#define STATE_VECTOR_M      10 //397 /* changes to STATE_VECTOR_LENGTH also require changes to this */

typedef struct tagMTRand {
  uint32_t mt[STATE_VECTOR_LENGTH];
  int32_t index;
} MTRand;

MTRand   seedRand(uint32_t seed);
uint32_t genRandLong(MTRand* rand);
double   genRand(MTRand* rand);
float    genRandFloat(MTRand* rand);


/**
 * PULP-TrainLib's definitions
 */

/**
 * @brief Sets up a variable to generate a random integer number
 * @param seed seed from which to generate a pseudo-random number
 * @param probability probability of 0 or 1 (Bernoulli distribution). Specify a number in [0,1]
 * @param output output generated number
 */
struct integer_random_args{
  int seed;
  float probability;
  int* output;
};

/**
 * @brief Generates a pseudo-random number from a seed
 * @return pseudo-random float number
 */
float pulp_generate_float_seed(int seed);

/**
 * @brief Generates a scalar Bernoulli distribution
 * @param integer_random_args pointer to integer_random_args structure
 */
void pulp_random_bernoulli(void * integer_random_args);


#endif /* #ifndef __MTWISTER_H */
