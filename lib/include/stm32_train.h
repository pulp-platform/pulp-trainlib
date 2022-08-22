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
 * =====> GENERAL INCLUDES <=====
 */

// FP32 structures
#include "stm32_train_utils.h"
// FP32 primitives
#include "stm32_matmul.h"
#include "stm32_im2col.h"
#include "stm32_linear.h"
#include "stm32_conv2d.h"
#include "stm32_conv_pw.h"
#include "stm32_conv_dw.h"
#include "stm32_losses.h"
#include "stm32_act.h"
#include "stm32_pooling.h"
#include "stm32_optimizers.h"

