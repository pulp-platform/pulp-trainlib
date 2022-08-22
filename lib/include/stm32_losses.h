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
 * Loss functions
 */

/**
 * @brief Standard Cross Entropy Loss function 
 * @param output pointer to output data
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void stm32_CrossEntropyLoss (
    struct blob * output, 
    float * target,
    float * wr_loss
);

/**
 * @brief Standard Mean Squared Error Loss function 
 * @param output pointer to output data
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void stm32_MSELoss (
    struct blob * output, 
    float * target, 
    float * wr_loss
);