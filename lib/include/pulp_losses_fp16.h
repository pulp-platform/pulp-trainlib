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

#include "pulp_train_defines.h"

/**
 * Loss functions configuration structure
 */

/**
 * @brief Structure to configure the loss functions
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target current sample's label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
struct loss_args_fp16 {
    struct blob_fp16 * output;
    fp16 * target;
    fp16 * wr_loss;
};

/**
 * @brief Structure to configure the berHu loss function
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target current sample's label
 * @param wr_loss variable to retrieve the value of the calculated loss
 * @param alpha alpha value of the berHu loss
 */
struct berHu_loss_args_fp16 {
    struct blob_fp16 * output;
    fp16 * target;
    fp16 * wr_loss;
    fp16 alpha;
};



/**
 * Loss functions
 */

/**
 * @brief Cross Entropy Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void pulp_CrossEntropyLoss_fp16( void * loss_args_fp16 );

/**
 * @brief Cross Entropy Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void pulp_CrossEntropyLoss_backward_fp16( void * loss_args_fp16 );

/**
 * @brief Mean Absolute Error Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void pulp_L1Loss_fp16( void * loss_args_fp16 );

/**
 * @brief Mean Absolute Error Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void pulp_L1Loss_backward_fp16( void * loss_args_fp16 );

/**
 * @brief Mean Squared Error Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void pulp_MSELoss_fp16( void * loss_args_fp16 );

/**
 * @brief Mean Squared Error Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 */
void pulp_MSELoss_backward_fp16( void * loss_args_fp16 );

/**
 * @brief berHu Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 * @param alpha alpha value of berHu loss
 */
void pulp_berHuLoss_fp16( void * berHu_loss_args_fp16 );

/**
 * @brief berHu Loss function 
 * @param output pointer to the blob structure of the last DNN's layer (loss computation + calculation of the output gradient)
 * @param target output label
 * @param wr_loss variable to retrieve the value of the calculated loss
 * @param alpha alpha value of berHu loss
 */
void pulp_berHuLoss_backward_fp16( void * berHu_loss_args_fp16 );
