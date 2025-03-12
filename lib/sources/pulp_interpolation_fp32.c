/*
 * Copyright (C) 2021-2024 ETH Zurich and University of Bologna
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

#include "math.h"
#include "pulp_interpolation_fp32.h"
#include "pulp_train_utils_fp32.h"



void pulp_nearest_neighbour_core_fp32 (void * interpolation_args) 
{
    struct interpolation_args * args = (struct interpolation_args *) interpolation_args;
    float * inData  = args->input->data;
    float * inDiff  = args->input->diff;
    int C_in        = args->input->C;
    int H_in        = args->input->H;
    int W_in        = args->input->W;
    float * outData = args->output->data;
    float * outDiff = args->output->diff;
    int C_out       = args->output->C;
    int H_out       = args->output->H;
    int W_out       = args->output->W;
    int intp_data   = args->interpolate_data;
    int intp_grad   = args->interpolate_gradient;

    // Internal parameters
    float H_scale = (float) H_out / H_in;
    float W_scale = (float) W_out / W_in;

    if (C_in != C_out) {
        printf("[pulp_nearest_neighbour_core_fp32] Input and output channels not matching!!\n");
        return;
    }

    // Interpolation core
    const int blockSize=(C_in+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > C_in ? C_in : start+blockSize;    

    if (intp_data == 1) {
        for (int c = start; c < stop; ++c) {
            // Iterate over output height
            for (int h = 0; h < H_out; ++h) {
                int nearest_h = (int)(h / H_scale);  // Find nearest height index in the input
                if (nearest_h >= H_in) nearest_h = H_in - 1;  // Boundary check

                // Iterate over output width
                for (int w = 0; w < W_out; ++w) {
                    int nearest_w = (int)(w / W_scale);  // Find nearest width index in the input
                    if (nearest_w >= W_in) nearest_w = W_in - 1;  // Boundary check

                    // Copy the value from the nearest neighbor in the input to the output
                    outData[c*H_out*W_out + h*W_out + w] = inData[c*H_in*W_in + nearest_h*W_in + nearest_w];
                }
            }
        } 
    }

    if (intp_grad == 1) {
        for (int c = start; c < stop; ++c) {
            // Iterate over output height
            for (int h = 0; h < H_out; ++h) {
                int nearest_h = (int)(h / H_scale);  // Find nearest height index in the input
                if (nearest_h >= H_in) nearest_h = H_in - 1;  // Boundary check

                // Iterate over output width
                for (int w = 0; w < W_out; ++w) {
                    int nearest_w = (int)(w / W_scale);  // Find nearest width index in the input
                    if (nearest_w >= W_in) nearest_w = W_in - 1;  // Boundary check

                    // Copy the value from the nearest neighbor in the input to the output
                    outDiff[c*H_out*W_out + h*W_out + w] = inDiff[c*H_in*W_in + nearest_h*W_in + nearest_w];
                }
            }
        }  
    }
}

void pulp_nearest_neighbour_interpolation_fp32_cl (void * interpolation_args) 
{
    struct interpolation_args * args = (struct interpolation_args *) interpolation_args;
    pi_cl_team_fork(NUM_CORES, pulp_nearest_neighbour_core_fp32, interpolation_args);
}



void pulp_bilinear_core_fp32 (void * interpolation_args) 
{
    struct interpolation_args * args = (struct interpolation_args *) interpolation_args;
    float * inData  = args->input->data;
    float * inDiff  = args->input->diff;
    int C_in        = args->input->C;
    int H_in        = args->input->H;
    int W_in        = args->input->W;
    float * outData = args->output->data;
    float * outDiff = args->output->diff;
    int C_out       = args->output->C;
    int H_out       = args->output->H;
    int W_out       = args->output->W;
    int intp_data   = args->interpolate_data;
    int intp_grad   = args->interpolate_gradient;

    // Internal parameters
    float H_scale = (float)(H_in) / (H_out);
    float W_scale = (float)(W_in) / (W_out);

    if (C_in != C_out) {
        printf("[pulp_nearest_neighbour_core_fp32] Input and output channels not matching!!\n");
        return;
    }

    // Interpolation core
    const int blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > C_in ? C_in : start + blockSize;

    if (intp_data == 1) {
        // Calculate the scale factors for height and width
        float scale_y = 1;
        if (H_in > 1) {scale_y = (float)(H_in) / (float)(H_out);}
        float scale_x = 1;
        if (W_in > 1) {scale_x = (float)(W_in) / (float)(W_out);}

        for (int c = start; c < stop; c++) {
            for (int y = 0; y < H_out; y++) {
                float in_y; 
                if (y == 0) {
                    // Special case for the first point
                    in_y = 0.0f;
                } else {
                    in_y = (y + 0.5f) * scale_y - 0.5f;
                }
                int top_index = (int)in_y;
                int bottom_index = top_index + 1;
                float y_weight = in_y - top_index;

                if (bottom_index >= H_in) {
                    bottom_index = H_in - 1;
                }

                for (int x = 0; x < W_out; x++) {
                    float in_x;
                    if (x == 0) {
                        // Special case for the first point
                        in_x = 0.0f;
                    } else {
                        in_x = (x + 0.5f) * scale_x - 0.5f;
                    }

                    int left_index = (int)in_x;
                    int right_index = left_index + 1;
                    float x_weight = in_x - left_index;

                    if (right_index >= W_in) {
                        right_index = W_in - 1;
                    }

                    float top_interp = inData[c*H_in*W_in + top_index * W_in + left_index] * (1.0f - x_weight)
                                    + inData[c*H_in*W_in + top_index * W_in + right_index] * x_weight;

                    float bottom_interp = inData[c*H_in*W_in + bottom_index * W_in + left_index] * (1.0f - x_weight)
                                        + inData[c*H_in*W_in + bottom_index * W_in + right_index] * x_weight;

                    outData[c*H_out*W_out + y * W_out + x] = top_interp * (1.0f - y_weight) + bottom_interp * y_weight;
                }
            }
        }
    }


    if (intp_grad == 1) {
        // Calculate the scale factors for height and width
        float scale_y = 1;
        if (H_in > 1) {scale_y = (float)(H_in) / (H_out);}
        float scale_x = 1;
        if (W_in > 1) {scale_x = (float)(W_in) / (W_out);}

        for (int c = start; c < stop; c++) {
            for (int y = 0; y < H_out; y++) {
                float in_y; 
                if (y == 0) {
                    // Special case for the first point
                    in_y = 0.0f;
                } else {
                    in_y = (y + 0.5f) * scale_y - 0.5f;
                }
                int top_index = (int)in_y;
                int bottom_index = top_index + 1;
                float y_weight = in_y - top_index;

                if (bottom_index >= H_in) {
                    bottom_index = H_in - 1;
                }

                for (int x = 0; x < W_out; x++) {
                    float in_x;
                    if (x == 0) {
                        // Special case for the first point
                        in_x = 0.0f;
                    } else {
                        in_x = (x + 0.5f) * scale_x - 0.5f;
                    }

                    int left_index = (int)in_x;
                    int right_index = left_index + 1;
                    float x_weight = in_x - left_index;

                    if (right_index >= W_in) {
                        right_index = W_in - 1;
                    }

                    float top_interp = inDiff[c*H_in*W_in + top_index * W_in + left_index] * (1.0f - x_weight)
                                    + inDiff[c*H_in*W_in + top_index * W_in + right_index] * x_weight;

                    float bottom_interp = inDiff[c*H_in*W_in + bottom_index * W_in + left_index] * (1.0f - x_weight)
                                        + inDiff[c*H_in*W_in + bottom_index * W_in + right_index] * x_weight;

                    outDiff[c*H_out*W_out + y * W_out + x] = top_interp * (1.0f - y_weight) + bottom_interp * y_weight;
                }
            }
        }
    }
}

void pulp_bilinear_interpolation_fp32_cl (void * interpolation_args) 
{
    struct interpolation_args * args = (struct interpolation_args *) interpolation_args;
    pi_cl_team_fork(NUM_CORES, pulp_bilinear_core_fp32, interpolation_args);
}
