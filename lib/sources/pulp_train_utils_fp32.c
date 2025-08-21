/*
 * Copyright (C) 2021-2025 ETH Zurich and University of Bologna
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

 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino, Calin Diaconu
*/


#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_matmul_fp32.h"
#include <math.h>

int verify_tensor(float *tensor_out, float *tensor_ref, int size, float tolerance) {
    int error_flag = 0;
    for (int i = 0; i < size; i++) {
        if (ABS(tensor_out[i] - tensor_ref[i]) > tolerance) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i,
                   tensor_ref[i], *(unsigned int *) &tensor_ref[i], tensor_out[i], *(unsigned int *) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}


void transpose_matrix(void *void_args) 
{
    struct transp_args args = *((struct transp_args *)void_args);
    float * matrix = args.in_matrix;
    float * transp_matrix = args.out_matrix;
    int N = args.N;
    int M = args.M;

    // Parallelize on N or M depending on the wides available dimension
    if (N > M) 
    {
        int blockSize = (N+NUM_CORES-1) / NUM_CORES;
        int start = pi_core_id()*blockSize;
        int stop = start+blockSize > N ? N : start+blockSize;

        for (int i=start; i<stop; i++)
        {
            for (int j=0; j<M; j++)
            {
                transp_matrix[j*N+i] = matrix[i*M+j];
            }
        }
    }
    else 
    {
        int blockSize = (M+NUM_CORES-1) / NUM_CORES;
        int start = pi_core_id()*blockSize;
        int stop = start+blockSize > M ? M : start+blockSize;

        for (int j=start; j<stop; j++)
        {
            for (int i=0; i<N; i++)
            {
                transp_matrix[j*N+i] = matrix[i*M+j];
            }
        }
    }    
}


void transpose(void *void_args) {
    // Extract passed arguments
    struct transp_args args = *((struct transp_args *) void_args);

    float *in_matrix = args.in_matrix;
    float *out_matrix = args.out_matrix;

    int *dim = args.dim;
    int *transposed_axes = args.transposed_axes;

    int n_dim = args.n_dim;

    // Get total number of elements
    int n_elements = 1;
    for (int i = 0; i < n_dim; i++) {
        n_elements *= dim[i];
    }

    // Find indexes transposed dimensions
    int tr_indexes[n_dim];
    for (int i = 0; i < n_dim; i++) {
        tr_indexes[transposed_axes[i]] = i;
    }

    // Prepare look-up table for new index computation
    int prod[n_dim];
    prod[n_dim - 1] = 1;
    for (int i = n_dim - 2; i >= 0; i--) {
        prod[i] = prod[i + 1] * dim[transposed_axes[i + 1]];
    }

    // Store last dimension for step size
    int step_size = dim[n_dim - 1];

    // Compute first n - 1 dimensions
    int dim_n_1 = n_elements / dim[n_dim - 1];

    // Prepare new index variables
    int idx;
    int new_index = 0;
    int new_index_increment = prod[tr_indexes[n_dim - 1]];

    // Share work among cores "line"-wise, where a line comprises the first n - 1 dimensions
    if (step_size < (1.15 * dim_n_1)) {
        int blockSize = (dim_n_1 + NUM_CORES - 1) / NUM_CORES;
        int start = pi_core_id() * blockSize * step_size;
        int stop = start + (blockSize * step_size) > n_elements ? n_elements : start + (blockSize * step_size);

        // Iterate through elements
        for (int i = start; i < stop; i += step_size) {
            // Store original index and prepare new index
            idx = i;
            new_index = 0;

            // Iterate through axes to compute first new index
            for (int j = (n_dim - 1); j >= 0; j--) {
                // Update new index
                new_index += idx % dim[j] * prod[tr_indexes[j]];

                // Get remainder of original index
                idx /= dim[j];
            }

            // Iterate through elements in line
            for (int j = i; j < i + step_size; j++) {
                // Store element in new position
                out_matrix[new_index] = in_matrix[j];

                // Increment new index
                new_index += new_index_increment;
            }
        }
    }
        // Share work among cores "column"-wise
    else {
        int blockSize = (step_size + NUM_CORES - 1) / NUM_CORES;
        int start = pi_core_id() * blockSize;
        int stop = start + blockSize > step_size ? step_size : start + blockSize;

        // Iterate through elements
        for (int i = 0; i < n_elements; i += step_size) {
            // Store original index and prepare new index
            idx = i + start;
            new_index = 0;

            // Iterate through axes to compute first new index
            for (int j = (n_dim - 1); j >= 0; j--) {
                // Update new index
                new_index += idx % dim[j] * prod[tr_indexes[j]];

                // Get remainder of original index
                idx /= dim[j];
            }

            // Iterate through elements in line
            for (int j = i + start; j < i + stop; j++) {
                // Store element in new position
                out_matrix[new_index] = in_matrix[j];

                // Increment new index
                new_index += new_index_increment;
            }
        }
    }
}


void copy(void *void_args) {
    struct copy_args args = *((struct copy_args *) void_args);
    int blockSize = (args.size + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > args.size ? args.size : start + blockSize;

    for (int i = start; i < stop; i++)
        args.to[i] = args.from[i];
}


void set_to_value(void *void_args) {
    struct set_to_value_args args = *((struct set_to_value_args *) void_args);
    int blockSize = (args.size + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > args.size ? args.size : start + blockSize;

    for (int i = start; i < stop; i++)
        args.to[i] = args.value;
}


void vect_sum(void *vect_sum_args) {
    struct vect_sum_args *args = (struct vect_sum_args *) vect_sum_args;
    float *op_1 = args->op_1;
    float *op_2 = args->op_2;
    float *dest = args->dest;
    int size = args->size;

    int blockSize = (size + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > size ? size : start + blockSize;

    for (int i = start; i < stop; i++) {
        dest[i] = op_1[i] + op_2[i];
    }
}


void array_broadcast_sum_fp32(void *arr_bc_args) {
    // Extract passed arguments
    struct array_broadcast_sum_fp32_args *args = (struct array_broadcast_sum_fp32_args *) arr_bc_args;

    float *op_1 = args->op_1;
    float *op_2 = args->op_2;
    float *dest = args->dest;

    int *op_1_dims = args->op_1_dims;
    int *op_2_dims = args->op_2_dims;

    int op_1_dims_len = args->op_1_dims_len;
    int op_2_dims_len = args->op_2_dims_len;

    // Compute output dimensions size
    int out_dims_len, min_dims_len;
    int output_size = 1;
    if (op_1_dims_len > op_2_dims_len) {
        out_dims_len = op_1_dims_len;
        min_dims_len = op_2_dims_len;
    } else {
        min_dims_len = op_1_dims_len;
        out_dims_len = op_2_dims_len;
    }

    // Compute output dimensions
    int out_dims[out_dims_len];
    for (int i = 0; i < min_dims_len; i++) {
        int out_dims_idx = out_dims_len - i - 1;
        int op_1_dims_idx = op_1_dims_len - i - 1;
        int op_2_dims_idx = op_2_dims_len - i - 1;

        if (op_1_dims[op_1_dims_idx] > op_2_dims[op_2_dims_idx])
            out_dims[out_dims_idx] = op_1_dims[op_1_dims_idx];
        else
            out_dims[out_dims_idx] = op_2_dims[op_2_dims_idx];

        output_size *= out_dims[out_dims_idx];
    }

    if (op_1_dims_len > op_2_dims_len) {
        for (int i = 0; i < (out_dims_len - min_dims_len); i++) {
            out_dims[i] = op_1_dims[i];
            output_size *= out_dims[i];
        }
    } else {
        for (int i = 0; i < (out_dims_len - min_dims_len); i++) {
            out_dims[i] = op_2_dims[i];
            output_size *= out_dims[i];
        }
    }

    // Prepare look-up tables for new indexes computation
    int prod_1[op_1_dims_len];
    int prod_so_far = 1;

    if (op_1_dims[op_1_dims_len - 1] == 1)
        prod_1[op_1_dims_len - 1] = 0;
    else
        prod_1[op_1_dims_len - 1] = 1;

    for (int i = op_1_dims_len - 2; i >= 0; i--) {
        prod_so_far *= op_1_dims[i + 1];

        if (op_1_dims[i] == 1)
            prod_1[i] = 0;
        else
            prod_1[i] = prod_so_far;
    }

    int prod_2[op_2_dims_len];

    prod_so_far = 1;

    if (op_2_dims[op_2_dims_len - 1] == 1)
        prod_2[op_2_dims_len - 1] = 0;
    else
        prod_2[op_2_dims_len - 1] = 1;

    for (int i = op_2_dims_len - 2; i >= 0; i--) {
        prod_so_far *= op_2_dims[i + 1];

        if (op_2_dims[i] == 1)
            prod_2[i] = 0;
        else
            prod_2[i] = prod_so_far;
    }

    int last_dim = out_dims[out_dims_len - 1];
    int first_n_1_dims = output_size / last_dim;

    // Split "lines" among cores
    if (1.1 * first_n_1_dims > last_dim) {
        // Compute core split
        int blockSize = ((first_n_1_dims + NUM_CORES - 1) / NUM_CORES) * last_dim;
        int start = pi_core_id() * blockSize;
        int stop = start + blockSize > output_size ? output_size : start + blockSize;

        // Compute output
        if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2 + j];
                }
            }

        } else if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] == 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2];
                }
            }

        } else if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] == 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2 + j];
                }
            }

        } else if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] == 1) &&
                (op_2_dims[op_2_dims_len - 1] == 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2];
                }
            }

        } else if (
                (op_1_dims_len <= op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2 + j];
                }
            }

        } else if (
                (op_1_dims_len <= op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] == 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2];
                }
            }

        } else if (
                (op_1_dims_len <= op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] == 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2 + j];
                }
            }

        } else {

            for (int i = start; i < stop; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = 0; j < last_dim; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2];
                }
            }
        }
    }
        // Split "columns" among cores
    else {
        // Compute core split
        int blockSize = (last_dim + NUM_CORES - 1) / NUM_CORES;
        int start = pi_core_id() * blockSize;
        int stop = start + blockSize > last_dim ? last_dim : start + blockSize;

        // Compute output
        if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2 + j];
                }
            }

        } else if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] == 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2];
                }
            }

        } else if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] == 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2 + j];
                }
            }

        } else if (
                (op_1_dims_len > op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] == 1) &&
                (op_2_dims[op_2_dims_len - 1] == 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2];
                }
            }

        } else if (
                (op_1_dims_len <= op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2 + j];
                }
            }

        } else if (
                (op_1_dims_len <= op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] > 1) &&
                (op_2_dims[op_2_dims_len - 1] == 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1 + j] + op_2[idx_2];
                }
            }

        } else if (
                (op_1_dims_len <= op_2_dims_len) &&
                (op_1_dims[op_1_dims_len - 1] == 1) &&
                (op_2_dims[op_2_dims_len - 1] > 1)
                ) {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2 + j];
                }
            }

        } else {

            for (int i = 0; i < output_size; i += last_dim) {
                // Compute inputs ids
                int idx_1 = 0;
                int idx_2 = 0;
                int idx = i;

                for (int j = 0; j < min_dims_len; j++) {
                    idx_1 += (idx % out_dims[out_dims_len - j - 1]) * prod_1[op_1_dims_len - j - 1];
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];

                    idx /= out_dims[out_dims_len - j - 1];
                }

                for (int j = min_dims_len; j < out_dims_len; j++) {
                    idx_2 += (idx % out_dims[out_dims_len - j - 1]) * prod_2[op_2_dims_len - j - 1];
                    idx /= out_dims[out_dims_len - j - 1];
                }

                // Update output
                for (int j = start; j < stop; j++) {
                    dest[i + j] = op_1[idx_1] + op_2[idx_2];
                }
            }
        }
    }
}


void cast_fp16_tensor_to_fp32(void *cast_16t32_args) {
    struct cast_16t32_args args = *((struct cast_16t32_args *) cast_16t32_args);
    int blockSize = (args.size + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > args.size ? args.size : start + blockSize;

    for (int i = start; i < stop; i++) {
        args.destination[i] = (float) args.source[i];
    }
}


void HWC_to_CHW(void *layout_args) {
    struct layout_args *args = (struct layout_args *) layout_args;
    float *data = args->tensor->data;
    float *grad = args->tensor->diff;
    uint16_t C = args->tensor->C;
    uint16_t H = args->tensor->H;
    uint16_t W = args->tensor->W;
    float *buff = args->transp_buffer;
    uint8_t transpose_data = args->transpose_data;
    uint8_t transpose_grad = args->transpose_grad;

    struct transp_args tr_args;
    struct copy_args cpy_args;

    if (transpose_data == 1) {
        // Transpose data
        int dims[] = {H * W, C};
        int t_axes[] = {1, 0};

        tr_args.in_matrix = data;
        tr_args.out_matrix = buff;
        tr_args.dim = dims;
        tr_args.transposed_axes = t_axes;
        tr_args.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);

        cpy_args.from = buff;
        cpy_args.to = data;
        cpy_args.size = C * H * W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    }

    if (transpose_grad == 1) {
        // Transpose grad
        int dims[] = {H * W, C};
        int t_axes[] = {1, 0};

        tr_args.in_matrix = grad;
        tr_args.out_matrix = buff;
        tr_args.dim = dims;
        tr_args.transposed_axes = t_axes;
        tr_args.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);

        cpy_args.from = buff;
        cpy_args.to = grad;
        cpy_args.size = C * H * W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    }
}


void CHW_to_HWC(void *layout_args) {
    struct layout_args *args = (struct layout_args *) layout_args;
    float *data = args->tensor->data;
    float *grad = args->tensor->diff;
    uint16_t C = args->tensor->C;
    uint16_t H = args->tensor->H;
    uint16_t W = args->tensor->W;
    float *buff = args->transp_buffer;
    uint8_t transpose_data = args->transpose_data;
    uint8_t transpose_grad = args->transpose_grad;

    struct transp_args tr_args;
    struct copy_args cpy_args;

    if (transpose_data == 1) {
        // Transpose data
        int dims[] = {C, H * W};
        int t_axes[] = {1, 0};

        tr_args.in_matrix = data;
        tr_args.out_matrix = buff;
        tr_args.dim = dims;
        tr_args.transposed_axes = t_axes;
        tr_args.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);

        cpy_args.from = buff;
        cpy_args.to = data;
        cpy_args.size = C * H * W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    }

    if (transpose_grad == 1) {
        // Transpose grad
        int dims[] = {C, H * W};
        int t_axes[] = {1, 0};

        tr_args.in_matrix = grad;
        tr_args.out_matrix = buff;
        tr_args.dim = dims;
        tr_args.transposed_axes = t_axes;
        tr_args.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);

        cpy_args.from = buff;
        cpy_args.to = grad;
        cpy_args.size = C * H * W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    }
}


void pad_tensor(void *pad_args) {
    struct pad_args *args = (struct pad_args *) pad_args;
    float *source = args->source;
    float *dest = args->dest;
    int C = args->C;
    int H = args->H;
    int W = args->W;
    int L_PAD = args->T_LPAD;
    int R_PAD = args->T_RPAD;
    int U_PAD = args->T_UPAD;
    int D_PAD = args->T_DPAD;
    int HWC = args->HWC_lay;

    int H_out = H + U_PAD + D_PAD;
    int W_out = W + L_PAD + R_PAD;

    int blockSize = (C + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > C ? C : start + blockSize;

    if (HWC == 0) {
        for (int ch = 0; ch < C; ch++) {
            for (int ht = 0; ht < H_out; ht++) {
                for (int wt = 0; wt < W_out; wt++) {
                    // Compute matrix idx
                    int in_t_idx = (wt - L_PAD) + (ht - U_PAD) * W + ch * H * W;
                    int out_t_idx = wt + ht * W_out + ch * H_out * W_out;
                    // Padding conditions
                    int zero_cond = (wt < L_PAD || wt > W) || (ht < U_PAD || ht > H);
                    if (zero_cond == 1) { dest[out_t_idx] = 0; }
                    else {
                        dest[out_t_idx] = source[in_t_idx];
                    }
                }
            }
        }
    } else if (HWC == 1) {
        for (int ht = 0; ht < H_out; ht++) {
            for (int wt = 0; wt < W_out; wt++) {
                for (int ch = 0; ch < C; ch++) {
                    // Compute matrix idx
                    int in_t_idx = ch + (wt - L_PAD) * C + (ht - U_PAD) * C * W;
                    int out_t_idx = ch + wt * C + ht * C * W_out;
                    // Padding conditions
                    int zero_cond = (wt < L_PAD || wt > W) || (ht < U_PAD || ht > H);
                    if (zero_cond == 1) { dest[out_t_idx] = 0; }
                    else {
                        dest[out_t_idx] = source[in_t_idx];
                    }
                }
            }
        }
    } else {
        printf("[pad_tensor] HWC layout not implemented!!");
    }
}


void pulp_max_fp32_cl(void *void_args) {
    struct max_args *args = (struct max_args *) void_args;

    float *input = args->input;

    float max;
    int WIDTH = args->W;

    const int blockSize = (WIDTH + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > WIDTH ? WIDTH : start + blockSize;

    max = input[start];

    for (int i = start; i < stop; i++)
        if (max < input[i])
            max = input[i];

    args->maxes[pi_core_id()] = max;
}


float threshold(float x) {
    /*
    float log2 = 0.6931471805599453f;
    float log2_2 = 0.4804530139182014f;
    float log2_3 = 0.3330246519889294f;
    float log2_4 = 0.2308350985830834f;
    float log2_5 = 0.1600026977571413f;
    */

    if (x >= 3.14f)
        return 0.0f;

    float x_2 = x * x;
    float x_3 = x * x * x;
    float x_4 = x * x * x * x;
    float x_5 = x * x * x * x * x;

    return (T1 - LOG2 * x + T2 * x_2 * LOG2_2 - T3 * x_3 * LOG2_3 + T4 * x_4 * LOG2_4 - T5 * x_5 * LOG2_5);
}


int clamp_int(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}


// ~~~~~~~~~~~~~~~~~~ SOFTMAX FUNCTIONS ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~      FORWARD      ~~~~~~~~~~~~~~~~~~
// Find the maximum value from each row of the passed matrix
void pulp_row_max_fp32_cl(void *void_args) {
    // Extract variables from function arguments
    struct max_args *args = (struct max_args *) void_args;

    float *input = args->input;
    int HEIGHT = args->H;
    int WIDTH = args->W;
    int i, j;
    float *max = args->maxes;

    // Split work row-wise (each worker will receive a number of rows)
    const int blockSize = (HEIGHT + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > HEIGHT ? HEIGHT : start + blockSize;

    // Set pointer at start position
    input = input + start * WIDTH;

    // Iterate through allocated rows
    for (i = start; i < stop; i++) {
        // Iterate through the rest of the elements in the row and keep the maximum up to date
        for (j = 0; j < WIDTH; j++) {
            if (max[i] < *input)
                max[i] = *input;
            input++;
        }
    }
}


// Row-wisely compute the sum of exponentials required for the softmax activation
void pulp_exp_sum_fp32_cl(void *void_args) {
    // Extract variable from function arguments
    struct exp_sum_args *args = (struct exp_sum_args *) void_args;

    float *input = args->input;
    float *output = args->output;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    float *sums = args->sums;
    float *maxes = args->maxes;

    // Split work row-wise (each worker will receive a number of rows)
    const int blockSize = (HEIGHT + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > HEIGHT ? HEIGHT : start + blockSize;

    // Iterate through allocated rows
    for (int i = start; i < stop; i++) {
        // Iterate through each element and update the sum accordingly
        for (int j = 0; j < WIDTH; j++) {
            float o = fastexp_gist(input[i * WIDTH + j] - maxes[i]);
            //float o = expf(input[i * WIDTH + j] - maxes[i]);

            output[i * WIDTH + j] = o;
            sums[i] += o;
        }
    }
}


// Divide each element in a row with a value given in a sums array, used in the softmax activation
void pulp_row_div_fp32_cl(void *void_args) {
    // Extract variable from function arguments
    struct row_div_args *args = (struct row_div_args *) void_args;

    float *input = args->input;
    float *sums = args->sums;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    // Split work row-wise (each worker will receive a number of rows)
    const int blockSize = (HEIGHT + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > HEIGHT ? HEIGHT : start + blockSize;

    // For each element in a row, divide with the corresponding precomputed sum
    for (int i = start; i < stop; i++) {
        int row = i * WIDTH;
        for (int j = 0; j < WIDTH; j++) {
            input[row + j] = input[row + j] / sums[i];
        }
    }
}


// ~~~~~~~~~~~~~~~~~~      BACKWARD     ~~~~~~~~~~~~~~~~~~
void pulp_sm_bw_op_1(void *void_args) {
    // Extract variable from function arguments
    struct sm_bw_op_1_args *args = (struct sm_bw_op_1_args *) void_args;

    float *input_A = args->A;
    float *input_B = args->B;
    float *output_S = args->S;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    // Split work row-wise (each worker will receive a number of rows)
    const int blockSize = (HEIGHT + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > HEIGHT ? HEIGHT : start + blockSize;

    // For each row, compute the sum of the element-wise products of matrices A and B into array S
    for (int i = start; i < stop; i++) {
        int row = i * WIDTH;
        output_S[i] = 0;

        for (int j = 0; j < WIDTH; j++) {
            output_S[i] += (input_A[row + j] * input_B[row + j]);
        }
    }
}


void pulp_sm_bw_op_2(void *void_args) {
    // Extract variable from function arguments
    struct sm_bw_op_2_args *args = (struct sm_bw_op_2_args *) void_args;

    float *input_A = args->A;
    float *input_B = args->B;
    float *sums = args->S;
    float *result = args->output;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    // Split work row-wise (each worker will receive a number of rows)
    const int blockSize = (HEIGHT + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > HEIGHT ? HEIGHT : start + blockSize;

    // For each row, do the necessary computation
    for (int i = start; i < stop; i++) {
        int row = i * WIDTH;

        for (int j = 0; j < WIDTH; j++) {
            result[row + j] = (input_A[row + j] - sums[i]) * input_B[row + j];
        }
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void pulp_shift_sum_fp32_cl(void *void_args) {
    struct shift_sum_args *args = (struct shift_sum_args *) void_args;

    float *input = args->input;
    float *output = args->output;
    float *sums = args->sums;
    int dim = args->dim;
    float *maxes = args->maxes;

    const int blockSize = (dim + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > dim ? dim : start + blockSize;

    int row = 0;

    for (int i = start; i < stop; i++) {
        sums[i] = 0;
        row = i * dim;
        for (int j = 0; j < dim; j++) {
            float o = threshold(maxes[i] - input[row + j]);
            /*float o = 1.0f - 0.5f *(maxes[i] - input[row + j]);
            if(o < 0.0f)
                o = 0.0f;*/
            output[row + j] = o;
            sums[i] += o;
        }
    }
}


float fastexp_gist(float x) {
    x = GIST_A * x + GIST_B;

    if (x < GIST_C || x > GIST_D)
        x = (x < GIST_C) ? 0.0f : GIST_D;

    uint32_t n = (uint32_t)(x);
    return *(float *) &n;
}


float q_rsqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5f;

    x2 = number * 0.5f;
    y = number;
    i = *(long *) &y;                       // evil floating point bit level hacking
    i = 0x5f3759df - (i >> 1);               // what the fuck?
    y = *(float *) &i;
    y = y * (threehalfs - (x2 * y * y));   // 1st iteration

    return y;
}


void pulp_div_fp32_cl(void *void_args) {
    struct div_args *args = (struct div_args *) void_args;

    float *input = args->input;
    float n = args->n;
    int dim = args->dim;

    const int blockSize = (dim + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > dim ? dim : start + blockSize;

    for (int i = start; i < stop; i++) {
        input[i] = input[i] / n;
    }
}


void pulp_scalar_mul_fp32_cl(void *void_args) {
    struct scalar_mul_args *args = (struct scalar_mul_args *) void_args;

    float *input = args->input;
    float scalar = args->scalar;
    int dim = args->dim;

    const int blockSize = (dim + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > dim ? dim : start + blockSize;

    for (int i = start; i < stop; i++) {
        input[i] = input[i] * scalar;
    }
}

/**
 * Bias addition.
 */
void mm_bias_add_transposed(void *void_args) {
    // Extract variable from function arguments
    struct mm_bias_add_args *args = (struct mm_bias_add_args *) void_args;

    int t = args->t;
    float *mat = args->mat;
    float *bias = args->bias;
    float temp;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    // Split work row-wise (each worker will receive a number of rows)
    const int blockSize = (HEIGHT + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start+blockSize > HEIGHT ? HEIGHT : start+blockSize;

    if(t == 0){
        // For each row, sum it with the bias
        for (int i = start; i < stop; i++) {
            int row = i * WIDTH;
            for (int j = 0; j < WIDTH; j++) {
                temp = mat[row + j]+bias[j];
                mat[row + j] = temp;
            }
        }
    }
    else{
       // For each column, sum it with the bias
       for (int i = start; i < stop; i++){
        int row = i*WIDTH;
        for(int j = 0; j<WIDTH; j++){
            temp = mat[row+j] + bias[i];
            mat[row+j] = temp;
        }
       } 
    }
    
}


/**
 * Choose the user-selected matmul for the chosen layer.
 */
void mm_manager(void *void_args) {
    struct mm_manager_args *args = (struct mm_manager_args *) void_args;

    struct matMul_args *matMul_args = args->mm_args;
    struct matMul_DW_args *matMul_DW_args = args->mm_dw_args;
    int layer_type = args->layer_type;
    int step_type = args->step_type;
    int matmul_type = args->matmul_type;
    int use_bias = args->mm_args->USE_BIASES;


#ifdef DEBUG
    printf("Running layer %d, step %d, matmul %d\n", layer_type, step_type, matmul_type);
#endif

    // =====> CONV2D
    if (layer_type == LAYER_CONV2D) {
        // Select step type
        if (step_type == STEP_FW) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else if (step_type == STEP_WGT_GRAD) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else if (step_type == STEP_IN_GRAD) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
    }
        // =====> POINTWISE CONVOLUTION
    else if (layer_type == LAYER_PW_CONV) {
        // Select step type
        if (step_type == STEP_FW) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else if (step_type == STEP_WGT_GRAD) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else if (step_type == STEP_IN_GRAD) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
    }
        // =====> LINEAR LAYER
    else if (layer_type == LAYER_LINEAR) {
        // Select step type
        if (step_type == STEP_FW) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else if (step_type == STEP_WGT_GRAD) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else if (step_type == STEP_IN_GRAD) {
            // Select matmul type
            // Naive
            if (matmul_type == 0) { mm((void *) matMul_args); }
            else if (matmul_type == 1) { mm_M((void *) matMul_args); }
                // Parallelism on N
            else if (matmul_type == 2) { mm_u2((void *) matMul_args); }
            else if (matmul_type == 3) { mm_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 4) { mm_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 5) { mm_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 6) { mm_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 7) { mm_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 8) { mm_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 9) { mm_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 10) { mm_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 11) { mm_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 12) { mm_unroll_4x4((void *) matMul_args); }
                // Parallelism on M
            else if (matmul_type == 13) { mm_M_u2((void *) matMul_args); }
            else if (matmul_type == 14) { mm_M_unroll_1x2((void *) matMul_args); }
            else if (matmul_type == 15) { mm_M_unroll_1x4((void *) matMul_args); }
            else if (matmul_type == 16) { mm_M_unroll_1x8((void *) matMul_args); }
            else if (matmul_type == 17) { mm_M_unroll_2x1((void *) matMul_args); }
            else if (matmul_type == 18) { mm_M_unroll_4x1((void *) matMul_args); }
            else if (matmul_type == 19) { mm_M_unroll_8x1((void *) matMul_args); }
            else if (matmul_type == 20) { mm_M_unroll_2x2((void *) matMul_args); }
            else if (matmul_type == 21) { mm_M_unroll_2x4((void *) matMul_args); }
            else if (matmul_type == 22) { mm_M_unroll_4x2((void *) matMul_args); }
            else if (matmul_type == 23) { mm_M_unroll_4x4((void *) matMul_args); }
            else {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        } else {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
    }
        // =====> WRONG LAYER SELECTION
    else {
        printf("\nWrong layer_type selection!!\n");
    }

    if(use_bias==1){
        // Bias_addition
        pi_cl_team_barrier();
        struct mm_bias_add_args mm_bias_add_args_q;
        mm_bias_add_args_q.mat = args->mm_args->C;
        mm_bias_add_args_q.bias = args->mm_args->bias;
        mm_bias_add_args_q.H = args->mm_args->N;
        mm_bias_add_args_q.W = args->mm_args->M;
        mm_bias_add_args_q.t = args->mm_args->bias_transposed;

        mm_bias_add_transposed((void *) &mm_bias_add_args_q);
    }
}

void pulp_mean_std_fp32_cl(void *mean_std_args) {
    struct mean_std_args *args = (struct mean_std_args *) mean_std_args;

    float *data = args->input;
    int D = args->dim;
    float D_inverse = (1 / (float) D);
    float *mean = args->mean;
    float *std = args->std;
    float *var = args->var;
    float epsilon = args->epsilon;

    float m = 0;
    float v = 0;
    float s = 0;

    int var_was_infinite = 0;

    for (int d = 0; d < D; d++) {
        float t = data[d];
        m += t;
        v += t * t;
    }
    m = m * D_inverse;
    v = v * D_inverse;

    // Test for infinite variance
    if (*(int *) &v == 0x7f80000) {
        var_was_infinite = 1;
        v = 0;
        for (int d = 0; d < D; d++) {
            float t = data[d];
            float temp = t - m;
            v += temp * temp * D_inverse;
        }
    }

    if (!var_was_infinite) v -= m * m;
    v = v + epsilon;
    if ((v) < 0) v = epsilon;
    *mean = m;
    *var = v;
    *std = sqrtf(v);
}


void vector_exp_sum_fp32_cl(void *vector_exp_sum_args) {
    struct vector_exp_sum_args *args = (struct vector_exp_sum_args *) vector_exp_sum_args;

    float *input = args->input;
    float *output = args->output;
    float *sums = args->sums;
    int dim = args->dim;
    float max = args->max;

    int id = pi_core_id();

    const int blockSize = (dim + NUM_CORES - 1) / NUM_CORES;
    const int start = id * blockSize;
    const int stop = start + blockSize > dim ? dim : start + blockSize;

    sums[id] = 0;

    for (int i = start; i < stop; i++) {
#ifdef FASTEXPF
        float o = fastexp_gist(input[i] - max);
#else
        float o = expf(input[i] - max);
#endif
        output[i] = o;
        sums[id] += o;
    }
}


#define CORDIC_N_ITERATION 12
#define CORDIC_SCALING_FACTOR_14 0.6072529365170104
#define CORDIC_SCALING_FACTOR_12 0.607252959138945
#define CORDIC_SCALING_FACTOR_10 0.6072533210898753
#define CORDIC_SCALING_FACTOR_8 0.6072591122988928


const float atan_pow_2[14] = {
        0.7853981633974483f,
        0.4636476090008061f,
        0.24497866312686414f,
        0.12435499454676144f,
        0.06241880999595735f,
        0.031239833430268277f,
        0.015623728620476831f,
        0.007812341060101111f,
        0.0039062301319669718f,
        0.0019531225164788188f,
        0.0009765621895593195f,
        0.0004882812111948983f,
        0.00024414062014936177f,
        0.00012207031189367021f};


void cordic_cos_sin_fp32(float angle, float *cos, float *sin) {
    int inv_tan_theta = 1;
    float x = CORDIC_SCALING_FACTOR_12;
    float y = 0;
    float x_n;
    int cos_sign = 1;

    angle -= ((int) (angle / (2 * M_PI))) * (2 * M_PI);

    if (angle > M_PI)
        angle -= 2 * M_PI;
    else if (angle < -M_PI)
        angle += 2 * M_PI;

    if (angle > M_PI_2) {
        angle = M_PI - angle;
        cos_sign = -1;
    } else if (angle < -M_PI_2) {
        angle = -M_PI - angle;
        cos_sign = -1;
    }

    for (int i = 0; i < CORDIC_N_ITERATION; i++) {
        x_n = x;
        if (angle > 0) {
            x -= y / inv_tan_theta;
            y += x_n / inv_tan_theta;
            angle -= atan_pow_2[i];
        } else {
            x += y / inv_tan_theta;
            y -= x_n / inv_tan_theta;
            angle += atan_pow_2[i];
        }
        inv_tan_theta <<= 1;
    }
    *cos = cos_sign * x;
    *sin = y;
}

void reduce_mean_fp32(void *void_args) {
    // Extract args
    struct reduce_mean_args_fp32 *args = (struct reduce_mean_args_fp32 *) void_args;

    float *input = args->input;
    float *output = args->output;

    int *dims = args->dims;
    int dims_len = args->dims_len;

    int reduce_axis = args->reduce_axis;

    // Compute necessary variables
    int dim_to_reduce = dims[reduce_axis];

    int axis_to_keep = 1;
    for (int i = 0; i < dims_len; i++) {
        if (i != reduce_axis) {
            axis_to_keep *= dims[i];
        }
    }

    int step = 1;
    for (int i = reduce_axis + 1; i < dims_len; i++) {
        step *= dims[i];
    }

    // Prepare work split variables
    int id = pi_core_id();
    int blockSize, start, stop;

    if (dim_to_reduce > axis_to_keep) {
        // Split summing work
        blockSize = (dim_to_reduce + NUM_CORES - 1) / NUM_CORES;
        start = id * blockSize;
        stop = start + blockSize > dim_to_reduce ? dim_to_reduce : start + blockSize;

        // Compute output sum
        for (int i = 0; i < axis_to_keep; i++) {
            int start_idx = (i / step) * step * dim_to_reduce + i % step;

            for (int j = 0; j < dim_to_reduce; j++) {
                output[i] += input[start_idx + j * step];
            }
        }

        // Split division work
        blockSize = (axis_to_keep + NUM_CORES - 1) / NUM_CORES;
        start = id * blockSize;
        stop = start + blockSize > axis_to_keep ? axis_to_keep : start + blockSize;

        // Compute output
        for (int i = start; i < stop; i++) {
            output[i] /= dim_to_reduce;
        }
    } else {
        // Split work output-wise
        blockSize = (axis_to_keep + NUM_CORES - 1) / NUM_CORES;
        start = id * blockSize;
        stop = start + blockSize > axis_to_keep ? axis_to_keep : start + blockSize;

        // Compute output
        for (int i = start; i < stop; i++) {
            int start_idx = (i / step) * step * dim_to_reduce + i % step;

            for (int j = 0; j < dim_to_reduce; j++) {
                output[i] += input[start_idx + j * step];
            }

            output[i] /= dim_to_reduce;
        }
    }
}
