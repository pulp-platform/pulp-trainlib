#ifndef TENSOR_CHECKERS_H
#define TENSOR_CHECKERS_H

// Constants definition
#if DATA_TYPE == 32
#define CHECK_TOLERANCE 0.001
#define ERROR_TOLERANCE 0.001
#elif DATA_TYPE == 16
#define CHECK_TOLERANCE 0x00000001
#define ERROR_TOLERANCE 0x00000001
#endif

// Includes
#include "math.h"


// Functions definition
// Mean error checker
#if DATA_TYPE == 32
void mean_error_checker(float *A, float *B, int length) {
    float mean_err_rel = 0.0;
    float diff;
    float mean_abs_value = 0.0;
#elif DATA_TYPE == 16
void mean_error_checker(fp16 *A, fp16 *B, int length) {
    fp16 mean_err_rel = 0.0;
    fp16 diff;
    fp16 mean_abs_value = 0.0;
#endif

    double err_variance = 0;
    double abs_value_variance = 0;

    for (int i = 0; i < length; i++) {
        diff = A[i] - B[i];
        if (diff > 0) diff = diff;
        else diff = -diff;
        mean_err_rel = mean_err_rel + diff;
        if (B[i] > 0)
            mean_abs_value = mean_abs_value + B[i] / length;
        else
            mean_abs_value = mean_abs_value - B[i] / length;
    }

    mean_err_rel = mean_err_rel / length;


    for (int i = 0; i < length; i++) {
        diff = A[i] - B[i];
        if (diff > 0) diff = diff;
        else diff = -diff;

        err_variance = err_variance + pow((diff - mean_err_rel), 2) / length;

        if (B[i] > 0)
            abs_value_variance = abs_value_variance + pow((B[i] - mean_abs_value), 2) / length;
        else
            abs_value_variance = abs_value_variance + pow(((-B[i]) - mean_abs_value), 2) / length;
    }

#if DATA_TYPE == 32
    float std_err = sqrt(err_variance);
    float std_abs = sqrt(abs_value_variance);
#elif DATA_TYPE == 16
    fp16 std_err = sqrt(err_variance);
    fp16 std_abs = sqrt(abs_value_variance);
#endif

    if (mean_err_rel < ERROR_TOLERANCE) printf("\n>>>TENSOR MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
    else printf("\n>>>TENSOR NOT MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);

    printf("\n>>>MEAN ERROR:%f MEAN GM ABS OUTPUT:%f\n", mean_err_rel, mean_abs_value);
    printf("\n>>>MEAN ERROR / MEAN GM OUTPUT ABS VALUE:%f\n", mean_err_rel / mean_abs_value);
    printf("\n>>>ERROR VARIANCE:%f ABS GM OUTPUT VARIANCE:%f\n", err_variance, abs_value_variance);
    printf("\n>>>STD DEVIATIONS: ERROR->%f  ABS ->%f\n", std_err, std_abs);
}


// Elementwise checker
#if DATA_TYPE == 32
int elementwise_checker(float *tensor_out, float *tensor_ref, int size) {
#elif DATA_TYPE == 16
int elementwise_checker(fp16 *tensor_out, fp16 *tensor_ref, int size) {
#endif
    int error_flag = 0;

    for (int i = 0; i < size; i++) {
        if (ABS(tensor_out[i] - tensor_ref[i]) > CHECK_TOLERANCE) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i,
                   tensor_ref[i], *(unsigned int *) &tensor_ref[i], tensor_out[i], *(unsigned int *) &tensor_out[i]);
            error_flag = 1;
        }
    }

    return error_flag;
}

#endif
