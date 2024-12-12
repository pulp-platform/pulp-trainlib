#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#include "net_args.h"
#include "intp_data.h"

#ifdef FLOAT32
PI_L1 float near_intp_data[IN_CH*OUT_H*OUT_W];
PI_L1 float bil_intp_data[IN_CH*OUT_H*OUT_W];
#elif defined(BFLOAT16)
PI_L1 fp16 near_intp_data[IN_CH*OUT_H*OUT_W];
PI_L1 fp16 bil_intp_data[IN_CH*OUT_H*OUT_W];
#endif


#ifdef FLOAT32

static inline void compare_tensors(float *A, float *B, int length){

  float mean_err_rel = 0.0f;
  float diff = 0.0f;
  float den = 0.000001f;

  for(int i=0; i<length; i++){
     if (A[i]>B[i] && A[i]>0.0f){
        diff = A[i]-B[i];
        if (diff>0) diff = diff;
        else diff=-diff;
        if (A[i]>0) den = A[i];
        else den = -A[i]; // missing A = 0
        mean_err_rel = mean_err_rel + (diff / den)/length;
     }
     else{
       diff = A[i]-B[i];
       if (diff>0) diff = diff;
       else diff=-diff;
       if (A[i]>0) den = A[i];
       else den = -A[i];
       mean_err_rel = mean_err_rel + (diff / den)/length;
     }
  }
  if (mean_err_rel<ERROR_TOLERANCE) printf(">>>TENSOR MATCHING!\n");
  else printf(">>>TENSOR NOT MATCHING!\n");

}


// Elementwise checker
int check_tensor(float * tensor_out, float * tensor_ref, int size){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}

#elif defined(BFLOAT16)

static inline void compare_tensors(fp16 *A, fp16 *B, int length){

  float mean_err_rel = 0.0f;
  float diff = 0.0f;
  float den = 0.000001f;

  for(int i=0; i<length; i++){
     if (A[i]>B[i] && A[i]>0.0f){
        diff = A[i]-B[i];
        if (diff>0) diff = diff;
        else diff=-diff;
        if (A[i]>0) den = A[i];
        else den = -A[i]; // missing A = 0
        mean_err_rel = mean_err_rel + (diff / den)/length;
     }
     else{
       diff = A[i]-B[i];
       if (diff>0) diff = diff;
       else diff=-diff;
       if (A[i]>0) den = A[i];
       else den = -A[i];
       mean_err_rel = mean_err_rel + (diff / den)/length;
     }
  }
  if (mean_err_rel<ERROR_TOLERANCE) printf(">>>TENSOR MATCHING!\n");
  else printf(">>>TENSOR NOT MATCHING!\n");

}


// Elementwise checker
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned short int*) &tensor_ref[i], tensor_out[i], *(unsigned short int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}

#endif


// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    #ifdef FLOAT32
    struct blob input_blob;
    struct blob near_output_blob;
    struct blob bil_output_blob;
    #elif defined(BFLOAT16)
    struct blob_fp16 input_blob;
    struct blob_fp16 near_output_blob;
    struct blob_fp16 bil_output_blob;
    #endif

    input_blob.data = INDATA;
    input_blob.C = IN_CH;
    input_blob.H = IN_H;
    input_blob.W = IN_W;
    
    near_output_blob.data = near_intp_data;
    near_output_blob.C = IN_CH;
    near_output_blob.H = OUT_H;
    near_output_blob.W = OUT_W;
    
    bil_output_blob.data = bil_intp_data;
    bil_output_blob.C = IN_CH;
    bil_output_blob.H = OUT_H;
    bil_output_blob.W = OUT_W;

    #ifdef FLOAT32
    struct interpolation_args near_args;
    struct interpolation_args bil_args;
    #elif defined(BFLOAT16)
    struct interpolation_args_fp16 near_args;
    struct interpolation_args_fp16 bil_args;
    #endif

    near_args.input  = &input_blob;
    near_args.output = &near_output_blob;
    near_args.interpolate_data = 1;
    near_args.interpolate_gradient = 0;
    
    bil_args.input  = &input_blob;
    bil_args.output = &bil_output_blob;
    bil_args.interpolate_data = 1;
    bil_args.interpolate_gradient = 0;

    #if INTP_TYPE == 0  // Nearest Neighbour

    printf("Nearest Neigbour:\n");
    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifdef FLOAT32
    pulp_nearest_neighbour_interpolation_fp32_cl(&near_args);
    #elif defined(BFLOAT16)
    pulp_nearest_neighbour_interpolation_fp16_cl(&near_args);
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("NEAREST NEIGHBOUR CHECK: \n");
    compare_tensors(near_intp_data, OUTDATA_NEAR, IN_CH*OUT_H*OUT_W);
    check_tensor(near_intp_data, OUTDATA_NEAR, IN_CH*OUT_H*OUT_W);

    printf("\n\nTotal operations: %d\n", IN_CH*OUT_H*OUT_W);
    // // TEST
    // printf("\nIN SIZES: [%d, %d, %d]\n", IN_CH, IN_H, IN_W);
    // printf("IN_ELEMENTS: %d\n", IN_CH*IN_H*IN_W);
    // for (int index=0; index<IN_CH*IN_H*IN_W; index++) {
    //   if (!(index%IN_W)) printf("\n");
    //   printf("%f ", INDATA[index]);
    // }
    // printf("\n");
    // printf("\nOUT SIZES: [%d, %d, %d]\n", IN_CH, OUT_H, OUT_W);
    // printf("OUT_ELEMENTS: %d\n", IN_CH*OUT_H*OUT_W);
    // for (int index=0; index<IN_CH*OUT_H*OUT_W; index++) {
    //   if (!(index%OUT_W)) printf("\n");
    //   printf("%f ", near_intp_data[index]);
    // }
    // printf("\n");

    #elif INTP_TYPE == 1    // Bilinear

    printf("Bilinear Interpolation:\n");
    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifdef FLOAT32
    pi_cl_team_fork(NUM_CORES, pulp_bilinear_interpolation_fp32_cl, &bil_args);
    #elif defined(BFLOAT16)
    pi_cl_team_fork(NUM_CORES, pulp_bilinear_interpolation_fp16_cl, &bil_args);
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("BILINEAR CHECK: \n");
    compare_tensors(bil_intp_data, OUTDATA_BIL, IN_CH*OUT_H*OUT_W);
    check_tensor(bil_intp_data, OUTDATA_BIL, IN_CH*OUT_H*OUT_W);
    // // TEST
    // printf("\nIN SIZES: [%d, %d, %d]\n", IN_CH, IN_H, IN_W);
    // printf("IN_ELEMENTS: %d\n", IN_CH*IN_H*IN_W);
    // for (int index=0; index<IN_CH*IN_H*IN_W; index++) {
    //   if (!(index%IN_W)) printf("\n");
    //   printf("%f ", INDATA[index]);
    // }
    // printf("\n");
    // printf("\nOUT SIZES: [%d, %d, %d]\n", IN_CH, OUT_H, OUT_W);
    // printf("OUT_ELEMENTS: %d\n", IN_CH*OUT_H*OUT_W);
    // for (int index=0; index<IN_CH*OUT_H*OUT_W; index++) {
    //   if (!(index%OUT_W)) printf("\n");
    //   printf("%f ", bil_intp_data[index]);
    // }
    // printf("\n");

    #endif

    return;
}
