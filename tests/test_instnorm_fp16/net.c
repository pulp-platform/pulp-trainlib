/**
 * INCLUDES
**/

#include "pulp_train.h"
#include "net.h"
#include "stats.h"

#include "init-defines.h"
#include "io_data.h"



/**
 * DATA
**/

// Define loss
PI_L1 fp16 loss = 0;

// Define DNN blobs
PI_L1 struct blob_fp16 layer1_in, layer1_wgt, layer1_out;

// Define DNN layer structures
PI_L1 struct InstNorm_args_fp16 l1_args;

// Define kernel tensors
PI_L1 fp16 l1_ker[2*Tin_C_l1];

// Define kernel grad tensors
PI_L1 fp16 l1_ker_diff[2*Tin_C_l1];

// Define I/O tensors
PI_L1 fp16 l1_in[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L1 fp16 l1_out[Tout_C_l1 * Tout_H_l1 * Tout_W_l1];

// Define error propagation tensors
PI_L1 fp16 l1_in_diff[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L1 fp16 l1_out_diff[Tout_C_l1 * Tout_H_l1 * Tout_W_l1];

// Define running params arrays
PI_L1 fp16 running_mean[Tin_C_l1];
PI_L1 fp16 running_var[Tin_C_l1];
PI_L1 fp16 running_stdev[Tin_C_l1];

// Loss function configuration structure
PI_L1 struct loss_args_fp16 loss_args;



/**
 * DNN BACKEND FUNCTIONS
**/

// DNN initialization function
void DNN_init()
{
  // Layer 1
  for(int i=0; i<Tin_C_l1*Tin_H_l1*Tin_W_l1; i++)			l1_in[i] = INPUT[i];
  for(int i=0; i<2*Tin_C_l1; i++)		                  l1_ker[i] = init_WGT_l1[i];

  // Connect tensors to blobs

//Connecting InstNorm
  // Layer 1
  layer1_in.data = l1_in;
  layer1_in.diff = l1_in_diff;
  layer1_in.dim = Tin_C_l1*Tin_H_l1*Tin_W_l1;
  layer1_in.C = Tin_C_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.W = Tin_W_l1;
  layer1_wgt.data = l1_ker;
  layer1_wgt.diff = l1_ker_diff;
  layer1_wgt.dim = 2*Tin_C_l1;
  layer1_wgt.C = Tin_C_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_out.data = l1_out;
  layer1_out.diff = l1_out_diff;
  layer1_out.dim = Tout_C_l1*Tout_H_l1*Tout_W_l1;
  layer1_out.C = Tout_C_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.W = Tout_W_l1;

  // Configure layer structures
  // Layer 1
  l1_args.input = &layer1_in;
  l1_args.coeff = &layer1_wgt;
  l1_args.output = &layer1_out;
  l1_args.running_mean = running_mean;
  l1_args.running_var = running_var;
  l1_args.running_stdev = running_stdev;
  l1_args.freeze_running_params = 0;
  l1_args.skip_wg_grad = 0;
  l1_args.skip_in_grad = 0;
}


// Forward pass function
void forward()
{
  pulp_instnorm_fp16_fw_cl(&l1_args);
}

void forward_print()
{
  #ifdef PROF_NET
  printf("\nForward Stats:\n");
  START_STATS();
  #endif
  pulp_instnorm_fp16_fw_cl(&l1_args);
  #ifdef PROF_NET
  STOP_STATS();
  #endif
}

// Backward pass function
void backward()
{
  loss_args.output = &layer1_out;
  loss_args.target = LABEL;
  loss_args.wr_loss = &loss;
  pulp_MSELoss_backward(&loss_args);
  pulp_instnorm_fp16_bw_param_grads_cl(&l1_args);
  pulp_instnorm_fp16_bw_input_grads_cl(&l1_args);
}

void backward_print()
{
  loss_args.output = &layer1_out;
  loss_args.target = LABEL;
  loss_args.wr_loss = &loss;
  pulp_MSELoss_backward_fp16(&loss_args);

  #if defined(PROF_NET) && defined(BACKWARD_GRAD)
  printf("\nBackward Stats:\n");
  START_STATS();
  pulp_instnorm_fp16_bw_param_grads_cl(&l1_args);
  STOP_STATS();
  #endif

  #if defined(PROF_NET) && defined(BACKWARD_ERROR)
  printf("\nBackward Stats:\n");
  START_STATS();
  pulp_instnorm_fp16_bw_input_grads_cl(&l1_args);
  STOP_STATS();
  #endif
}

// Compute loss and output gradient
void compute_loss()
{
  loss_args.output = &layer1_out;
  loss_args.target = LABEL;
  loss_args.wr_loss = &loss;
  pulp_MSELoss_fp16(&loss_args);
}


/**
 * DATA VISUALIZATION AND CHECK TOOLS
**/

// Function to print FW output
void print_output()
{
  #ifdef FORWARD
  printf("\nLayer 1 output:\n");
  for (int i=0; i<Tout_C_l1*Tout_H_l1*Tout_W_l1; i++)
  {
    printf("%f ", l1_out[i]);
    if(!(i%Tout_W_l1*Tout_H_l1)) printf("\n");
  }
  #endif

  #ifdef BACKWARD_GRAD
  printf("\nBatchnorm Weight Grad:\n");
  for (int i=0; i<2*Tin_C_l1; i++)
  {
    printf("%f ", l1_ker_diff[i]);
    if(!(i%2)) printf("\n");
  }
  #endif

  #ifdef BACKWARD_ERROR
  printf("\nLayer 1 in diff:\n");
  for (int i=0; i<Tin_C_l1*Tin_H_l1*Tin_W_l1; i++)
  {
    printf("%.10f ", l1_in_diff[i]);
    if(!(i%Tin_W_l1*Tin_H_l1)) printf("\n");
  }
  #endif
}

// Function to check post-training output wrt Golden Model (GM)
void check_post_training_output()
{
  int integrity_check = 0;
  integrity_check = verify_tensor_fp16(l1_out, REFERENCE_OUTPUT, Tout_C_l1*Tout_H_l1*Tout_W_l1, TOLERANCE);
  if (integrity_check > 0)
    printf("\n*** UPDATED OUTPUT NOT MATCHING GOLDEN MODEL ***\n");
}

// Checks forward, weight and input grads of the InstanceNorm
void check_instancenorm() 
{
  int integrity_check = 0;
  #ifdef BACKWARD_ERROR
  integrity_check = verify_tensor_fp16(l1_in_diff, INSTN_IN_GRAD, Tin_C_l1*Tin_H_l1*Tin_W_l1, TOLERANCE);
  #elif defined(BACKWARD_GRAD)
  integrity_check = verify_tensor_fp16(l1_ker_diff, INSTN_WGT_GRAD, 2*Tin_C_l1, TOLERANCE);
  #endif
}



/**
 * DNN MODEL TRAINING
**/

// Call for a complete training step
void net_step()
{
 
  printf("Initializing network..\n");
  DNN_init();
  printf("Initializing Instance Normalization test\n");
  forward();
  compute_loss();

  #ifdef FORWARD
  printf("\nProfiling FORWARD step..\n");
  #endif
  #if defined(BACKWARD_GRAD) || defined(BACKWARD_ERROR)
  printf("\nProfiling BACKWARD step..\n");
  #endif

  #ifdef PROF_NET
  INIT_STATS();
  PRE_START_STATS();
  #endif

  #ifdef FORWARD
  forward_print();
  #endif

  #if defined(BACKWARD_GRAD) || defined(BACKWARD_ERROR)
  backward_print();
  #endif

  // Check and print updated output
  forward();
  printf("Checking updated output..\n");
  #ifdef FORWARD
  check_post_training_output();
  #else
  check_instancenorm();
  #endif
  print_output();
}
