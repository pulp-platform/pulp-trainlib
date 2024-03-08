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
PI_L1 float loss = 0;

// Define DNN blobs
PI_L1 struct blob layer0_in, layer0_wgt, layer0_out;
PI_L1 struct blob layer1_in, layer1_wgt, layer1_out;
PI_L1 struct blob layer2_in, layer2_wgt, layer2_out;

// Define DNN layer structures
PI_L1 struct vect_sum_args vect_sum_args;
PI_L1 struct vect_sum_args_fp16 vect_sum_args_fp16;
PI_L1 struct PointWise_Conv_args l0_args;
PI_L1 struct InstNorm_args l1_args;
PI_L1 struct PointWise_Conv_args l2_args;

// Define kernel tensors
PI_L1 float l0_ker[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L1 float l1_ker[2*Tin_C_l1];
PI_L1 float l2_ker[Tin_C_l2 * Tout_C_l2 * Tker_H_l2 * Tker_W_l2];

// Define kernel grad tensors
PI_L1 float l0_ker_diff[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L1 float l1_ker_diff[2*Tin_C_l1];
PI_L1 float l2_ker_diff[Tin_C_l2 * Tout_C_l2 * Tker_H_l2 * Tker_W_l2];

// Define I/O tensors
PI_L1 float l0_in[Tin_C_l0 * Tin_H_l0 * Tin_W_l0];
PI_L1 float l1_in[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L1 float l2_in[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L1 float l2_out[Tout_C_l2 * Tout_H_l2 * Tout_W_l2];

// Define transposition / block transposition buffer for all conv2d and PW layers
PI_L1 float bt_buffer[Tin_C_l2*Tout_C_l2*Tker_H_l2*Tker_W_l2];

// Define error propagation tensors
PI_L1 float l1_in_diff[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L1 float l2_in_diff[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L1 float l2_out_diff[Tout_C_l2 * Tout_H_l2 * Tout_W_l2];

// Loss function configuration structure
PI_L1 struct loss_args loss_args;



/**
 * DNN BACKEND FUNCTIONS
**/

// DNN initialization function
void DNN_init()
{
  // Layer 0
  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)			l0_in[i] = INPUT[i];
  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)		l0_ker[i] = init_WGT_l0[i];
  // Layer 1
  for(int i=0; i<2*Tin_C_l1; i++)		l1_ker[i] = init_WGT_l1[i];
  // Layer 2
  for(int i=0; i<Tin_C_l2*Tout_C_l2*Tker_H_l2*Tker_W_l2; i++)		l2_ker[i] = init_WGT_l2[i];

  // Connect tensors to blobs


//Connecting PW
  // Layer 0
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_C_l0*Tin_H_l0*Tin_W_l0;
  layer0_in.C = Tin_C_l0;
  layer0_in.H = Tin_H_l0;
  layer0_in.W = Tin_W_l0;
  layer0_wgt.data = l0_ker;
  layer0_wgt.diff = l0_ker_diff;
  layer0_wgt.dim = Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0;
  layer0_wgt.C = Tin_C_l0;
  layer0_wgt.H = Tker_H_l0;
  layer0_wgt.W = Tker_W_l0;
  layer0_out.data = l1_in;
  layer0_out.diff = l1_in_diff;
  layer0_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;
  layer0_out.C = Tout_C_l0;
  layer0_out.H = Tout_H_l0;
  layer0_out.W = Tout_W_l0;


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
  layer1_out.data = l2_in;
  layer1_out.diff = l2_in_diff;
  layer1_out.dim = Tout_C_l1*Tout_H_l1*Tout_W_l1;
  layer1_out.C = Tout_C_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.W = Tout_W_l1;


//Connecting PW
  // Layer 2
  layer2_in.data = l2_in;
  layer2_in.diff = l2_in_diff;
  layer2_in.dim = Tin_C_l2*Tin_H_l2*Tin_W_l2;
  layer2_in.C = Tin_C_l2;
  layer2_in.H = Tin_H_l2;
  layer2_in.W = Tin_W_l2;
  layer2_wgt.data = l2_ker;
  layer2_wgt.diff = l2_ker_diff;
  layer2_wgt.dim = Tin_C_l2*Tout_C_l2*Tker_H_l2*Tker_W_l2;
  layer2_wgt.C = Tin_C_l2;
  layer2_wgt.H = Tker_H_l2;
  layer2_wgt.W = Tker_W_l2;
  layer2_out.data = l2_out;
  layer2_out.diff = l2_out_diff;
  layer2_out.dim = Tout_C_l2*Tout_H_l2*Tout_W_l2;
  layer2_out.C = Tout_C_l2;
  layer2_out.H = Tout_H_l2;
  layer2_out.W = Tout_W_l2;

  // Configure layer structures
  // Layer 0
  l0_args.input = &layer0_in;
  l0_args.coeff = &layer0_wgt;
  l0_args.output = &layer0_out;
  l0_args.transpose_buffer = (float*) bt_buffer;
  l0_args.skip_in_grad = 1;
  l0_args.opt_matmul_type_fw = 0;
  l0_args.opt_matmul_type_wg = 0;
  l0_args.opt_matmul_type_ig = 0;
  l0_args.HWC = 0;
  // Layer 1
  l1_args.input = &layer1_in;
  l1_args.coeff = &layer1_wgt;
  l1_args.output = &layer1_out;
  l1_args.skip_in_grad = 0;
  // Layer 2
  l2_args.input = &layer2_in;
  l2_args.coeff = &layer2_wgt;
  l2_args.output = &layer2_out;
  l2_args.transpose_buffer = (float*) bt_buffer;
  l2_args.skip_in_grad = 0;
  l2_args.opt_matmul_type_fw = 0;
  l2_args.opt_matmul_type_wg = 0;
  l2_args.opt_matmul_type_ig = 0;
  l2_args.HWC = 0;
}


// Forward pass function
void forward()
{
  pulp_conv_pw_fp32_fw_cl(&l0_args);
  pulp_instnorm_fp32_fw_cl(&l1_args);
  pulp_conv_pw_fp32_fw_cl(&l2_args);
}

void forward_print()
{
  pulp_conv_pw_fp32_fw_cl(&l0_args);

  #ifdef PROF_NET
  printf("\nForward Stats:\n");
  START_STATS();
  #endif
  pulp_instnorm_fp32_fw_cl(&l1_args);
  #ifdef PROF_NET
  STOP_STATS();
  #endif

  pulp_conv_pw_fp32_fw_cl(&l2_args);
}

// Backward pass function
void backward()
{
  pulp_conv_pw_fp32_bw_cl(&l2_args);
  pulp_instnorm_fp32_bw_cl(&l1_args);
  pulp_conv_pw_fp32_bw_cl(&l0_args);
}

void backward_print()
{
  pulp_conv_pw_fp32_bw_cl(&l2_args);

  #if defined(PROF_NET) && defined(BACKWARD_GRAD)
  printf("\nBackward Stats:\n");
  START_STATS();
  #endif
  pulp_instnorm_fp32_bw_param_grads_cl(&l1_args);
  #if defined(PROF_NET) && defined(BACKWARD_GRAD)
  STOP_STATS();
  #endif

  #if defined(PROF_NET) && defined(BACKWARD_ERROR)
  printf("\nBackward Stats:\n");
  START_STATS();
  #endif
  pulp_instnorm_fp32_bw_input_grads_cl(&l1_args);
  #if defined(PROF_NET) && defined(BACKWARD_ERROR)
  STOP_STATS();
  #endif

  pulp_conv_pw_fp32_bw_cl(&l0_args);
}

// Compute loss and output gradient
void compute_loss()
{
  loss_args.output = &layer2_out;
  loss_args.target = LABEL;
  loss_args.wr_loss = &loss;
  pulp_MSELoss(&loss_args);
}

// Function to update the network
void update_weights()
{
  struct optim_args opt_l0;
  opt_l0.weights = &layer0_wgt;
  opt_l0.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0);
  struct optim_args opt_l1;
  opt_l1.weights = &layer1_wgt;
  opt_l1.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l1);
  struct optim_args opt_l2;
  opt_l2.weights = &layer2_wgt;
  opt_l2.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l2);
}



/**
 * DATA VISUALIZATION AND CHECK TOOLS
**/

// Function to print FW output
void print_output()
{
  printf("\nLayer 2 output:\n");

  for (int i=0; i<Tout_C_l2*Tout_H_l2*Tout_W_l2; i++)
  {
    printf("%f ", l2_out[i]);
    // Newline when an output row ends
    // if(!(i%Tout_W_l2)) printf("\n");
    // Newline when an output channel ends
    if(!(i%Tout_W_l2*Tout_H_l2)) printf("\n");
  }
}

// Function to check post-training output wrt Golden Model (GM)
void check_post_training_output()
{
  int integrity_check = 0;
  integrity_check = verify_tensor(l2_out, REFERENCE_OUTPUT, Tout_C_l2*Tout_H_l2*Tout_W_l2, TOLERANCE);
  if (integrity_check > 0)
    printf("\n*** UPDATED OUTPUT NOT MATCHING GOLDEN MODEL ***\n");
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
  update_weights();
  #endif

  // Check and print updated output
  forward();
  printf("Checking updated output..\n");
  check_post_training_output();
  print_output();
}
