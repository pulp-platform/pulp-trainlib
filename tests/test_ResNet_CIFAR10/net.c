/**
 * INCLUDES
**/

#include "pulp_train.h"
#include "net.h"
#include "stats.h"

#include "init-defines.h"
#include "io_data.h"

// Define structures and pointers to data in L1 memory
PI_L1 float * IN_DATA , * IN_DIFF, * W_DATA, * W_DIFF, * OUT_DATA, * OUT_DIFF;
PI_L1 float BUFF[MAX_SIZE];
PI_L1 struct blob input_blob;
PI_L1 struct blob weight_blob;
PI_L1 struct blob output_blob;
PI_L1 struct blob temp_blob;
PI_L1 struct Linear_args linear_args;
PI_L1 struct Conv2D_args conv2d_args;
PI_L1 struct PointWise_Conv_args PW_args;
PI_L1 struct DepthWise_Conv_args DW_args;
PI_L1 struct act_args act_args;
PI_L1 struct InstNorm_args InstNorm_args;
PI_L1 struct SkipConn_args resconn_args;
PI_L1 float * t;
PI_L1 pi_cl_dma_cmd_t * cmd_store;
PI_L1 pi_cl_dma_cmd_t * cmd_load;



/**
 * DATA
**/

// Define loss
PI_L1 float loss = 0;
PI_L1 float train_loss = 0;
PI_L1 float test_loss = 0;

// Define DNN blobs
PI_L2 struct blob layer0_in, layer0_wgt, layer0_out;
PI_L2 struct blob layer1_in, layer1_wgt, layer1_out;
PI_L2 struct blob layer2_in, layer2_wgt, layer2_out;
PI_L2 struct blob layer3_in, layer3_wgt, layer3_out;
PI_L2 struct blob layer4_in, layer4_wgt, layer4_out;
PI_L2 struct blob layer5_in, layer5_wgt, layer5_out;
PI_L2 struct blob layer6_in, layer6_wgt, layer6_out;
PI_L2 struct blob layer7_in, layer7_wgt, layer7_out;
PI_L2 struct blob layer8_1_in, layer8_1_wgt, layer8_1_out;
PI_L2 struct blob layer8_2_in, layer8_2_wgt, layer8_2_out;
PI_L2 struct blob layer9_in, layer9_wgt, layer9_out;
PI_L2 struct blob layer10_in, layer10_wgt, layer10_out;
PI_L2 struct blob layer11_in, layer11_wgt, layer11_out;
PI_L2 struct blob layer12_in, layer12_wgt, layer12_out;
PI_L2 struct blob layer13_in, layer13_wgt, layer13_out;
PI_L2 struct blob layer14_in, layer14_wgt, layer14_out;
PI_L2 struct blob layer15_in, layer15_wgt, layer15_out;
PI_L2 struct blob layer16_in, layer16_wgt, layer16_out;
PI_L2 struct blob layer17_in, layer17_wgt, layer17_out;
PI_L2 struct blob layer18_in, layer18_wgt, layer18_out;
PI_L2 struct blob layer19_in, layer19_wgt, layer19_out;

// Define DNN layer structures
PI_L1 struct array_broadcast_sum_fp32_args vect_sum_args;
PI_L2 struct DepthWise_Conv_args l0_args;
PI_L2 struct PointWise_Conv_args l1_args;
PI_L2 struct InstNorm_args l2_args;
PI_L2 struct act_args l3_args;
PI_L2 struct DepthWise_Conv_args l4_args;
PI_L2 struct PointWise_Conv_args l5_args;
PI_L2 struct InstNorm_args l6_args;
PI_L2 struct act_args l7_args;
PI_L2 struct DepthWise_Conv_args l8_1_args;
PI_L2 struct PointWise_Conv_args l8_2_args;
PI_L2 struct DepthWise_Conv_args l9_args;
PI_L2 struct PointWise_Conv_args l10_args;
PI_L2 struct InstNorm_args l11_args;
PI_L2 struct act_args l12_args;
PI_L2 struct DepthWise_Conv_args l13_args;
PI_L2 struct PointWise_Conv_args l14_args;
PI_L2 struct InstNorm_args l15_args;
PI_L2 struct act_args l16_args;
PI_L2 struct SkipConn_args l17_args;
PI_L2 struct Linear_args l18_args;
PI_L2 struct act_args l19_args;

// Define kernel tensors
PI_L2 float l0_ker[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L2 float l1_ker[Tin_C_l1 * Tout_C_l1 * Tker_H_l1 * Tker_W_l1];
PI_L2 float l2_ker[2*Tin_C_l2];
PI_L2 float l3_ker[Tin_C_l3 * Tout_C_l3 * Tker_H_l3 * Tker_W_l3];
PI_L2 float l4_ker[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L2 float l5_ker[Tin_C_l5 * Tout_C_l5 * Tker_H_l5 * Tker_W_l5];
PI_L2 float l6_ker[2*Tin_C_l6];
PI_L2 float l7_ker[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L2 float l8_1_ker[Tin_C_l8_1 * Tout_C_l8_1 * Tker_H_l8_1 * Tker_W_l8_1];
PI_L2 float l8_2_ker[Tin_C_l8_2 * Tout_C_l8_2 * Tker_H_l8_2 * Tker_W_l8_2];
PI_L2 float l9_ker[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L2 float l10_ker[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L2 float l11_ker[2*Tin_C_l11];
PI_L2 float l12_ker[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L2 float l13_ker[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L2 float l14_ker[Tin_C_l14 * Tout_C_l14 * Tker_H_l14 * Tker_W_l14];
PI_L2 float l15_ker[2*Tin_C_l15];
PI_L2 float l16_ker[Tin_C_l16 * Tout_C_l16 * Tker_H_l16 * Tker_W_l16];
PI_L2 float l18_ker[Tin_C_l18 * Tout_C_l18 * Tker_H_l18 * Tker_W_l18];

// Define kernel grad tensors
PI_L2 float l0_ker_diff[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L2 float l1_ker_diff[Tin_C_l1 * Tout_C_l1 * Tker_H_l1 * Tker_W_l1];
PI_L2 float l2_ker_diff[2*Tin_C_l2];
PI_L2 float l3_ker_diff[Tin_C_l3 * Tout_C_l3 * Tker_H_l3 * Tker_W_l3];
PI_L2 float l4_ker_diff[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L2 float l5_ker_diff[Tin_C_l5 * Tout_C_l5 * Tker_H_l5 * Tker_W_l5];
PI_L2 float l6_ker_diff[2*Tin_C_l6];
PI_L2 float l7_ker_diff[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L2 float l8_1_ker_diff[Tin_C_l8_1 * Tout_C_l8_1 * Tker_H_l8_1 * Tker_W_l8_1];
PI_L2 float l8_2_ker_diff[Tin_C_l8_2 * Tout_C_l8_2 * Tker_H_l8_2 * Tker_W_l8_2];
PI_L2 float l9_ker_diff[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L2 float l10_ker_diff[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L2 float l11_ker_diff[2*Tin_C_l11];
PI_L2 float l12_ker_diff[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L2 float l13_ker_diff[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L2 float l14_ker_diff[Tin_C_l14 * Tout_C_l14 * Tker_H_l14 * Tker_W_l14];
PI_L2 float l15_ker_diff[2*Tin_C_l15];
PI_L2 float l16_ker_diff[Tin_C_l16 * Tout_C_l16 * Tker_H_l16 * Tker_W_l16];
PI_L2 float l18_ker_diff[Tin_C_l18 * Tout_C_l18 * Tker_H_l18 * Tker_W_l18];

// Define I/O tensors
PI_L2 float l0_in[Tin_C_l0 * Tin_H_l0 * Tin_W_l0];
PI_L2 float l1_in[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L2 float l2_in[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L2 float l3_in[Tin_C_l3 * Tin_H_l3 * Tin_W_l3];
PI_L2 float l4_in[Tin_C_l4 * Tin_H_l4 * Tin_W_l4];
PI_L2 float l5_in[Tin_C_l5 * Tin_H_l5 * Tin_W_l5];
PI_L2 float l6_in[Tin_C_l6 * Tin_H_l6 * Tin_W_l6];
PI_L2 float l7_in[Tin_C_l7 * Tin_H_l7 * Tin_W_l7];
PI_L2 float l8_1_in[Tin_C_l8_1 * Tin_H_l8_1 * Tin_W_l8_1];
PI_L2 float l8_2_in[Tin_C_l8_2 * Tin_H_l8_2 * Tin_W_l8_2];
PI_L2 float l9_in[Tin_C_l9 * Tin_H_l9 * Tin_W_l9];
PI_L2 float l10_in[Tin_C_l10 * Tin_H_l10 * Tin_W_l10];
PI_L2 float l11_in[Tin_C_l11 * Tin_H_l11 * Tin_W_l11];
PI_L2 float l12_in[Tin_C_l12 * Tin_H_l12 * Tin_W_l12];
PI_L2 float l13_in[Tin_C_l13 * Tin_H_l13 * Tin_W_l13];
PI_L2 float l14_in[Tin_C_l14 * Tin_H_l14 * Tin_W_l14];
PI_L2 float l15_in[Tin_C_l15 * Tin_H_l15 * Tin_W_l15];
PI_L2 float l16_in[Tin_C_l16 * Tin_H_l16 * Tin_W_l16];
PI_L2 float l17_in[Tin_C_l17 * Tin_H_l17 * Tin_W_l17];
PI_L2 float l18_in[Tin_C_l18 * Tin_H_l18 * Tin_W_l18];
PI_L2 float l18_out[Tout_C_l18 * Tout_H_l18 * Tout_W_l18];
PI_L2 float l19_out[Tout_C_l18 * Tout_H_l18 * Tout_W_l18];

// Define transposition / block transposition buffer for all conv2d and PW layers
PI_L1 float bt_buffer[1200];

// Define error propagation tensors
PI_L2 float l1_in_diff[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L2 float l2_in_diff[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L2 float l3_in_diff[Tin_C_l3 * Tin_H_l3 * Tin_W_l3];
PI_L2 float l4_in_diff[Tin_C_l4 * Tin_H_l4 * Tin_W_l4];
PI_L2 float l5_in_diff[Tin_C_l5 * Tin_H_l5 * Tin_W_l5];
PI_L2 float l6_in_diff[Tin_C_l6 * Tin_H_l6 * Tin_W_l6];
PI_L2 float l7_in_diff[Tin_C_l7 * Tin_H_l7 * Tin_W_l7];
PI_L2 float l8_1_in_diff[Tin_C_l8_1 * Tin_H_l8_1 * Tin_W_l8_1];
PI_L2 float l8_2_in_diff[Tin_C_l8_2 * Tin_H_l8_2 * Tin_W_l8_2];
PI_L2 float l9_in_diff[Tin_C_l9 * Tin_H_l9 * Tin_W_l9];
PI_L2 float l10_in_diff[Tin_C_l10 * Tin_H_l10 * Tin_W_l10];
PI_L2 float l11_in_diff[Tin_C_l11 * Tin_H_l11 * Tin_W_l11];
PI_L2 float l12_in_diff[Tin_C_l12 * Tin_H_l12 * Tin_W_l12];
PI_L2 float l13_in_diff[Tin_C_l13 * Tin_H_l13 * Tin_W_l13];
PI_L2 float l14_in_diff[Tin_C_l14 * Tin_H_l14 * Tin_W_l14];
PI_L2 float l15_in_diff[Tin_C_l15 * Tin_H_l15 * Tin_W_l15];
PI_L2 float l16_in_diff[Tin_C_l16 * Tin_H_l16 * Tin_W_l16];
PI_L2 float l17_in_diff[Tin_C_l17 * Tin_H_l17 * Tin_W_l17];
PI_L2 float l18_in_diff[Tin_C_l18 * Tin_H_l18 * Tin_W_l18];
PI_L2 float l18_out_diff[Tout_C_l18 * Tout_H_l18 * Tout_W_l18];
PI_L2 float l19_out_diff[Tout_C_l18 * Tout_H_l18 * Tout_W_l18];
// Loss function configuration structure
PI_L1 struct loss_args loss_args;



/**
 * DNN BACKEND FUNCTIONS
**/

// DNN initialization function
void DNN_init()
{

// Assign pointers in L1
IN_DATA = BUFF;
IN_DIFF = BUFF;
W_DATA = BUFF;
W_DIFF = BUFF;
OUT_DATA = BUFF;
OUT_DIFF = BUFF;
update_blob();
reset_arguments();

  // Layer 0
  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)		l0_ker[i] = init_WGT_l0[i];
  // Layer 1
  for(int i=0; i<Tin_C_l1*Tout_C_l1*Tker_H_l1*Tker_W_l1; i++)		l1_ker[i] = init_WGT_l1[i];
  // Layer 2
  for(int i=0; i<2*Tin_C_l2; i++)		l2_ker[i] = init_WGT_l2[i];
  // Layer 3
  for(int i=0; i<Tin_C_l3*Tout_C_l3*Tker_H_l3*Tker_W_l3; i++)		l3_ker[i] = init_WGT_l3[i];
  // Layer 4
  for(int i=0; i<Tin_C_l4*Tker_H_l4*Tker_W_l4; i++)		l4_ker[i] = init_WGT_l4[i];
  // Layer 5
  for(int i=0; i<Tin_C_l5*Tout_C_l5*Tker_H_l5*Tker_W_l5; i++)		l5_ker[i] = init_WGT_l5[i];
  // Layer 6
  for(int i=0; i<2*Tin_C_l6; i++)		l6_ker[i] = init_WGT_l6[i];
  // Layer 7
  for(int i=0; i<Tin_C_l7*Tout_C_l7*Tker_H_l7*Tker_W_l7; i++)		l7_ker[i] = init_WGT_l7[i];
  // Layer 8
  for(int i=0; i<Tin_C_l8_1*Tker_H_l8_1*Tker_W_l8_1; i++)		l8_1_ker[i] = init_WGT_l8_1[i];
  for(int i=0; i<32*Tin_C_l8_2*Tker_H_l8_2*Tker_W_l8_2; i++)		l8_2_ker[i] = init_WGT_l8_2[i];
  // Layer 9
  for(int i=0; i<Tin_C_l9*Tker_H_l9*Tker_W_l9; i++)		l9_ker[i] = init_WGT_l9[i];
  // Layer 10
  for(int i=0; i<Tin_C_l10*Tout_C_l10*Tker_H_l10*Tker_W_l10; i++)		l10_ker[i] = init_WGT_l10[i];
  // Layer 11
  for(int i=0; i<2*Tin_C_l11; i++)		l11_ker[i] = init_WGT_l11[i];
  // Layer 12
  for(int i=0; i<Tin_C_l12*Tout_C_l12*Tker_H_l12*Tker_W_l12; i++)		l12_ker[i] = init_WGT_l12[i];
  // Layer 13
  for(int i=0; i<Tin_C_l13*Tker_H_l13*Tker_W_l13; i++)		l13_ker[i] = init_WGT_l13[i];
  // Layer 14
  for(int i=0; i<Tin_C_l14*Tout_C_l14*Tker_H_l14*Tker_W_l14; i++)		l14_ker[i] = init_WGT_l14[i];
  // Layer 15
  for(int i=0; i<2*Tin_C_l15; i++)		l15_ker[i] = init_WGT_l15[i];
  // Layer 16
  for(int i=0; i<Tin_C_l16*Tout_C_l16*Tker_H_l16*Tker_W_l16; i++)		l16_ker[i] = init_WGT_l16[i];
  // Layer 17
  //   Resconn layer (no parameters)
  // Layer 18
  for(int i=0; i<Tin_C_l18*Tout_C_l18*Tker_H_l18*Tker_W_l18; i++)		l18_ker[i] = init_WGT_l18[i];

  // Connect tensors to blobs


//Connecting DW
  // Layer 0
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_C_l0*Tin_H_l0*Tin_W_l0;
  layer0_in.C = Tin_C_l0;
  layer0_in.H = Tin_H_l0;
  layer0_in.W = Tin_W_l0;
  layer0_wgt.data = l0_ker;
  layer0_wgt.diff = l0_ker_diff;
  layer0_wgt.dim = Tin_C_l0*Tker_H_l0*Tker_W_l0;
  layer0_wgt.C = Tin_C_l0;
  layer0_wgt.H = Tker_H_l0;
  layer0_wgt.W = Tker_W_l0;
  layer0_out.data = l1_in;
  layer0_out.diff = l1_in_diff;
  layer0_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;
  layer0_out.C = Tout_C_l0;
  layer0_out.H = Tout_H_l0;
  layer0_out.W = Tout_W_l0;


//Connecting PW
  // Layer 1
  layer1_in.data = l1_in;
  layer1_in.diff = l1_in_diff;
  layer1_in.dim = Tin_C_l1*Tin_H_l1*Tin_W_l1;
  layer1_in.C = Tin_C_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.W = Tin_W_l1;
  layer1_wgt.data = l1_ker;
  layer1_wgt.diff = l1_ker_diff;
  layer1_wgt.dim = Tin_C_l1*Tout_C_l1*Tker_H_l1*Tker_W_l1;
  layer1_wgt.C = Tin_C_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_out.data = l2_in;
  layer1_out.diff = l2_in_diff;
  layer1_out.dim = Tout_C_l1*Tout_H_l1*Tout_W_l1;
  layer1_out.C = Tout_C_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.W = Tout_W_l1;


//Connecting InstNorm
  // Layer 2
  layer2_in.data = l2_in;
  layer2_in.diff = l2_in_diff;
  layer2_in.dim = Tin_C_l2*Tin_H_l2*Tin_W_l2;
  layer2_in.C = Tin_C_l2;
  layer2_in.H = Tin_H_l2;
  layer2_in.W = Tin_W_l2;
  layer2_wgt.data = l2_ker;
  layer2_wgt.diff = l2_ker_diff;
  layer2_wgt.dim = 2*Tin_C_l2;
  layer2_wgt.C = Tin_C_l2;
  layer2_wgt.H = Tker_H_l2;
  layer2_wgt.W = Tker_W_l2;
  layer2_out.data = l3_in;
  layer2_out.diff = l3_in_diff;
  layer2_out.dim = Tout_C_l2*Tout_H_l2*Tout_W_l2;
  layer2_out.C = Tout_C_l2;
  layer2_out.H = Tout_H_l2;
  layer2_out.W = Tout_W_l2;


//Connecting ReLU
  // Layer 3
  layer3_in.data = l3_in;
  layer3_in.diff = l3_in_diff;
  layer3_in.dim = Tin_C_l3*Tin_H_l3*Tin_W_l3;
  layer3_in.C = Tin_C_l3;
  layer3_in.H = Tin_H_l3;
  layer3_in.W = Tin_W_l3;
  layer3_wgt.data = l3_ker;
  layer3_wgt.diff = l3_ker_diff;
  layer3_wgt.dim = Tin_C_l3*Tout_C_l3*Tker_H_l3*Tker_W_l3;
  layer3_wgt.C = Tin_C_l3;
  layer3_wgt.H = Tker_H_l3;
  layer3_wgt.W = Tker_W_l3;
  layer3_out.data = l4_in;
  layer3_out.diff = l4_in_diff;
  layer3_out.dim = Tout_C_l3*Tout_H_l3*Tout_W_l3;
  layer3_out.C = Tout_C_l3;
  layer3_out.H = Tout_H_l3;
  layer3_out.W = Tout_W_l3;


//Connecting DW
  // Layer 4
  layer4_in.data = l4_in;
  layer4_in.diff = l4_in_diff;
  layer4_in.dim = Tin_C_l4*Tin_H_l4*Tin_W_l4;
  layer4_in.C = Tin_C_l4;
  layer4_in.H = Tin_H_l4;
  layer4_in.W = Tin_W_l4;
  layer4_wgt.data = l4_ker;
  layer4_wgt.diff = l4_ker_diff;
  layer4_wgt.dim = Tin_C_l4*Tker_H_l4*Tker_W_l4;
  layer4_wgt.C = Tin_C_l4;
  layer4_wgt.H = Tker_H_l4;
  layer4_wgt.W = Tker_W_l4;
  layer4_out.data = l5_in;
  layer4_out.diff = l5_in_diff;
  layer4_out.dim = Tout_C_l4*Tout_H_l4*Tout_W_l4;
  layer4_out.C = Tout_C_l4;
  layer4_out.H = Tout_H_l4;
  layer4_out.W = Tout_W_l4;


//Connecting PW
  // Layer 5
  layer5_in.data = l5_in;
  layer5_in.diff = l5_in_diff;
  layer5_in.dim = Tin_C_l5*Tin_H_l5*Tin_W_l5;
  layer5_in.C = Tin_C_l5;
  layer5_in.H = Tin_H_l5;
  layer5_in.W = Tin_W_l5;
  layer5_wgt.data = l5_ker;
  layer5_wgt.diff = l5_ker_diff;
  layer5_wgt.dim = Tin_C_l5*Tout_C_l5*Tker_H_l5*Tker_W_l5;
  layer5_wgt.C = Tin_C_l5;
  layer5_wgt.H = Tker_H_l5;
  layer5_wgt.W = Tker_W_l5;
  layer5_out.data = l6_in;
  layer5_out.diff = l6_in_diff;
  layer5_out.dim = Tout_C_l5*Tout_H_l5*Tout_W_l5;
  layer5_out.C = Tout_C_l5;
  layer5_out.H = Tout_H_l5;
  layer5_out.W = Tout_W_l5;


//Connecting InstNorm
  // Layer 6
  layer6_in.data = l6_in;
  layer6_in.diff = l6_in_diff;
  layer6_in.dim = Tin_C_l6*Tin_H_l6*Tin_W_l6;
  layer6_in.C = Tin_C_l6;
  layer6_in.H = Tin_H_l6;
  layer6_in.W = Tin_W_l6;
  layer6_wgt.data = l6_ker;
  layer6_wgt.diff = l6_ker_diff;
  layer6_wgt.dim = 2*Tin_C_l6;
  layer6_wgt.C = Tin_C_l6;
  layer6_wgt.H = Tker_H_l6;
  layer6_wgt.W = Tker_W_l6;
  layer6_out.data = l7_in;
  layer6_out.diff = l7_in_diff;
  layer6_out.dim = Tout_C_l6*Tout_H_l6*Tout_W_l6;
  layer6_out.C = Tout_C_l6;
  layer6_out.H = Tout_H_l6;
  layer6_out.W = Tout_W_l6;


//Connecting ReLU
  // Layer 7
layer7_in.data = l7_in;
  layer7_in.diff = l7_in_diff;
  layer7_in.dim = Tin_C_l7*Tin_H_l7*Tin_W_l7;
  layer7_in.C = Tin_C_l7;
  layer7_in.H = Tin_H_l7;
  layer7_in.W = Tin_W_l7;
  layer7_wgt.data = l7_ker;
  layer7_wgt.diff = l7_ker_diff;
  layer7_wgt.dim = 2*Tin_C_l7;
  layer7_wgt.C = Tin_C_l7;
  layer7_wgt.H = Tker_H_l7;
  layer7_wgt.W = Tker_W_l7;
  layer7_out.data = l8_1_in;
  layer7_out.diff = l9_in_diff;
  layer7_out.dim = Tout_C_l7*Tout_H_l7*Tout_W_l7;
  layer7_out.C = Tout_C_l7;
  layer7_out.H = Tout_H_l7;
  layer7_out.W = Tout_W_l7;


  layer8_1_in.data = l8_1_in;
  layer8_1_in.diff = l8_1_in_diff;
  layer8_1_in.dim = Tin_C_l8_1*Tin_H_l8_1*Tin_W_l8_1;
  layer8_1_in.C = Tin_C_l8_1;
  layer8_1_in.H = Tin_H_l8_1;
  layer8_1_in.W = Tin_W_l8_1;
  layer8_1_wgt.data = l8_1_ker;
  layer8_1_wgt.diff = l8_1_ker_diff;
  layer8_1_wgt.dim = Tin_C_l8_1*Tker_H_l8_1*Tker_W_l8_1;
  layer8_1_wgt.C = Tin_C_l8_1;
  layer8_1_wgt.H = Tker_H_l8_1;
  layer8_1_wgt.W = Tker_W_l8_1;
  layer8_1_out.data = l8_2_in;
  layer8_1_out.diff = l8_2_in_diff;
  layer8_1_out.dim = Tout_C_l8_1*Tout_H_l8_1*Tout_W_l8_1;
  layer8_1_out.C = Tout_C_l8_1;
  layer8_1_out.H = Tout_H_l8_1;
  layer8_1_out.W = Tout_W_l8_1;


  layer8_2_in = layer8_1_out;
  layer8_2_wgt.data = l8_2_ker;
  layer8_2_wgt.diff = l8_2_ker_diff;
  layer8_2_wgt.dim = 32*Tin_C_l8_2*Tker_H_l8_2*Tker_W_l8_2;
  layer8_2_wgt.C = Tin_C_l8_2;
  layer8_2_wgt.H = Tker_H_l8_2;
  layer8_2_wgt.W = Tker_W_l8_2;
  layer8_2_out.data = l9_in;
  layer8_2_out.diff = l17_in_diff;
  layer8_2_out.dim = Tout_C_l8_2*Tout_H_l8_2*Tout_W_l8_2;
  layer8_2_out.C = Tout_C_l8_2;
  layer8_2_out.H = Tout_H_l8_2;
  layer8_2_out.W = Tout_W_l8_2;


  layer9_in.data = l8_1_in;
  layer9_in.diff = l9_in_diff;
  layer9_in.dim = Tin_C_l9*Tin_H_l9*Tin_W_l9;
  layer9_in.C = Tin_C_l9;
  layer9_in.H = Tin_H_l9;
  layer9_in.W = Tin_W_l9;
  layer9_wgt.data = l9_ker;
  layer9_wgt.diff = l9_ker_diff;
  layer9_wgt.dim = Tin_C_l9*Tker_H_l9*Tker_W_l9;
  layer9_wgt.C = Tin_C_l9;
  layer9_wgt.H = Tker_H_l9;
  layer9_wgt.W = Tker_W_l9;
  layer9_out.data = l10_in;
  layer9_out.diff = l10_in_diff;
  layer9_out.dim = Tout_C_l9*Tout_H_l9*Tout_W_l9;
  layer9_out.C = Tout_C_l9;
  layer9_out.H = Tout_H_l9;
  layer9_out.W = Tout_W_l9;



//Connecting PW
  // Layer 10
  layer10_in.data = l10_in;
  layer10_in.diff = l10_in_diff;
  layer10_in.dim = Tin_C_l10*Tin_H_l10*Tin_W_l10;
  layer10_in.C = Tin_C_l10;
  layer10_in.H = Tin_H_l10;
  layer10_in.W = Tin_W_l10;
  layer10_wgt.data = l10_ker;
  layer10_wgt.diff = l10_ker_diff;
  layer10_wgt.dim = Tin_C_l10*Tout_C_l10*Tker_H_l10*Tker_W_l10;
  layer10_wgt.C = Tin_C_l10;
  layer10_wgt.H = Tker_H_l10;
  layer10_wgt.W = Tker_W_l10;
  layer10_out.data = l11_in;
  layer10_out.diff = l11_in_diff;
  layer10_out.dim = Tout_C_l10*Tout_H_l10*Tout_W_l10;
  layer10_out.C = Tout_C_l10;
  layer10_out.H = Tout_H_l10;
  layer10_out.W = Tout_W_l10;


//Connecting InstNorm
  // Layer 11
  layer11_in.data = l11_in;
  layer11_in.diff = l11_in_diff;
  layer11_in.dim = Tin_C_l11*Tin_H_l11*Tin_W_l11;
  layer11_in.C = Tin_C_l11;
  layer11_in.H = Tin_H_l11;
  layer11_in.W = Tin_W_l11;
  layer11_wgt.data = l11_ker;
  layer11_wgt.diff = l11_ker_diff;
  layer11_wgt.dim = 2*Tin_C_l11;
  layer11_wgt.C = Tin_C_l11;
  layer11_wgt.H = Tker_H_l11;
  layer11_wgt.W = Tker_W_l11;
  layer11_out.data = l12_in;
  layer11_out.diff = l12_in_diff;
  layer11_out.dim = Tout_C_l11*Tout_H_l11*Tout_W_l11;
  layer11_out.C = Tout_C_l11;
  layer11_out.H = Tout_H_l11;
  layer11_out.W = Tout_W_l11;


//Connecting ReLU
  // Layer 12
  layer12_in.data = l12_in;
  layer12_in.diff = l12_in_diff;
  layer12_in.dim = Tin_C_l12*Tin_H_l12*Tin_W_l12;
  layer12_in.C = Tin_C_l12;
  layer12_in.H = Tin_H_l12;
  layer12_in.W = Tin_W_l12;
  layer12_wgt.data = l12_ker;
  layer12_wgt.diff = l12_ker_diff;
  layer12_wgt.dim = Tin_C_l12*Tout_C_l12*Tker_H_l12*Tker_W_l12;
  layer12_wgt.C = Tin_C_l12;
  layer12_wgt.H = Tker_H_l12;
  layer12_wgt.W = Tker_W_l12;
  layer12_out.data = l13_in;
  layer12_out.diff = l13_in_diff;
  layer12_out.dim = Tout_C_l12*Tout_H_l12*Tout_W_l12;
  layer12_out.C = Tout_C_l12;
  layer12_out.H = Tout_H_l12;
  layer12_out.W = Tout_W_l12;


//Connecting DW
  // Layer 13
  layer13_in.data = l13_in;
  layer13_in.diff = l13_in_diff;
  layer13_in.dim = Tin_C_l13*Tin_H_l13*Tin_W_l13;
  layer13_in.C = Tin_C_l13;
  layer13_in.H = Tin_H_l13;
  layer13_in.W = Tin_W_l13;
  layer13_wgt.data = l13_ker;
  layer13_wgt.diff = l13_ker_diff;
  layer13_wgt.dim = Tin_C_l13*Tker_H_l13*Tker_W_l13;
  layer13_wgt.C = Tin_C_l13;
  layer13_wgt.H = Tker_H_l13;
  layer13_wgt.W = Tker_W_l13;
  layer13_out.data = l14_in;
  layer13_out.diff = l14_in_diff;
  layer13_out.dim = Tout_C_l13*Tout_H_l13*Tout_W_l13;
  layer13_out.C = Tout_C_l13;
  layer13_out.H = Tout_H_l13;
  layer13_out.W = Tout_W_l13;


//Connecting PW
  // Layer 14
  layer14_in.data = l14_in;
  layer14_in.diff = l14_in_diff;
  layer14_in.dim = Tin_C_l14*Tin_H_l14*Tin_W_l14;
  layer14_in.C = Tin_C_l14;
  layer14_in.H = Tin_H_l14;
  layer14_in.W = Tin_W_l14;
  layer14_wgt.data = l14_ker;
  layer14_wgt.diff = l14_ker_diff;
  layer14_wgt.dim = Tin_C_l14*Tout_C_l14*Tker_H_l14*Tker_W_l14;
  layer14_wgt.C = Tin_C_l14;
  layer14_wgt.H = Tker_H_l14;
  layer14_wgt.W = Tker_W_l14;
  layer14_out.data = l15_in;
  layer14_out.diff = l15_in_diff;
  layer14_out.dim = Tout_C_l14*Tout_H_l14*Tout_W_l14;
  layer14_out.C = Tout_C_l14;
  layer14_out.H = Tout_H_l14;
  layer14_out.W = Tout_W_l14;


//Connecting InstNorm
  // Layer 15
  layer15_in.data = l15_in;
  layer15_in.diff = l15_in_diff;
  layer15_in.dim = Tin_C_l15*Tin_H_l15*Tin_W_l15;
  layer15_in.C = Tin_C_l15;
  layer15_in.H = Tin_H_l15;
  layer15_in.W = Tin_W_l15;
  layer15_wgt.data = l15_ker;
  layer15_wgt.diff = l15_ker_diff;
  layer15_wgt.dim = 2*Tin_C_l15;
  layer15_wgt.C = Tin_C_l15;
  layer15_wgt.H = Tker_H_l15;
  layer15_wgt.W = Tker_W_l15;
  layer15_out.data = l16_in;
  layer15_out.diff = l16_in_diff;
  layer15_out.dim = Tout_C_l15*Tout_H_l15*Tout_W_l15;
  layer15_out.C = Tout_C_l15;
  layer15_out.H = Tout_H_l15;
  layer15_out.W = Tout_W_l15;


//Connecting ReLU
  // Layer 16
  layer16_in.data = l16_in;
  layer16_in.diff = l16_in_diff;
  layer16_in.dim = Tin_C_l16*Tin_H_l16*Tin_W_l16;
  layer16_in.C = Tin_C_l16;
  layer16_in.H = Tin_H_l16;
  layer16_in.W = Tin_W_l16;
  layer16_wgt.data = l16_ker;
  layer16_wgt.diff = l16_ker_diff;
  layer16_wgt.dim = Tin_C_l16*Tout_C_l16*Tker_H_l16*Tker_W_l16;
  layer16_wgt.C = Tin_C_l16;
  layer16_wgt.H = Tker_H_l16;
  layer16_wgt.W = Tker_W_l16;
  layer16_out.data = l17_in;
  layer16_out.diff = l17_in_diff;
  layer16_out.dim = Tout_C_l16*Tout_H_l16*Tout_W_l16;
  layer16_out.C = Tout_C_l16;
  layer16_out.H = Tout_H_l16;
  layer16_out.W = Tout_W_l16;


//Connecting Sumnode
  // Layer 17
  layer17_in.data = l17_in;
  layer17_in.diff = l17_in_diff;
  layer17_in.dim = Tin_C_l17*Tin_H_l17*Tin_W_l17;
  layer17_in.C = Tin_C_l17;
  layer17_in.H = Tin_H_l17;
  layer17_in.W = Tin_W_l17;
  layer17_wgt.data = layer8_2_out.data;
  layer17_wgt.diff = layer8_2_out.diff;
  layer17_wgt.C = layer8_2_out.C;
  layer17_wgt.H = layer8_2_out.H;
  layer17_wgt.W = layer8_2_out.W;
  layer17_wgt.dim = layer8_2_out.C*layer8_2_out.H*layer8_2_out.W;
  layer17_out.data = l18_in;
  layer17_out.diff = l18_in_diff;
  layer17_out.dim = Tout_C_l17*Tout_H_l17*Tout_W_l17;
  layer17_out.C = Tout_C_l17;
  layer17_out.H = Tout_H_l17;
  layer17_out.W = Tout_W_l17;


//Connecting linear
  // Layer 18
  layer18_in.data = l18_in;
  layer18_in.diff = l18_in_diff;
  layer18_in.dim = Tin_C_l18*Tin_H_l18*Tin_W_l18;
  layer18_in.C = Tin_C_l18;
  layer18_in.H = Tin_H_l18;
  layer18_in.W = Tin_W_l18;
  layer18_wgt.data = l18_ker;
  layer18_wgt.diff = l18_ker_diff;
  layer18_wgt.dim = Tin_C_l18*Tout_C_l18*Tker_H_l18*Tker_W_l18;
  layer18_wgt.C = Tin_C_l18;
  layer18_wgt.H = Tker_H_l18;
  layer18_wgt.W = Tker_W_l18;
  layer18_out.data = l18_out;
  layer18_out.diff = l18_out_diff;
  layer18_out.dim = Tout_C_l18*Tout_H_l18*Tout_W_l18;
  layer18_out.C = Tout_C_l18;
  layer18_out.H = Tout_H_l18;
  layer18_out.W = Tout_W_l18;


//Connecting softmax
  // Layer 19
  layer19_in.data = l18_out;
  layer19_in.diff = l18_out_diff;
  layer19_in.dim = Tout_C_l18*Tout_H_l18*Tout_W_l18;
  layer19_in.C = Tout_C_l18;
  layer19_in.H = Tin_H_l18;
  layer19_in.W = Tin_W_l18;
  layer19_out.data = l19_out;
  layer19_out.diff = l19_out_diff;
  layer19_out.dim = Tout_C_l18*Tout_H_l18*Tout_W_l18;
  layer19_out.C = Tout_C_l18;
  layer19_out.H = Tout_H_l18;
  layer19_out.W = Tout_W_l18;
  // Configure layer structures
  // Layer 0
  l0_args.input = &input_blob;
  l0_args.coeff = &weight_blob;
  l0_args.output = &output_blob;
  l0_args.skip_in_grad = 1;
  l0_args.Lpad = 0;
  l0_args.Rpad = 0;
  l0_args.Upad = 0;
  l0_args.Dpad = 0;
  l0_args.HWC = 0;
  // Layer 1
  l1_args.input = &input_blob;
  l1_args.coeff = &weight_blob;
  l1_args.output = &output_blob;
  l1_args.transpose_buffer = (float*) bt_buffer;
  l1_args.skip_wg_grad = 0;
  l1_args.skip_in_grad = 0;
  l1_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L1;
  l1_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L1;
  l1_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L1;
  l1_args.HWC = 0;
  // Layer 2
  l2_args.input = &input_blob;
  l2_args.coeff = &weight_blob;
  l2_args.output = &output_blob;
  l1_args.skip_wg_grad = 0;
  l2_args.skip_in_grad = 0;
  // Layer 3
  l3_args.input = &input_blob;
  l3_args.output = &output_blob;
  // Layer 4
  l4_args.input = &input_blob;
  l4_args.coeff = &weight_blob;
  l4_args.output = &output_blob;
  l4_args.skip_wg_grad = 0;
  l4_args.skip_in_grad = 0;
  l4_args.Lpad = 0;
  l4_args.Rpad = 0;
  l4_args.Upad = 0;
  l4_args.Dpad = 0;
  l4_args.HWC = 0;
  // Layer 5
  l5_args.input = &input_blob;
  l5_args.coeff = &weight_blob;
  l5_args.output = &output_blob;
  l5_args.transpose_buffer = (float*) bt_buffer;
  l5_args.skip_wg_grad = 0;
  l5_args.skip_in_grad = 0;
  l5_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L5;
  l5_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L5;
  l5_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L5;
  l5_args.HWC = 0;
  // Layer 6
  l6_args.input = &input_blob;
  l6_args.coeff = &weight_blob;
  l6_args.output = &output_blob;
  l6_args.skip_wg_grad = 0;
  l6_args.skip_in_grad = 0;
  // Layer 7
  l7_args.input = &input_blob;
  l7_args.output = &output_blob;
  // Layer 8
  l8_1_args.input = &input_blob;
  l8_1_args.coeff = &weight_blob;
  l8_1_args.output = &output_blob;
  l8_1_args.skip_wg_grad = 0;
  l8_1_args.skip_in_grad = 0;
  l8_1_args.Lpad = 0;
  l8_1_args.Rpad = 0;
  l8_1_args.Upad = 0;
  l8_1_args.Dpad = 0;
  l8_1_args.HWC = 0;

  l8_2_args.input = &input_blob;
  l8_2_args.coeff = &weight_blob;
  l8_2_args.output = &output_blob;
  l8_2_args.transpose_buffer = (float*) bt_buffer;
  l8_2_args.skip_wg_grad = 0;
  l8_2_args.skip_in_grad = 0;
  l8_2_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L1;
  l8_2_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L1;
  l8_2_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L1;
  l8_2_args.HWC = 0;
  // Layer 9
  l9_args.input = &input_blob;
  l9_args.coeff = &weight_blob;
  l9_args.output = &output_blob;
  l9_args.skip_wg_grad = 0;
  l9_args.skip_in_grad = 0;
  l9_args.Lpad = 0;
  l9_args.Rpad = 0;
  l9_args.Upad = 0;
  l9_args.Dpad = 0;
  l9_args.HWC = 0;
  // Layer 10
  l10_args.input = &input_blob;
  l10_args.coeff = &weight_blob;
  l10_args.output = &output_blob;
  l10_args.transpose_buffer = (float*) bt_buffer;
  l10_args.skip_wg_grad = 0;
  l10_args.skip_in_grad = 0;
  l10_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L10;
  l10_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L10;
  l10_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L10;
  l10_args.HWC = 0;
  // Layer 11
  l11_args.input = &input_blob;
  l11_args.coeff = &weight_blob;
  l11_args.output = &output_blob;
  l11_args.skip_wg_grad = 0;
  l11_args.skip_in_grad = 0;
  // Layer 12
  l12_args.input = &input_blob;
  l12_args.output = &output_blob;
  // Layer 13
  l13_args.input = &input_blob;
  l13_args.coeff = &weight_blob;
  l13_args.output = &output_blob;
  l13_args.skip_wg_grad = 0;
  l13_args.skip_in_grad = 0;
  l13_args.Lpad = 0;
  l13_args.Rpad = 0;
  l13_args.Upad = 0;
  l13_args.Dpad = 0;
  l13_args.HWC = 0;
  // Layer 14
  l14_args.input = &input_blob;
  l14_args.coeff = &weight_blob;
  l14_args.output = &output_blob;
  l14_args.transpose_buffer = (float*) bt_buffer;
  l14_args.skip_wg_grad = 0;
  l14_args.skip_in_grad = 0;
  l14_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L14;
  l14_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L14;
  l14_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L14;
  l14_args.HWC = 0;
  // Layer 15
  l15_args.input = &input_blob;
  l15_args.coeff = &weight_blob;
  l15_args.output = &output_blob;
  l15_args.skip_wg_grad = 0;
  l15_args.skip_in_grad = 0;
  // Layer 16
  l16_args.input = &input_blob;
  l16_args.output = &output_blob;
  // Layer 17
  l17_args.lout = &input_blob;
  l17_args.skip = &weight_blob;
  l17_args.output = &output_blob;
  l17_args.skip_wg_grad = 0;
  l17_args.skip_in_grad = 0;
  // Layer 18
  l18_args.input = &input_blob;
  l18_args.coeff = &weight_blob;
  l18_args.output = &output_blob;
  l18_args.skip_wg_grad = 0;
  l18_args.skip_in_grad = 0;
  l18_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L18;
  l18_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L18;
  l18_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L18;
  // Layer 19
  l19_args.input = &input_blob;
  l19_args.output = &output_blob;
}


// Forward pass function
void forward()
{
	reset_dim();
	load_input(&layer0_in, 1);
	load_coeff(&layer0_wgt, 1);
	copy_struct_param((unsigned int) &l0_args, (unsigned int) &DW_args, sizeof(DW_args));
	get_output_dim(&layer0_out);
	pulp_conv_dw_fp32_fw_cl(&DW_args);
	store_output(&layer0_out, 1);

	reset_dim();
	load_input(&layer1_in, 1);
	load_coeff(&layer1_wgt, 1);
	copy_struct_param((unsigned int) &l1_args, (unsigned int) &PW_args, sizeof(PW_args));
	get_output_dim(&layer1_out);
	pulp_conv_pw_fp32_fw_cl(&PW_args);
	store_output(&layer1_out, 1);

	reset_dim();
	load_input(&layer2_in, 1);
	load_coeff(&layer2_wgt, 1);
	get_output_dim(&layer2_out);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);
	store_output(&layer2_out, 1);

	reset_dim();
	load_input(&layer3_in, 1);
	get_output_dim(&layer3_out);
	pulp_relu_fp32_fw_cl(&act_args);
	store_output(&layer3_out, 1);

	reset_dim();
	load_input(&layer4_in, 1);
	load_coeff(&layer4_wgt, 1);
	copy_struct_param((unsigned int) &l4_args, (unsigned int) &DW_args, sizeof(DW_args));
	get_output_dim(&layer4_out);
	pulp_conv_dw_fp32_fw_cl(&DW_args);
	store_output(&layer4_out, 1);

	reset_dim();
	load_input(&layer5_in, 1);
	load_coeff(&layer5_wgt, 1);
	copy_struct_param((unsigned int) &l5_args, (unsigned int) &PW_args, sizeof(PW_args));
	get_output_dim(&layer5_out);
	pulp_conv_pw_fp32_fw_cl(&PW_args);
	store_output(&layer5_out, 1);

	reset_dim();
	load_input(&layer6_in, 1);
	load_coeff(&layer6_wgt, 1);
	get_output_dim(&layer6_out);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);
	store_output(&layer6_out, 1);

	reset_dim();
	load_input(&layer7_in, 1);
	get_output_dim(&layer7_out);
	pulp_relu_fp32_fw_cl(&act_args);
	store_output(&layer7_out, 1);
  

	reset_dim();
	load_input(&layer8_1_in, 1);
	load_coeff(&layer8_1_wgt, 1);
	copy_struct_param((unsigned int) &l8_1_args, (unsigned int) &DW_args, sizeof(DW_args));
	get_output_dim(&layer8_1_out);
	pulp_conv_dw_fp32_fw_cl(&DW_args);
	store_output(&layer8_1_out, 1);
  
  reset_dim();
	load_input(&layer8_2_in, 1);
	load_coeff(&layer8_2_wgt, 1);
	copy_struct_param((unsigned int) &l8_2_args, (unsigned int) &PW_args, sizeof(PW_args));
	get_output_dim(&layer8_2_out);
	pulp_conv_pw_fp32_fw_cl(&PW_args);
	store_output(&layer8_2_out, 1);

	reset_dim();
	load_input(&layer9_in, 1);
	load_coeff(&layer9_wgt, 1);
	copy_struct_param((unsigned int) &l9_args, (unsigned int) &DW_args, sizeof(DW_args));
	get_output_dim(&layer9_out);
	pulp_conv_dw_fp32_fw_cl(&DW_args);
	store_output(&layer9_out, 1);

	reset_dim();
	load_input(&layer10_in, 1);
	load_coeff(&layer10_wgt, 1);
	copy_struct_param((unsigned int) &l10_args, (unsigned int) &PW_args, sizeof(PW_args));
	get_output_dim(&layer10_out);
	pulp_conv_pw_fp32_fw_cl(&PW_args);
	store_output(&layer10_out, 1);

	reset_dim();
	load_input(&layer11_in, 1);
	load_coeff(&layer11_wgt, 1);
	get_output_dim(&layer11_out);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);
	store_output(&layer11_out, 1);

	reset_dim();
	load_input(&layer12_in, 1);
	get_output_dim(&layer12_out);
	pulp_relu_fp32_fw_cl(&act_args);
	store_output(&layer12_out, 1);

	reset_dim();
	load_input(&layer13_in, 1);
	load_coeff(&layer13_wgt, 1);
	copy_struct_param((unsigned int) &l13_args, (unsigned int) &DW_args, sizeof(DW_args));
	get_output_dim(&layer13_out);
	pulp_conv_dw_fp32_fw_cl(&DW_args);
	store_output(&layer13_out, 1);

	reset_dim();
	load_input(&layer14_in, 1);
	load_coeff(&layer14_wgt, 1);
	copy_struct_param((unsigned int) &l14_args, (unsigned int) &PW_args, sizeof(PW_args));
	get_output_dim(&layer14_out);
	pulp_conv_pw_fp32_fw_cl(&PW_args);
	store_output(&layer14_out, 1);

	reset_dim();
	load_input(&layer15_in, 1);
	load_coeff(&layer15_wgt, 1);
	get_output_dim(&layer15_out);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);
	store_output(&layer15_out, 1);

	reset_dim();
	load_input(&layer16_in, 1);
	get_output_dim(&layer16_out);
	pulp_relu_fp32_fw_cl(&act_args);
	store_output(&layer16_out, 1);

	reset_dim();
	load_input(&layer17_in, 1);
	load_coeff(&layer17_wgt, 1);
	get_output_dim(&layer17_out);
	resconn_args.skip = &weight_blob;
	resconn_args.output = &output_blob;
	resconn_args.lout = &input_blob;
	pulp_residualconn_fp32_fw(&resconn_args);
	store_output(&layer17_out, 1);

	reset_dim();
	load_input(&layer18_in, 1);
	load_coeff(&layer18_wgt, 1);
	copy_struct_param((unsigned int) &l18_args, (unsigned int) &linear_args, sizeof(linear_args));
	get_output_dim(&layer18_out);
	pulp_linear_fp32_fw_cl(&linear_args);
	store_output(&layer18_out, 1);

  //printf("\n%.5f\t%.5f",OUT_DATA[0],OUT_DATA[1]);

  reset_dim();
	load_input(&layer19_in, 1);
	get_output_dim(&layer19_out);
	pulp_softmax_fp32_fw_cl(&act_args);
	store_output(&layer19_out, 1);
  //printf("\t%.5f\t%.5f\n",OUT_DATA[0],OUT_DATA[1]);
}

// Backward pass function
void backward() {
    loss_args.output = &output_blob;
    loss_args.target = output_blob.diff;
    loss_args.wr_loss = &loss;
    // FIX THIS!!
    //pi_cl_dma_cmd((uint32_t) (LABEL + idx*NUM_CLASSES), (uint32_t) (output_blob.diff), 4*NUM_CLASSES, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
    pi_cl_dma_cmd_wait(cmd_load);

    //for(int i=0; i<NUM_CLASSES; i++) printf("\n%.5f",*(OUT_DATA + i));
    //for(int i=0; i<NUM_CLASSES; i++) printf("\n%.5f",*(OUT_DIFF + i));
    pulp_CrossEntropyLoss_backward(&loss_args);
    //pulp_MSELoss(&loss_args);
    store_output(&layer18_out, 2);
    //printf("\n%.5f", loss);

    reset_dim();
    load_input(&layer18_in, 1);
    load_coeff(&layer18_wgt, 1);
    load_output(&layer18_out, 2);
    copy_struct_param((unsigned int) &l18_args, (unsigned int) &linear_args, sizeof(l18_args));
    pulp_linear_fp32_bw_cl(&linear_args);
    store_coeff(&layer18_wgt, 0);
    store_input(&layer18_in, 0);
    //PrintBlob(&output_blob, 0);

    reset_dim();
    load_output(&layer17_out, 2);
    store_output(&layer17_in, 0);

    reset_dim();
    load_input(&layer16_in, 1);
    load_output(&layer16_out, 2);
    pulp_relu_fp32_bw_cl(&act_args);
    store_input(&layer16_in, 0);

    reset_dim();
    load_input(&layer15_in, 1);
    load_coeff(&layer15_wgt, 1);
    load_output(&layer15_out, 2);
    copy_struct_param((unsigned int) &l15_args, (unsigned int) &InstNorm_args, sizeof(l15_args));
    pulp_instnorm_fp32_bw_cl(&InstNorm_args);
    store_coeff(&layer15_wgt, 0);
    store_input(&layer15_in, 0);

    reset_dim();
    load_input(&layer14_in, 1);
    load_coeff(&layer14_wgt, 1);
    load_output(&layer14_out, 2);
    copy_struct_param((unsigned int) &l14_args, (unsigned int) &PW_args, sizeof(l14_args));
    pulp_conv_pw_fp32_bw_cl(&PW_args);
    store_coeff(&layer14_wgt, 0);
    store_input(&layer14_in, 0);

    reset_dim();
    load_input(&layer13_in, 1);
    load_coeff(&layer13_wgt, 1);
    load_output(&layer13_out, 2);
    copy_struct_param((unsigned int) &l13_args, (unsigned int) &DW_args, sizeof(l13_args));
    pulp_conv_dw_fp32_bw_cl(&DW_args);
    store_coeff(&layer13_wgt, 0);
    store_input(&layer13_in, 0);

    reset_dim();
    load_input(&layer12_in, 1);
    load_output(&layer12_out, 2);
    pulp_relu_fp32_bw_cl(&act_args);
    store_input(&layer12_in, 0);

    reset_dim();
    load_input(&layer11_in, 1);
    load_coeff(&layer11_wgt, 1);
    load_output(&layer11_out, 2);
    copy_struct_param((unsigned int) &l11_args, (unsigned int) &InstNorm_args, sizeof(l11_args));
    pulp_instnorm_fp32_bw_cl(&InstNorm_args);
    store_coeff(&layer11_wgt, 0);
    store_input(&layer11_in, 0);

    reset_dim();
    load_input(&layer10_in, 1);
    load_coeff(&layer10_wgt, 1);
    load_output(&layer10_out, 2);
    copy_struct_param((unsigned int) &l10_args, (unsigned int) &PW_args, sizeof(l10_args));
    pulp_conv_pw_fp32_bw_cl(&PW_args);
    store_coeff(&layer10_wgt, 0);
    store_input(&layer10_in, 0);

    reset_dim();
    load_input(&layer9_in, 1);
    load_coeff(&layer9_wgt, 1);
    load_output(&layer9_out, 2);
    copy_struct_param((unsigned int) &l9_args, (unsigned int) &DW_args, sizeof(l9_args));
    pulp_conv_dw_fp32_bw_cl(&DW_args);
    store_coeff(&layer9_wgt, 0);
    store_input(&layer9_in, 0);

    reset_dim();
    load_input(&layer9_in, 1);
    load_coeff(&layer9_wgt, 1);
    load_output(&layer9_out, 2);
    copy_struct_param((unsigned int) &l9_args, (unsigned int) &DW_args, sizeof(l9_args));
    pulp_conv_dw_fp32_bw_cl(&DW_args);
    store_coeff(&layer9_wgt, 0);
    store_input(&layer9_in, 0);


    reset_dim();
    load_input(&layer8_2_in, 1);
    load_coeff(&layer8_2_wgt, 1);
    load_output(&layer8_2_out, 2);
    copy_struct_param((unsigned int) &l8_2_args, (unsigned int) &PW_args, sizeof(l8_2_args));
    pulp_conv_pw_fp32_bw_cl(&PW_args);
    store_coeff(&layer8_2_wgt, 0);
    store_input(&layer8_2_in, 0);

    reset_dim();
    load_input(&layer8_1_in, 1);
    load_coeff(&layer8_1_wgt, 1);
    load_output(&layer8_1_out, 2);
    copy_struct_param((unsigned int) &l8_1_args, (unsigned int) &DW_args, sizeof(l8_1_args));
    pulp_conv_dw_fp32_bw_cl(&DW_args);
    store_coeff(&layer8_1_wgt, 0);
    store_input(&layer8_1_in, 0);


    reset_dim();
    load_input(&layer9_in, 0);
    load_output(&layer8_1_in, 0);

    vect_sum_args.op_1 = output_blob.diff;
    vect_sum_args.op_2 = input_blob.diff;
    vect_sum_args.dest = input_blob.diff;

    vect_sum_args.op_1_dims = {input_blob.dim};
    vect_sum_args.op_2_dims = {input_blob.dim};

    vect_sum_args.op_1_dims_len = 1;
    vect_sum_args.op_2_dims_len = 1;

    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &vect_sum_args);

    store_input(&layer9_in, 0);

    reset_dim();
    load_input(&layer7_in, 1);
    load_output(&layer7_out, 2);
    pulp_relu_fp32_bw_cl(&act_args);
    store_input(&layer7_in, 0);

    reset_dim();
    load_input(&layer6_in, 1);
    load_coeff(&layer6_wgt, 1);
    load_output(&layer6_out, 2);
    copy_struct_param((unsigned int) &l6_args, (unsigned int) &InstNorm_args, sizeof(l6_args));
    pulp_instnorm_fp32_bw_cl(&InstNorm_args);
    store_coeff(&layer6_wgt, 0);
    store_input(&layer6_in, 0);

    reset_dim();
    load_input(&layer5_in, 1);
    load_coeff(&layer5_wgt, 1);
    load_output(&layer5_out, 2);
    copy_struct_param((unsigned int) &l5_args, (unsigned int) &PW_args, sizeof(l5_args));
    pulp_conv_pw_fp32_bw_cl(&PW_args);
    store_coeff(&layer5_wgt, 0);
    store_input(&layer5_in, 0);

    reset_dim();
    load_input(&layer4_in, 1);
    load_coeff(&layer4_wgt, 1);
    load_output(&layer4_out, 2);
    copy_struct_param((unsigned int) &l4_args, (unsigned int) &DW_args, sizeof(l4_args));
    pulp_conv_dw_fp32_bw_cl(&DW_args);
    store_coeff(&layer4_wgt, 0);
    store_input(&layer4_in, 0);

    reset_dim();
    load_input(&layer3_in, 1);
    load_output(&layer3_out, 2);
    pulp_relu_fp32_bw_cl(&act_args);
    store_input(&layer3_in, 0);

    reset_dim();
    load_input(&layer2_in, 1);
    load_coeff(&layer2_wgt, 1);
    load_output(&layer2_out, 2);
    copy_struct_param((unsigned int) &l2_args, (unsigned int) &InstNorm_args, sizeof(l2_args));
    pulp_instnorm_fp32_bw_cl(&InstNorm_args);
    store_coeff(&layer2_wgt, 0);
    store_input(&layer2_in, 0);

    reset_dim();
    load_input(&layer1_in, 1);
    load_coeff(&layer1_wgt, 1);
    load_output(&layer1_out, 2);
    copy_struct_param((unsigned int) &l1_args, (unsigned int) &PW_args, sizeof(l1_args));
    pulp_conv_pw_fp32_bw_cl(&PW_args);
    store_coeff(&layer1_wgt, 0);
    store_input(&layer1_in, 0);

    reset_dim();
    load_input(&layer0_in, 1);
    load_coeff(&layer0_wgt, 1);
    load_output(&layer0_out, 2);
    copy_struct_param((unsigned int) &l0_args, (unsigned int) &DW_args, sizeof(l0_args));
    pulp_conv_dw_fp32_bw_cl(&DW_args);
    store_coeff(&layer0_wgt, 0);
}

// Compute loss and output gradient
void compute_loss(int idx)
{
  //printf("\nLOSS");
  loss_args.output = &output_blob;
  loss_args.target = output_blob.diff;
  loss_args.wr_loss = &loss;
  pi_cl_dma_cmd((uint32_t) (LABEL + idx*NUM_CLASSES), (uint32_t) (output_blob.diff), 4*NUM_CLASSES, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
  pi_cl_dma_cmd_wait(cmd_load);
  
  //for(int i=0; i<NUM_CLASSES; i++) printf("\n%.5f",*(OUT_DATA + i));
  //for(int i=0; i<NUM_CLASSES; i++) printf("\n%.5f",*(OUT_DIFF + i));
  pulp_CrossEntropyLoss(&loss_args);
  //pulp_MSELoss(&loss_args);
  store_output(&layer18_out, 2);
  //printf("\n%.5f", loss);
}

// Function to update the network
void update_weights()
{
  struct optim_args opt_l0;
  opt_l0.weights = &weight_blob;
  opt_l0.learning_rate = LEARNING_RATE;
  load_coeff(&layer0_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0);
  store_coeff(&layer0_wgt, 2);

  struct optim_args opt_l1;
  opt_l1.weights = &weight_blob;
  opt_l1.learning_rate = LEARNING_RATE;
  load_coeff(&layer1_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l1);
  store_coeff(&layer1_wgt, 2);

  struct optim_args opt_l2;
  opt_l2.weights = &weight_blob;
  opt_l2.learning_rate = LEARNING_RATE;
  load_coeff(&layer2_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l2);
  store_coeff(&layer2_wgt, 2);

  struct optim_args opt_l4;
  opt_l4.weights = &weight_blob;
  opt_l4.learning_rate = LEARNING_RATE;
  load_coeff(&layer4_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l4);
  store_coeff(&layer4_wgt, 2);

  struct optim_args opt_l5;
  opt_l5.weights = &weight_blob;
  opt_l5.learning_rate = LEARNING_RATE;
  load_coeff(&layer5_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l5);
  store_coeff(&layer5_wgt, 2);

  struct optim_args opt_l6;
  opt_l6.weights = &weight_blob;
  opt_l6.learning_rate = LEARNING_RATE;
  load_coeff(&layer6_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l6);
  store_coeff(&layer6_wgt, 2);

  struct optim_args opt_l8;
  opt_l8.weights = &weight_blob;
  opt_l8.learning_rate = LEARNING_RATE;
  load_coeff(&layer8_1_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l8);
  store_coeff(&layer8_1_wgt, 2);
  load_coeff(&layer8_2_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l8);
  store_coeff(&layer8_2_wgt, 2);

  struct optim_args opt_l9;
  opt_l9.weights = &weight_blob;
  opt_l9.learning_rate = LEARNING_RATE;
  load_coeff(&layer9_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l9);
  store_coeff(&layer9_wgt, 2);

  struct optim_args opt_l10;
  opt_l10.weights = &weight_blob;
  opt_l10.learning_rate = LEARNING_RATE;
  load_coeff(&layer10_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l10);
  store_coeff(&layer10_wgt, 2);

  struct optim_args opt_l11;
  opt_l11.weights = &weight_blob;
  opt_l11.learning_rate = LEARNING_RATE;
  load_coeff(&layer11_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l11);
  store_coeff(&layer11_wgt, 2);

  struct optim_args opt_l13;
  opt_l13.weights = &weight_blob;
  opt_l13.learning_rate = LEARNING_RATE;
  load_coeff(&layer13_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l13);
  store_coeff(&layer13_wgt, 2);

  struct optim_args opt_l14;
  opt_l14.weights = &weight_blob;
  opt_l14.learning_rate = LEARNING_RATE;
  load_coeff(&layer14_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l14);
  store_coeff(&layer14_wgt, 2);

  struct optim_args opt_l15;
  opt_l15.weights = &weight_blob;
  opt_l15.learning_rate = LEARNING_RATE;
  load_coeff(&layer15_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l15);
  store_coeff(&layer15_wgt, 2);

  struct optim_args opt_l18;
  opt_l18.weights = &weight_blob;
  opt_l18.learning_rate = LEARNING_RATE;
  load_coeff(&layer18_wgt, 2);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l18);
  store_coeff(&layer18_wgt, 2);

}



/**
 * DATA VISUALIZATION AND CHECK TOOLS
**/


/**
 * DNN MODEL TRAINING
**/

// Call for a complete training step
void read_img(int idx)
{
  //printf("\nREAD");
  for(int i=0; i<IMG_SIZE; i++) l0_in[i] = INPUT[i + idx*IMG_SIZE];

}
void train_epoch()
{
  //printf("\nTRAIN");
  train_loss=0;
  for(int i=0; i<NUM_TRAIN; i++)
  {
    read_img(i);
    forward();
    compute_loss(i);
    train_loss += loss;
    backward();
    update_weights();
  }
  train_loss = train_loss/NUM_TRAIN;
}
void test_epoch()
{
 // printf("\nTEST");
  test_loss=0;
  for(int i=0; i<NUM_TEST; i++)
  {
    read_img(i + NUM_TRAIN);
    forward();
    compute_loss(i + NUM_TRAIN);
    test_loss += loss;
  }
  test_loss = test_loss/NUM_TEST;
}
// Call for a complete training step
void net_step()
{
  printf("Initializing network AAAAA    ..\n");
  DNN_init();

  for (int epoch=0; epoch<EPOCHS; epoch++)
  {
    train_epoch();
    test_epoch();
    printf("\n%d, train_loss: %.10f, test_loss: %.10f", epoch, train_loss, test_loss);
  }
 printf("\nPappappero");
}

// Functions for DMA managment

void load_coeff(void * src_blob, uint8_t data_diff_both){
	struct blob * b = (struct blob *) src_blob;
	get_weight_dim(src_blob);
	if (data_diff_both == 0) // Load only .diff
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	if (data_diff_both == 1) // Load only .data
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	if (data_diff_both > 1) { // Load both .data and .diff
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	pi_cl_dma_cmd_wait(cmd_load);
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);}
	pi_cl_dma_cmd_wait(cmd_load);} 

void load_input(void * src_blob, uint8_t data_diff_both){
	struct blob * b = (struct blob *) src_blob;
	get_input_dim(src_blob);
	if (data_diff_both == 0) // Load only .diff
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	if (data_diff_both == 1) // Load only .data
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	if (data_diff_both > 1) { // Load both .data and .diff
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	pi_cl_dma_cmd_wait(cmd_load);
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);}
	pi_cl_dma_cmd_wait(cmd_load);} 

void load_output(void * src_blob, uint8_t data_diff_both){
	struct blob * b = (struct blob *) src_blob;
	get_output_dim(src_blob);
	if (data_diff_both == 0) // Load only .diff
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	if (data_diff_both == 1) // Load only .data
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	if (data_diff_both > 1) { // Load both .data and .diff
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	pi_cl_dma_cmd_wait(cmd_load);
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), 4*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);}
	pi_cl_dma_cmd_wait(cmd_load);} 

void store_output(void * dest_blob, uint8_t data_diff_both){ 
	struct blob * b = (struct blob *) dest_blob;
	if (data_diff_both == 0) // Store only .diff
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	if (data_diff_both == 1) // Store only .data
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	if (data_diff_both > 1) { // Store both .data and .diff
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	pi_cl_dma_cmd_wait(cmd_store);
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);}
	pi_cl_dma_cmd_wait(cmd_store);} 

void store_coeff(void * dest_blob, uint8_t data_diff_both){ 
	struct blob * b = (struct blob *) dest_blob;
	if (data_diff_both == 0) // Store only .diff
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	if (data_diff_both == 1) // Store only .data
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	if (data_diff_both > 1) { // Store both .data and .diff
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	pi_cl_dma_cmd_wait(cmd_store);
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);}
	pi_cl_dma_cmd_wait(cmd_store);} 

void store_input(void * dest_blob, uint8_t data_diff_both){ 
	struct blob * b = (struct blob *) dest_blob;
	if (data_diff_both == 0) // Store only .diff
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	if (data_diff_both == 1) // Store only .data
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	if (data_diff_both > 1) { // Store both .data and .diff
	pi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);
	pi_cl_dma_cmd_wait(cmd_store);
	pi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), 4*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);}
	pi_cl_dma_cmd_wait(cmd_store);} 

void get_input_dim(void * b){
	struct blob * src = (struct blob *) b;
	input_blob.C = src->C;
	input_blob.H = src->H;
	input_blob.W = src->W;
	input_blob.dim = src->dim;
	IN_DIFF = BUFF + input_blob.dim;
	W_DATA = BUFF + 2*input_blob.dim;
	update_blob();}

void get_output_dim(void * b){
	struct blob * src = (struct blob *) b;
	output_blob.C = src->C;
	output_blob.H = src->H;
	output_blob.W = src->W;
	output_blob.dim = src->dim;
	OUT_DIFF = BUFF + 2*weight_blob.dim + 2*input_blob.dim + output_blob.dim;
	update_blob();}

void get_weight_dim(void * b){
	struct blob * src = (struct blob *) b;
	weight_blob.C = src->C;
	weight_blob.H = src->H;
	weight_blob.W = src->W;
	weight_blob.dim = src->dim;
	W_DIFF = BUFF + weight_blob.dim + 2*input_blob.dim;
	OUT_DATA = BUFF + 2*weight_blob.dim + 2*input_blob.dim;
	update_blob();}

void copy_struct_param(unsigned int from, unsigned int to, int size){
	pi_cl_dma_cmd(from, to, size, PI_CL_DMA_DIR_EXT2LOC , cmd_load);
	pi_cl_dma_cmd_wait(cmd_load);}

void reset_arguments(){
	linear_args.output = &output_blob;
	linear_args.input = &input_blob;
	linear_args.coeff = &weight_blob;
	conv2d_args.output = &output_blob;
	conv2d_args.input = &input_blob;
	conv2d_args.coeff = &weight_blob;
	PW_args.output = &output_blob;
	PW_args.input = &input_blob;
	PW_args.coeff = &weight_blob;
	DW_args.output = &output_blob;
	DW_args.input = &input_blob;
	DW_args.coeff = &weight_blob;
	act_args.output = &output_blob;
	act_args.input = &input_blob;
	resconn_args.output = &output_blob;
	resconn_args.lout = &input_blob;
	resconn_args.skip = &weight_blob;
	InstNorm_args.output = &output_blob;
	InstNorm_args.input = &input_blob;
	InstNorm_args.coeff = &weight_blob;
}


void update_blob(){
	input_blob.data = IN_DATA;
	input_blob.diff = IN_DIFF;
	output_blob.data = OUT_DATA;
	output_blob.diff = OUT_DIFF;
	weight_blob.data = W_DATA;
	weight_blob.diff = W_DIFF;}

void reset_dim(){
	input_blob.dim = 0;
	weight_blob.dim = 0;
	output_blob.dim = 0;}


#define FLOAT32
void PrintBlob(void * b, int step)
{
    #ifdef FLOAT32
    struct blob * B = (struct blob *) b;
    float * Data = step ? B->data : B->diff;
    #else 
    struct blob_fp16 * B = (struct blob_fp16 *) b;
    fp16 * Data = step ? B->data : B->diff;
    #endif
   
    int widht = B->W;
    int channels = B->C;
    int height = B->H;
    int dim = B->dim;
    int N = dim/(widht*channels*height);
    int indice=0;
    printf("N:%d D:%d C:%d H:%d W:%d \n",N, dim, channels, height, widht);
   
    for(int n=0; n<N ; n++)
    {
        for(int c=0; c<channels; c++)
        {
            printf("Channel %d:\n",c);
            for(int h=0; h<height; h++)
            {
                for(int w=0; w<widht; w++)
                {
                    indice = h*widht + w + c*widht*height + n*height*widht*channels;
                    printf("%.5f, ",Data[indice]*1e5);
                }
                printf("\n");
            }
            printf("_________________________________________________________________________\n");
        }
    }
}