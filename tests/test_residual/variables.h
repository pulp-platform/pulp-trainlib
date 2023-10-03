#ifdef FLOAT32
PI_L1 struct blob input; 
PI_L1 float input_diff[CI*WI*HI];

struct blob expected_input;

PI_L1 struct blob output_conv1;
PI_L1 float output_conv1_data[CO*HO*WO];
PI_L1 float output_conv1_diff[CO*WO*HO];

PI_L1 struct blob coeff_conv1;
PI_L1 float coeff_conv1_data[KER_DIM];
PI_L1 float coeff_conv1_diff[KER_DIM];

PI_L1 struct blob output_residual;
PI_L1 float output_residual_data[CO*HO*WO];
PI_L1 float output_residual_diff[CO*WO*HO];

 struct blob relu_output;
 float relu_output_data[CO*WO*HO];
 float relu_output_diff[CO*WO*HO];

 PI_L1 struct Conv2D_args conv1_args; 
 struct act_args relu_args; 
 PI_L1 struct SkipConn_args residual_args;

 struct loss_args loss_args;

 float calculated_loss=0;

 PI_L1 float im2col_buff[I2C_SIZE];
 PI_L1 float bt_buffer[KER_SIZE*KER_SIZE*CI*CO];




#else //FLOAT16
PI_L1 struct blob_fp16 input; 
PI_L1 fp16 input_diff[CI*WI*HI];

PI_L1 struct blob_fp16 expected_input;

PI_L1 struct blob_fp16 output_conv1;
PI_L1 fp16 output_conv1_data[CO*HO*WO];
PI_L1 fp16 output_conv1_diff[CO*WO*HO];

PI_L1 struct blob_fp16 coeff_conv1;
PI_L1 fp16 coeff_conv1_data[KER_DIM];
PI_L1 fp16 coeff_conv1_diff[KER_DIM];

PI_L1 struct blob_fp16 output_residual;
PI_L1 fp16 output_residual_data[CO*HO*WO];
PI_L1 fp16 output_residual_diff[CO*WO*HO];

 struct blob_fp16 relu_output;
 fp16 relu_output_data[CO*WO*HO];
 fp16 relu_output_diff[CO*WO*HO];

 PI_L1 struct Conv2D_args_fp16 conv1_args; 
 struct act_args_fp16 relu_args; 
 PI_L1 struct SkipConn_args_fp16 residual_args;

 struct loss_args_fp16 loss_args;

 fp16 calculated_loss=0;

 PI_L1 fp16 im2col_buff[I2C_SIZE];
 PI_L1 fp16 bt_buffer[KER_SIZE*KER_SIZE*CI*CO];
#endif



void prepare_data()
{
    printf("\nPreparing Data...\n");


    expected_input.diff = expected_input_diff;
    expected_input.C = CI;
    expected_input.W = WI;
    expected_input.H = HI;
    expected_input.dim = CI*HI*WI;

    // Input
    input.C = CI;
    input.H = HI;
    input.W = WI;
    input.dim = CI*HI*WI;
    input.data = input_data;
    input.diff = input_diff;

    //Assigning Conv2d arguments
    conv1_args.input = &input;

    coeff_conv1.C = CI;
    coeff_conv1.H = KER_SIZE;
    coeff_conv1.W = KER_SIZE;
    coeff_conv1.dim = KER_DIM;
    coeff_conv1.data = coeff_conv1_data;
    coeff_conv1.diff = coeff_conv1_diff;
   
    conv1_args.coeff = &coeff_conv1;

    output_conv1.C = CO;
    output_conv1.H = HO;
    output_conv1.W = WO;
    output_conv1.dim = CO*HO*WO;
    output_conv1.data = output_conv1_data;
    output_conv1.diff = output_conv1_diff;
   
    conv1_args.output = &output_conv1;

    conv1_args.Upad = PAD_SIZE;
    conv1_args.Dpad = PAD_SIZE;
    conv1_args.Lpad = PAD_SIZE;
    conv1_args.Rpad = PAD_SIZE;
    
    conv1_args.stride_h=1;
    conv1_args.stride_w=1;

    conv1_args.bt_buffer = bt_buffer;
    conv1_args.i2c_buffer = im2col_buff;

    conv1_args.skip_in_grad = 0;
    conv1_args.HWC = HWC;
    conv1_args.opt_matmul_type_fw = MATMUL_TYPE;
    conv1_args.opt_matmul_type_wg = MATMUL_TYPE;
    conv1_args.opt_matmul_type_ig = MATMUL_TYPE;
    conv1_args.USE_IM2COL = USE_IM2COL;
    conv1_args.USE_DMA_IM2COL = USE_DMA;

   
    //Assigning SkipCon args
    residual_args.lout = &output_conv1;
    residual_args.skip = &input;

    output_residual.data = output_residual_data;
    output_residual.diff = output_residual_diff;
    output_residual.C = CO;
    output_residual.W = WO;
    output_residual.H = HO;
    output_residual.dim = CO*WO*HO;

    residual_args.output = &output_residual;
    residual_args.skip_in_grad = 0;
    //RELU

    relu_output.data = relu_output_data;
    relu_output.diff = relu_output_diff;
    relu_output.C = CO;
    relu_output.W = WO;
    relu_output.H = HO;
    relu_output.dim = CO*WO*HO;

    relu_args.input = residual_args.output;
    relu_args.output = &relu_output;

    loss_args.target = labels;
    loss_args.output = &relu_output;
    loss_args.wr_loss = &calculated_loss;

}
