// ~~~~~~~~~~ INCLUDES ~~~~~~~~~~
#include "pulp_train.h"

#include "model_components.h"

#include "stats.h"
#include "net.h"

// ~~~~~~~~~~ FORWARD AND MAIN FUNCS ~~~~~~~~~~
// Define forward step function
void forward() {
    // Shape: b, c, h, w
    // patch_embedding fw pass
//    C2D_args.Lpad = PAD_L;
//    C2D_args.Rpad = PAD_R;
//    C2D_args.Upad = PAD_U;
//    C2D_args.Dpad = PAD_D;
//    C2D_args.stride_h = STRIDE_H;
//    C2D_args.stride_w = STRIDE_W;
//    C2D_args.i2c_buffer = im2col_buffer;
//    C2D_args.bt_buffer = bt_buffer;
//    C2D_args.skip_wg_grad = 0;
//    C2D_args.skip_in_grad = 0;
//    C2D_args.HWC = HWC_LAYOUT;
//    C2D_args.opt_matmul_type_fw = MATMUL_TYPE;
//    C2D_args.opt_matmul_type_wg = MATMUL_TYPE;
//    C2D_args.opt_matmul_type_ig = MATMUL_TYPE;
//    C2D_args.USE_IM2COL = IM2COL;
//    C2D_args.USE_DMA_IM2COL = DMA;

    pulp_conv2d_fp32_fw_cl(&patch_embedding_conv2d_args);

    return;
}

// Main function
void net_step() {
    // Initialize performance counters
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif

    // Initialize model components
    printf("ViT test:\n");

    // Forward pass
#ifdef PROF_NET
    START_STATS();
#endif
    forward();
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Perform forward check
    printf("\nFORWARD CHECK: \n");
    // TODO: Get from utils/tensor_checkers.c

    return;
}
