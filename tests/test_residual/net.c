#include "pmsis.h"
#include "pulp_train.h"

#include "data.h"
#include "init_defines.h"

#include "net.h"

#include "stats.h"
#include <math.h>

#include "variables.h"

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
    int N = B->dim/(widht*channels*height);
    int indice=0;
    printf("N:%d C:%d H:%d W:%d \n",N, channels, height, widht);
   
    for(int n=0; n<N ; n++)
    {
        for(int c=0; c<channels; c++)
        {
            printf("Channel %d:\n",c);
            for(int h=0; h<height; h++)
            {
                for(int w=0; w<widht; w++)
                {
                    indice = HWC? h*channels*widht + w*channels + c +n*height*widht*channels : h*widht + w + c*widht*height + n*height*widht*channels;
                    printf("%.9f, ",Data[indice]);
                }
                printf("\n");
            }
            printf("\n\n");
        }
    }
}

void forward()
{

    #ifdef FLOAT32 
    pulp_conv2d_fp32_fw_cl(&conv1_args);
    pulp_residualconn_fp32_fw(&residual_args);
    pulp_relu_fp32_fw_cl(&relu_args);
    pulp_MSELoss(&loss_args);
    pulp_MSELoss_backward(&loss_args);
    
    #else //FLOAT16
    pulp_conv2d_fp16_fw_cl(&conv1_args);
    pulp_residualconn_fp16_fw(&residual_args);
    pulp_relu_fp16_fw_cl(&relu_args);
    pulp_MSELoss_fp16(&loss_args);
    pulp_MSELoss_backward_fp16(&loss_args);
    #endif

    #ifdef FORWARD
    #ifdef PROF_NET
    STOP_STATS();
    #endif
    #endif

    calculated_loss = *(loss_args.wr_loss); 

    #ifdef DEBUG
    printf("Input:\n");
    PrintBlob(conv1_args.input, 1);
    printf("Output Conv1:\n");
    PrintBlob(conv1_args.output, 1);
    printf("Output Residual:\n");
    PrintBlob(residual_args.output, 1);
    printf("Output ReLU:\n");
    PrintBlob(relu_args.output, 1);
    #endif

    //Error calculation
    printf("\nExpected Loss: %.5f, Calculated Loss: %.5f\n", expected_loss, calculated_loss);
    #ifdef FLOAT32
    printf("Error: %.4fppm\n\n",1000000*fabs(1 - expected_loss/calculated_loss));
    #else
    printf("Error: %.4f%%\n\n",100*fabs(1 - expected_loss/calculated_loss));
    #endif
}

void backward()
{
 
    #ifdef FLOAT32
    pulp_relu_fp32_bw_cl(&relu_args);
    pulp_residualconn_fp32_bw(&residual_args);
    pulp_conv2d_fp32_bw_input_grads_cl(&conv1_args);
    pulp_sumnode_fp32_bw(&residual_args);

    #else //FLOAT16
    pulp_relu_fp16_bw_cl(&relu_args);
    pulp_residualconn_fp16_bw(&residual_args);
    pulp_conv2d_fp16_bw_input_grads_cl(&conv1_args);
    pulp_sumnode_fp16_bw(&residual_args);   
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef DEBUG
    printf("Relu Gradient:\n");
    PrintBlob(relu_args.input, 0);
    printf("Residual Gradient:\n");
    PrintBlob(residual_args.lout, 0);
    printf("Calculated Input Gradient:\n");
    PrintBlob(conv1_args.input, 0);
    printf("Expected Input Gradient:\n");
    PrintBlob(&expected_input, 0);
    #endif

    #ifdef FLOAT32
    verify_tensor(input.diff, expected_input.diff, input.dim, (float) 1e-5);
    #else
    verify_tensor_fp16(input.diff, expected_input.diff, input.dim, (fp16) 1e-5);
    #endif

    //Error calculation
    float ppm=0;
    for(int i=0; i<input.dim; i++)
    {
        ppm += ABS(1 - expected_input_diff[i]/input.diff[i]);
    
    }
    ppm = 1000000*ppm/input.dim;

   printf("Average ppm difference: %.3fppm (%.3f%% average error)\n", ppm, ppm*0.0001);

}


void net_step()
{
    #ifdef FLOAT32
    printf("\nData type is float32\n");
    #else
    printf("\nData type is float16\n");
    #endif

    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    prepare_data();

    #ifdef BACKWARD
    forward();
    #endif

    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifdef FORWARD
    forward();
    #else
    backward();
    #endif

}
