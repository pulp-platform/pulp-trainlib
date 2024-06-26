#include "pmsis.h"
#include "pulp_train.h"

#include "net.h"
#include "stats.h"
#include "cordic_data.h"
#include "math.h"

#define N_PRINT 10

PI_L1 float cosines[N_TEST];
PI_L1 float sines[N_TEST];

PI_L1 fp16 cosines_fp16[N_TEST];
PI_L1 fp16 sines_fp16[N_TEST];

void net_step () {
    printf("Using standard math.h cosf and sinf:\n");
    
    INIT_STATS();
    PRE_START_STATS();
    START_STATS();

    for(int i=0;i<N_TEST;i++){
        cosines[i] = cosf(gm_angles[i]);
        sines[i] = sinf(gm_angles[i]);
    }

    STOP_STATS();

    printf("\n%10s %10s %10s %10s %10s\n", "angle", "gm_cos", "cosf", "gm_sin", "sinf");
    for(int i=0;i<N_PRINT;i++)
        printf("%10f %10f %10f %10f %10f\n", gm_angles[i], gm_cos[i], cosines[i], gm_sin[i], sines[i]);
    printf("...\n");

    printf("\n\nUsing cordic cos and sin in fp32: \n");

    INIT_STATS();
    PRE_START_STATS();
    START_STATS();

    for(int i=0;i<N_TEST;i++)
        cordic_cos_sin_fp32(gm_angles[i], &cosines[i], &sines[i]);

    STOP_STATS();

    float diff_c_tot = 0, diff_s_tot = 0;

    printf("\n%10s %10s %10s %10s %10s\n", "angle", "gm_cos", "cordic cos", "gm_sin", "cordic sin");
    for(int i=0;i<N_TEST;i++){
        float diff_c = gm_cos[i] - cosines[i];
        float diff_s = gm_sin[i] - sines[i];
        if(diff_c > 0) diff_c_tot += diff_c;
        else diff_c_tot -= diff_c;

        if(diff_s > 0) diff_s_tot += diff_s;
        else diff_s_tot -= diff_s;

        if(i<N_PRINT)
            printf("%10f %10f %10f %10f %10f\n", gm_angles[i], gm_cos[i], cosines[i], gm_sin[i], sines[i]);
    }
    if(N_PRINT < N_TEST)
        printf("...\n\n");

    printf("Mean Error cos in fp32: %f\n", diff_c_tot / N_TEST);
    printf("Mean error sin in fp32: %f\n", diff_s_tot / N_TEST);



    printf("\n\nUsing cordic cos and sin in fp16: \n");

    INIT_STATS();
    PRE_START_STATS();
    START_STATS();

    for(int i=0;i<N_TEST;i++)
        cordic_cos_sin_fp16((fp16)gm_angles[i], &cosines_fp16[i], &sines_fp16[i]);

    STOP_STATS();

    fp16 diff_c_tot_fp16 = 0, diff_s_tot_fp16 = 0;

    printf("\n%10s %10s %10s %10s %10s\n", "angle", "gm_cos", "cordic cos", "gm_sin", "cordic sin");
    for(int i=0;i<N_TEST;i++){
        fp16 diff_c_fp16 = gm_cos[i] - cosines_fp16[i];
        fp16 diff_s_fp16 = gm_sin[i] - sines_fp16[i];
        if(diff_c_fp16 > 0) diff_c_tot_fp16 += diff_c_fp16;
        else diff_c_tot_fp16 -= diff_c_fp16;

        if(diff_s_fp16 > 0) diff_s_tot_fp16 += diff_s_fp16;
        else diff_s_tot_fp16 -= diff_s_fp16;

        if(i<N_PRINT)
            printf("%10f %10f %10f %10f %10f\n", gm_angles[i], gm_cos[i], cosines_fp16[i], gm_sin[i], sines_fp16[i]);
    }
    if(N_PRINT < N_TEST)
        printf("...\n\n");

    printf("Mean Error cos in fp16: %f\n", diff_c_tot_fp16 / N_TEST);
    printf("Mean Error sin in fp16: %f\n", diff_s_tot_fp16 / N_TEST);


    printf("\n\nCycles analys:\n");

    INIT_STATS();
    PRE_START_STATS();
    START_STATS();

    printf("Angle: \t\t tmp_m: \t tmp_c_fp32  tmp_c_fp16:\n");

    float angle = 0.1f;
    float cos, sin, cos_c, sin_c;
    fp16 cos_c_fp16, sin_c_fp16;   
    unsigned long tmp_c, tmp_m, tmp_fp16;
    for(int i=0;i<100;i++){
        tmp_m = pi_perf_read(PI_PERF_CYCLES);
        cos = cosf(angle);
        sin = sinf(angle);
        tmp_m = pi_perf_read(PI_PERF_CYCLES) - tmp_m;

        tmp_c = pi_perf_read(PI_PERF_CYCLES);
        cordic_cos_sin_fp32(angle, &cos_c, &sin_c);
        tmp_c = pi_perf_read(PI_PERF_CYCLES) - tmp_c;

        tmp_fp16 = pi_perf_read(PI_PERF_CYCLES);
        cordic_cos_sin_fp16((fp16)angle, &cos_c_fp16, &sin_c_fp16);
        tmp_fp16 = pi_perf_read(PI_PERF_CYCLES) - tmp_fp16;

        printf("%10f   %lu  \t %lu  \t\t %lu\n", angle, tmp_m, tmp_c, tmp_fp16);

        angle *= 1.1f;
    }

    printf("\n\nTest finished\n\n");
    return;
}