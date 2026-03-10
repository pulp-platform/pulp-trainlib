#ifdef TEST

#include <stdlib.h>
#include "unity.h"
#include "test_utils.h"

#include "pmsis.h"
#include "pulp_train_defines.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_matmul_fp32.h"

#define DELTA   1e-12

static float A[] = { 1.0, 4.3, 2.1, 0.9, -1.5, 2.9, -1.2, 7.8, 9.3, -2.3, 5.8, 0.6, 1.4, 8.5, -8.6, -8.3, -9.6, 6.7, 5.6, 7.4, 9.6, 6.0, -0.8, 5.6, -7.6, 2.8, -7.1, 8.9, 0.4, -1.7, -4.7, 5.5, -0.9, 1.4, -9.6, 2.4, 2.2, 2.3, 8.9, 3.6, -2.8, -1.3, 4.0, -8.8, 3.3, 3.4, -5.8, -7.4, -3.7, -2.7, 1.4, -1.2, 9.8, -8.0, -5.8, -6.8, 3.1, -4.9, -0.7, -5.1, -6.8, -7.8, 3.1, -7.2, -6.1, -2.6, 6.4, -8.1, 6.8, -8.1, 9.5, -0.6, 9.5, 2.1, 4.8, -9.2, -4.3, -7.6, -4.1, -7.6, -3.6 };
static float B[] = { -1.7, -8.7, 3.8, 1.3, -4.7, 0.5, -8.1, 1.5, 8.6, -3.6, 3.3, -7.4, 4.3, -4.2, -6.3, 1.7, -9.6, 6.6, -9.9, 3.6, -4.6, 4.7, 9.2, -5.0, 1.5, 1.8, 1.4, -5.5, 9.1, -1.1, 6.9, 4.0, -4.1, 6.3, -2.1, 7.6, 1.6, 7.6, 3.9, 4.5, 0.0, 9.1, 2.9, -1.5, 2.1, -9.6, -4.0, 3.2, -4.2, 2.4, -1.4, -7.3, -4.0, 1.4, 1.8, 1.5, 3.1, 3.0, -1.4, 7.9, -2.6, -1.3, 7.8, 6.1, 4.1, -8.0, 8.4, 4.3, 10.0, -7.0, 7.4, -6.8, 2.3, -7.5, 7.0, 6.1, 1.4, -1.9, -8.6, 3.9, -0.9 };
static float Bt[] = { -1.7, -3.6, -9.9, -5.5, 1.6, -9.6, 1.8, 6.1, 2.3, -8.7, 3.3, 3.6, 9.1, 7.6, -4.0, 1.5, 4.1, -7.5, 3.8, -7.4, -4.6, -1.1, 3.9, 3.2, 3.1, -8.0, 7.0, 1.3, 4.3, 4.7, 6.9, 4.5, -4.2, 3.0, 8.4, 6.1, -4.7, -4.2, 9.2, 4.0, 0.0, 2.4, -1.4, 4.3, 1.4, 0.5, -6.3, -5.0, -4.1, 9.1, -1.4, 7.9, 10.0, -1.9, -8.1, 1.7, 1.5, 6.3, 2.9, -7.3, -2.6, -7.0, -8.6, 1.5, -9.6, 1.8, -2.1, -1.5, -4.0, -1.3, 7.4, 3.9, 8.6, 6.6, 1.4, 7.6, 2.1, 1.4, 7.8, -6.8, -0.9 };
static float C[] = { -6.35, -41.33, -36.26, 135.59, 55.36, -7.64, -148.95, 48.31, -23.1, 7.46, 50.99, 47.64, 44.13, -43.35, -131.34, 156.8, -73.46, 30.3, -232.17, 89.71, -165.1, 55.91, 81.1, -150.09, 37.44, -76.25, 90.27, -6.97, -134.86, 160.36, -60.4, -119.99, -8.49, -38.13, -51.83, 125.41, -150.82, -17.66, 37.26, 30.49, 34.34, -158.98, 46.31, -58.37, 154.29, 130.17, -36.77, -50.22, -68.94, -38.52, 167.6, -86.54, 96.56, -120.03, 25.75, -55.76, 63.38, -104.76, 32.82, -92.31, 90.02, 57.11, -139.06, -11.51, 33.93, -92.44, -16.82, 128.81, -29.49, -29.04, 93.08, -191.91, -16.57, -154.75, -9.6, -105.07, -96.33, -124.8, 13.69, -6.05, 35.63 };

static float A_k1[] = { 4.4, 7.3, 9.5, 7.1, -9.8, -2.8, 4.6, -6.6, 0.4 };
static float B_k1[] = { -8.9, -6.0, -9.6, 5.9, -5.5, -3.1, 8.6, 4.1, -9.4 };
static float C_k1[] = { -39.16, -26.4, -42.24, 25.96, -24.2, -13.64, 37.84, 18.04, -41.36, -64.97, -43.8, -70.08, 43.07, -40.15, -22.63, 62.78, 29.93, -68.62, -84.55, -57.0, -91.2, 56.05, -52.25, -29.45, 81.7, 38.95, -89.3, -63.19, -42.6, -68.16, 41.89, -39.05, -22.01, 61.06, 29.11, -66.74, 87.22, 58.8, 94.08, -57.82, 53.9, 30.38, -84.28, -40.18, 92.12, 24.92, 16.8, 26.88, -16.52, 15.4, 8.68, -24.08, -11.48, 26.32, -40.94, -27.6, -44.16, 27.14, -25.3, -14.26, 39.56, 18.86, -43.24, 58.74, 39.6, 63.36, -38.94, 36.3, 20.46, -56.76, -27.06, 62.04, -3.56, -2.4, -3.84, 2.36, -2.2, -1.24, 3.44, 1.64, -3.76 };

static struct matMul_args test_vectors[] = {
    { // 9x9 x 9x9
        .A = A,
        .B = B,
        .C = C,
        .N = 9,
        .K = 9,
        .M = 9,
        .trans_B = 0
    },
    { // 9x9 x 9x9 transposed
        .A = A,
        .B = Bt,
        .C = C,
        .N = 9,
        .K = 9,
        .M = 9,
        .trans_B = 1
    },
    { // 9x1 x 1x9
        .A = A_k1,
        .B = B_k1,
        .C = C_k1,
        .N = 9,
        .K = 1,
        .M = 9,
        .trans_B = 0
    },
    { // 9x1 x 1x9 transposed
        .A = A_k1,
        .B = B_k1,
        .C = C_k1,
        .N = 9,
        .K = 1,
        .M = 9,
        .trans_B = 1
    }
};

static struct matMul_args mm_args;
static struct matMul_args mm_expected;

void create_test_vectors(int transp, int use_k1)
{
    int idx = transp + use_k1*2;
    mm_expected = test_vectors[idx];

    mm_args = mm_expected;
    mm_args.C = malloc(mm_args.N * mm_args.M * sizeof(float));
    set_array_fp32(mm_args.C, mm_args.N * mm_args.M, 0.0);
}

void free_test_vectors(void)
{
    if (mm_args.C != NULL) {
        free(mm_args.C);
        mm_args.C = NULL;
    }
}

// called before each test
void setUp(void)
{
}

// called after each test
void tearDown(void)
{
    free_test_vectors();
}

TEST_CASE(0,0)
TEST_CASE(1,0)
TEST_CASE(0,1)
TEST_CASE(1,1)
void test_pulp_matmul_fp32_mm(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_u2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_u2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
TEST_CASE(0,1)
TEST_CASE(1,1)
void test_pulp_matmul_fp32_mm_add(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    set_array_fp32(mm_args.C, mm_args.N * mm_args.M, 1.0);

    // add 1.0 to expected and result buffer
    float expected_result[mm_expected.N * mm_expected.M];
    for (int i=0; i<(mm_expected.N * mm_expected.M); i++) {
        expected_result[i] = mm_expected.C[i] + 1.0;
    }
    pi_cl_team_fork(NUM_CORES, mm_add, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected_result, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_1x2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_1x2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_1x4(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_1x4, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_1x8(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_1x8, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_2x1(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_2x1, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_4x1(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_8x1(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_8x1, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_2x2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_2x2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_2x4(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_2x4, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_4x2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_unroll_4x4(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x4, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_u2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_u2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_2x1(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_2x1, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_4x1(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x1, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_8x1(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_8x1, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_1x2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_1x2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_1x4(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_1x4, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_1x8(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_1x8, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_2x2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_2x2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_4x2(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x2, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_2x4(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_2x4, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

TEST_CASE(0,0)
TEST_CASE(1,0)
void test_pulp_matmul_fp32_mm_M_unroll_4x4(int transp, int use_k1)
{
    create_test_vectors(transp, use_k1);
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x4, &mm_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, mm_expected.C, mm_args.C, mm_args.N*mm_args.M);
}

#endif // TEST
