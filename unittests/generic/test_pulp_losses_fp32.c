#ifdef TEST

#include <stdlib.h>
#include <string.h>
#include "unity.h"
#include "test_utils.h"

#include "pmsis.h"
#include "pulp_train_defines.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_losses_fp32.h"
#include "pulp_matmul_fp32.h"

#define DELTA   1e-12

// known model output and label
static float ONES[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, };
static float ONE_HOT[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, };
static float ZEROS[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, };
static float RANDN[] = {1.318617343902588f, 0.747639000415802f, -1.3264806270599365f, -1.2412971258163452f, -0.10280492901802063f, -0.9497631192207336f, 0.6181049346923828f, -0.2384750097990036f, 0.024509044364094734f, -0.49122968316078186f, };
static float OUTPUT[] = {0.39229682087898254f, -0.223564013838768f, -0.31950026750564575f, -1.2050371170043945f, 1.0444635152816772f, -0.6332277059555054f, 0.5731067657470703f, 0.540947437286377f, -0.39190584421157837f, -1.0426788330078125f, };

// expected loss and output gradients
static float EXPECTED_CE_OUTPUT_GRAD_ONES[] = {0.32192298769950867f, -0.28592920303344727f, -0.3512510657310486f, -0.7323958277702332f, 1.5376904010772705f, -0.5259473323822021f, 0.5839117765426636f, 0.5337845683097839f, -0.3965638279914856f, -0.6852221488952637f, };
static float EXPECTED_CE_OUTPUT_GRAD_ONE_HOT[] = {0.13219229876995087f, 0.07140707969665527f, 0.06487489491701126f, 0.026760416105389595f, 0.25376904010772705f, -0.9525947570800781f, 0.15839117765426636f, 0.1533784568309784f, 0.06034361571073532f, 0.03147778660058975f, };
static float EXPECTED_MSE_OUTPUT_GRAD_RANDN[] = {-0.18526411056518555f, -0.19424061477184296f, 0.20139609277248383f, 0.007252001669257879f, 0.2294536828994751f, 0.06330708414316177f, -0.008999633602797985f, 0.1558844894170761f, -0.08328297734260559f, -0.11028983443975449f, };
static float EXPECTED_L1_OUTPUT_GRAD_RANDN[] = {-0.10000000149011612f, -0.10000000149011612f, 0.10000000149011612f, 0.10000000149011612f, 0.10000000149011612f, 0.10000000149011612f, -0.10000000149011612f, 0.10000000149011612f, -0.10000000149011612f, -0.10000000149011612f, };
static float EXPECTED_BERHU_OUTPUT_GRAD_ONES[] = {-0.13779886066913605f, -0.27744749188423157f, -0.2992013394832611f, -0.5f, 0.10000000149011612f, -0.37034016847610474f, -0.10000000149011612f, -0.10409180074930191f, -0.3156195878982544f, -0.4631846845149994f, };

struct TestVector
{
    float* label;
    float* out;
    float* expected_out_grad;
    float expected_loss;
    float alpha;
    int dim;
};

static struct TestVector test_vectors[] = {
    { // CrossEntropy
        .label = ONES,
        .out = OUTPUT,
        .expected_out_grad = EXPECTED_CE_OUTPUT_GRAD_ONES,
        .expected_loss = 25.423044204711914f,
        .dim = 10,
        .alpha = 0.0f, // not used
    },
    { // CrossEntropy
        .label = ONE_HOT,
        .out = OUTPUT,
        .expected_out_grad = EXPECTED_CE_OUTPUT_GRAD_ONE_HOT,
        .expected_loss = 3.0490219593048096f,
        .dim = 10,
        .alpha = 0.0f, // not used
    },
    { // CrossEntropy
        .label = ZEROS,
        .out = OUTPUT,
        .expected_out_grad = ZEROS, // zero gradient expected for zero labels
        .expected_loss = 0.0f,
        .dim = 10,
        .alpha = 0.0f, // not used
    },
    { // MSE
        .label = RANDN,
        .out = OUTPUT,
        .expected_out_grad = EXPECTED_MSE_OUTPUT_GRAD_RANDN,
        .expected_loss = 0.5320070385932922f,
        .dim = 10,
        .alpha = 0.0f, // not used
    },
    { // L1Loss
        .label = RANDN,
        .out = OUTPUT,
        .expected_out_grad = EXPECTED_L1_OUTPUT_GRAD_RANDN,
        .expected_loss = 0.6196852326393127f,
        .dim = 10,
        .alpha = 0.0f, // not used
    },
    { // L1Loss
        .label = RANDN,
        .out = RANDN,
        .expected_out_grad = ZEROS, // zero gradient expected for identical out and label
        .expected_loss = 0.0f, // zero loss expected for identical out and label
        .dim = 10,
        .alpha = 0.0f, // not used
    },
    { // berHuLoss
        .label = ONES,
        .out = OUTPUT,
        .expected_out_grad = EXPECTED_BERHU_OUTPUT_GRAD_ONES,
        .expected_loss = 1.8500397205352783f,
        .dim = 10,
        .alpha = 0.2f,
    },

};

static struct blob out_blob;
static float loss;
static struct loss_args args;
static struct berHu_loss_args berhu_args;
static struct TestVector expected;

void create_test_vectors(int test_vector_id)
{
    expected = test_vectors[test_vector_id];

    out_blob.data = expected.out;
    out_blob.diff = calloc(expected.dim, sizeof(float));
    out_blob.dim = expected.dim;

    if (expected.alpha == 0.0f)
    {
        args.output = &out_blob;
        args.target = expected.label;
        args.wr_loss = &loss;
    }
    else
    {
        //berHu loss test vector
        berhu_args.output = &out_blob;
        berhu_args.target = expected.label;
        berhu_args.wr_loss = &loss;
        berhu_args.alpha = expected.alpha;
    }
}

void free_test_vectors(void)
{
    free(out_blob.diff);
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

TEST_CASE(0) // label = all ones
TEST_CASE(1) // label = one-hot
TEST_CASE(2) // label = all zeros
void test_pulp_CrossEntropyLoss(int test_vector_id)
{
    // for some test vectors the delta is larger
    float delta = test_vector_id != 2 ? 1e-5 : DELTA;

    create_test_vectors(test_vector_id);
    pulp_CrossEntropyLoss(&args);
    TEST_ASSERT_FLOAT_WITHIN(delta, expected.expected_loss, *args.wr_loss);
}

TEST_CASE(0) // label = all ones
TEST_CASE(1) // label = one-hot
TEST_CASE(2) // label = all zeros
void test_pulp_CrossEntropyLoss_backward(int test_vector_id)
{
    create_test_vectors(test_vector_id);
    pulp_CrossEntropyLoss_backward(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.expected_out_grad, args.output->diff, args.output->dim);
}

void test_pulp_MSELoss(void)
{
    create_test_vectors(3);
    pulp_MSELoss(&args);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, expected.expected_loss, *args.wr_loss); // larger delta was needed
}

void test_pulp_MSELoss_backward(void)
{
    create_test_vectors(3);
    pulp_MSELoss_backward(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.expected_out_grad, args.output->diff, args.output->dim);
}

TEST_CASE(4) // label = RANDN
TEST_CASE(5) // label = out = RANDN
void test_pulp_L1Loss(int test_vector_id)
{
    create_test_vectors(test_vector_id);
    pulp_L1Loss(&args);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, expected.expected_loss, *args.wr_loss);
}

TEST_CASE(4) // label = RANDN
TEST_CASE(5) // label = out = RANDN
void test_pulp_L1Loss_backward(int test_vector_id)
{
    create_test_vectors(test_vector_id);
    pulp_L1Loss_backward(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.expected_out_grad, args.output->diff, args.output->dim);
}

void test_pulp_berHuLoss(void)
{
    create_test_vectors(6);
    pulp_berHuLoss(&berhu_args);
    TEST_ASSERT_FLOAT_WITHIN(DELTA, expected.expected_loss, *berhu_args.wr_loss);
}

void test_pulp_berHuLoss_backward(void)
{
    create_test_vectors(6);
    pulp_berHuLoss_backward(&berhu_args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.expected_out_grad, berhu_args.output->diff, berhu_args.output->dim);
}

#endif // TEST
