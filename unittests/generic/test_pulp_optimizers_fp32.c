#ifdef TEST

#include <stdlib.h>
#include <string.h>
#include "unity.h"
#include "test_utils.h"

#include "pmsis.h"
#include "pulp_train_defines.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_optimizers_fp32.h"
#include "pulp_matmul_fp32.h"

#define DELTA   1e-12

// known initial weights + biases and their gradients
static float WEIGHTS_BEFORE[] = {0.4414065182209015f, 0.47920528054237366f, -0.1352572739124298f, 0.5303604602813721f, -0.1264995038509369f, 0.11650390923023224f};
static float BIASES_BEFORE[] = {-0.28108587861061096f, 0.3390677869319916f};
static float WEIGHTS_GRAD[] = {0.2129594087600708f, 0.4259188175201416f, 0.6388782262802124f, 1.4659409523010254f, 2.931881904602051f, 4.397822856903076f};
static float BIASES_GRAD[] =  {0.2129594087600708f, 1.4659409523010254f};

// expected updated weights and biases after one SGD step with lr=0.1
static float WEIGHTS_AFTER[] = {0.4201105833053589f, 0.43661341071128845f, -0.1991450935602188f, 0.3837663531303406f, -0.4196876883506775f, -0.32327839732170105f};
static float BIASES_AFTER[] = {-0.30238181352615356f, 0.19247369468212128f};
static float WEIGHTS_AFTER_WEIGHT_DECAY[] = {0.41966918110847473f, 0.4361341893672943f, -0.19900983572006226f, 0.38323599100112915f, -0.4195612072944641f, -0.32339486479759216f};
static float BIASES_AFTER_WEIGHT_DECAY[] = {-0.3021007478237152f, 0.19213461875915527f};

struct TestVector
{
    struct blob weights;
    struct blob biases;
    float learning_rate;
    float weight_decay_lambda;
    int use_biases;
    float* expected_weights;
    float* expected_biases;
};

static struct TestVector test_vectors[] = {
    { // no biases, no weight decay
        .weights = {
            .data = WEIGHTS_BEFORE,
            .diff = WEIGHTS_GRAD,
            .dim = 6
        },
        .biases = {
            .data = BIASES_BEFORE,
            .diff = BIASES_GRAD,
            .dim = 2
        },
        .learning_rate = 0.1f,
        .weight_decay_lambda = 0.0f,
        .use_biases = 0,
        .expected_weights = WEIGHTS_AFTER,
        .expected_biases = BIASES_BEFORE, // biases should remain unchanged
    },
    { // no biases, with weight decay
        .weights = {
            .data = WEIGHTS_BEFORE,
            .diff = WEIGHTS_GRAD,
            .dim = 6
        },
        .biases = {
            .data = BIASES_BEFORE,
            .diff = BIASES_GRAD,
            .dim = 2
        },
        .learning_rate = 0.1f,
        .weight_decay_lambda = 0.01f,
        .use_biases = 0,
        .expected_weights = WEIGHTS_AFTER_WEIGHT_DECAY,
        .expected_biases = BIASES_BEFORE, // biases should remain unchanged
    },
    { // with biases, no weight decay
        .weights = {
            .data = WEIGHTS_BEFORE,
            .diff = WEIGHTS_GRAD,
            .dim = 6
        },
        .biases = {
            .data = BIASES_BEFORE,
            .diff = BIASES_GRAD,
            .dim = 2
        },
        .learning_rate = 0.1f,
        .weight_decay_lambda = 0.0f,
        .use_biases = 1,
        .expected_weights = WEIGHTS_AFTER,
        .expected_biases = BIASES_AFTER,
    },
    { // with biases, with weight decay
        .weights = {
            .data = WEIGHTS_BEFORE,
            .diff = WEIGHTS_GRAD,
            .dim = 6
        },
        .biases = {
            .data = BIASES_BEFORE,
            .diff = BIASES_GRAD,
            .dim = 2
        },
        .learning_rate = 0.1f,
        .weight_decay_lambda = 0.01f,
        .use_biases = 1,
        .expected_weights = WEIGHTS_AFTER_WEIGHT_DECAY,
        .expected_biases = BIASES_AFTER_WEIGHT_DECAY,
    }
};

static struct blob weights, biases;
static struct optim_args args;
static struct TestVector expected;

void create_test_vectors(int use_biases, int use_weight_decay)
{
    int idx = 2 * use_biases + use_weight_decay; // index to select the correct test vector based on the flags
    expected = test_vectors[idx];

    weights = expected.weights;
    weights.data = calloc(weights.dim, sizeof(float));  
    weights.diff = calloc(weights.dim, sizeof(float));
    memcpy(weights.data, expected.weights.data, weights.dim * sizeof(float));
    memcpy(weights.diff, expected.weights.diff, weights.dim * sizeof(float));
    biases = expected.biases;
    biases.data = calloc(biases.dim, sizeof(float));
    biases.diff = calloc(biases.dim, sizeof(float));
    memcpy(biases.data, expected.biases.data, biases.dim * sizeof(float));
    memcpy(biases.diff, expected.biases.diff, biases.dim * sizeof(float));

    args.weights = &weights;
    args.biases = &biases;
    args.learning_rate = expected.learning_rate;
    args.weight_decay_lambda = expected.weight_decay_lambda;
    args.use_biases = expected.use_biases;
}

void free_test_vectors(void)
{
    free(weights.data);
    free(weights.diff);
    free(biases.data);
    free(biases.diff);
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

TEST_CASE(0, 0) // no biases, no weight decay
TEST_CASE(0, 1) // no biases, with weight decay
TEST_CASE(1, 0) // with biases, no weight decay
TEST_CASE(1, 1) // with biases, with weight decay
void test_pulp_gradient_descent_fp32(int use_biases, int use_weight_decay)
{
    create_test_vectors(use_biases, use_weight_decay);
    pulp_gradient_descent_fp32(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.expected_weights, args.weights->data, args.weights->dim);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.expected_biases, args.biases->data, args.biases->dim);
    // also ensure that grads are not modified by the optimizer
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.weights.diff, args.weights->diff, args.weights->dim);
}



#endif // TEST
