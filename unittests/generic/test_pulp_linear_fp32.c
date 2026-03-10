#ifdef TEST

#include <stdlib.h>
#include <string.h>
#include "unity.h"
#include "test_utils.h"

#include "pmsis.h"
#include "pulp_train_defines.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_linear_fp32.h"
#include "pulp_matmul_fp32.h"

#define DELTA   1e-12


// known parameters
static float WEIGHTS[] = {0.009999999776482582f, 0.019999999552965164f, 0.029999999329447746f, 0.03999999910593033f, 0.05000000074505806f, 0.05999999865889549f, 0.07000000029802322f, 0.07999999821186066f, 0.09000000357627869f, 0.10000000149011612f, 0.10999999940395355f, 0.11999999731779099f, 0.12999999523162842f, 0.14000000059604645f, 0.15000000596046448f, 0.1599999964237213f, 0.17000000178813934f, 0.18000000715255737f, 0.1899999976158142f, 0.20000000298023224f, 0.20999999344348907f, 0.2199999988079071f, 0.23000000417232513f, 0.23999999463558197f, 0.25f, 0.25999999046325684f, 0.27000001072883606f, 0.2800000011920929f, 0.28999999165534973f, 0.30000001192092896f, 0.3100000023841858f, 0.3199999928474426f, 0.33000001311302185f, 0.3400000035762787f, 0.3499999940395355f, 0.36000001430511475f, 0.3700000047683716f, 0.3799999952316284f, 0.38999998569488525f, 0.4000000059604645f, 0.4099999964237213f, 0.41999998688697815f, 0.4300000071525574f, 0.4399999976158142f, 0.44999998807907104f, 0.46000000834465027f, 0.4699999988079071f, 0.47999998927116394f, 0.49000000953674316f, 0.5f, 0.5099999904632568f, 0.5199999809265137f, 0.5299999713897705f, 0.5400000214576721f, 0.550000011920929f, 0.5600000023841858f, 0.5699999928474426f, 0.5799999833106995f, 0.5899999737739563f, 0.6000000238418579f, 0.6100000143051147f, 0.6200000047683716f, 0.6299999952316284f, 0.6399999856948853f};
static float BIASES[] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

// known input and output gradient data
static float INPUT[] = {0.6613521575927734f, 0.266924113035202f, 0.06167725846171379f, 0.6213173270225525f, -0.4519059658050537f, -0.16613022983074188f, -1.522768497467041f, 0.38168391585350037f};
static float OUTPUT_GRAD[] = {-0.14249178767204285f, -0.02044878900051117f, 0.10159420967102051f, 0.223637193441391f, 0.34568023681640625f, 0.4677232503890991f, 0.5897662043571472f, 0.7118092179298401f};

// expected output and expected weight/bias/input gradients
static float EXPECTED_OUTPUT[] = {0.4300328195095062f, 0.9182048439979553f, 1.406376838684082f, 1.894548773765564f, 2.382720947265625f, 2.8708930015563965f, 3.359064817428589f, 3.8472368717193604f};
static float EXPECTED_WEIGHT_GRAD[] = {-0.09423725306987762f, -0.03803449496626854f, -0.008788502775132656f, -0.08853261917829514f, 0.06439288705587387f, 0.0236721932888031f, 0.21698200702667236f, -0.05438682436943054f, -0.013523850589990616f, -0.005458274856209755f, -0.0012612252030521631f, -0.012705187313258648f, 0.00924092996865511f, 0.0033971620723605156f, 0.031138772144913673f, -0.007804973982274532f, 0.06718955188989639f, 0.02711794339120388f, 0.00626605236902833f, 0.06312224268913269f, -0.0459110289812088f, -0.01687786914408207f, -0.15470446646213531f, 0.03877687454223633f, 0.1479029357433319f, 0.059694159775972366f, 0.013793328776955605f, 0.1389496624469757f, -0.10106298327445984f, -0.03715289756655693f, -0.34054768085479736f, 0.08535871654748917f, 0.2286163717508316f, 0.0922703891992569f, 0.021320609375834465f, 0.2147771269083023f, -0.15621496737003326f, -0.057427939027547836f, -0.5263909697532654f, 0.1319405883550644f, 0.3093297779560089f, 0.12484661489725113f, 0.028847888112068176f, 0.2906045615673065f, -0.2113669216632843f, -0.07770296931266785f, -0.7122342586517334f, 0.17852243781089783f, 0.39004313945770264f, 0.15742282569408417f, 0.03637516126036644f, 0.36643195152282715f, -0.26651886105537415f, -0.09797799587249756f, -0.8980773687362671f, 0.22510427236557007f, 0.47075656056404114f, 0.1899990439414978f, 0.0439024418592453f, 0.44225940108299255f, -0.3216708302497864f, -0.11825302988290787f, -1.0839205980300903f, 0.2716861367225647f};
static float EXPECTED_BIAS_GRAD[] = {-0.14249178767204285f, -0.02044878900051117f, 0.10159420967102051f, 0.223637193441391f, 0.34568023681640625f, 0.4677232503890991f, 0.5897662043571472f, 0.7118092179298401f};
static float EXPECTED_INPUT_GRAD[] = {1.0704727172851562f, 1.0932453870773315f, 1.1160180568695068f, 1.1387908458709717f, 1.161563515663147f, 1.1843361854553223f, 1.2071088552474976f, 1.2298816442489624f};

struct TestVector
{
    struct blob in;         // in + expected in grad
    struct blob out;        // out_grad + expected out
    struct blob weight;     // weight + expected weight grad
    struct blob bias;       // bias + expected bias grad
};

static struct TestVector test_vectors[] = {
    { // in = 8, out = 8
        .in = {
            .data = INPUT,
            .diff = EXPECTED_INPUT_GRAD,
            .dim = 8
        },
        .out = {
            .diff = OUTPUT_GRAD,
            .data = EXPECTED_OUTPUT,
            .dim = 8
        },
        .weight = {
            .data = WEIGHTS,
            .diff = EXPECTED_WEIGHT_GRAD,
            .dim = 8*8
        },
        .bias = {
            .data = BIASES,
            .diff = EXPECTED_BIAS_GRAD,
            .dim = 8
        }
    },
};

// create a deep copy of a test vector
void copy_test_vector(const struct TestVector *src, struct TestVector* dst)
{
    *dst = *src;

    dst->in.data = malloc(src->in.dim * sizeof(float));
    dst->in.diff = malloc(src->in.dim * sizeof(float));
    dst->out.data = malloc(src->out.dim * sizeof(float));
    dst->out.diff = malloc(src->out.dim * sizeof(float));
    dst->weight.data = malloc(src->weight.dim * sizeof(float));
    dst->weight.diff = malloc(src->weight.dim * sizeof(float));
    dst->bias.data = malloc(src->bias.dim * sizeof(float));
    dst->bias.diff = malloc(src->bias.dim * sizeof(float));
    memcpy(dst->in.data, src->in.data, src->in.dim * sizeof(float));
    memcpy(dst->in.diff, src->in.diff, src->in.dim * sizeof(float));
    memcpy(dst->out.diff, src->out.diff, src->out.dim * sizeof(float));
    memcpy(dst->out.data, src->out.data, src->out.dim * sizeof(float));
    memcpy(dst->weight.data, src->weight.data, src->weight.dim * sizeof(float));
    memcpy(dst->weight.diff, src->weight.diff, src->weight.dim * sizeof(float));
    memcpy(dst->bias.data, src->bias.data, src->bias.dim * sizeof(float));
    memcpy(dst->bias.diff, src->bias.diff, src->bias.dim * sizeof(float));
}

// free a copied test vector
void free_test_vector(struct TestVector *v)
{
    free(v->in.data);
    free(v->in.diff);
    free(v->out.data);
    free(v->out.diff);
    free(v->weight.data);
    free(v->weight.diff);
    free(v->bias.data);
    free(v->bias.diff);
}

void create_test_vectors(struct Linear_args* args, struct TestVector* expected)
{
    static struct TestVector a;
    struct TestVector v = test_vectors[0];

    // create two deep copies of the test vector so that we don't overwrite the
    // original one. One copy to populate args and one as expected values
    copy_test_vector(&v, expected);
    copy_test_vector(&v, &a);

    // set some buffers to zero for the argument blobs
    set_array_fp32(a.in.diff, a.in.dim, 0.0);
    set_array_fp32(a.out.data, a.out.dim, 0.0);
    set_array_fp32(a.weight.diff, a.weight.dim, 0.0);
    set_array_fp32(a.bias.diff, a.bias.dim, 0.0);

    // potulate linear parameter struct
    args->input = &a.in;
    args->coeff = &a.weight;
    args->bias = &a.bias;
    args->output = &a.out;
    args->skip_wg_grad = 0;
    args->skip_in_grad = 0;
    args->opt_matmul_type_fw = 0;
    args->opt_matmul_type_wg = 0;
    args->opt_matmul_type_ig = 0;
    args->use_biases = 1;
}

void free_test_vectors(struct Linear_args* args, struct TestVector* expected)
{
    free(args->input->data);
    free(args->input->diff);
    free(args->output->data);
    free(args->output->diff);
    free(args->coeff->data);
    free(args->coeff->diff);
    free(args->bias->data);
    free(args->bias->diff);
    free_test_vector(expected);
}

// called before each test
void setUp(void)
{
}

// called after each test
void tearDown(void)
{
}

void test_pulp_linear_fp32_fw_cl(void)
{
    struct Linear_args args;
    struct TestVector expected;
    create_test_vectors(&args, &expected);

    pulp_linear_fp32_fw_cl(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.out.data, args.output->data, args.output->dim);

    free_test_vectors(&args, &expected);
}

void test_pulp_linear_fp32_bw_param_grads_cl(void)
{
    struct Linear_args args;
    struct TestVector expected;
    create_test_vectors(&args, &expected);

    pulp_linear_fp32_bw_param_grads_cl(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.weight.diff, args.coeff->diff, args.coeff->dim);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.bias.diff, args.bias->diff, args.bias->dim);

    free_test_vectors(&args, &expected);
}

void test_pulp_linear_fp32_bw_input_grads_cl(void)
{
    struct Linear_args args;
    struct TestVector expected;
    create_test_vectors(&args, &expected);

    pulp_linear_fp32_bw_input_grads_cl(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.in.diff, args.input->diff, args.input->dim);

    free_test_vectors(&args, &expected);
}

TEST_CASE(0, 0) // calculate both weight and input gradients
TEST_CASE(1, 0) // skip weight gradient calculation
TEST_CASE(0, 1) // skip input gradient calculation
void test_pulp_linear_fp32_bw_cl(int skip_wg_grad, int skip_in_grad)
{
    struct Linear_args args;
    struct TestVector expected;
    create_test_vectors(&args, &expected);

    // test skip grad calculations
    args.skip_wg_grad = skip_wg_grad;
    args.skip_in_grad = skip_in_grad;

    if (skip_wg_grad) {
        set_array_fp32(expected.weight.diff, expected.weight.dim, 0);
        set_array_fp32(expected.bias.diff, expected.bias.dim, 0);
    }
    if (skip_in_grad) {
        set_array_fp32(expected.in.diff, expected.in.dim, 0);
    }

    pulp_linear_fp32_bw_cl(&args);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.in.diff, args.input->diff, args.input->dim);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.weight.diff, args.coeff->diff, args.coeff->dim);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(DELTA, expected.bias.diff, args.bias->diff, args.bias->dim);

    free_test_vectors(&args, &expected);
}


#endif // TEST
