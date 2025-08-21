APP = linear_test

# User settings
IN_CH?=8
OUT_CH?=8
NUM_CORES?=8
STEP?='FORWARD' # Possible steps: 'FORWARD', 'BACKWARD_GRAD', 'BACKWARD_ERROR'
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=0
USE_BIASES_LINEAR?=0	# Allocate biases (1) or not (0)
NUM_MATMULS?=24		# When profiling with multiple matmul algorithms
NUM_SIZES?=3		# When profiling multiple sizes of the network
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c

DATA_TYPE?='fp32'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 
#APP_CFLAGS += -Wno-error
APP_CFLAGS += -DFABRIC 
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DMEMOCC_COMP
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
APP_CFLAGS += -DUSE_BIASES_LINEAR=${USE_BIASES_LINEAR}
APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 utils/GM.py --in_size $(IN_CH) --out_size $(OUT_CH) --step $(STEP) --use_bias $(USE_BIASES_LINEAR)

profile_all_optim:
	python3 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --in_size ${IN_CH} --out_size ${OUT_CH} --use_bias ${USE_BIASES_LINEAR}

profile_all_sizes:
	python3 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE} --use_bias ${USE_BIASES_LINEAR}

include $(RULES_DIR)/pmsis_rules.mk
