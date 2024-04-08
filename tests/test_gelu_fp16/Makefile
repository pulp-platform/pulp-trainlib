APP = gelu_fp16

# User settings
IN_H?=7 
IN_W?=7 
IN_CH?=160
OUT_CH?=160
NUM_CORES?=8
STEP?='FORWARD' # Possible steps: 'FORWARD', 'BACKWARD'
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=3
NUM_MATMULS?=24		# When profiling with multiple matmul algorithms
NUM_SIZES?=3		# When profiling multiple sizes of the network
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c

DATA_TYPE?='fp16'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g
APP_CFLAGS += -DFABRIC 
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DMEMOCC_COMP
APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
#APP_CFLAGS += -DDEBUG
APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 ./utils/GM.py --step $(STEP) --in_width $(IN_W) --in_height $(IN_H) --ch_in ${IN_CH} --ch_out ${OUT_CH}

profile_all_optim:
	python3 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --in_width $(IN_W) --in_height $(IN_H) --ch_in ${IN_CH} --ch_out ${OUT_CH} --n_heads $(N_HEADS) --att_dim $(ATT_DIM)

profile_all_sizes:
	python3 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE}

include $(RULES_DIR)/pmsis_rules.mk
