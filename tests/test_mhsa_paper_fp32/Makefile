APP = mhsa_fp32

# User settings
IN_H?=64 # Sequence Length (L)
IN_W?=512 # Token Size (E)
ATT_DIM?=512 #Hidden dimension (F)
N_HEADS?=4 # Number of heads (H = F / N_HEADS)


TILE_H?=4
TILE_W?=4
TILE_DIM?=16

# Projections are done on F x L
TILE_H_P?=16
TILE_W_P?=32
TILE_DIM_P?=512

# Softmax are done on L x L
TILE_H_SM?=64
TILE_W_SM?=64
TILE_DIM_SM?=4096

# Transpose are done on L x H
TILE_H_TR?=64
TILE_W_TR?=128
TILE_DIM_TR?=8192

# Att * V are done on H x L
TILE_H_ATTV?=128
TILE_W_ATTV?=64
TILE_DIM_ATTV?=8192

# Output Transpose are done on L x F
TILE_H_OUT_TR?=64
TILE_W_OUT_TR?=128
TILE_DIM_OUT_TR?=8192

IN_CH?=1
OUT_CH?=1


IN_CH?=1
OUT_CH?=1

NUM_CORES?=8
STEP?='FORWARD' # Possible steps: 'FORWARD', 'BACKWARD'

APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=10
NUM_MATMULS?=24		# When profiling with multiple matmul algorithms
NUM_SIZES?=3		# When profiling multiple sizes of the network
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_random.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_nonorm_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_mhsa_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp32.c

DATA_TYPE?='fp32'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g
APP_CFLAGS += -DFABRIC 
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DMEMOCC_COMP
APP_CFLAGS += -mhwloopalign
#APP_CFLAGS += -mnohwloop
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
#APP_CFLAGS += -DDEBUG
APP_LDFLAGS += -lm 

APP_CFLAGS += -DTILE_H=$(TILE_H)
APP_CFLAGS += -DTILE_W=$(TILE_W)
APP_CFLAGS += -DTILE_DIM=$(TILE_DIM)

APP_CFLAGS += -DTILE_H_P=$(TILE_H_P)
APP_CFLAGS += -DTILE_W_P=$(TILE_W_P)
APP_CFLAGS += -DTILE_DIM_P=$(TILE_DIM_P)

APP_CFLAGS += -DTILE_H_SM=$(TILE_H_SM)
APP_CFLAGS += -DTILE_W_SM=$(TILE_W_SM)
APP_CFLAGS += -DTILE_DIM_SM=$(TILE_DIM_SM)

APP_CFLAGS += -DTILE_H_TR=$(TILE_H_TR)
APP_CFLAGS += -DTILE_W_TR=$(TILE_W_TR)
APP_CFLAGS += -DTILE_DIM_TR=$(TILE_DIM_TR)

APP_CFLAGS += -DTILE_H_ATTV=$(TILE_H_ATTV)
APP_CFLAGS += -DTILE_W_ATTV=$(TILE_W_ATTV)
APP_CFLAGS += -DTILE_DIM_ATTV=$(TILE_DIM_ATTV)

APP_CFLAGS += -DTILE_H_OUT_TR=$(TILE_H_OUT_TR)
APP_CFLAGS += -DTILE_W_OUT_TR=$(TILE_W_OUT_TR)
APP_CFLAGS += -DTILE_DIM_OUT_TR=$(TILE_DIM_OUT_TR)

APP_CFLAGS += -DSTEP=$(STEP)

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	rm -rf BUILD/
	python3 ./utils/GM.py --step $(STEP) --in_width $(IN_W) --in_height $(IN_H) --ch_in ${IN_CH} --ch_out ${OUT_CH} --n_heads $(N_HEADS) --att_dim $(ATT_DIM)

profile_all_optim:
	python3 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --in_width $(IN_W) --in_height $(IN_H) --ch_in ${IN_CH} --ch_out ${OUT_CH} --n_heads $(N_HEADS) --att_dim $(ATT_DIM)

profile_all_sizes:
	python3 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE}

include $(RULES_DIR)/pmsis_rules.mk
