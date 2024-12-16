APP = mobilebert_fp16

# User code
NUM_CORES?=8
DATA_TYPE?=fp16
VOCAB_SIZE?=30522
EMBEDDING_SIZE??=128
INTERMEDIATE_SIZE=512
N_HEADS?=4
N_LAYERS?=1
SEQ_LEN?=128
TILE_H?= 32
TILE_W?= 32
TILE_DIM?= 1024

# Projections are done on F x L
TILE_H_P?=32
TILE_W_P?=32
TILE_DIM_P?=1024

# Softmax are done on L x L
TILE_H_SM?=32
TILE_W_SM?=32
TILE_DIM_SM?=1024

# Transpose are done on L x H
TILE_H_TR?=32
TILE_W_TR?=32
TILE_DIM_TR?=1024

# Att * V are done on H x L
TILE_H_ATTV?=32
TILE_W_ATTV?=32
TILE_DIM_ATTV?=1024

# Output Transpose are done on L x F
TILE_H_OUT_TR?=32
TILE_W_OUT_TR?=32
TILE_DIM_OUT_TR?=1024
# End of user code

BERT_BASE_DIR=./utils/MobileBert_PyTorch/prev_trained_model/mobilebert
DATA_DIR=./utils/MobileBert_PyTorch/GLUE
OUTPUT_DIR=./utils/MobileBert_PyTorch/outputs
TASK_NAME=sst-2
TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources

FILE0_NAME = weights.bin
FILE0 = $(CURDIR)/files/$(FILE0_NAME)
APP_CFLAGS += -DFS_READ_FS
FILE0_PATH = $(FILE0_NAME)
APP_CFLAGS += -DFILE0="$(FILE0_PATH)"

FILES = $(FILE0)
READFS_FILES = $(FILES)

PLPBRIDGE_FLAGS += -f -jtag
CONFIG_HYPERFLASH=1
USE_PMSIS_BSP=1

APP_SRCS = main.c net_l1.c
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DUSE_CLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DN_HEADS=$(N_HEADS)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -mhwloopalign
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

APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=3
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}

# STATISTICS
APP_CFLAGS += -DSTATS


APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_nonorm_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_mhsa_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_embedding_fp16.c

include $(RULES_DIR)/pmsis_rules.mk

get_golden:
	rm -rf BUILD/
	python3 utils/GM.py --type=$(DATA_TYPE) --model_type=mobilebert --model_name_or_path=$(BERT_BASE_DIR) --task_name=$(TASK_NAME) --data_dir=$(DATA_DIR)/$(TASK_NAME)/ --output_dir=$(OUTPUT_DIR)/$(TASK_NAME)_output/ --eval_model_dir=$(OUTPUT_DIR)/$(TASK_NAME)_output/mobilebert --n_layers=$(N_LAYERS) --seq_len=$(SEQ_LEN) --skipgen --skipembed --skipgenio
