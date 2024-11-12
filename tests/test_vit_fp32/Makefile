APP = vit_fp32

# User code
NUM_CORES?=8
DATA_TYPE?=fp32

WEIGHTS_PATH?=None
#WEIGHTS_PATH?=sample_data/lite_weights_sample.pth
#WEIGHTS_PATH?=sample_data/weights_sample.pth

INPUT_PATH?=None
#INPUT_PATH?=sample_data/input_sample.png

# Will be ignored if weights path is None
NUM_CLASSES?=50
ORIGINAL_IMAGE_WIDTH?=350
ORIGINAL_IMAGE_HEIGHT?=350
INPUT_IMAGE_WIDTH?=384
INPUT_IMAGE_HEIGHT?=384

MATMUL_TYPE?=9
NUM_MATMULS?=24		# When profiling with multiple matmul algorithms
NUM_SIZES?=3		# When profiling multiple sizes of the network
# End of user code

TASK_NAME=sst-2
TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources

#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DN_HEADS=$(N_HEADS)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -mhwloopalign
APP_LDFLAGS += -lm
APP_CFLAGS += -DMEMOCC_COMP

APP_CFLAGS += -DTILE_H=$(TILE_H)
APP_CFLAGS += -DTILE_W=$(TILE_W)
APP_CFLAGS += -DTILE_DIM=$(TILE_DIM)

APP_CFLAGS += -DOPTIMIZE
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}

# STATISTICS
APP_CFLAGS += -DSTATS

# =============== SOURCES ===============
APP_SRCS = main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_layernorm_fp32.c

# CONV2D REQUIRED SOURCES
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_naive_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_mhsa_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c

include $(RULES_DIR)/pmsis_rules.mk

get_golden:
	rm -rf BUILD/
	python3 utils/GM.py --weights_path $(WEIGHTS_PATH) --data_type $(DATA_TYPE) --num_classes $(NUM_CLASSES) --original_image_width $(ORIGINAL_IMAGE_WIDTH) --original_image_height $(ORIGINAL_IMAGE_HEIGHT) --input_image_width $(INPUT_IMAGE_WIDTH) --input_image_height $(INPUT_IMAGE_HEIGHT)
