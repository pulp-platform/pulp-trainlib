APP = test_matmul

# User settings
# Standard matmul arguments
IN_CH?=32			# Used also to define the number of channels of DW Conv
MID_CH?=32
OUT_CH?=32
# General arguments
DATA_TYPE?='float' 	# float, fp16 (=>float16), bf16 (=>float16alt)  to select the desired format
DIVIDER?=100000000	# Scaling factor for data initialization in golden model
TRANSP?=0		# Matrix B is transposed if = 1, not transposed if = 0.
NUM_CORES?=8
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET

APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 utils/GM.py --in_size $(IN_CH) --out_size $(OUT_CH) --mid_size $(MID_CH) --type $(DATA_TYPE) --init_value_div $(DIVIDER) --transpose $(TRANSP) 

profile_fastest:
	python3 utils/profile_fastest.py

include $(RULES_DIR)/pmsis_rules.mk
