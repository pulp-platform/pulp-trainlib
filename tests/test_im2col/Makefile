APP = im2col_fp32

# User settings
DATA_BITS?=32				# 32=FP32, 16=FP16
NUM_CORES?=8
IM2ROW?=0					# 0=Im2Col, 1=Im2Row
MOD?=0 						# 0=Forward/WeightGrad, 1=InGrad
DMA_ENABLE?=0				# Use Cluster DMA to move data or not
HWC_format?=0				# CHW (=0) or HWC (=1) data format for the im2col
# Layer padding and stride
LPAD?=0
RPAD?=0
UPAD?=0
DPAD?=0
HSTR?=1
WSTR?=1
# Layer sizes
Tin_H_l1?=3
Tin_W_l1?=3
Tin_C_l1?=2
Tker_W_l1?=2
Tout_C_l1?=2
APP_CFLAGS += -DPRINT_OUTPUT
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 -mno-memcpy
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DDATA_BITS=$(DATA_BITS)
APP_CFLAGS += -DMOD=$(MOD)
APP_CFLAGS += -DDMA_ENABLE=$(DMA_ENABLE)
APP_CFLAGS += -DLPAD=$(LPAD)
APP_CFLAGS += -DRPAD=$(RPAD)
APP_CFLAGS += -DUPAD=$(UPAD)
APP_CFLAGS += -DDPAD=$(DPAD)
APP_CFLAGS += -DHSTR=$(HSTR)
APP_CFLAGS += -DWSTR=$(WSTR)
APP_CFLAGS += -DTin_H_l1=$(Tin_H_l1)
APP_CFLAGS += -DTin_W_l1=$(Tin_W_l1)
APP_CFLAGS += -DTin_C_l1=$(Tin_C_l1)
APP_CFLAGS += -DTker_W_l1=$(Tker_W_l1)
APP_CFLAGS += -DTout_C_l1=$(Tout_C_l1)
APP_CFLAGS += -DHWC_format=$(HWC_format)
APP_CFLAGS += -DIM2ROW=$(IM2ROW)
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -mhwloopalign
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

# Sources
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c

include $(RULES_DIR)/pmsis_rules.mk
