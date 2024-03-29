NNTOOL=nntool
MODEL_SQ8=1
# MODEL_POW2=1
# MODEL_FP16=1
# MODEL_NE16=1


MODEL_SUFFIX_NAVIGATION?=_NAVIGATION
MODEL_SUFFIX_CLASSIFICATION?=_CLASSIFICATION
MODEL_PREFIX_NAVIGATION?=gate_navigator_model_v22
MODEL_PREFIX_CLASSIFICATION?=gate_classifier_model_v51
MODEL_PYTHON=python3
MODEL_BUILD=BUILD_MODEL

AT_MODEL_NAVIGATION_PATH=Model_$(MODEL_PREFIX_NAVIGATION).c
AT_MODEL_CLASSIFICATION_PATH=Model_$(MODEL_PREFIX_CLASSIFICATION).c

MODEL_EXPRESSIONS = Expression_Kernels.c

ifndef MODEL_HAS_NO_CONSTANTS
  MODEL_TENSORS=$(MODEL_BUILD)/$(MODEL_PREFIX_NAVIGATION)_L3_Flash_Const.dat $(MODEL_BUILD)/$(MODEL_PREFIX_CLASSIFICATION)_L3_Flash_Const.dat
else
  MODEL_TENSORS=
endif


# Memory sizes for cluster L1, SoC L2 and Flash
TARGET_L1_SIZE = 64000
TARGET_L2_SIZE = 512000
TARGET_L3_SIZE = 8000000

# Options for the memory settings: will require
# set l3_flash_device $(MODEL_L3_FLASH)
# set l3_ram_device $(MODEL_L3_RAM)
# in the nntool_script
# FLASH and RAM type
FLASH_TYPE = HYPER
RAM_TYPE   = HYPER

ifeq '$(FLASH_TYPE)' 'HYPER'
    MODEL_L3_FLASH=AT_MEM_L3_HFLASH
else ifeq '$(FLASH_TYPE)' 'MRAM'
    MODEL_L3_FLASH=AT_MEM_L3_MRAMFLASH
    READFS_FLASH = target/chip/soc/mram
else ifeq '$(FLASH_TYPE)' 'QSPI'
    MODEL_L3_FLASH=AT_MEM_L3_QSPIFLASH
    READFS_FLASH = target/board/devices/spiflash
else ifeq '$(FLASH_TYPE)' 'OSPI'
    MODEL_L3_FLASH=AT_MEM_L3_OSPIFLASH
else ifeq '$(FLASH_TYPE)' 'DEFAULT'
    MODEL_L3_FLASH=AT_MEM_L3_DEFAULTFLASH
endif

ifeq '$(RAM_TYPE)' 'HYPER'
    MODEL_L3_RAM=AT_MEM_L3_HRAM
else ifeq '$(RAM_TYPE)' 'QSPI'
    MODEL_L3_RAM=AT_MEM_L3_QSPIRAM
else ifeq '$(RAM_TYPE)' 'OSPI'
    MODEL_L3_RAM=AT_MEM_L3_OSPIRAM
else ifeq '$(RAM_TYPE)' 'DEFAULT'
    MODEL_L3_RAM=AT_MEM_L3_DEFAULTRAM
endif

ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
    FREQ_CL?=370
    FREQ_FC?=370
    FREQ_PE?=370
else
    ifeq '$(TARGET_CHIP)' 'GAP8_V3'
    FREQ_CL?=175
    else
    FREQ_CL?=250
    endif
    FREQ_FC?=250
    FREQ_PE?=250
endif

# Cluster stack size for master core and other cores
CLUSTER_STACK_SIZE=4096
CLUSTER_SLAVE_STACK_SIZE=1024

$(info GEN ... $(CNN_GEN))
