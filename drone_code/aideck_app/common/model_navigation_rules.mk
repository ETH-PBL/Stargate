# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

# The training of the model is slightly different depending on
# the quantization. This is because in 8 bit mode we used signed
# 8 bit so the input to the model needs to be shifted 1 bit

USE_DISP=1
ifdef USE_DISP
  SDL_FLAGS= -lSDL2 -lSDL2_ttf
else
  SDL_FLAGS=
endif

ifdef MODEL_L1_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L1 $(MODEL_L1_MEMORY)
endif

ifdef MODEL_L2_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L2 $(MODEL_L2_MEMORY)
endif

ifdef MODEL_L3_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L3 $(MODEL_L3_MEMORY)
endif

$(MODEL_BUILD):
	mkdir $(MODEL_BUILD)

# Build the code generator from the model code
$(MODEL_NAVIGATION_GEN_EXE): $(CNN_GEN) $(AT_MODEL_NAVIGATION_PATH) $(EXTRA_GENERATOR_SRC) | $(MODEL_BUILD)
	echo "COMPILING AUTOTILER MODEL"
	gcc -g -o $(MODEL_NAVIGATION_GEN_EXE) -I. -I$(TILER_INC) -I$(TILER_EMU_INC) $(CNN_GEN_INCLUDE) $(CNN_LIB_INCLUDE) $^ $(TILER_LIB) $(SDL_FLAGS)

compile_model_navigation: $(MODEL_NAVIGATION_GEN_EXE)

# Run the code generator to generate GAP graph and kernel code
$(MODEL_NAVIGATION_GEN_C): $(MODEL_NAVIGATION_GEN_EXE)
	echo "RUNNING AUTOTILER MODEL - START"
	$(MODEL_NAVIGATION_GEN_EXE) -o $(MODEL_BUILD) -c $(MODEL_BUILD) $(MODEL_GEN_EXTRA_FLAGS)
	echo "RUNNING AUTOTILER MODEL - FINISH"

# A phony target to simplify including this in the main Makefile
model_navigation: $(MODEL_NAVIGATION_GEN_C) $(MODEL_EXPRESSIONS)

clean_model_navigation:
	$(RM) -rf BUILD
	$(RM) -rf $(MODEL_BUILD)

.PHONY: model_navigation clean_model_navigation compile_model_navigation