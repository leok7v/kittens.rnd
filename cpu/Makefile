# cpu/Makefile -- build kt_tensor + tests outside Xcode for fast iteration.
#
# Usage:
#   make             -- build and run all tests
#   make test_basic  -- standalone sanity tests
#   make test_ggml   -- ggml-parity tests (needs vendors/llama.cpp built)
#   make clean

CC      ?= clang
ROOT    := $(abspath ..)
INC     := -I include -I $(ROOT)/ggml/include
CFLAGS  := -Wall -Wextra -Wno-unused-parameter -O2 -std=c11 $(INC)
LDFLAGS := -framework Accelerate -lm

GGML_BUILD := $(ROOT)/vendors/llama.cpp/build-cpu/ggml/src
GGML_INC   := $(ROOT)/vendors/llama.cpp/ggml/include
GGML_LIBS  := $(GGML_BUILD)/libggml.a \
              $(GGML_BUILD)/libggml-cpu.a \
              $(GGML_BUILD)/libggml-base.a \
              $(GGML_BUILD)/ggml-blas/libggml-blas.a

# Build output lives OUTSIDE cpu/ so Xcode's
# PBXFileSystemSynchronizedRootGroup for the cpu folder never picks
# our macOS-compiled object files up by mistake. Previously a `make`
# invocation would drop cpu/build/*.o, and a subsequent iOS app build
# would fail with "Building for 'iOS', but linking in object file
# (cpu/build/gguf_read.o) built for 'macOS'".
BUILD := ../tmp/cpu_build

SRC_KT  := kt_tensor.c gguf_read.c kittens-tts-cpu.c
OBJ_KT  := $(BUILD)/kt_tensor.o $(BUILD)/gguf_read.o \
           $(BUILD)/kittens-tts-cpu.o

all: test

test: test_basic test_ggml test_gguf test_cpu_link

test_basic: $(BUILD)/test_basic
	@$(BUILD)/$@

test_ggml: $(BUILD)/test_ggml
	@$(BUILD)/$@

test_gguf: $(BUILD)/test_gguf
	@cd .. && tmp/cpu_build/test_gguf app/Resources/nano/kitten_full.gguf

test_cpu_link: $(BUILD)/test_cpu_link
	@cd .. && tmp/cpu_build/test_cpu_link app/Resources/nano/kitten_full.gguf

$(BUILD)/kt_tensor.o: kt_tensor.c include/kt_tensor.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/gguf_read.o: gguf_read.c include/kt_gguf.h include/kt_tensor.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/kittens-tts-cpu.o: kittens-tts-cpu.c \
	    include/KittensCPU.h include/kt_gguf.h include/kt_tensor.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/test_basic: tests/test_kt_basic.c $(OBJ_KT) | $(BUILD)
	$(CC) $(CFLAGS) tests/test_kt_basic.c $(OBJ_KT) $(LDFLAGS) -o $@

$(BUILD)/test_ggml: tests/test_kt_vs_ggml.c $(OBJ_KT) $(GGML_LIBS) | $(BUILD)
	clang++ -Wall -Wextra -Wno-unused-parameter -O2 -std=c++17 \
	    $(INC) -I $(GGML_INC) -x c++ tests/test_kt_vs_ggml.c -x none \
	    $(OBJ_KT) $(GGML_LIBS) $(LDFLAGS) -o $@

$(BUILD)/test_gguf: tests/test_gguf.c $(OBJ_KT) $(GGML_LIBS) | $(BUILD)
	clang++ -Wall -Wextra -Wno-unused-parameter -O2 -std=c++17 \
	    $(INC) -I $(GGML_INC) -x c++ tests/test_gguf.c -x none \
	    $(OBJ_KT) $(GGML_LIBS) $(LDFLAGS) -o $@

$(BUILD)/test_cpu_link: tests/test_cpu_link.c $(OBJ_KT) | $(BUILD)
	$(CC) $(CFLAGS) tests/test_cpu_link.c $(OBJ_KT) $(LDFLAGS) -o $@

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	@rm -rf $(BUILD)

.PHONY: all test test_basic test_ggml clean
