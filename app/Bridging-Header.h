// Umbrella bridging header for the KittensRnD Xcode target.
// Exposes the CEPhonemizer C API (implemented in C++), the KittensGGML
// C API (ggml/kittens-tts.c, ggml/llama.cpp backend), and the
// KittensCPU C API (cpu/kittens-tts-cpu.c, pure-C + cblas backend) to
// Swift.

#ifndef KITTENSRND_BRIDGING_HEADER_H
#define KITTENSRND_BRIDGING_HEADER_H

#include "CEPhonemizer.h"
#include "KittensGGML.h"
#include "KittensCPU.h"

#endif /* KITTENSRND_BRIDGING_HEADER_H */
