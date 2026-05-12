// test_cpu_link.c -- smoke test that the kt_cpu_* C surface compiles
// and links, opens the bundled GGUF, and returns a non-NULL audio
// buffer from the stub synth. Catches any link-time mismatch before
// the Swift side touches it.

#include "KittensCPU.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    const char * path = argc > 1
        ? argv[1]
        : "app/Resources/nano/kitten_full.gguf";
    kt_cpu_ctx * ctx = kt_cpu_create(path);
    int rc = 1;
    if (ctx == NULL) {
        fprintf(stderr, "kt_cpu_create failed: %s\n",
                kt_cpu_last_error(NULL));
    } else {
        const int32_t ids[] = { 0, 10, 0 };
        float style[256];
        for (int i = 0; i < 256; i++) { style[i] = 0.0f; }
        kt_cpu_audio a = kt_cpu_synthesize(ctx, ids, 3, style, 1.0f);
        if (a.samples == NULL) {
            fprintf(stderr, "kt_cpu_synthesize returned NULL: %s\n",
                    kt_cpu_last_error(ctx));
        } else {
            printf("kt_cpu link smoke: ctx OK, audio %llu samples "
                   "(stub silence)\n",
                   (unsigned long long)a.n_samples);
            rc = 0;
            kt_cpu_audio_free(a);
        }
        kt_cpu_destroy(ctx);
    }
    return rc;
}
