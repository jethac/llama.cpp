#include "../src/llama-ext.h"
#include "llama.h"

#include <cstdio>

int main(void) {
    if (llama_ftype_get_default_type(LLAMA_FTYPE_MOSTLY_NVFP4) != GGML_TYPE_COUNT) {
        printf("NVFP4 ftype must not be a default llama-quantize output target\n");
        return 1;
    }

    return 0;
}
