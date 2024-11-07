#include <R_ext/Rdynload.h>
#include "hann.h"

static R_CallMethodDef Call_entries[] = {
    {"test_7", (DL_FUNC) &test_7, 11},
    {"test_6", (DL_FUNC) &test_6, 14},
    {"E", (DL_FUNC) &E, 3},
    {"updateSigma_Call", (DL_FUNC) &updateSigma_Call, 4},
    {NULL, NULL, 0}
};

void R_init_hann(DllInfo *info)
{
    R_registerRoutines(info, NULL, Call_entries, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
