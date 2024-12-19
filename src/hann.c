/* hann.c    2024-10-31 */

/* Copyright 2024 Emmanuel Paradis */

/* This file is part of the R-package `hann'. */
/* See the file ../DESCRIPTION for licensing issues. */

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
