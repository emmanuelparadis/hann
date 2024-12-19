/* hann.h    2024-10-31 */

/* Copyright 2024 Emmanuel Paradis */

/* This file is part of the R-package `hann'. */
/* See the file ../DESCRIPTION for licensing issues. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Lapack.h>

#define ZETA0 0.1
#define c1 0.0001
#define c2 0.9

#define bar(x) (1 - x * x)

double runif_local();
void fast_mat_prod_4(double *X, double *Y, double *Z, int n, int m, int p);
void fast_mat_prod_0(double *X, double *Y, double *Z, int n, int m, int p);
int fast_mat_inv(double *X, int n);
int do_error_rate(int *E, double *O, int K, int C);
SEXP E(SEXP SIGMA, SEXP XI, SEXP n4F);
SEXP updateSigma_Call(SEXP SIGMA, SEXP XI, SEXP UPSIGMA, SEXP n4F);
SEXP test_7(SEXP W, SEXP BIAS, SEXP SIGMA, SEXP XI, SEXP EXPEC,
	    SEXP ITERLIM, SEXP QUIET, SEXP CTRL,
	    SEXP CONVERGENCE, SEXP beta, SEXP MC_CORES);
SEXP test_6(SEXP W1, SEXP W2, SEXP W3, SEXP BIAS_HH, SEXP BIAS_HC,
	    SEXP SIGMA, SEXP XI, SEXP EXPEC, SEXP ITERLIM, SEXP QUIET,
	    SEXP CTRL, SEXP CONVERGENCE, SEXP beta, SEXP MC_CORES);
