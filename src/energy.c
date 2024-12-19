/* energy.c    2024-10-31 */

/* Copyright 2024 Emmanuel Paradis */

/* This file is part of the R-package `hann'. */
/* See the file ../DESCRIPTION for licensing issues. */

#include "hann.h"

inline double F(double x, int n)
{
    double z;
    z = x < 0 ? 0 : pow(x, n);
    return z;
}

inline int sign(double x)
{
    if (x == 0) return 0;
    if (x < 0) return -1;
    return 1;
}

SEXP E(SEXP SIGMA, SEXP XI, SEXP n4F)
{
    int i, mu, *sigma, *xi, N, K, n;
    double v = 0, s;

    PROTECT(SIGMA = coerceVector(SIGMA, INTSXP));
    PROTECT(XI = coerceVector(XI, INTSXP));
    PROTECT(n4F = coerceVector(n4F, INTSXP));
    N = ncols(XI);
    K = nrows(XI);
    sigma = INTEGER(SIGMA);
    xi = INTEGER(XI);
    n = INTEGER(n4F)[0];

    for (mu = 0; mu < K; mu++) {
	s = 0;
	for (i = 0; i < N; i++)
	    s += xi[mu + i * K] * sigma[i];
	v -= F(s, n);
    }

    UNPROTECT(3);
    return ScalarReal(v);
}

void updateSigma(int *sigma, int *xi, int N, int K,
		 int *upsigma, int n)
{
    int i, j, mu;
    double s, a, *S;

    S = (double*)R_alloc(K, sizeof(double));

    for (mu = 0; mu < K; mu++) {
	s = 0;
	for (i = 0; i < N; i++)
	    s += xi[mu + i * K] * sigma[i];
	S[mu] = s;
    }

    j = 0;
    for (i = 0; i < N; i++) {
	for (mu = 0; mu < K; mu++) {
	    a = xi[j++]; // j = mu + i * K
	    s = S[mu] - a * sigma[i];
	    upsigma[i] = sign(F(a + s, n) - F(-a + s, n));
	}
    }
}

SEXP updateSigma_Call(SEXP SIGMA, SEXP XI, SEXP UPSIGMA, SEXP n4F)
{
    int *sigma, *upsigma, *xi, N, K;

    PROTECT(SIGMA = coerceVector(SIGMA, INTSXP));
    PROTECT(XI = coerceVector(XI, INTSXP));
    PROTECT(UPSIGMA = coerceVector(UPSIGMA, INTSXP));
    PROTECT(n4F = coerceVector(n4F, INTSXP));

    N = ncols(XI);
    K = nrows(XI);

    sigma = INTEGER(SIGMA);
    upsigma = INTEGER(UPSIGMA);
    xi = INTEGER(XI);

    updateSigma(sigma, xi, N, K, upsigma, INTEGER(n4F)[0]);

    UNPROTECT(4);
    return ScalarInteger(0);
}
