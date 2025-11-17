/* prototype_7.c    2025-11-17 */

/* Copyright 2024-2025 Emmanuel Paradis */

/* This file is part of the R-package `hann'. */
/* See the file ../DESCRIPTION for licensing issues. */

#ifndef R_NO_REMAP
# define R_NO_REMAP
#endif

#include "hann.h"

/* global parameters: */
static int N, K, C, np1, npar, KC, mc_cores;
static double target, BETA;

/*
  control_list[0]: quasiNewton
  control_list[1]: fullHessian
  control_list[2]: trace
  control_list[3]: checkWolfe
*/
static int control_list[4];

/* inline int cumulate_count(int *count) */
/* { */
/*     int s = 0, i = 0; */

/*     while (i < mc_cores) s += count[i++]; */

/*     return s; */
/* } */

/* inline void init_count(int *count) */
/* { */
/*     memset(count, 0, mc_cores * sizeof(int)); */
/* } */

/* use the next #undef to check that the code works for machines
   without omp.h (2025-07-23) */
/* #undef _OPENMP */

#ifdef _OPENMP
#include <omp.h>

/* arrays used at each iteration */
double *O7, *Deviation, *bar_O7;

double objfun_7_OMP(double *PARA, double *sigma_xi, int *E, double *GRAD, int eval_grad)
{
    omp_set_num_threads(mc_cores);
    /* int count[mc_cores]; */

    int mu, i, j, k, idx;
    double val = 0, s, tmp;

    double *w, *bias, *grad_W, *grad_bias;

    w = PARA;
    bias = PARA + np1;

    if (eval_grad) {
	grad_W = GRAD;
	grad_bias = GRAD + np1;
    }

    /* 2. integrate the signals from the N input neurons to the C
       classification neurons */
    fast_mat_prod_0(sigma_xi, w, O7, K, C, N);

#pragma omp parallel for// shared(KC, O, bias, BETA) private(i)
    for (i = 0; i < KC; i++)
	O7[i] = my_tanh(BETA * O7[i] + bias[i/K]); // integer division (i/K)

    if (control_list[2]) {
	k = do_error_rate(E, O7, K, C);
	Rprintf("Error rate = %d / %d\n", k, K);
    }

    if (eval_grad) {
#pragma omp parallel for// shared(KC, O, bar_O) private(i)
	for (i = 0; i < KC; i++)
	    bar_O7[i] = bar(O7[i]);
    }

/*
  SO FAR, PARALLELIZING THE SUMS DOES NOT WORK WELL, SO WE KEEP THE
  SERIAL CODE. THIS APPLIES ALSO TO THE CALCULATIONS OF THE GRADIENTS
  BELOW. THE GAIN FROM PARALLELIZING THE ARRAY-TO ARRAY OPERATIONS
  (ABOVE) IS ALREADY SUBSTANTIAL, ESPECIALLY IF 'K' IS LARGE.
*/

/*     if (eval_grad) { */
/* 	init_count(count); */
/* #pragma omp parallel for private(s) */
/* 	for (i = 0; i < KC; i++) { */
/* 	    Deviation[i] = s = O[i] - E[i]; */
/* 	    count[omp_get_thread_num()] += s * s; */
/* 	} */
/* 	val = cumulate_count(count); */
/*     } else { */
/* 	init_count(count); */
/* #pragma omp parallel for */
/* 	for (i = 0; i < KC; i++) */
/* 	    count[omp_get_thread_num()] += pow(O[i] - E[i], 2); */
/* 	val = cumulate_count(count); */
/* 	return val; */
/*     } */

    if (eval_grad) {
	for (i = 0; i < KC; i++) {
	    Deviation[i] = s = O7[i] - E[i];
	    val += s * s;
	}
    } else {
	for (i = 0; i < KC; i++)
	    val += pow(O7[i] - E[i], 2);
	return val;
    }

    /****************************/
    /* the gradients start here */
    /****************************/

    /* Note: BLAS/LAPACK are not useful here. */

    /* gradients of W */
    tmp = 2 * BETA;
    k = 0;
    for (j = 0; j < C; j++) {
	for (i = 0; i < N; i++) {
	    s = 0;
	    idx = j * K;
	    for (mu = 0; mu < K; mu++) {
		// idx = mu + j*K;
		s += sigma_xi[mu + i*K] * bar_O7[idx] * Deviation[idx];
		idx++;
	    }
	    grad_W[k++] = tmp * s; // k = i + j*N
	}
    }

    /* gradients of B */
    for (j = 0; j < C; j++) {
	s = 0;
	for (mu = 0; mu < K; mu++) {
	    idx = mu + j*K;
	    s += bar_O7[idx] * Deviation[idx];
	}
	grad_bias[j]  = 2 * s;
    }

/*     /\* gradients of W *\/ */
/*     tmp = 2 * BETA; */
/*     k = 0; */
/*     for (j = 0; j < C; j++) { */
/* 	for (i = 0; i < N; i++) { */
/* 	init_count(count); */
/* #pragma omp parallel for private(idx) */
/* 	    for (mu = 0; mu < K; mu++) { */
/* 		idx = mu + j*K; */
/* 		count[omp_get_thread_num()] += */
/* 		    sigma_xi[mu + i*K] * bar_O[idx] * Deviation[idx]; */
/* 	    } */
/* 	    grad_W[k++] = tmp * cumulate_count(count); // k = i + j*N */
/* 	} */
/*     } */

/*     /\* gradients of B *\/ */
/*     for (j = 0; j < C; j++) { */
/* 	init_count(count); */
/* #pragma omp parallel for private(idx) */
/* 	for (mu = 0; mu < K; mu++) { */
/* 	    idx = mu + j*K; */
/* 	    count[omp_get_thread_num()] += bar_O[idx] * Deviation[idx]; */
/* 	} */
/* 	grad_bias[j]  = 2 * cumulate_count(count); */
/*     } */

    return val;
}
#endif

double objfun_7(double *PARA, double *sigma_xi, int *E, double *GRAD, int eval_grad)
{
    int mu, i, j, k, idx;
    double val = 0, s, tmp;

    double *w, *bias, *grad_W, *grad_bias;

    w = PARA;
    bias = PARA + np1;

    if (eval_grad) {
	grad_W = GRAD;
	grad_bias = GRAD + np1;
    }

    /* 2. integrate the signals from the N input neurons to the C
       classification neurons */
    fast_mat_prod_0(sigma_xi, w, O7, K, C, N);

    for (i = 0; i < KC; i++)
	O7[i] = my_tanh(BETA * O7[i] + bias[i/K]); // integer division (i/K)

    if (control_list[2]) {
	k = do_error_rate(E, O7, K, C);
	Rprintf("Error rate = %d / %d\n", k, K);
    }

    if (eval_grad) {
	for (i = 0; i < KC; i++)
	    bar_O7[i] = bar(O7[i]);
    }

    if (eval_grad) {
	for (i = 0; i < KC; i++) {
	    Deviation[i] = s = O7[i] - E[i];
	    val += s * s;
	}
    } else {
	for (i = 0; i < KC; i++)
	    val += pow(O7[i] - E[i], 2);
	return val;
    }

    /****************************/
    /* the gradients start here */
    /****************************/

    /* Note: BLAS/LAPACK are not useful here. */

    /* gradients of W */
    tmp = 2 * BETA;
    k = 0;
    for (j = 0; j < C; j++) {
	for (i = 0; i < N; i++) {
	    s = 0;
	    idx = j * K;
	    for (mu = 0; mu < K; mu++) {
		// idx = mu + j*K;
		s += sigma_xi[mu + i*K] * bar_O7[idx] * Deviation[idx];
		idx++;
	    }
	    grad_W[k++] = tmp * s; // k = i + j*N
	}
    }

    /* gradients of B */
    for (j = 0; j < C; j++) {
	s = 0;
	for (mu = 0; mu < K; mu++) {
	    idx = mu + j*K;
	    s += bar_O7[idx] * Deviation[idx];
	}
	grad_bias[j]  = 2 * s;
    }

    /*
    FILE *fw, *fb, *fo;
    fw = fopen("w_", "w");
    fb = fopen("b_", "w");
    fo = fopen("o_", "w");
    fwrite(grad_W, sizeof(double), np1, fw);
    fwrite(grad_bias, sizeof(double), C, fb);
    fwrite(O, sizeof(double), KC, fo);
    fclose(fw);
    fclose(fb);
    fclose(fo);
    */

    return val;
}

void do_Hessian(double *PARA, double *sigma_xi, int *E,
		double *GRAD, double *hessian)
{
    int i, j, k = 0;
    double delta = 1E-8, *new_grad, tmp;
    new_grad = malloc(npar * sizeof(double));

    for (i = 0; i < npar; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	objfun_7(PARA, sigma_xi, E, new_grad, 1);
	for (j = 0; j < npar; j++)
	    hessian[k++] = (new_grad[j] - GRAD[j]) / delta;
	PARA[i] = tmp;
    }

    free(new_grad);
}

#define COMPUTE_PROPOSED_PARA				\
    memcpy(ptr_para[!Switch], ptr_para[Switch],		\
	   npar * sizeof(double));			\
    for (i = 0; i < npar; i++)				\
	ptr_para[!Switch][i] += alpha * p_by_grad[i]

double optimize_7(double *PARA, int *sigma, int *xi,
		  int *expec, int iterlim, int quiet)
{
    int i, k, iter = 0, mu, withHessian;
    double s, sb, res, resb, alpha, *sigma_xi, *p_by_grad;

    double (*FUN)(double *PARA, double *sigma_xi, int *E, double *GRAD, int eval_grad);

#ifdef _OPENMP
    if (mc_cores == 1) FUN = &objfun_7; else FUN = &objfun_7_OMP;
#else
    FUN = &objfun_7;
#endif

    sigma_xi = (double*)R_alloc(K * N, sizeof(double));

    /* gradients: */
    double *new_PARA, *GRAD, *new_GRAD, *p;
    new_PARA = (double*)R_alloc(npar, sizeof(double));
    GRAD = (double*)R_alloc(npar, sizeof(double));
    new_GRAD = (double*)R_alloc(npar, sizeof(double));
    p = (double*)R_alloc(npar, sizeof(double));
    p_by_grad = (double*)R_alloc(npar, sizeof(double));

    /* Hessian */
    double *hessian;
    if (control_list[0])
	hessian = (double*)R_alloc(npar * npar, sizeof(double));

    _Bool Switch = 0;
    double *ptr_para[2], *ptr_grad[2];
    ptr_para[0] = PARA;
    ptr_grad[0] = GRAD;
    ptr_para[1] = new_PARA;
    ptr_grad[1] = new_GRAD;

    /* 1. calculate the intermediate products 'sigma_i * xi_{mu,i}'
       only once */
    k = 0;
    for (i = 0; i < N; i++) {
	s = (double)sigma[i];
	for (mu = 0; mu < K; mu++) {
	    sigma_xi[k] = s * (double)xi[k];
	    // xi[k] *= res; // <- if overwrite
	    k++;
	}
    }

    /* do a first iteration to get the gradients: */
    res = FUN(PARA, sigma_xi, expec, GRAD, 1);

    if (!quiet) Rprintf("INITIALIZATION -- iteration %d\tobj_fun = %f\n", iter, res);
    if (!quiet) Rprintf("Gradients done.\n");

    for (;;) {
	if (iter >= iterlim) break;
	if (!quiet) Rprintf("\riteration %d\tobj_fun = %f", iter + 1, res);

	if (control_list[0]) { // try quasi-Newton step
	    mu = control_list[2];
	    control_list[2] = 0;
	    do_Hessian(ptr_para[Switch], sigma_xi, expec, ptr_grad[Switch], hessian);
	    control_list[2] = mu;
	    k = fast_mat_inv(hessian, npar);
	    withHessian = k == 0 ? 1 : 0;
	} else {
	    withHessian = 0;
	}

	/* compute 'p' */
start:
	if (withHessian) { // p = -H^{-1} GRAD
	    fast_mat_prod_0(hessian, ptr_grad[Switch], p, npar, 1, npar);
	    /* store the products p * GRAD */
	    memcpy(p_by_grad, ptr_para[Switch], npar * sizeof(double));
	    for (i = 0; i < npar; i++) p_by_grad[i] *= p[i];
	} else { // p = -GRAD
	    memcpy(p, ptr_grad[Switch], npar * sizeof(double));
	    for (i = 0; i < npar; i++) p[i] *= -1;
	    /* store p * GRAD which is simply a copy of p */
	    p_by_grad = p;
	}

	mu = control_list[2];
	control_list[2] = 0;

	alpha = ZETA0;
	for (;;) {
	    /* compute the proposed parameters */
	    COMPUTE_PROPOSED_PARA;
	    /* get the obj_fun value using the proposed parameters */
	    resb = FUN(ptr_para[!Switch], sigma_xi, expec,
		       ptr_grad[!Switch], 0);
	    if (resb < res) {
		/* Rprintf("   GOOD: alpha = %.11f\n", alpha); */
		goto update;
	    }
	    alpha /= 2;
	    if (alpha < 1E-10) {
		if (!withHessian) {
		    /* Rprintf("  WRONG: alpha = %.11f\n", alpha); */
		    goto update;
		} else {
		    /* use p = -GRAD instead */
		    withHessian = 0;
		    goto start;
		}
	    }
	}

update:
	control_list[2] = mu;
	/* check Wolfe's conditions */
	if (control_list[3]) {
	    fast_mat_prod_4(p, ptr_grad[Switch], &s, npar, 1, 1); // s = p^t GRAD
	    fast_mat_prod_4(p, ptr_grad[!Switch], &sb, npar, 1, 1);
	    Rprintf("Wolfe's conditions 1 ");
	    if (resb <= res + c1 * alpha * s)
		Rprintf("good\n"); else Rprintf("bad\n");
	    Rprintf("Wolfe's conditions 2 ");
	    if (sb <= -c2 * s)
		Rprintf("good\n"); else Rprintf("bad\n");
	}

	if (resb < res) {
	    Switch = !Switch;
	} else {
	    /* reinitialize the parameters */
	    for (i = 0; i < np1; i++)
		ptr_para[Switch][i] = runif_local();
	    memset(ptr_para[Switch] + np1, 0,
		   C * sizeof(double));
	}
	/* update the gradients (and 'res') */
	res = FUN(ptr_para[Switch], sigma_xi, expec,
		  ptr_grad[Switch], 1);
	if (res < target) break;
	R_CheckUserInterrupt();
	iter++;
    }

    if (!quiet && iterlim) Rprintf("\n");

    if (Switch) memcpy(PARA, new_PARA, npar * sizeof(double));

    return res;
}

SEXP test_7(SEXP W, SEXP BIAS, SEXP SIGMA, SEXP XI, SEXP EXPEC,
	    SEXP ITERLIM, SEXP QUIET, SEXP CTRL,
	    SEXP CONVERGENCE, SEXP beta, SEXP MC_CORES)
{
    int *sigma, *xi, *expec;
    double *w, *bias, val;

    PROTECT(W = Rf_coerceVector(W, REALSXP));
    PROTECT(BIAS = Rf_coerceVector(BIAS, REALSXP));
    PROTECT(SIGMA = Rf_coerceVector(SIGMA, INTSXP));
    PROTECT(XI = Rf_coerceVector(XI, INTSXP));
    PROTECT(EXPEC = Rf_coerceVector(EXPEC, INTSXP));
    PROTECT(ITERLIM = Rf_coerceVector(ITERLIM, INTSXP));
    PROTECT(QUIET = Rf_coerceVector(QUIET, INTSXP));
    PROTECT(CTRL = Rf_coerceVector(CTRL, INTSXP));
    PROTECT(CONVERGENCE = Rf_coerceVector(CONVERGENCE, REALSXP));
    PROTECT(beta = Rf_coerceVector(beta, REALSXP));
    PROTECT(MC_CORES = Rf_coerceVector(MC_CORES, INTSXP));

    /* set the global parameters */
    N = Rf_ncols(XI);
    K = Rf_nrows(XI);
    C = Rf_ncols(EXPEC);
    np1 = N * C;
    npar = np1 + C;
    KC = K * C;

    memcpy(&control_list, INTEGER(CTRL), 4 * sizeof(int));

    target = REAL(CONVERGENCE)[0];
    BETA = REAL(beta)[0];
    mc_cores = INTEGER(MC_CORES)[0];

    w = REAL(W);
    bias = REAL(BIAS);
    sigma = INTEGER(SIGMA);
    xi = INTEGER(XI);
    expec = INTEGER(EXPEC);

    double *PARA;
    PARA = (double*)R_alloc(npar, sizeof(double));
    memcpy(PARA, w, np1 * sizeof(double));
    memcpy(PARA + np1, bias, C * sizeof(double));

    /* global arrays */
    O7 = (double*)R_alloc(KC, sizeof(double));
    Deviation = (double*)R_alloc(KC, sizeof(double));
    bar_O7 = (double*)R_alloc(KC, sizeof(double));

    val = optimize_7(PARA, sigma, xi, expec, INTEGER(ITERLIM)[0],
		     INTEGER(QUIET)[0]);

    /* copy back the estimates into the original objects */
    memcpy(w, PARA, np1 * sizeof(double));
    memcpy(bias, PARA + np1, C * sizeof(double));

    UNPROTECT(11);
    return Rf_ScalarReal(val);
}
