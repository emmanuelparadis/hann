#include "hann.h"

/* global parameters: */
static int N, K, H, C, np1, np2, np3, npar, np_cumul2,
    np_cumul3, np_cumul4, KH, HC, KC, mc_cores;
static double target, BETA, twiceBETA, twiceBETA_SQUARE,
    twiceBETA_CUBE;

/*
  control_list[0]: quasiNewton
  control_list[1]: fullHessian
  control_list[2]: trace
  control_list[3]: checkWolfe
*/
static int control_list[4];

/* 24 bits to control the computation of gradients */
unsigned char eval_grad[3];
/* eval_grad[0]: specifies which groups of parameters to compute their
   gradients using bit-masking (from right to left):
   1st bit: W1
   2nd bit: W2
   3rd bit: W3
   4th bit: B1
   5th bit: B2

   eval_grad[1]: specifies which neuron(s) of W1 to compute the
   gradients neurons
   0: all
   1, ..., 255: 1st, ..., 255th neuron

   eval_grad[2]: id. for neurons of W2
*/

void do_Gradients_W1(double *sigma_xi, double *w2, double *w3,
		     double *bar_O2, double *O1, double *S,
		     double *gradient_W1, int what_k)
{
    int h, i, j, k, mu, idx, from, to;
    double a, s;

    if (what_k == -1 || what_k > H) { /* do all */
	from = 0;
	to = H;
	idx = 0;
    } else {
	from = what_k;
	to = what_k + 1;
	idx = what_k * N;
    }

    for (k = from; k < to; k++) {
	for (i = 0; i < N; i++) {
	    a = 0;
	    for (j = 0; j < C; j++) {
		for (mu = 0; mu < K; mu++) {
		    s = 0;
		    for (h = 0; h < H; h++)
			s += w3[h + j*H] * w2[k + h*H] * bar_O2[mu + h*K];
		    a += twiceBETA_CUBE * S[mu + j*K] *
			sigma_xi[mu + i*K] * bar(O1[mu + k*K]) * s;;
		}
	    }
	    gradient_W1[idx] = a;
	    idx++;
	}
    }
}

void do_Gradients_W2(double *w3, double *bar_O2, double *O1,
		     double *S, double *gradient_W2, int what_h)
{
    int h, i, j, k, mu, idx, from, to;
    double a, s;

    if (what_h == -1 || what_h > H) { /* do all */
	from = 0;
	to = H;
    } else {
	from = what_h;
	to = what_h + 1;
    }

    for (h = from; h < to; h++) {
	for (k = 0; k < H; k++) {
	    idx = k + h*H;
	    a = 0;
	    for (j = 0; j < C; j++) {
		s = twiceBETA_SQUARE * w3[h + j * H];
		for (i = 0, mu = j*K; i < K; i++, mu++) {
		    a += s * S[mu] * bar_O2[i + h*K] * O1[i + k*K];
		}
	    }
	    gradient_W2[idx] = a;
	}
    }
}

double objfun_6(double *PARA, double *sigma_xi, int *E, double *GRAD)
{
    int i, j, k, h, from, to;
    double val = 0, s, tmp;

    double *w1, *w2, *w3, *bias_HH, *bias_HC,
	*gradient_W1, *gradient_W2, *gradient_W3,
	*gradient_bias_HH, *gradient_bias_HC;

    w1 = PARA;
    w2 = PARA + np1;
    w3 = w2 + np2;
    bias_HH = w3 + np3;
    bias_HC = bias_HH + H;

    gradient_W1 = GRAD;
    gradient_W2 = GRAD + np1;
    gradient_W3 = gradient_W2 + np2;
    gradient_bias_HH = gradient_W3 + np3;
    gradient_bias_HC = gradient_bias_HH + H;

    double *O1, *O2, *O, *DEVIATION;
    O1 = (double*)R_alloc(KH, sizeof(double));
    O2 = (double*)R_alloc(KH, sizeof(double));
    O = (double*)R_alloc(KC, sizeof(double));

    /* 2. integrate the signals from the N input neurons to the H
       hidden neurons */
    fast_mat_prod_0(sigma_xi, w1, O1, K, H, N);
    for (i = 0; i < KH; i++)
	O1[i] = tanh(BETA * O1[i]);

    /* 3. modulate the signals of the H hidden neurons by their
       convolutions */
    fast_mat_prod_0(O1, w2, O2, K, H, H);
    k = 0;
    for (h = 0; h < H; h++) {
	s = bias_HH[h];
	for (i = 0; i < K; i++) {
	    O2[k] = tanh(BETA * O2[k] + s);
	    k++;
	}
    }

    /* 4. integrate the signals of the H hidden neurons
       towards the C classif. neurons */
    fast_mat_prod_0(O2, w3, O, K, C, H);
    k = 0;
    for (j = 0; j < C; j++) {
	s = bias_HC[j];
	for (i = 0; i < K; i++) {
	    O[k] = tanh(BETA * O[k] + s);
	    k++;
	}
    }

    if (control_list[2]) {
	k = do_error_rate(E, O, K, C);
	Rprintf("  Error rate = %d / %d\n", k, K);
    }

    if (eval_grad[0]) {
	DEVIATION = (double*)R_alloc(KC, sizeof(double));
	for (i = 0; i < KC; i++) {
	    DEVIATION[i] = s = O[i] - E[i];
	    val += s * s;
	}
    } else {
	for (i = 0; i < KC; i++)
	    val += pow(O[i] - E[i], 2);
	return val;
    }

    /****************************/
    /* the gradients start here */
    /****************************/

    double *bar_O, *bar_O2, *S;
    bar_O2 = (double*)R_alloc(KH, sizeof(double));
    bar_O = (double*)R_alloc(KC, sizeof(double));
    S = (double*)R_alloc(KC, sizeof(double));

    for (i = 0; i < KC; i++) {
	tmp = bar_O[i] = bar(O[i]);
	S[i] = DEVIATION[i] * tmp;
    }

    for (i = 0; i < KH; i++)
	bar_O2[i] = bar(O2[i]);

    /* gradients of B2 */
    if (eval_grad[0] & 0x10) { // 0x10 = 0001 0000
	from = 0; to = K;
	for (j = 0; j < C; j++) {
	    s = 0;
	    for (i = from; i < to; i++)
		s += S[i];
	    gradient_bias_HC[j] = 2 * s;
	    from += K; to += K;
	}
    }

    // matrix versions (only for W3 and B1)

    /* gradients of W3 */
    if (eval_grad[0] & 0x04) { // 0x04 = 0000 0100
	fast_mat_prod_4(O2, S, gradient_W3, K, C, H);
	for (i = 0; i < HC ; i++)
	    gradient_W3[i] *= twiceBETA;
    }

    /* gradients of B1 */
    if (eval_grad[0] & 0x08) { // 0x08 = 0000 1000
	double *RES;
	RES = (double*)R_alloc(HC, sizeof(double));

	fast_mat_prod_4(S, bar_O2, RES, K, H, C);
	/* we could do:
	   `fast_mat_prod_4(bar_O2, S, RES, K, C, H);`
	   in which case RES would the transpose and
	   we'd have to do `RES[i]...; i += H;`,
	   i.e., scan its columns rowwise */

	i = 0;
	for (h = 0; h < H; h++) {
	    s = 0;
	    for (j = 0; j < C; j++)
		s += RES[i++] * w3[h + j * H];
	    gradient_bias_HH[h] = twiceBETA * s;
	}
    }
    // end of matrix versions

    /* gradients of W2 */
    if (eval_grad[0] & 0x02) { // 0x02 = 0000 0010
	h = ((int)eval_grad[2]) - 1;
	do_Gradients_W2(w3, bar_O2, O1, S, gradient_W2, h);
    }

    /* gradients of W1 */
    if (eval_grad[0] & 0x01) { // 0x01 = 0000 0001
	i = ((int)eval_grad[1]) - 1;
	do_Gradients_W1(sigma_xi, w2, w3, bar_O2, O1, S, gradient_W1, i);
    }

    /* FILE *fo/\* , *fb, *fb2, *fb3, *fb4, *fb5 *\/; */
    /* fo = fopen("o_", "w"); */
    /* fb = fopen("b_", "w"); */
    /* fb2 = fopen("b2_", "w"); */
    /* fb3 = fopen("b3_", "w"); */
    /* fb4 = fopen("b4_", "w"); */
    /* fb5 = fopen("b5_", "w"); */
    /* fwrite(O, sizeof(double), K*C, fo); */
    /* fwrite(gradient_W1, sizeof(double), np1, fb); */
    /* fwrite(gradient_W2, sizeof(double), np2, fb2); */
    /* fwrite(gradient_W3, sizeof(double), np3, fb3); */
    /* fwrite(gradient_bias_HH, sizeof(double), H, fb4); */
    /* fwrite(gradient_bias_HC, sizeof(double), C, fb5); */
    /* fclose(fo); */
    /* fclose(fb); */
    /* fclose(fb2); */
    /* fclose(fb3); */
    /* fclose(fb4); */
    /* fclose(fb5); */

    return val;
}

void do_Hessian_block(double *PARA, double *sigma_xi, int *E,
		      double *GRAD, double *hessian)
{
    int i, j, from = 0;
    double delta = 1E-8, *new_grad, tmp, s;
    new_grad = (double*)R_alloc(npar, sizeof(double));

    memset(hessian, 0, npar * npar * sizeof(double));

    /* do the diagonal of blocks for W1 */
    for (i = 0; i < np1; i++) { // go down the rows of hessian[]
	tmp = PARA[i];
	PARA[i] += delta;
	eval_grad[0] = 0x01;
	eval_grad[1] = (unsigned char) 1 + i/N;
	objfun_6(PARA, sigma_xi, E, new_grad);
	PARA[i] = tmp;
	/* start at the border of the block until the diagonal and
	   fill the row: */
	for (j = from; j <= i; j++) {
	    s = (new_grad[j] - GRAD[j]) / delta;
	    hessian[i + j*npar] = hessian[j + i*npar] = s;
	}
	if (i > 0 && !(i % N)) from += N;
    }

    from = np1;
    // do the diagonal of blocks for W2
    for (; i < np_cumul2; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	eval_grad[0] = 0x02;
	eval_grad[2] = (unsigned char) 1 + i/H;
	objfun_6(PARA, sigma_xi, E, new_grad);
	PARA[i] = tmp;
	for (j = from; j <= i; j++) {
	    s = (new_grad[j] - GRAD[j]) / delta;
	    hessian[i + j*npar] = hessian[j + i*npar] = s;
	}
	if (i > np1 && !(i % H)) from += H;
    }

    from = np_cumul2;
    // do  W3
    for (; i < np_cumul3; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	eval_grad[0] = 0x04;
	// eval_grad[2] = (unsigned char) 1 + i/H;
	objfun_6(PARA, sigma_xi, E, new_grad);
	PARA[i] = tmp;
	for (j = from; j <= i; j++) {
	    s = (new_grad[j] - GRAD[j]) / delta;
	    hessian[i + j*npar] = hessian[j + i*npar] = s;
	}
	// if (i > np_cumul2 && !(i % H)) from += H;
    }

    from = np_cumul3;
    // do  B1
    for (; i < np_cumul4; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	eval_grad[0] = 0x08;
	// eval_grad[2] = (unsigned char) 1 + i/H;
	objfun_6(PARA, sigma_xi, E, new_grad);
	PARA[i] = tmp;
	for (j = from; j <= i; j++) {
	    s = (new_grad[j] - GRAD[j]) / delta;
	    hessian[i + j*npar] = hessian[j + i*npar] = s;
	}
	// if (i > np_cumul3 && !(i % H)) from += H;
    }

    from = np_cumul4;
    // do  B2
    for (; i < npar; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	eval_grad[0] = 0x10;
	// eval_grad[2] = (unsigned char) 1 + i/H;
	objfun_6(PARA, sigma_xi, E, new_grad);
	PARA[i] = tmp;
	for (j = from; j <= i; j++) {
	    s = (new_grad[j] - GRAD[j]) / delta;
	    hessian[i + j*npar] = hessian[j + i*npar] = s;
	}
	// if (i > np_cumul4 && !(i % H)) from += H;
    }

    return;

    /* do the rest of the parameters "full row" */
    for (; i < npar; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	eval_grad[0] = 0x1f;
	eval_grad[1] = 0x00;
	eval_grad[2] = 0x00;
	objfun_6(PARA, sigma_xi, E, new_grad);
	PARA[i] = tmp;
	/* start at 1st column (j = 0) and fill the row until the
	   diagonal (included) */
	for (j = 0; j <= i; j++) {
	    s = (new_grad[j] - GRAD[j]) / delta;
	    hessian[i + j*npar] = hessian[j + i*npar] = s;
	}
    }
}

void do_Hessian_full(double *PARA, double *sigma_xi, int *E,
		     double *GRAD, double *hessian)
{
    int i, j, k = 0;
    double delta = 1E-8, *new_grad, tmp;
    new_grad = (double*)R_alloc(npar, sizeof(double));

    for (i = 0; i < npar; i++) {
	tmp = PARA[i];
	PARA[i] += delta;
	objfun_6(PARA, sigma_xi, E, new_grad);
	for (j = 0; j < npar; j++)
	    hessian[k++] = (new_grad[j] - GRAD[j]) / delta;
	PARA[i] = tmp;
    }
}

#define COMPUTE_PROPOSED_PARA				\
    memcpy(ptr_para[!Switch], ptr_para[Switch],		\
	   npar * sizeof(double));			\
    for (i = 0; i < npar; i++)				\
	ptr_para[!Switch][i] += alpha * p_by_grad[i]

double optimize_6(double *PARA, int *sigma, int *xi,
		  int *expec, int iterlim, int quiet)
{
    int i, k, iter = 0, mu, withHessian;
    double s, sb, res, resb, *sigma_xi, alpha, *p_by_grad;

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
       only once (may take a lot of memory if K is large...
       => can overwrite 'xi' instead???) */
    k = 0;
    for (i = 0; i < N; i++) {
	s = (double)sigma[i];
	for (mu = 0; mu < K; mu++) {
	    sigma_xi[k] = s * (double)xi[k];
	    k++;
	}
    }

    /* do a first iteration to get the gradients: */
    eval_grad[0] = 0x1f;
    eval_grad[1] = 0x00;
    eval_grad[2] = 0x00;
    res = objfun_6(PARA, sigma_xi, expec, GRAD);

    /*
    FILE *fgrad;
    fgrad = fopen("GRAD", "w");
    fwrite(GRAD, sizeof(double), npar, fgrad);
    fclose(fgrad);
    */

    /*
    do_Hessian_block(PARA, sigma_xi, expec, GRAD, hessian);

    FILE *fh;
    fh = fopen("hessian", "w");
    fwrite(hessian, sizeof(double), npar*npar, fh);
    fclose(fh);
    */

    if (!quiet) {
	Rprintf("INITIALIZATION -- iteration %d\tobj_fun = %.3f\n", iter, res);
	Rprintf("gradients done.\n");
    }

    for (;;) {
	if (iter >= iterlim) break;

	if (control_list[0]) { // try quasi-Newton
	    mu = control_list[2];
	    control_list[2] = 0;
	    if (control_list[1]) {
		do_Hessian_full(ptr_para[Switch], sigma_xi, expec, ptr_grad[Switch], hessian);
	    } else {
		do_Hessian_block(ptr_para[Switch], sigma_xi, expec, ptr_grad[Switch], hessian);
	    }
	    control_list[2] = mu;
	    k = fast_mat_inv(hessian, npar);
	    withHessian = k == 0 ? 1 : 0;
	} else {
	    withHessian = 0;
	}

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
	    /* compute the proposed new parameters */
	    COMPUTE_PROPOSED_PARA;
	    /* get the obj_fun value using the proposed parameters */
	    eval_grad[0] = 0x00;
	    resb = objfun_6(ptr_para[!Switch], sigma_xi, expec,
			    ptr_grad[!Switch]);
	    if (resb < res) {
		/* Rprintf("  Hessian "); */
		/* if (withHessian) Rprintf("good"); else Rprintf("bad"); */
		/* Rprintf(": alpha=%.11f\n", alpha); */
		goto update;
	    }
	    alpha /= 2;
	    if (alpha < 1E-10) {
		if (!withHessian) {
		    /* Rprintf("  WRONG: alpha=%.11f\n", alpha); */
		    goto update;
		}
		/* use p = -GRAD instead */
		withHessian = 0;
		goto start;
	    }
	}

	/* check Wolfe's conditions */
	if (control_list[3]) {
	    fast_mat_prod_4(p, ptr_grad[Switch], &s, npar, 1, 1); // s = p^t GRAD
	    fast_mat_prod_4(p, ptr_grad[!Switch], &sb, npar, 1, 1);
	    Rprintf("Wolfe's conditions 1 \n");
	    if (resb <= res + c1 * alpha * s)
		Rprintf("good\n"); else Rprintf("bad\n");
	    Rprintf("Wolfe's conditions 2 \n");
	    if (sb <= -c2 * s)
		Rprintf("good\n"); else Rprintf("bad\n");
	}

update:
	control_list[2] = mu;
	if (resb < res) {
	    Switch = !Switch;
	} else {
	    /* reinitialize the parameters */
	    for (i = 0; i < np_cumul3; i++)
		ptr_para[Switch][i] = runif_local();
	    memset(ptr_para[Switch] + np_cumul3, 0,
		   (H + C) * sizeof(double));
	}

	/* update the gradients (and 'res') */
	eval_grad[0] = 0x1f;
	eval_grad[1] = 0x00;
	eval_grad[2] = 0x00;
	res = objfun_6(ptr_para[Switch], sigma_xi, expec,
		       ptr_grad[Switch]);

	if (!quiet) Rprintf("\riteration %d\tobj_fun = %.3f", iter + 1, res);

	if (res < target) break;
	R_CheckUserInterrupt();
	iter++;
    }

    if (!quiet && iterlim) Rprintf("\n");
    if (Switch) memcpy(PARA, new_PARA, npar * sizeof(double));
    return res;
}

SEXP test_6(SEXP W1, SEXP W2, SEXP W3, SEXP BIAS_HH, SEXP BIAS_HC,
	    SEXP SIGMA, SEXP XI, SEXP EXPEC, SEXP ITERLIM, SEXP QUIET,
	    SEXP CTRL, SEXP TARGET, SEXP beta, SEXP MC_CORES)
{
    int *sigma, *xi, *expec, iterlim;
    double *w1, *w2, *w3, *bias_HH, *bias_HC, res;

    PROTECT(W1 = coerceVector(W1, REALSXP));
    PROTECT(W2 = coerceVector(W2, REALSXP));
    PROTECT(W3 = coerceVector(W3, REALSXP));
    PROTECT(BIAS_HH = coerceVector(BIAS_HH, REALSXP));
    PROTECT(BIAS_HC = coerceVector(BIAS_HC, REALSXP));
    PROTECT(SIGMA = coerceVector(SIGMA, INTSXP));
    PROTECT(XI = coerceVector(XI, INTSXP));
    PROTECT(EXPEC = coerceVector(EXPEC, INTSXP));
    PROTECT(ITERLIM = coerceVector(ITERLIM, INTSXP));
    PROTECT(QUIET = coerceVector(QUIET, INTSXP));
    PROTECT(CTRL = coerceVector(CTRL, INTSXP));
    PROTECT(TARGET = coerceVector(TARGET, REALSXP));
    PROTECT(beta = coerceVector(beta, REALSXP));
    PROTECT(MC_CORES = coerceVector(MC_CORES, INTSXP));

    /* assign the global parameters */
    N = ncols(XI);
    K = nrows(XI);
    H = ncols(W2);
    C = ncols(EXPEC);
    np1 = N * H;
    np2 = H * H;
    np3 = H * C;
    np_cumul2 = np1 + np2;
    np_cumul3 = np_cumul2 + np3;
    np_cumul4 = np_cumul3 + H;
    npar = np_cumul4 + C;
    KH = K * H;
    KC = K * C;
    HC = H * C;

    memcpy(&control_list, INTEGER(CTRL), 4 * sizeof(int));

    target = REAL(TARGET)[0];
    BETA = REAL(beta)[0];
    mc_cores = INTEGER(MC_CORES)[0];

    twiceBETA = 2 * BETA;
    twiceBETA_SQUARE = 2 * pow(BETA, 2);
    twiceBETA_CUBE = 2 * pow(BETA, 3);

    w1 = REAL(W1);
    w2 = REAL(W2);
    w3 = REAL(W3);
    bias_HH = REAL(BIAS_HH);
    bias_HC = REAL(BIAS_HC);
    sigma = INTEGER(SIGMA);
    xi = INTEGER(XI);
    expec = INTEGER(EXPEC);

    iterlim = INTEGER(ITERLIM)[0];

    double *PARA;
    PARA = (double*)R_alloc(npar, sizeof(double));
    memcpy(PARA, w1, np1 * sizeof(double));
    memcpy(PARA + np1, w2, np2 * sizeof(double));
    memcpy(PARA + np_cumul2, w3, np3 * sizeof(double));
    memcpy(PARA + np_cumul3, bias_HH, H * sizeof(double));
    memcpy(PARA + np_cumul4, bias_HC, C * sizeof(double));

    res = optimize_6(PARA, sigma, xi, expec, iterlim,
		     INTEGER(QUIET)[0]);

    /* for the moment: copy back the estimates into the original
       objects */
    memcpy(w1, PARA, np1 * sizeof(double));
    memcpy(w2, PARA + np1, np2 * sizeof(double));
    memcpy(w3, PARA + np_cumul2, np3 * sizeof(double));
    memcpy(bias_HH, PARA + np_cumul3, H * sizeof(double));
    memcpy(bias_HC, PARA + np_cumul4, C * sizeof(double));

    UNPROTECT(14);

    return ScalarReal(res);
}
