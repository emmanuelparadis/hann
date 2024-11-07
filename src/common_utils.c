#ifndef  USE_FC_LEN_T
# define USE_FC_LEN_T
#endif
#include <Rconfig.h>
#include <R_ext/BLAS.h>
#ifndef FCONE
# define FCONE
#endif

#include "hann.h"

/* #include <stdint.h> */
/* #define SCALING ((double)UINT32_MAX) / 2 */

/* double runif_local() // x ~ U(-1, 1) */
/* { */
/*     double x; */
/*     x = (double)arc4random(); */
/*     x /= SCALING; */
/*     x -= 1; */
/*     return x; */
/* } */

double runif_local() // x ~ U(-1, 1)
{
    double x;

    GetRNGstate();
    x = unif_rand();
    PutRNGstate();

    x *= 2;
    x -= 1;
    return x;
}

/* X: n*p matrix
   Y: n*m matrix
   Z: p*m matrix (Z = X^t Y) */
void fast_mat_prod_4(double *X, double *Y, double *Z, int n, int m, int p)
{
    double one = 1, zero = 0;

    F77_CALL(dgemm)("T", "N", &p, &m, &n, &one, X,
		    &n, Y, &n, &zero, Z, &p FCONE FCONE);
}

/* X: n*p matrix
   Y: p*m matrix
   Z: n*m matrix (Z = XY) */
void fast_mat_prod_0(double *X, double *Y, double *Z, int n, int m, int p)
{
    double one = 1, zero = 0;

    F77_CALL(dgemm)("N", "N", &n, &m, &p, &one, X,
		    &n, Y, &p, &zero, Z, &n FCONE FCONE);
}

int fast_mat_inv(double *X, int n)
{
    int info, *ipiv, lwork = n * n, i, j, i0, k1, k2;
    double *work;

    /* lwork=n^2 makes it faster if n is large (compared to lwork=n) */

    ipiv = (int*)R_alloc(n, sizeof(int));
    work = (double*)R_alloc(lwork, sizeof(double));

    /* F77_CALL(dsytrf)("L", &n, X, &n, ipiv, work, &lwork, &info FCONE); */
    /* if (info) error("DSYTRF had error code %d", info); */
    /* F77_CALL(dsytri)("L", &n, X, &n, ipiv, work, &info FCONE); */
    /* if (info) error("DSYTRI had error code %d", info); */

    F77_CALL(dgetrf)(&n, &n, X, &n, ipiv, &info);
    if (info) return info;
    F77_CALL(dgetri)(&n, X, &n, ipiv, work, &lwork, &info);
    if (info) return info;

    i0 = 1;
    for (j = 0; j < n; j++) {
	k1 = i0 + j * n;
	k2 = j + i0 * n;
	for (i = i0; i < n; i++) {
	    X[k2] = X[k1];
	    k1++;
	    k2 += n;
	}
	i0++;
    }
    return 0;
}

int do_error_rate(int *E, double *O, int K, int C)
{
    int i, j, k, c = 0, jmax, emax, max_E;
    double max_O;

    for (i = 0; i < K; i++) {
	jmax = emax = 0;
	max_O = O[i];
	max_E = E[i];
	k = i;
	for (j = 1; j < C; j++) {
	    k = i + j*K;
	    if (O[k] > max_O) {
		jmax = j;
		max_O = O[k];
	    }
	    if (E[k] > max_E) {
		emax = j;
		max_E = E[k];
	    }
	    k += K;
	}
	if (jmax != emax) c++;
    }

    return c;
}


