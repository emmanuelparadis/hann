# hann: Hopfield Artificial Neural Networks

'hann' is a R package for building and optimizing Hopfield-based artificial neural networks. Its implementation was made from scratch with (notably):

* Use of BLAS/LAPACK for most matrix operations.
* All optimizations are coded in C (with efficient use of arrays).
* Some parallel code make use of OpenMP (although this is limited for the moment).
* A gradient-based optimization algorithm is used, as well as a quasi-Newton version (the latter is recommended for small networks only).

Hopfield networks have the property to be able to learn many patterns with a small number of input neurons making possible to build parsimonious neural networks.

## Installation

hann must be compiled in the standard way for R packages. For the moment, there are no pre-compiled packages. A submission to CRAN is planned to happen soon.

It is recommended to have an efficient BLAS/LAPACK library installed on your system (e.g., OpenBLAS).

If OpenMP is available on your system, parallel (multicore) code is compiled. For the moment, this is available only for the function hann1().

## Examples

There is a vignette with several examples. All functions are documented with small examples.	
