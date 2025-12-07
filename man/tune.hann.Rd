\name{tune.hann}
\alias{tune.hann}
\title{Tune Hyperparameters}
\description{
  Tuning the two hyperparameters of the neural nets: the number of
  hidden neurons and beta (slope of the activation function).
}
\usage{
tune.hann(xi, sigma, classes,
          ranges = list(H = seq(10, 50, by = 10),
                        beta = seq(0.2, 0.8, by = 0.1)),
          nrepeat = 10,
          control = control.hann(iterlim = 20))
}
\arguments{
  \item{xi}{a matrix of patterns with K rows.}
  \item{sigma}{a vector coding the Hopfield network.}
  \item{classes}{the classes of the patterns (vector of length K).}
  \item{ranges}{a list giving the values of the parameters to be
    tested.}
  \item{nrepeat}{the number of repeats}
  \item{control}{the control parameters.}
}
\details{
  The effect of the hyperparameters is usually visible with a small
  number of iterations of. The fitting process is repeat several times
  for each combination of hyperparameter values.
}
\value{
  a data frame with four columns:

  \item{H}{the number of hidden neurons}
  \item{beta}{the values of the slope of the activation function}
  \item{mean}{the mean of the error rate computed over the repeats}
  \item{sd}{the standard-deviation}
}
\author{Emmanuel Paradis}
\seealso{\code{\link{hann}}}
\examples{
## simulate 200 random patterns with 30 pixels:
v <- c(-1L, 1L)
K <- 200L
N <- 30L
xi <- matrix(sample(v, K*N, TRUE), K, N)
stopifnot(nrow(unique(xi)) == K)
## build the vector sigma:
sig <- buildSigma(xi, quiet = TRUE)
## define the classes:
cl <- rep(1:2, each = K/2)
## the ranges:
ranges <- list(H = seq(10, 60, by = 10),
               beta = seq(0.1, 1, .1))
ctr <- control.hann(iterlim = 10)
res <- tune.hann(xi, sig, cl, ranges, control = ctr, nrepeat = 5)
str(res)
## visualize the results:
library(lattice)
levelplot(mean ~ beta * H, data = res, main = "Mean")
levelplot(sd ~ beta * H, data = res, main = "Standard-deviation")
}
\keyword{hmodel}
