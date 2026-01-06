\name{buildSigma}
\alias{buildSigma}
\title{Hopfield Network Energy}
\description{
  Minimize the energy of the Hopfield network.
}
\usage{
buildSigma(xi, n = 20, nrep = 100, quiet = FALSE)
}
\arguments{
  \item{xi}{a matrix of patterns coded with 1 and -1.}
  \item{n}{the parameter of the energy function (integer).}
  \item{nrep}{the number of attempts.}
  \item{quiet}{a logical value indicating whether to print the details
    for each attempt.}
}
\details{
  The number of columns in \code{xi} is equal to the size of the
  Hopfield network (the number of input neurons denoted as N),
  whereas the number of columns is the number of memories denoted as K
  (Krotov and Hopfield, 2016).

  A random vector `sigma' is first generated and then updated in order
  to minimize the energy level of the Hopfield network. The convergence
  to a low energy level depends on the initial values in `sigma', so the
  procedure is repeated several times. The vector with the lowest energy
  level is returned.
}
\value{
  a vector of integers (-1/1). The length of this vector (N) is equal to
  the number of columns in \code{xi}.}
\references{
  Krotov, D. and Hopfield, J. J. (2016) Dense associative memory for
  pattern recognition. \doi{10.48550/ARXIV.1606.01164}.
}
\seealso{\code{\link{hann}}}
\examples{
xi <- matrix(NA, K <- 1000, N <- 60)
xi[] <- sample(c(1L, -1L), K * N, TRUE)
(sigma <- buildSigma(xi))
}
\keyword{manip}
