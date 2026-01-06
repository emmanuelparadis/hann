\name{hann.R}
\alias{hann}
\alias{print.hann}
\alias{summary.hann}
\alias{str.hann}
\alias{plot.hann}
\alias{coef.hann}
\alias{fitted.hann}
\alias{labels.hann}
\alias{predict.hann}
\title{Method Top-Level Functions}
\description{
  Functions to fit Hopfield artificial neural networks or to access the
  results.
}
\usage{
hann(xi, sigma, classes, H = NULL, labels = NULL, net = NULL,
     control = control.hann())
\method{print}{hann}(x, ...)
\method{summary}{hann}(object, ...)
\method{str}{hann}(object, ...)
\method{plot}{hann}(x, y, type = "h", ...)
\method{coef}{hann}(object, ...)
\method{fitted}{hann}(object, ...)
\method{labels}{hann}(object, ...)
\method{predict}{hann}(object, ...)
}
\arguments{
  \item{xi}{a matrix of patterns with K rows and N columns.}
  \item{sigma}{a vector coding the Hopfield network (length N).}
  \item{classes}{the classes of the patterns (vector of length K).}
  \item{H}{the number of neurons in the hidden layer (can be 0).}
  \item{labels}{a vector of labels used for the classes.}
  \item{net, x, object}{an object inheriting class \code{"hann"}.}
  \item{control}{the control parameters.}
  \item{y}{(unused).}
  \item{type}{the type of plot for vectors of parameters (biases).}
  \item{\dots}{options passed to other methods.}
}
\details{
  \code{hann()} calls either \code{hann1()} or \code{hann3()} depending
  on the value given to the argument \code{H} (the number of hidden
  neurons).

  The other functions are (standard) methods for accessing the results.
}
\value{
  \code{hann} returns an object of class \code{c("hann", "hann1")} or
  \code{c("hann", "hann3")}; see the links below for their description.
}
\seealso{\code{\link{hann1}}, \code{\link{hann3}}}
\examples{
## function to create 'images' default size is 9x9 pixels,
## with 4 possible shapes ("V"ertical, "H"orizontal, "U"p-diag.
## or "D"own-diag)
rpat <- function(type, nr = 9L, nc = 9L,
                 signal = c(200, 255), noise = c(0, 50))
{
    ij <- switch(type,
                 "V" = cbind(1:nr, ceiling(nc/2)),
                 "H" = cbind(ceiling(nr/2), 1:nc),
                 "U" = cbind(nr:1, 1:nc),
                 "D" = cbind(1:nr, 1:nc))
    x <- matrix(runif(nr * nc, noise[1], noise[2]), nr, nc)
    x[ij] <- runif(nr, signal[1], signal[2])
    round(x)
}

## the 4 types of patterns to simulate:
labs <- c("V", "H", "U", "D")
## repeat them 40 times, this will be used as class-vector:
cl <- rep(labs, each = 40)
## simulate the images:
xi <- t(sapply(cl, rpat))
## binarize the patterns (-1/+1):
xi <- binarize(xi)
## build the sigma vector:
sig <- buildSigma(xi, quiet = TRUE)

## optimize the neural net with 10 hidden neurons:
ctr <- control.hann(quiet = TRUE)
nt <- hann(xi, sig, cl, H = 10, control = ctr)

## convergence depends on the initial parameter values, so it might be
## needed to repeat the previous command a few times so that the next
## one shows only values on the diagonal (which can be reached with
## the default 100 iterations)

table(cl, predict(nt, xi, rawsignal = FALSE))

## now generate 10 new patterns...
new_cl <- rep(labs, each = 10)
new_xi <- binarize(t(sapply(new_cl, rpat)))
## ... and see how well they are predicted:
table(new_cl, predict(nt, new_xi, rawsignal = FALSE))

## visualize the optimized neural net
layout(matrix(1:6, 2, 3, TRUE))
plot(nt)
layout(1)
}
\keyword{manip}
