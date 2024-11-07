\name{hann1}
\alias{hann1}
\alias{print.hann1}
\title{One-layer Hopfield ANN}
\description{
  This optimizes a one-layer Hopfield-based artificial neural
  network. The structure of the network is quite simple: a Hopfield
  network with N input neurons all connected to C output neurons. The
  number of parameters (N and C) is determined by the input data:
  \code{xi} has N columns (which is also the length of \code{sigma}) and
  the number of unique values of \code{classes} is equal to C.

  See the vignette of this package for an example and some background.
}
\usage{
hann1(xi, sigma, classes, net = NULL, control = control.hann())

\method{print}{hann1}(x, details = FALSE, ...)
}
\arguments{
  \item{xi}{a matrix of patterns with K rows.}
  \item{sigma}{a vector coding the Hopfield network.}
  \item{classes}{the classes of the patterns (vector of length K).}
  \item{net, x}{an object of class \code{"hann1"}.}
  \item{control}{the control parameters.}
  \item{details}{a logical value (whether to print the parameter values
    of the network).}
  \item{\dots}{further arguments passed to \code{print.default}.}
}
\details{
  By default, the parameters of the neural network are initialized with
  random values from a uniform distribution between -1 and 1 (except the
  biases which are initialized to zero).

  If an object of \code{"hann1"} is given to the argument \code{net},
  then its parameter values are used to initialize the parameters of the
  network.

  The main control parameters are given as a list to the \code{control}
  argument. They are detailed in the page of the function
  \code{\link{control.hann}()}.

}
\value{
  an object of class \code{"hann1"} with the following elements:

  \item{parameters}{a list with one matrix, \code{W}, and one vector,
    \code{bias}.}
  \item{sigma}{the Hopfield network.}
  \item{beta}{the hyperparameter of the activation function.}
  \item{call}{the function call.}
}
\references{
  Hopfield, J. J. (1982) Neural networks and physical systems with
  emergent collective computational abilities. \emph{Proceedings of the
    National Academy of Sciences, USA}, \bold{79}, 2554--2558.
  \doi{10.1073/pnas.79.8.2554}.

  Krotov, D. and Hopfield, J. J. (2016) Dense associative memory for
  pattern recognition. \doi{10.48550/ARXIV.1606.01164}.
}
\author{Emmanuel Paradis}
\seealso{\code{\link{buildSigma}}, \code{\link{predict.hann1}}}
\keyword{hmodel}
