\name{hann3}
\alias{hann3}
\alias{print.hann3}
\title{Three-layer Hopfield ANN}
\description{
  This optimizes a three-layer Hopfield-based artificial neural
  network. The network is made of a Hopfield network with N input
  neurons all connected to H hidden neurons. The latter are all
  connected together (convolution) which is equivalent to defining two
  hidden layers. Each hidden neuron is connected to C output
  neurons. The values of the parameters N and C are determined by the
  input data: \code{xi} has N columns (which is also the length of
  \code{sigma}) and the number of unique values of \code{classes} is
  equal to C. The value of H must be given by the user (a default of
  half the number of input neurons is defined).

  See the vignette of this package for an example.
}
\usage{
hann3(xi, sigma, classes,  H = 0.5 * length(sigma),
      labels = NULL, net = NULL, control = control.hann())

\method{print}{hann3}(x, details = FALSE, ...)
}
\arguments{
  \item{xi}{a matrix of patterns with K rows.}
  \item{sigma}{a vector coding the Hopfield network.}
  \item{classes}{the classes of the patterns (vector of length K).}
  \item{H}{the number of neurons in the hidden layer; by default half
    the number of input neurons (rounded to the lowest integer if the
    latter is odd).}
  \item{labels}{a vector of labels used for the classes.}
  \item{net, x}{an object of class \code{"hann3"}.}
  \item{control}{the control parameters.}
  \item{details}{a logical value (whether to print the parameter values
    of the network).}
  \item{\dots}{further arguments passed to \code{print.default}.}
}
\details{
  By default, the parameters of the neural network are initialized with
  random values from a uniform distribution between -1 and 1 (expect the
  biases which are initialized to zero).

  If an object of \code{"hann3"} is given to the argument \code{net},
  then its parameter values are used to initialize the parameters of the
  network.

  The main control parameters are given as a list to the \code{control}
  argument. They are detaild in the page of the function
  \code{\link{control.hann}()}.
}
\value{
  an object of class \code{"hann3"} with the following elements:

  \item{parameters}{a list with three matrices, \code{W1}, \code{W2},
    and \code{W3}, and two vectors, \code{bias1} and \code{bias3}.}
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
\seealso{
  \code{\link{buildSigma}}, \code{\link{control.hann}},
  \code{\link{predict.hann3}}
}
\keyword{hmodel}
