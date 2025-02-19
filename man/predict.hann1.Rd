\name{predict.hann1}
\alias{predict.hann1}
\alias{predict.hann3}
\title{Prediction}
\description{
  Classification of patterns with Hopfield-based artificial neural
  networks.
}
\usage{
\method{predict}{hann1}(object, patterns, rawsignal = TRUE, ...)
\method{predict}{hann3}(object, patterns, rawsignal = TRUE, ...)
}
\arguments{
  \item{object}{an object of class \code{"hann1"} or \code{"hann3"}.}
  \item{patterns}{the patterns to be classified.}
  \item{rawsignal}{a logical value (see details).}
  \item{\dots}{(ignored).}
}
\details{
  The patterns have to be coded in the same way than the matrix
  \code{xi} used to train the networks.

  If \code{rawsignal = TRUE}, the raw signal of each neuron is output
  for each pattern. Otherwise, a classification of each pattern is done
  by finding the neuron with the largest signal.
}
\value{
  If \code{rawsignal = TRUE} a matrix; if \code{rawsignal = FALSE} a vector.
}
\author{Emmanuel Paradis}
\seealso{\code{\link{hann1}}, \code{\link{hann3}}}
\keyword{hmodel}
