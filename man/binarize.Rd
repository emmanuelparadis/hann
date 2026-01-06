\name{binarize}
\alias{binarize}
\title{Helper Function to Prepare Data From Images}
\description{
  Take one or more images coded as numeric values (pixels) and return
  them coded as -1 or +1 depending on a threshold value.
}
\usage{
binarize(x, threshold = median(x))
}
\arguments{
  \item{x}{a vector or a matrix of numeric values.}
  \item{threshold}{a numeric value: values in \code{x} below (or equal)
    to this threshold are coded -1, values above are coded +1.}
}
\details{
  It is recommended to check that the default of the argument
  \code{threshold} is appropriate or not. For instance, if an image has
  many pixels dark and a few light, this default might not be a good
  choice.

  If the default is a good choice (e.g., good balance between the dark
  and light pixels), it can be applied globally or separately for each
  image. Suppose the images (patterns) are arranged along the rows of a
  matrix \code{X}, the binarization can be done either with:

  \code{xi <- t(apply(X, 1, binarize))}

  where the threshold might be different for each image, or with
  \code{xi <- binarize(X)} in which case the threshold will be the same
  for all images.
}
\value{
  an object with integers (-1/1) with the same attributes (dim, names,
  etc.) than the input object \code{x}.
}
\seealso{\code{\link{hann}}}
\examples{
## a plus (+) sign on a 3x3 grid:
x <- matrix(runif(9, 150, 255), 3, 3)
## make the corners lighter to draw the "+":
x[c(1, 3, 7, 9)] <- runif(4, 0, 20)
x <- round(x)
xi <- binarize(x)
## compare:
xi; x
layout(matrix(1:2, 1))
image(x, asp = 1, main = "Original image")
image(xi, asp = 1, main = "Binarized image")
layout(1)
}
\keyword{manip}
