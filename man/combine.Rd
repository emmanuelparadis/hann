\name{combine}
\alias{combine}
\title{Combine Several Neural Nets for Prediction}
\description{
  Combine several neural networks optimized with different sets of
  (overlapping) classes. The patterns (\code{xi}) are classified with
  all networks and the signal are combined to make a final prediction.
}
\usage{
combine(nets, xi)
}
\arguments{
  \item{nets}{a list of objects inheriting class \code{"hann"}.}
  \item{xi}{a matrix of patterns.}
}
\details{
  Suppose we want to classify (predict) patterns labelled A, B, and C,
  then we can optimize two networks: one classifying A and B, and
  another one classifying B and C. The present function combines the
  predictions from both networks to classify patterns into the three
  classes.

  Since optimizing a network with many classes can be complicated, it
  might be a better strategy to optimize separate networks with two
  classes each, and then combine them to make predictions over all
  classes.
}
\value{
  a vector of predicted classes (as coded by the labels).
}
\seealso{
  \code{\link{hann}}, \code{\link{predict.hann1}}
}
\examples{
## see ?hann for explanation on rpat():
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

## take 3 patterns:
labs <- c("V", "H", "U")
## repeat each label 40 times:
cl <- rep(labs, each = 40)
## simulate a pattern for each label:
xi <- t(sapply(cl, rpat))
## binarize the patterns:
xi <- binarize(xi)

## we take only 2 patterns in turn
## S: list with the indices for each pair
S <- list(1:80, c(1:40, 81:120), 41:120)
P <- list(c("V", "H"), c("V", "U"), c("H", "U"))

cl2 <- gl(2, 40) # only 2 classes for each net
## so the class (1 or 2) will be the same for each net
## but the labels will be different (from the list 'P')

ctr <- control.hann(quiet = TRUE)

NT <- vector("list", 3) # to store the optimized nets
for (i in 1:3) {
    s <- S[[i]]
    sig <- buildSigma(xi[s, ], quiet = TRUE)
    NT[[i]] <- hann3(xi[s, ], sig, cl2, H = 10,
                     labels = P[[i]], control = ctr)
}

## final predictions:
table(cl, combine(NT, xi))
}
\keyword{hmodel}
