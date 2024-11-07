\name{control.hann}
\alias{control.hann}
\title{Parameters for Neural Network Optimization}
\description{
  Set the parameters for the Hopfield artificial neural network
  optimization.
}
\usage{
control.hann(...)
}
\arguments{
  \item{\dots}{named arguments to be modified (see examples).}
}
\details{
  When the user modifies one or several parameters by giving them as
  named arguments, if some names are incorrect they are ignored with a
  warning.

  The parameters with their default values are:

  \itemize{
    \item \code{iterlim = 100}: an integer giving the number of
    iterations.

    \item \code{quiet = FALSE}: a logical controlling whether to print
    the value of the objective (loss) function at each iteration.

    \item \code{quasinewton = FALSE}: a logical. If \code{TRUE},
    quasi-Newton steps are performed (not recommended unless for
    networks with few parameters and/or for a small number of
    iterations).

    \item \code{fullhessian = FALSE}: (ignored if \code{quasinewton =
    FALSE}) a logical, by default only some blocks of the Hessian matrix
    are computed. If \code{TRUE}, the full Hessian matrix is computed
    (very time consuming).

    \item \code{trace.error = FALSE}: a logical. If \code{TRUE}, the
    error rate is printed at each iteration of the optimization process.

    \item \code{wolfe = FALSE}: a logical. If \code{TRUE}, Wolfe's
    conditions are tested and printed at each iteration.

    \item \code{target = 0.001}: the target value of the loss function
    to stop the optimization.

    \item \code{beta = 0.2}: the hyperparameter of the activation
    function.

    \item \code{mc.cores = 1}: an integer. The number of cores used when
    computing the loss function.
  }

  If \code{mc.cores} is greater than one, the optimization process calls
  a multithreaded code using OMP. So, do \emph{not} do this together
  with functions from the package \pkg{parallel}. On the other hand, if
  you leave this parameter to its default value, you should be able to
  run several optimizations in parallel, for instance with
  \code{\link[parallel]{mclapply}}.

  See the vignette for applications.
}
\note{
  For the moment, the parameter \code{mc.cores} is accepted only by
  \code{\link{hann1}}.
}
\value{a list with named elements as detailed above.}
\author{Emmanuel Paradis}
\references{
  \url{https://en.wikipedia.org/wiki/Wolfe_conditions}
}
\seealso{\code{\link{hann1}}}
\examples{
control.hann() # default values
ctrl <- control.hann(iterlim = 1000)
ctrl

## verbose is not a parameter:
ctrl <- control.hann(iterlim = 1000, verbose = TRUE)
}
\keyword{manip}
