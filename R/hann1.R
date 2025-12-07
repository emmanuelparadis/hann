## hann1.R (2025-12-06)

##   One-layer Hopfield ANN

## Copyright 2024-2025 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

hann1 <- function(xi, sigma, classes, labels = NULL, net = NULL,
                  control = control.hann())
{
    K <- nrow(xi)
    N <- ncol(xi)
    expec <- initExpec(classes, K)
    C <- ncol(expec)

    if (is.null(labels)) labels <- as.character(unique(classes))
    if (length(labels) != C) stop("wrong number of labels")

    if (is.null(net)) {
        W <- initW(N, C)
        bias <- numeric(C)
    } else {
        if (!inherits(net, "hann1"))
            stop("argument 'net' not of the correct class")
        W <- net$parameters$W
        bias <- net$parameters$bias
    }

    iterlim <- as.integer(control$iterlim)
    quiet <- as.logical(control$quiet)
    ctrl <- control[c("quasinewton", "fullhessian", "trace.error", "wolfe")]
    ctrl <- as.logical(unlist(ctrl))
    target <- as.numeric(control$target)
    beta <- as.numeric(control$beta)
    mc.cores  <- as.integer(control$mc.cores)

    .Call(test_7, W, bias, sigma, xi, expec, iterlim,
          quiet, ctrl, target, beta, mc.cores)
    res <- list(parameters = list(W = W, bias = bias),
                sigma = sigma, beta = beta, labels = labels,
                call = match.call())
    res$fitted <- predict.hann1(res, xi)
    class(res) <- c("hann", "hann1")
    res
}

print.hann1 <- function(x, details = FALSE, ...)
{
    cat("*** Neural network of class \"hann1\" ***\n")
    cat("Number of input neurons: ", nrow(x$parameters$W), "\n", sep = "")
    cat("Number of output neurons: ", ncol(x$parameters$W), "\n", sep = "")
    if (details) print.default(x, ...)
}

predict.hann1 <- function(object, patterns, rawsignal = TRUE,
                          useLabels = TRUE, ...)
{
    if (missing(patterns)) {
        res <- object$fitted
    } else {
        K <- nrow(patterns)
        patterns <- patterns * rep(object$sigma, each = K)
        res <- patterns %*% object$parameters$W
        res <- tanh(object$beta * res + rep(object$parameters$bias, each = K))
    }
    if (rawsignal) {
        if (useLabels) colnames(res) <- object$labels
        return(res)
    }
    res <- apply(res, 1, which.max)
    if (useLabels) res <- object$labels[res]
    res
}
