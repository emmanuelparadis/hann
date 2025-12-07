## hann3.R (2025-12-06)

##   Three-layer Hopfield ANN

## Copyright 2024-2025 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

hann3 <- function(xi, sigma, classes, H = 0.5 * length(sigma),
                  labels = NULL, net = NULL,
                  control = control.hann())
{
    H <- as.integer(H)
    K <- nrow(xi)
    N <- ncol(xi)
    expec <- initExpec(classes, K)
    C <- ncol(expec)

    if (is.null(labels)) labels <- as.character(unique(classes))
    if (length(labels) != C) stop("wrong number of labels")

    if (is.null(net)) {
        W1 <- initW(N, H)
        W2 <- initW(H, H)
        W3 <- initW(H, C)
        bias1 <- numeric(H)
        bias2 <- numeric(C)
    } else {
        if (!inherits(net, "hann3"))
            stop("argument 'net' not of the correct class")
        W1 <- net$parameters$W1
        W2 <- net$parameters$W2
        W3 <- net$parameters$W3
        bias1 <- net$parameters$bias1
        bias2 <- net$parameters$bias2
    }

    iterlim <- as.integer(control$iterlim)
    quiet <- as.logical(control$quiet)
    ctrl <- control[c("quasinewton", "fullhessian", "trace.error", "wolfe")]
    ctrl <- as.logical(unlist(ctrl))
    target <- as.numeric(control$target)
    beta <- as.numeric(control$beta)
    mc.cores  <- as.integer(control$mc.cores)

    .Call(test_6, W1, W2, W3, bias1, bias2, sigma, xi, expec,
          iterlim, quiet, ctrl, target, beta, mc.cores)

    res <- list(parameters = list(W1 = W1, W2 = W2, W3 = W3,
                                  bias1 = bias1, bias2 = bias2),
                sigma = sigma, beta = beta,  labels = labels,
                call = match.call())
    res$fitted <- predict.hann3(res, xi)
    class(res) <- c("hann", "hann3")
    res
}

print.hann3 <- function(x, details = FALSE, ...)
{
    cat("*** Neural network of class \"hann3\" ***\n")
    cat("Number of input neurons: ", nrow(x$parameters$W1), "\n", sep = "")
    cat("Number of hidden neurons: ", ncol(x$parameters$W1), "\n", sep = "")
    cat("Number of output neurons: ",
        length(x$parameters$bias2), "\n", sep = "")
    if (details) print.default(x, ...)
}

predict.hann3 <- function(object, patterns, rawsignal = TRUE,
                          useLabels = TRUE, ...)
{
    if (missing(patterns)) {
        res <- object$fitted
    } else {
        K <- nrow(patterns)
        H <- nrow(object$parameters$W2)
        patterns <- patterns * rep(object$sigma, each = K)
        res <- tanh(patterns %*% object$parameters$W1)
        res <- tanh(object$beta * res %*% object$parameters$W2 +
                    rep(object$parameters$bias1, each = K))
        res <- tanh(object$beta * res %*% object$parameters$W3 +
                    rep(object$parameters$bias2, each = K))
    }
    if (rawsignal) {
        if (useLabels) colnames(res) <- object$labels
        return(res)
    }
    res <- apply(res, 1, which.max)
    if (useLabels) res <- object$labels[res]
    res
}
