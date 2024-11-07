hann1 <- function(xi, sigma, classes, net = NULL, control = control.hann())
{
    K <- nrow(xi)
    N <- ncol(xi)
    expec <- initExpec(classes, K)
    C <- ncol(expec)

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
                sigma = sigma, beta = beta, call = match.call())
    class(res) <- "hann1"
    res
}

print.hann1 <- function(x, details = FALSE, ...)
{
    cat("*** Neural network of class \"hann1\" ***\n")
    cat("Number of input neurons: ", nrow(x$parameters$W), "\n", sep = "")
    cat("Number of output neurons: ", ncol(x$parameters$W), "\n", sep = "")
    if (details) print.default(x, ...)
}

predict.hann1 <- function(object, patterns, rawsignal = TRUE, ...)
{
    K <- nrow(patterns)
    patterns <- patterns * rep(object$sigma, each = K)
    res <- patterns %*% object$parameters$W
    res <- tanh(object$beta * res + rep(object$parameters$bias, each = K))
    if (rawsignal) return(res)
    apply(res, 1, which.max)
}
