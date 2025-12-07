## hann.R (2025-12-06)

##   Generic (Top-Level) Functions

## Copyright 2025 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

hann <- function(xi, sigma, classes, H = NULL, labels = NULL, net = NULL,
                 control = control.hann())
{
    if (is.numeric(H) && H == 0) {
        res <- hann1(xi = xi, sigma = sigma, classes = classes,
                     net = net, labels = labels, control = control)
        class(res) <- c("hann", "hann1")
        return(res)
    }
    if (is.null(H)) H <- 0.5 * length(sigma)
    res <- hann3(xi = xi, sigma = sigma, classes = classes,
                 H = H, net = net, labels = labels, control = control)
    class(res) <- c("hann", "hann3")
    res
}

print.hann <- function(x, ...) NextMethod("print")

predict.hann <- function(object, ...) NextMethod("predict")

summary.hann <- function(object, ...)
{
    cat("Neural network of class: ", deparse(class(object)), "\n\n", sep = "")
    summary.default(object, ...)
}

str.hann <- function(object, ...)
{
    cat("Neural network of class: ", deparse(class(object)), "\n\n", sep = "")
    NextMethod("str")
}

plot.hann <- function(x, y, type = "h", ...)
{
    op <- par(ask = TRUE)
    on.exit(par(op))
    nms <- names(x$parameters)
    for (i in seq_along(nms)) {
        par <- x$parameters[[i]]
        if (is.vector(par)) plot(par, type = type, ...)
        if (is.matrix(par)) image(par, ...)
        title(nms[i])
    }
}

coef.hann <- function(object, ...) object$parameters

fitted.hann <- function(object, ...) object$fitted

labels.hann <- function(object, ...) object$labels
