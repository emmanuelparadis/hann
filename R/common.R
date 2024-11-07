.hann.ctrl <-
    list(iterlim = 100L, quiet = FALSE, quasinewton = FALSE,
         fullhessian = FALSE, trace.error = FALSE, wolfe = FALSE,
         target = 0.001, beta = 0.2, mc.cores = 1L)


control.hann <- function(...)
{
    dots <- list(...)
    x <- .hann.ctrl
    if (length(dots)) {
        chk.nms <- names(dots) %in% names(x)
        if (any(!chk.nms)) {
            warning("some control parameter names do not match: they were ignored")
            dots <- dots[chk.nms]
        }
        x[names(dots)] <- dots
    }
    x
}

initW <- function(NROW, NCOL, lower = -1, upper = 1)
{
    W <- runif(NROW * NCOL, lower, upper)
    dim(W) <- c(NROW, NCOL)
    W
}

initExpec <- function(cls, K)
{
    cls <- factor(cls)
    C <- nlevels(cls)
    expec <- matrix(-1L, K, C)
    expec[cbind(1:K, as.integer(cls))] <- 1L
    expec
}

