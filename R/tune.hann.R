## tune.hann.R (2025-12-07)

##   Tune Hyperparameters

## Copyright 2025 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

tune.hann <- function(xi, sigma, classes,
                      ranges = list(H = seq(10, 50, by = 10),
                                    beta = seq(0.2, 0.8, by = 0.1)),
                      nrepeat = 10,
                      control = control.hann(iterlim = 20))
{
    tr <- function(mat) sum(diag(mat))
    K <- nrow(xi)
    ## ranges$nrepeat <- 1:nrepeat
    PARAS <- do.call(expand.grid, ranges)
    Ncombpara <- nrow(PARAS)
    MEAN <- SD <- numeric(Ncombpara)
    control$quiet <- TRUE
    k <- 0L
    for (i in 1:Ncombpara) {
        cat(sprintf("\r%3d /%3d : ", i, Ncombpara))
        H <- PARAS$H[i]
        control$beta <- PARAS$beta[i]
        res <- numeric(nrepeat)
        cat("repl.        ")
        for (j in 1:nrepeat) {
            cat(sprintf("\b\b\b\b\b\b\b\b%3d /%3d", j, nrepeat))
            nt <- hann(xi, sigma, classes, H = H, control = control)
            pred <- predict(nt, xi, rawsignal = FALSE)
            res[j] <- K - tr(table(classes, pred))
        }
        MEAN[i] <- mean(res)
        SD[i] <- sd(res)
    }
    cat("\n")
    PARAS$mean <- MEAN
    PARAS$sd <- SD
    PARAS
}
