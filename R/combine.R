## hann.R (2025-12-07)

##   Combine Several Neural Nets for Prediction

## Copyright 2025 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

combine <- function(nets, xi)
{
    N_nets <- length(nets)
    cat("Computing the raw signals...")
    Wall <- lapply(nets, predict, xi)
    cat(" Done.\n")
    LABELS <- lapply(nets, "[[", "labels")
    if (any(sapply(LABELS, is.null))) stop("some networks have no labels")

    unique_labels <- sort(unique(unlist(LABELS)))
    C <- length(unique_labels)
    PRED <- matrix(0L, nrow(xi), C)
    colnames(PRED) <- unique_labels

    foo <- function(x, i) match(unique_labels[i], x, 0L)
    cat("Combining the signals...")
    for (i in 1:C) {
        jj <- lapply(LABELS, foo, i = i)
        for (k in 1:N_nets) {
            if (!(j <- jj[[k]])) next
            PRED[, i] <- PRED[, i] + Wall[[k]][, j]
        }
    }
    cat(" Done.\n")
    unique_labels[apply(PRED, 1, which.max)]
}
