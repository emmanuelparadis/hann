## binarize.R (2025-12-06)

##   Helper Function to Prepare Data From Images

## Copyright 2025 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

binarize <- function(x, threshold = median(x))
{
    res <- x # keep the attributes (dim, ...)
    storage.mode(res) <- "integer"
    s <- x < threshold
    res[s] <- -1L
    res[!s] <- 1L
    res
}

