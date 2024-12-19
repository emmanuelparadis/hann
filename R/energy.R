## energy.R (2024-10-31)

##   Energy Function for Hopfield ANN

## Copyright 2024 Emmanuel Paradis

## This file is part of the R-package `hann'.
## See the file ../DESCRIPTION for licensing issues.

buildSigma <- function(xi, n = 20, nrep = 100, quiet = FALSE)
{
    n <- as.integer(n)
    storage.mode(xi) <- "integer"
    N <- ncol(xi)
    upsigma <- integer(N)
    Lup <- L <- vector("list", nrep)
    EE <- EE2 <- numeric(nrep)
    fmt <- paste0("%",
                  nchar(as.integer(nrep)),
                  "d:  Initial energy = %.2e   Updated energy = %.3g\n")
    for (i in 1:nrep) {
        sigma <- sample(c(-1L, 1L), N, TRUE)
        L[[i]] <- sigma
        EE[i] <- .Call(E, sigma, xi, n)
        upsigma[] <- 0L
        .Call(updateSigma_Call, sigma, xi, upsigma, n)
        EE2[i] <- .Call(E, upsigma, xi, n)
        Lup[[i]] <- upsigma
        if (!quiet) cat(sprintf(fmt, i, EE[i], EE2[i]))
    }
    ## repeat until there is no 0 in upsigma ###
    for (i in which(EE2 == min(EE2))) {
        upsigma <- Lup[[i]]
        if (all(upsigma != 0) && !is.null(upsigma)) break
    }
    if (!quiet) cat(sprintf("\nFinal energy = %e\n\n", EE2[i]))
    upsigma
}
