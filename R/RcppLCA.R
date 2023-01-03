RcppLCA <- function(formula,
                  data,
                  nclass = 2,
                  maxiter = 1000,
                  tol = 1e-10,
                  na.rm = TRUE,
                  probs.start = NULL,
                  nrep = 1,
                  verbose = TRUE,
                  calc.se = TRUE,
                  num_thread = parallel::detectCores()) {
  starttime <- Sys.time()
  mframe <- model.frame(formula, data, na.action = NULL)
  mf <- model.response(mframe)
  if (any(mf < 1, na.rm = TRUE) | any(round(mf) != mf, na.rm = TRUE)) {
    stop("\n ALERT: some manifest variables contain values that are not
              positive integers. For poLCA to run, please recode categorical
              outcome variables to increment from 1 to the maximum number of
              outcome categories for each variable. \n\n")
  }
  data <- data[rowSums(is.na(model.matrix(formula, mframe))) == 0, ]
  if (na.rm) {
    mframe <- model.frame(formula, data)
    y <- model.response(mframe)
  } else {
    mframe <- model.frame(formula, data, na.action = NULL)
    y <- model.response(mframe)
    y[is.na(y)] <- 0
  }
  if (any(sapply(lapply(as.data.frame(y), table), length) == 1)) {
    y <- y[, !(sapply(apply(y, 2, table), length) == 1)]
    cat("\n ALERT: at least one manifest variable contained only one
             outcome category, and has been removed from the analysis. \n\n")
  }
  x <- model.matrix(formula, mframe)
  N <- nrow(y)
  J <- ncol(y)
  K.j <- t(matrix(apply(y, 2, max)))
  R <- nclass
  S <- ncol(x)
  eflag <- FALSE
  probs.start.ok <- TRUE
  ret <- list()
  
  if (R == 1) {
    ret$probs <- list()
    for (j in 1:J) {
      ret$probs[[j]] <- matrix(NA, nrow = 1, ncol = K.j[j])
      for (k in 1:K.j[j]) {
        ret$probs[[j]][k] <- sum(y[, j] == k) / sum(y[, j] > 0)
      }
    }
    ret$probs.start <- ret$probs
    ret$P <- 1
    prior <- matrix(1, nrow = N, ncol = 1)
    ret$predclass <- prior
    ret$posterior <- ret$predclass
    ret$llik <- sum(log(RcppLCA.y_likelihood.C(RcppLCA.vectorize(ret$probs), y)))
    if (calc.se) {
      se <- RcppLCA.se(y, x, ret$probs, prior, ret$posterior)
      ret$probs.se <- se$probs
      ret$P.se <- se$P
    } else {
      ret$probs.se <- NA
      ret$P.se <- NA
    }
    ret$numiter <- 1
    ret$probs.start.ok <- TRUE
    ret$coeff <- NA
    ret$coeff.se <- NA
    ret$coeff.V <- NA
    ret$eflag <- FALSE
    if (S > 1) {
      S <- 1
    }
  } else {
    if (!is.null(probs.start)) {
      if ((length(probs.start) != J) | (!is.list(probs.start))) {
        probs.start.ok <- FALSE
      } else {
        if (sum(sapply(probs.start, dim)[1, ] == R) != J) {
          probs.start.ok <- FALSE
        }
        if (sum(sapply(probs.start, dim)[2, ] == K.j) != J) {
          probs.start.ok <- FALSE
        }
        if (sum(round(sapply(probs.start, rowSums), 4) == 1)
            != (R * J)) {
          probs.start.ok <- FALSE
        }
      }
    }
    
    initial_prob <- list()
    initial_prob_vector <- c()
    irep <- 1
    
    if (probs.start.ok & !is.null(probs.start)) {
      initial_prob[[1]] <- RcppLCA.vectorize(probs.start)
      initial_prob_vector <- c(
        initial_prob_vector,
        initial_prob[[1]]$vecprobs
      )
      irep <- irep + 1
    }
    
    if (nrep > 1 | irep == 1) {
      for (repl in irep:nrep) {
        probs <- list()
        for (j in 1:J) {
          probs[[j]] <- matrix(
            runif(R * K.j[j]),
            nrow = R, ncol = K.j[j]
          )
          probs[[j]] <- probs[[j]] / rowSums(probs[[j]])
        }
        initial_prob[[repl]] <- RcppLCA.vectorize(probs)
        initial_prob_vector <- c(
          initial_prob_vector,
          initial_prob[[repl]]$vecprobs
        )
      }
    }
    
    seed <- sample.int(
      as.integer(.Machine$integer.max), 5,
      replace = TRUE
    )

    em_results <- EmAlgorithmRcpp(
      x,
      t(y),
      initial_prob_vector,
      N,
      S,
      J,
      K.j,
      R,
      nrep,
      num_thread,
      maxiter,
      tol,
      seed
    )
    rgivy <- em_results[[1]]
    prior <- em_results[[2]]
    estimated_prob <- em_results[[3]]
    b <- em_results[[4]]
    ret$attempts <- em_results[[5]]
    best_rep_index <- em_results[[6]]
    numiter <- em_results[[7]]
    best_initial_prob <- em_results[[8]]
    eflag <- em_results[[9]]
    
    llik <- ret$attempts[best_rep_index]
    
    vp <- initial_prob[[1]]
    vp$vecprobs <- estimated_prob
    
    ret$probs.start <- initial_prob[[1]]
    ret$probs.start$vecprobs <- best_initial_prob
    
    if (calc.se) {
      se <- RcppLCA.se(
        y, x, RcppLCA.unvectorize(vp),
        prior, rgivy
      )
      rownames(se$b) <- colnames(x)
    } else {
      se <- list(
        probs = NA, P = NA, b = matrix(nrow = S, ncol = R - 1),
        var.b = NA
      )
    }
    
    if (S > 1) {
      b <- matrix(b, nrow = S)
      rownames(b) <- colnames(x)
    } else {
      b <- NA
      se$b <- NA
      se$var.b <- NA
    }
    
    ret$llik <- llik
    ret$probs.start <- RcppLCA.unvectorize(ret$probs.start)
    ret$probs <- RcppLCA.unvectorize(vp)
    ret$probs.se <- se$probs
    ret$P.se <- se$P
    ret$posterior <- rgivy
    ret$predclass <- apply(ret$posterior, 1, which.max)
    ret$P <- colMeans(ret$posterior)
    ret$numiter <- numiter
    ret$probs.start.ok <- probs.start.ok
    
    ret$coeff <- b
    ret$coeff.se <- se$b
    ret$coeff.V <- se$var.b
    
    ret$eflag <- eflag
  }
  names(ret$probs) <- colnames(y)
  if (calc.se) {
    names(ret$probs.se) <- colnames(y)
  }
  
  ret$npar <- (R * sum(K.j - 1)) + (R - 1)
  if (S > 1) {
    ret$npar <- ret$npar + (S * (R - 1)) - (R - 1)
  }

  ret$aic <- (-2 * ret$llik) + (2 * ret$npar)
  ret$bic <- (-2 * ret$llik) + (log(N) * ret$npar)
  ret$Nobs <- sum(rowSums(y == 0) == 0)
  
  y[y == 0] <- NA
  ret$y <- data.frame(y)
  ret$x <- data.frame(x)
  for (j in 1:J) {
    rownames(ret$probs[[j]]) <- paste("class ", 1:R, ": ", sep = "")
    if (is.factor(data[, match(colnames(y), colnames(data))[j]])) {
      lev <- levels(data[, match(colnames(y), colnames(data))[j]])
      colnames(ret$probs[[j]]) <- lev
      ret$y[, j] <- factor(ret$y[, j], labels = lev)
    } else {
      colnames(ret$probs[[j]]) <-
        paste("Pr(", 1:ncol(ret$probs[[j]]), ")", sep = "")
    }
  }
  ret$N <- N
  
  ret$maxiter <- maxiter
  ret$resid.df <- min(ret$N, (prod(K.j) - 1)) - ret$npar
  class(ret) <- "RcppLCA"
  
  if (verbose) {
    print.RcppLCA(ret)
  }
  ret$time <- Sys.time() - starttime
  ret$call <- match.call()
  return(ret)
}