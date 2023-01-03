RcppLCA.y_likelihood.C <-
  function(vp,y) {
    likelihood <- ylik(vp$vecprobs,
                t(y),
                dim(y)[1],
                length(vp$numChoices),
                vp$numChoices,
                vp$classes
    )
    likelihood <- matrix(likelihood,ncol=vp$classes,byrow=TRUE)
    return(likelihood)
  }

RcppLCA.vectorize <- function(probs) {
  classes <- nrow(probs[[1]])
  vecprobs <- c()
  for (m in seq_len(classes)) {
    for (j in seq_len(length(probs))) {
      vecprobs <- c(vecprobs, probs[[j]][m, ])
    }
  }
  num_choices <- sapply(probs, ncol)
  return(list(
    vecprobs = vecprobs, numChoices = num_choices, classes = classes
  ))
}

RcppLCA.unvectorize <- function(vp) {
  num_choices <- vp$numChoices
  n_category <- length(num_choices)
  probs <- list()
  for (j in seq_len(n_category)) {
    probs[[j]] <- matrix(nrow = vp$classes, ncol = num_choices[j])
  }
  
  index <- 1
  for (m in 1:vp$classes) {
    for (j in seq_len(n_category)) {
      next_index <- index + num_choices[j] - 1
      probs[[j]][m, ] <- vp$vecprobs[index:next_index]
      index <- next_index + 1
    }
  }
  return(probs)
}

RcppLCA.se <-
  function(y,x,probs,prior,rgivy) {
    J <- ncol(y)
    R <- ncol(prior)
    K.j <- sapply(probs,ncol)
    N <- nrow(y)
    ymat <- y
    y <- list()
    for (j in 1:J) {
      y[[j]] <- matrix(0,nrow=N,ncol=K.j[j])
      y[[j]][cbind(c(1:N),ymat[,j])] <- 1
      y[[j]][ymat[,j]==0,] <- NA      
    }
    s <- NULL
   
    for (r in 1:R) {
      for (j in 1:J) {
        s <- cbind(s,rgivy[,r] * t(t(y[[j]][,2:K.j[j]]) - probs[[j]][r,2:K.j[j]]))
      }
    }
    
    ppdiff <- rgivy-prior
    if (R>1) for (r in 2:R) { s <- cbind(s,x*ppdiff[,r]) }
    
    s[is.na(s)] <- 0      
    info <- t(s) %*% s    
    VCE <- MASS::ginv(info)     
    
    
    VCE.lo <- VCE[1:sum(R*(K.j-1)),1:sum(R*(K.j-1))]
    Jac <- matrix(0,nrow=nrow(VCE.lo)+(J*R),ncol=ncol(VCE.lo))
    rpos <- cpos <- 1
    for (r in 1:R) {
      for (j in 1:J) {
        Jsub <- -(probs[[j]][r,] %*% t(probs[[j]][r,]))
        diag(Jsub) <- probs[[j]][r,]*(1-probs[[j]][r,])
        Jsub <- Jsub[,-1]
        Jac[rpos:(rpos+K.j[j]-1),cpos:(cpos+K.j[j]-2)] <- Jsub
        rpos <- rpos+K.j[j]
        cpos <- cpos+K.j[j]-1
      }
    }
    VCE.probs <- Jac %*% VCE.lo %*% t(Jac)
    
    maindiag <- diag(VCE.probs)
    maindiag[maindiag<0] <- 0 
    se.probs.vec <- sqrt(maindiag)
    se.probs <- list()
    for (j in 1:J) { se.probs[[j]] <- matrix(0,0,K.j[j]) }
    pos <- 1
    for (r in 1:R) {
      for (j in 1:J) {
        se.probs[[j]] <- rbind(se.probs[[j]],se.probs.vec[pos:(pos+K.j[j]-1)])
        pos <- pos+K.j[j]
      }
    }
    
    if (R>1) {
      VCE.beta <- VCE[(1+sum(R*(K.j-1))):dim(VCE)[1],(1+sum(R*(K.j-1))):dim(VCE)[2]]
      se.beta <- matrix(sqrt(diag(VCE.beta)),nrow=ncol(x),ncol=(R-1))
      
      ptp <- array(NA,dim=c(R,R,N))
      for (n in 1:N) {
        ptp[,,n] <- -(prior[n,] %*% t(prior[n,]))
        diag(ptp[,,n]) <- prior[n,] * (1-prior[n,])
      }
      Jac.mix <- NULL
      for (r in 2:R) {
        for (l in 1:ncol(x)) {
          Jac.mix <- cbind(Jac.mix,colMeans(t(ptp[,r,]) * x[,l]))
        }
      }
      VCE.mix <- Jac.mix %*% VCE.beta %*% t(Jac.mix)
      se.mix <- sqrt(diag(VCE.mix))
    } else {
      VCE.beta <- se.beta <- se.mix <- NULL
    }
    return( list(probs=se.probs,P=se.mix,b=se.beta,var.b=VCE.beta) )
  }