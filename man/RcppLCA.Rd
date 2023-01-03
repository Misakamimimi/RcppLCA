\name{RcppLCA}
\alias{RcppLCA}
\title{Latent class analysis of polytomous outcome variables}
\description{Parameter estimations of latent class analysis for polytomous outcome variables.}
\usage{
RcppLCA(formula, data, nclass = 2, maxiter = 1000, 
      tol = 1e-10, na.rm = TRUE, probs.start = NULL, nrep = 1, 
      verbose = TRUE, calc.se = TRUE,num_thread=parallel::detectCores())
}
\arguments{
\item{formula}{A formula expression of the form \code{response ~ predictors}. The details of model specification are given below.}
\item{data}{A data frame containing variables in \code{formula}. Manifest variables must contain \emph{only} integer values, and must be coded with consecutive values from 1 to the maximum number of outcomes for each variable. All missing values should be entered as \code{NA}.}
\item{nclass}{The number of latent classes to assume in the model. Setting \code{nclass=1} results in \code{RcppLCA} estimating the loglinear independence model. The default is two.}
\item{maxiter}{The maximum number of iterations through which the estimation algorithm will cycle.}
\item{tol}{A tolerance value for judging when convergence has been reached.  When the one-iteration change in the estimated log-likelihood is less than \code{tol}, the estimation algorithm stops updating and considers the maximum log-likelihood to have been found.}
\item{na.rm}{Logical, for how \code{RcppLCA} handles cases with missing values on the manifest variables.  If \code{TRUE}, those cases are removed (listwise deleted) before estimating the model. If \code{FALSE}, cases with missing values are retained. Cases with missing covariates are always removed. The default is \code{TRUE}.}
\item{probs.start}{A list of matrices of class-conditional response probabilities to be used as the starting values for the estimation algorithm.  Each matrix in the list corresponds to one manifest variable, with one row for each latent class, and one column for each outcome.  The default is \code{NULL}, producing random starting values.  Note that if \code{nrep>1}, then any user-specified \code{probs.start} values are only used in the first of the \code{nrep} attempts.}
\item{nrep}{Number of times to estimate the model, using different values of \code{probs.start}.  The default is one.  Setting \code{nrep}>1 automates the search for the global---rather than just a local---maximum of the log-likelihood function. \code{RcppLCA} returns the parameter estimates corresponding to the model with the greatest log-likelihood.}
\item{verbose}{Logical, indicating whether \code{RcppLCA} should output to the screen the results of the model.  If \code{FALSE}, no output is produced. The default is \code{TRUE}.}
\item{calc.se}{Logical, indicating whether \code{RcppLCA} should calculate the standard errors of the estimated class-conditional response probabilities and mixing proportions.  The default is \code{TRUE}; can only be set to \code{FALSE} if estimating a basic model with no concomitant variables specified in \code{formula}.}
\item{num_thread}{The number of cores used for parallel computing, which defaults to all}
}
\value{
\code{RcppLCA} returns an object of class RcppLCA; a list containing the following elements:
\item{y}{data frame of manifest variables.}
\item{x}{data frame of covariates, if specified.}
\item{N}{number of cases used in model.}
\item{Nobs}{number of fully observed cases (less than or equal to \code{N}).}
\item{probs}{estimated class-conditional response probabilities.}
\item{probs.se}{standard errors of estimated class-conditional response probabilities, in the same format as \code{probs}.}
\item{P}{sizes of each latent class; equal to the mixing proportions in the basic latent class model, or the mean of the priors in the latent class regression model.}
\item{P.se}{the standard errors of the estimated \code{P}.}
\item{posterior}{matrix of posterior class membership probabilities; also see function \code{\link{RcppLCA.posterior}}.}
\item{predclass}{vector of predicted class memberships, by modal assignment.}
\item{predcell}{table of observed versus predicted cell counts for cases with no missing values; also see functions \code{\link{RcppLCA.table}} and \code{\link{RcppLCA.predcell}}.}
\item{llik}{maximum value of the log-likelihood.}
\item{numiter}{number of iterations until reaching convergence.}
\item{maxiter}{maximum number of iterations through which the estimation algorithm was set to run.}
\item{coeff}{multinomial logit coefficient estimates on covariates (when estimated). \code{coeff} is a matrix with \code{nclass-1} columns, and one row for each covariate.  All logit coefficients are calculated for classes with respect to class 1.}
\item{coeff.se}{standard errors of coefficient estimates on covariates (when estimated), in the same format as \code{coeff}.}
\item{coeff.V}{covariance matrix of coefficient estimates on covariates (when estimated).}
\item{aic}{Akaike Information Criterion.}
\item{bic}{Bayesian Information Criterion.}
\item{time}{length of time it took to run the model.}
\item{npar}{number of degrees of freedom used by the model (estimated parameters).}
\item{resid.df}{number of residual degrees of freedom.}
\item{attempts}{a vector containing the maximum log-likelihood values found in each of the \code{nrep} attempts to fit the model.}
\item{eflag}{Logical, error flag. \code{TRUE} if estimation algorithm needed to automatically restart with new initial parameters. A restart is caused in the event of computational/rounding errors that result in nonsensical parameter estimates.}
\item{probs.start}{A list of matrices containing the class-conditional response probabilities used as starting values in the estimation algorithm. If the algorithm needed to restart (see \code{eflag}), then this contains the starting values used for the final, successful, run.}
\item{probs.start.ok}{Logical. \code{FALSE} if \code{probs.start} was incorrectly specified by the user, otherwise \code{TRUE}.}
\item{call}{function call to \code{RcppLCA}.}
}
\examples{
##
## Three models without covariates:
## M0: Loglinear independence model.
## M1: Two-class latent class model.
## M2: Three-class latent class model.
##
data(values)
f <- cbind(A,B,C,D)~1
M0 <- RcppLCA(f,values,nclass=1) # log-likelihood: -543.6498
M1 <- RcppLCA(f,values,nclass=2) # log-likelihood: -504.4677
M2 <- RcppLCA(f,values,nclass=3,maxiter=8000) # log-likelihood: -503.3011

##
## Three-class model with a single covariate.
##
data(election)
f2a <- cbind(MORALG,CARESG,KNOWG,LEADG,DISHONG,INTELG,
             MORALB,CARESB,KNOWB,LEADB,DISHONB,INTELB)~PARTY
nes2a <- RcppLCA(f2a,election,nclass=3,nrep=5)    # log-likelihood: -16222.32 
pidmat <- cbind(1,c(1:7))
exb <- exp(pidmat \%*\% nes2a$coeff)
matplot(c(1:7),(cbind(1,exb)/(1+rowSums(exb))),ylim=c(0,1),type="l",
    main="Party ID as a predictor of candidate affinity class",
    xlab="Party ID: strong Democratic (1) to strong Republican (7)",
    ylab="Probability of latent class membership",lwd=2,col=1)
text(5.9,0.35,"Other")
text(5.4,0.7,"Bush affinity")
text(1.8,0.6,"Gore affinity")
}
\keyword{methods}