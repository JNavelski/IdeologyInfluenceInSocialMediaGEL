#============================================================
# Title: Supplementary Example, 2024 January 20th
# Purpose: Fixed-Model Fit to Twitter Data on m=46 Experts
#          and n=3383 followers
#============================================================

#setwd("C:/Users/MyAcct/Demo")
# Set your directory to the "Demo" folder
# Note that the forward / is used instead of backward \
#

options(buildtools.check = function(action) TRUE )
setwd("~/GEL_Ideology_and_Influence_Tweet/")



#  These packages are used for all general applications and for estimation
#  Install them first before proceeding
library(readr)
library(rstan)
library(Rcpp)
library(ggplot2)
library(lattice)
library(reshape2)
library(Matrix)
library(gridExtra)

#  These packages are used for diagnostics and results
#  Install them first before proceeding
library(xtable)
library(shinystan)
library(pROC)
library(caret)
library(ggplot2)
library(dplyr)
library(patchwork)
library(ggpubr) 
# NOTE:  Some packages may install older versions of other packages which may 
# result in error messages indicating a newer version is required.  In this 
# case, we suggest updating the versions by install the packages individually.


#################################################################
# ------------ Begin STAGE 1:  Preliminary Setup ----------------
#################################################################

# Set the seeds to match results in the supplement
set.seed(123)
stan.seed <- 123


# Read in functions used here 
# "./" is the directory C:/Users/MyAcct/Demo
source("./Bayes_Tweet_Functions.R") # Contains functions used here.

# This option is for when running on iOS (Mac)f
options(buildtools.check = function(action) TRUE )

# Load connections matrix data
load('sim_y_mat.rdata')
# The data are in the matrix y with n rows and m columns with elements
#   1 if following, 0 or . otherwise

# Load expert Data
df  <- read.csv("./GEL_expert_profile_info.csv")


# Below we will assign negative ideal points to Anti-GEL, positive to Pro-GEL
df <- df[order(df$possition,decreasing = TRUE),]

y0 <- sim_y_mat
dim(y0)
dimnames(y0)[[2]]

# Setting parameters for the data structure
m <- ncol(y0) # No. of experts
n <- nrow(y0) # No. of followers
inf <- dimnames(y0)[[2]] # Expert names

mneg <- 12 # Number of "negatively positioned" experts (precoding)
mpos <- m-mneg

###############################################################
# ------------ End STAGE 1:  Preliminary Setup ----------------
###############################################################




#################################################################
# ---- Begin STAGE 2:  Initial Values, Priors Specification -----
#################################################################

# Setting Parameters for MC-MC sampling
n.chains <- 2 # Number of Markov chains
n.iter <- 500 # Number of iterations per chain
n.warmup <- n.iter/2 # number of burn-in iterations per chain
thin <- 1; # Period for saving samples

###############################################
###### Setting up initial Values for parameters
###############################################

### Initial values for mu, gamma and lambda
mu.init <- 0
gam.init <- 1
lgam.init <- log(gam.init) # Use log(gamma) which is any real number

# Adding in a fixed lambda value
# .5 implies an absolute distance
# 1 implies a squared disance
lambda.fit <- 1

################################################################
# Commented out lambda initial values ------- JN
################################################################

# lambda.init <- 1
# llambda.init <- log(lambda.init) # Use log(lambda) which is any real #.


### alpha ~ expert popularity (latent)
alpha <- c()
for(i in 1:m){
  # More followers, more popular
  alpha[i]  <-  sum(y0[,i]) # Number of followers of expert i
}
alpha.init <- 5*(alpha - mean(alpha))/(max(alpha)-min(alpha)) 
# Scale alphas and constrain sum at 0


### beta ~ follower engagement (latent)
beta <- c()
for(i in 1:n){
  # Follow more Pro than Anti, more positive ideal point
  beta[i]  <-  sum(y0[i,]) # Number of experts followed by follower j
}
beta.init <- 5*(beta - mean(beta))/(max(beta)-min(beta))  
# Scale betas and constrain sum to 0


### theta ~ follower ideology (latent)
theta0 <- c()
for(i in 1:n){
  theta0[i] <- sum(y0[i,(mneg+1):m]) - sum(y0[i,1:(mneg)]) # Pro minus Anti
}
theta <- theta0/((max(abs(theta0))*1.5)) 
# Standardize to get theta between -1 to 1

tneg <- (theta<0) # Which followers have negative ideal points?
n.neg <- sum(tneg)
# Sort the followers by ideal point, negative then positive
y0=rbind(y0[tneg,], y0[!tneg,])
theta.init=c(theta[tneg],theta[!tneg]) # Sort the initial theta values to match 

#sum(tneg) # Print number of followers with negative ideal points
#sum(!(tneg)) # Print number of followers with at least 0 ideal points


### phi ~ expert ideology (latent)

# Each experts is precoded as having either negative (left) or 
# positive (right) ideal points.  To address possible precode errors, 
# allow overlapping ideal point ranges so that negative falls inside
# (-1, cutphi) and positive inside (-cutphi, +1) where cutphi >= 0.

cutphi <- 0.6 # Ideal point cut-off  
tcutphi <- tan(cutphi*(pi/2)) # In the analysis below, phi is transformed 
# using tan/arctan so that phi is uncontrained.  

phi0 <- c()
for(i in 1:m){
  phi0[i] <- abs(sum(y0[1:n.neg],i)-sum(y0[(n.neg+1):n,i])) 
  # More followers aligned, the more intense the ideal point.
}
phi.init <- 0.9*phi0/max(abs(phi0)) # Scale ideal point INTENSITY from 0 to 1
# The sign of the ideal point is determined during expert precoding  
# Start all expert phis between -0.9 and 0.9

# Transform phi so it becomes any real number
tphi.init <- log(tan(pi*phi.init/2)+tcutphi)


################################################################
# Removed the lambda initial values ------- JN
################################################################

### Put all initial values in a list (input for stan)
inits <- rep(list(list(mu=mu.init, alpha=alpha.init,
                       beta=beta.init, tphi=tphi.init, theta=theta.init,
                       lgam=lgam.init)), n.chains)

# Data counts, indexes
N <- m*n # Total number of Bernoulli decisions
jj <- rep(1:n, times=m)
kk <- rep(1:m, each=n)


######################################################
##### Hyperparameters for Independence Jeffreys priors
######################################################
# Each prior is generalized to location-scale with location mu and scale sigma
# mu=0 and sigma=1 are default hyperparameter values
# Note that hyperparameters are all fixed as specified below.
# We specify hyperparameters at this level instead of within the model script 
#   StanModel-Full-Ident.R.  This allows us to efficiently perform multiple
#   Bayesian analyses and a robustness study focusing on sensitivity to priors. 

s.scale <- 1 # For convenience, common sigma hyperparameter

##### Hyperparameters for priors for mu, alpha, beta
mu_mu_pri <- mu.init; sigma_mu_pri <- 1*s.scale
mu_alpha_pri <- 0; sigma_alpha_pri <- 1*s.scale*(1/24)
mu_beta_pri <- 0; sigma_beta_pri <- 1*s.scale
# Require alpha and beta prior means to be 0.
# mean=mu_pri; sd = pi*sigma_pri
# sigma > 1 more diffuse.  sigma < 1 more informative.

##### Hyperparameters for prior for gamma (take log,lgam=log(gamma))
sigma_lgam_pri <- 1*s.scale # Need to define first 
mu_lgam_pri <- lgam.init + 0.331606*sigma_lgam_pri 
# mean = mu_lgam_pri - 0.331606*sigma_lgam_pri = lgam.init
# sd = sqrt(1.108939)*sigma_pri
# sigma_lgam_pri > 1 more diffuse, sigma_lgam_pri < 1 more informative

##### Hyperparameters for prior for phi
mu_phi_pri <- .5 # Some expert polarization
sigma_phi_pri <- 1*s.scale
# Assumes lambda=1/2 
# sigma_theta_pri=1/gamma
# Pick mu_phi_pri>0 to keep (precoded) experts (phis) on correct side
# Larger mu, more polarization
# phi pdf is maximum at +/-mu_phi_pri  
# Set at abs(mu_phi_pri) < 0.50 for more diffuse
# sigma_phi_pri>1 more diffuse, sigma_phi_pri<1 more informative

##### Hyperparameters for prior for theta
mu_theta_pri <- .5 # some polarization
sigma_theta_pri <- 1*s.scale
# Assumes lambda=1/2.  
# sigma_theta_pri=1/gamma
# This is always symmetric.  Can be unimodal or bimodal.
# mu<=0, large sigma is diffuse with symmetric (about 0) distribution of 
# theta values.
# mu>0 is bimodal. Larger mu, more values closer to -1 and 1.
# sigma_theta_pri<1 more informative, sigma_theta_pri>1 more diffuse.
# Smaller sigma_theta_pri is more polarized (more separation).  
# Larger sigma_theta_pri is polarized (less separation).  


################################################################
# Commented out lambda priors ------- JN
################################################################

# sigma_llambda_pri <- 1*s.scale; # Need to define first
# mu_llambda_pri <- llambda.init - (llambda.init - 0.361541)*sigma_llambda_pri
# mean =  mu_llambda_pri + (llambda.init - 0.361541)*sigma_llambda_pri = llambda.init
# sd = sqrt(1.388348)*sigma_llambda_pri
# sigma_llambda_pri < 1 more informative, sigma_llambda_pri > 1 more diffuse

################################################################
# ----- End STAGE 2:  Initial Values, Priors Specification -----
################################################################




###########################################################################
# -- Begin STAGE 3:  Visualize Data, Jeffreys Priors, and Initial Values --
###########################################################################

#########################################
##### Look at the input connection matrix
#########################################
#   (black, following = 1's, not, white = 0's)
x <- as.matrix(y0)
rownames(x) <- nrow(y0):1
colnames(x) <- 1:ncol(y0)
heat_map2(x,y0,legend = 'none')

# Sort followers by ideal point initial values
y00 <- y0[order(theta.init),]
y0_left <- y00[,order(colSums(y00[,1:mneg]),decreasing = TRUE)]
y0_right <- y00[,order(colSums(y00[,(mneg+1):ncol(y00)]))+mneg]
y00 <- cbind(y0_left,y0_right)

x <- as.matrix(y00)
rownames(x) <- nrow(y00):1
colnames(x) <- 1:ncol(y00)
heat_map2(x,y00,legend = 'none')



#######################################
##### Plot of Priors and Initial Values
#########################################

# mu, log(gamma), log(lambda)
par(mfrow=c(1,2))
#mu
cauchy_mu_prior(mu_pri=mu_mu_pri, sigma_pri=sigma_mu_pri, init = mu.init)

#log(gamma)
lgamma_prior(mu_pri=mu_lgam_pri, sigma_pri=sigma_lgam_pri, init=lgam.init)

################################################################
# Commented out lambda prior plots ------- JN
################################################################

#log(lambda)
# llambda_pri(mu_pri=mu_llambda_pri, sigma_pri=sigma_llambda_pri, init=llambda.init)


# alpha and beta
par(mfrow=c(2,2))
cauchy_type_prior(mu_pri=mu_alpha_pri, sigma_pri=sigma_alpha_pri, inits=alpha.init, label = "alpha")
hist(alpha.init,breaks = length(alpha.init)/2, xlab="alpha Initial Values")
cauchy_type_prior(mu_pri=mu_beta_pri, sigma_pri=sigma_beta_pri, inits=beta.init, label = "beta")
hist(beta.init, breaks = 40, xlab="beta Initial Values")


# phi and theta
par(mfrow=c(2,2))
phi_pri(cutphi,mu_phi_pri,sigma_phi_pri,inits=phi.init,mneg=mneg)
hist(c(-phi.init[1:mneg],phi.init[(mneg+1):m]),breaks = length(phi.init),
     main = 'Histogram of phi.init', xlab = 'phi.init')
theta_pri(mu=mu_theta_pri, sigma=sigma_theta_pri, inits=theta.init)
hist(theta.init, breaks = 100)
par(mfrow=c(1,1))


#########################################################################
# -- End STAGE 3:  Visualize Data, Jeffreys Priors, and Initial Values --
#########################################################################




#########################################################################
# ---------------- Begin STAGE 4:  Bayesian analysis --------------------
#########################################################################

################################
### Collect data inputs for stan
################################

################################################################
# Commented out stan.data lambda prior values ------- JN
################################################################

stan.data <- list(m=m, mcut=mneg, n=n, N=N, jj=jj, kk=kk, tcutphi=tcutphi, 
            y0=c(as.matrix(y0)), lambda=lambda.fit,
            mu_mu_pri=mu_mu_pri, sigma_mu_pri=sigma_mu_pri,
            mu_alpha_pri=mu_alpha_pri, sigma_alpha_pri=sigma_alpha_pri,
            mu_beta_pri=mu_beta_pri, sigma_beta_pri=sigma_beta_pri,
            mu_phi_pri=mu_phi_pri, sigma_phi_pri=sigma_phi_pri,
            mu_theta_pri=mu_theta_pri, sigma_theta_pri=sigma_theta_pri,
            mu_lgam_pri=mu_lgam_pri, sigma_lgam_pri=sigma_lgam_pri)
            # mu_llambda_pri=mu_llambda_pri, sigma_llambda_pri=sigma_llambda_pri)

# Stan code for the model is in this file
# source("StanModel-Full-Ident.R")

source("stanmodel-FixedLambda-Ident.R")

# Run this command if using multiple cores:
# options(mc.cores = parallel::detectCores())


######################################################################
### Compile model - to set up model, no results, ignore error messages
######################################################################
stan.model1 <- stan(model_code=stan.code, 
                    data = stan.data, init=inits, iter=1, warmup=0, 
                    chains=n.chains, seed = stan.seed)



#######################################################################
### MAIN BAYESIAN ANALYSIS
### Run stan to perform MC-MC sampling from the posterior distributions
#######################################################################
timestamp()
stan.fit1 <- stan(fit=stan.model1, data = stan.data, iter=n.iter, 
                  warmup=n.warmup, chains=n.chains, thin=thin, 
                  init=inits, seed = 123,
                  control=list(adapt_delta=0.82))
timestamp()

# Save stan.fit1 to preserve posterior samples for future work without
#   rerunning the MCMC sampling.  
save(stan.fit1, file =  'stan.fit1.rdata') 

# shinystan::launch_shinystan(stan.fit1)
# traceplot(stan.fit1)

#########################
### Diagnostic Rhat plots
#########################
# Note:  If earlier you issued this command:
#           save(stan.fit1, file =  'stan.fit1.rdata')
# to save the MCMC results, then you can load it with this 
# command:
          load("stan.fit1.rdata")
# You can access the posterior samples for further analysis without
# redoing the the MCMC sampling.

### Locations of original parameters in the output
### This follows the order of parameters as submitted to stan
imu <- 1
igam <- 4*m + 4*n + 10
ialpha1 <- 4
ialpha2 <- m + 3
iphi1 <- 1
iphi2 <- iphi1 + m - 1
ibeta1 <- 2*m + 4
ibeta2 <- ibeta1 + n - 1
itheta1 <- ibeta1 + n 
itheta2 <- ibeta2 + n

iorig <- c(imu,igam,ialpha1:ialpha2,iphi1:iphi2,ibeta1:itheta2)
# dimnames(summary(stan.fit1)$summary)[[1]][iorig]
range(summary(stan.fit1)$summary[iorig,'Rhat']) # Rhat range


### Rhat plot for original parameters
par(mfrow=c(2,1))
plot(summary(stan.fit1)$summary[iorig,'Rhat'],
     ylim = c(min(summary(stan.fit1)$summary[iorig,'Rhat']),
              max(summary(stan.fit1)$summary[iorig,'Rhat'])+.1),
     xlab="Parameter", ylab="", log="x")
title(ylab=expression(hat(R)), line=2)

ypos=max(c(1.10,summary(stan.fit1)$summary[iorig,'Rhat']))
text(2,ypos,"M,G",pos=1)
text(3+1+m/2,ypos,"Alpha",pos=1)
text(3+m+1+m/2,ypos,"Phi",pos=1)
text(3+m+m+1+n/2,ypos,"Beta",pos=1)
text(3+m+m+n+1+n/2,ypos,"Theta",pos=1)
title("Original Parameters")

abline(h = 1.1,col = 'red')
abline(v = c(1,3)) #mu, log(gamma), log(lambda)
abline(v = 3+m) #alpha
abline(v = 3+m+m) #phi
abline(v = 3+m+m+n) #beta
#abline(v = 3+m+m+n+n) #theta


### Locations of adjusted parameters in the output
### This follows the order of adjusted parameters as defined in stan
imu0 <- 2*m + 2*n + 4
igam0 <- imu0 + 1
ialpha01 <- 2*m + 2*n + 10
ialpha02 <- ialpha01 + (m-1)
iphi01 <- ialpha01 + m
iphi02 <- ialpha02 + m
ibeta01 <- iphi02 + 1
ibeta02 <- ibeta01 + (n-1)
itheta01 <- ibeta01 + n 
itheta02 <- ibeta02 + n
# ilambda <- 4*m + 4*n + 11
iadj <- c(imu0,igam0,ialpha01:iphi02,ibeta01:itheta02)

#dimnames(summary(stan.fit1)$summary)[[1]][iadj]
range(summary(stan.fit1)$summary[iadj,'Rhat']) # Rhat range

### Rhat plot for adjusted parameters
plot(summary(stan.fit1)$summary[iadj,'Rhat'],
     ylim = c(min(summary(stan.fit1)$summary[iadj,'Rhat']),
              max(summary(stan.fit1)$summary[iadj,'Rhat'])+.1),
     xlab="Parameter", ylab="", log="x")
title(ylab=expression(hat(R)), line=2)

ypos=max(c(1.1,summary(stan.fit1)$summary[iadj,'Rhat']))
text(2,ypos,"M,G",pos=1)
text(3+1+m/2,ypos,"Alpha",pos=1)
text(3+m+1+m/2,ypos,"Phi",pos=1)
text(3+m+m+1+n/2,ypos,"Beta",pos=1)
text(3+m+m+n+1+n/2,ypos,"Theta",pos=1)
title("Adjusted Parameters")

abline(h = 1.1,col = 'red')
abline(v = c(1,3)) #mu, log(gamma), log(lambda)
abline(v = 3+m) #alpha
abline(v = 3+m+m) #phi
abline(v = 3+m+m+n) #beta
#abline(v = 3+m+m+n+n) #theta



########################################################
##### Stan Output to Samples - Original Parameters/Model
########################################################

# Note:  If earlier you issued this command:
#           save(stan.fit1, file =  'stan.fit1.rdata')
# to save the MCMC results, then you can load it with this 
# command:
#           load("stan.fit1.rdata")
# You can access the posterior samples for further analysis without
# redoing the the MCMC sampling.

### Extract posterior draws using the exact parameter names defined above
### Using original parameters for estimation/prediction
### lp___ is the log posterior likelihood
samples <- rstan::extract(stan.fit1, pars=c("mu",'alpha', "phi",
                                      "beta","theta","gam","lp__"))

### Save mu, gamma, and expert estimates and statistics
results <- list(
  mu = mean(samples$mu),
  mu.sd = sd(samples$mu),
  alpha = apply(samples$alpha, 2, mean),
  alpha.sd = apply(samples$alpha, 2, sd),
  alpha.lower = apply(samples$alpha, 2, quantile,probs = .025),
  alpha.upper = apply(samples$alpha, 2, quantile,probs = .975),
  phi = apply(samples$phi, 2, mean),
  phi.sd = apply(samples$phi, 2, sd),
  phi.lower = apply(samples$phi, 2, quantile,probs = .025),
  phi.upper = apply(samples$phi, 2, quantile,probs = .975),
  gam = mean(samples$gam),
  gam.sd = sd(samples$gam),
  lp = mean(samples$lp__))

### Save followers estimates and statistics
results_followers <- data.frame(
  beta = apply(samples$beta, 2, mean),
  beta.sd = apply(samples$beta, 2, sd),
  theta = apply(samples$theta, 2, mean),
  theta.sd = apply(samples$theta, 2, sd),stringsAsFactors=F)

#########################################################################
# ----------------- End STAGE 4:  Bayesian analysis ---------------------
#########################################################################




#########################################################################
# ------------------- Begin STAGE 5:  Diagnostics------------------------
#########################################################################

#######################################################
##### MCMC Diagnostics:  Plots of samples and estimates
#######################################################

### Histograms for posterior draws: mu, gamma, lambda, log posterior
par(mfrow=c(2,2))
hist(samples$mu, main=expression(mu~Samples), xlab=expression(mu))
hist(samples$gam, main=expression(gamma~Samples), xlab=expression(gamma))
hist(samples$lp__, main = 'Log Posterior of Samples', xlab="Log Posterior")

### Histograms for posterior means (estimates) and sds: 
#  m=42 alphas, m=42 phis
par(mfrow=c(2,2))
hist(results$alpha, breaks = 100, xlim = c(-5,5), ylim = c(0,2), xlab=expression(hat(alpha)[j]), main = paste(m," Popularity Estimates"))
hist(results$alpha.sd, breaks= 100, xlab=expression(SD~of~alpha[j]), main = paste("SDs of ", m, " Popularity Samples"))
hist(results$phi, breaks = 100, xlim = c(-1.1,1.1), xlab=expression(hat(phi)[j]), main = paste(m," Ideal Point Estimates"))
hist(results$phi.sd, breaks = 100, xlab=expression(SD~of~phi[j]), main = paste("SDs of ",m," Ideal Point Samples"))

#  n betas, n thetas
par(mfrow=c(2,2))
hist(results_followers$beta, breaks=100, xlab=expression(hat(beta)[i]), main = paste(n," Engagement Estimates"))
hist(results_followers$beta.sd, breaks = 100, xlab=expression(SD~of~beta[i]), main = paste("SDs of ",n," Engagement Samples"))
hist(results_followers$theta, breaks = 100, xlim = c(-1.1,1.1),  xlab=expression(hat(theta)[i]), main = paste(n," Ideal Point Estimates"))
hist(results_followers$theta.sd, breaks = 100, xlab=expression(SD~of~theta[i]), main = paste("SDs of M",n," Ideal Point Samples"))

par(mfrow=c(2,1))
# Ideal point estimates by expert
plot(results$phi, xlab="Expert", ylab="", ylim = c(-1.1,1.1))
title(ylab=expression(hat(phi)[j]), line=2)
abline(h=-1,col="black")
abline(h=1,col="black")
abline(v=mneg,col="red")
abline(h=cutphi,col="red")
abline(h=(-cutphi),col="red")

# Ideal point estimates by follower
plot(results_followers$theta, xlab="Follower", ylab="", ylim = c(-1.1,1.1))
title(ylab=expression(hat(theta)[i]), line=2)
abline(h=-1,col="black")
abline(h=1,col="black")
abline(v=n.neg,col="red")

# Histograms for posterior samples of individual expert ideal points
par(mfrow=c(3,4))
for(i in 1:m){
  hist(samples$phi[,i], main=paste(colnames(y0)[i],"-","phi",i,sep=""), xlab=expression(phi[j]), breaks = 30)
}
par(mfrow=c(1,1))



#######################################################
##### Prediction Diagnostics
#######################################################

logit <- function(a){
  return(1/(1+exp(-a)))
}

p_mat <- as.matrix(y0)
p_hat <- matrix(0,nrow = length(results_followers$beta), ncol = length(results$alpha))
y_res <- matrix(0,nrow = length(results_followers$beta), ncol = length(results$alpha))
p_predict <- matrix(0,nrow = length(results_followers$beta), ncol = length(results$alpha))

### Estimate p_ij using posterior draws 
for(j in 1:length(results$alpha)){
  for(i in 1:length(results_followers$beta)){
    p_hat[i,j]  <- mean(logit(samples$mu+samples$alpha[,j]+samples$beta[,i]-
                                samples$gam*(abs(samples$theta[,i]-samples$phi[,j])^(2*lambda.fit)-1)/(lambda.fit)))
    # get posterior draws for pij, take mean
  }
}


#### Draw individual ROC curves for each expert
par(mfrow=c(3,4))
threshold_j <- c()
for(j in 1:length(results$alpha)){
  roc_obj <- roc(y0[,j], p_hat[,j]) # Create an ROC object
  # roc_obj  #review the roc object
  plot(roc_obj, main = dimnames(y0)[[2]][[j]])

  # Get the "threshold" that maximizes sensitivityy+specificity=2*(Youden's J) 
  # Standard threshold for prediction is 0.50
  print(coords(roc_obj, "best", "threshold"))
  threshold_j[j] <- coords(roc_obj, "best", "threshold")[1]
}


### Combine all data for one ROC curve
x <- cbind(y0[,1],p_hat[,1])
for(j in 2:length(results$alpha)){
  df <- cbind(y0[,j],p_hat[,j])
  x <- rbind(x,df)
}

roc_obj <- roc(x[,1], x[,2]) # Create an roc object
roc_obj  # Review the roc object

### Draw an overall ROC curve using optimal threshold for all observations
par(mfrow=c(1,1))
plot(roc_obj, main = "", asp=1, cex.lab=1.5, cex.main=1.5, cex.axis=1.5)
title("All Observations", adj = 0, line = 2.5, cex.main=2)
print(coords(roc_obj, "best", "threshold")) # Sensitivity and specificity at threshold
print(coords(roc_obj, "best", ret="threshold", transpose = FALSE, best.method="youden"))
print(coords(roc_obj, "best", ret="threshold", transpose = FALSE, best.method="closest.topleft"))

# Threshold-Youden method sanity check
coords(roc_obj, "best", "threshold")[1,1] == coords(roc_obj, "best", ret="threshold", transpose = FALSE, best.method="youden")[1,1]

# Store Youden threshold for overall prediction
threshold_j[m+1] <- coords(roc_obj, "best", "threshold")[1]



#####################################################
### Predictions (fitted values) with all observations
#####################################################

# Pick a threshold THRESH for predictions:
# THRESH <- .5  # Standard 
THRESH <- threshold_j[m+1]  # Youden threshold (max Sensitivity+Specificity)

# Predict will follow (1) if p_hat > threshold, otherwise not (0)
for(j in 1:length(results$alpha)){
  for(i in 1:length(results_followers$beta)){
    if( p_hat[i,j] > THRESH){
      p_predict[i,j] <- 1
    } else {
      p_predict[i,j] <-0
    }
  }
}

par(mfrow=c(2,3))
y_res <- p_mat - p_hat
hist(p_mat)
hist(p_predict)
hist(p_hat, breaks = 100)
hist(y_res, breaks = 100)

### Fit diagnostics 
BIC <- (3+2*m+2*n)*log(N)-2*mean(samples$lp__) # Bayesian information criterion, small ideally
SSEp <- sum((y_res)^2) # SS error in p estimation, small ideally
SSE <- SSEp/(m*n) # Average SSEp across all observations
print(paste0("BIC: ",round(BIC,4)," | SSEp: ",round(SSEp,4)," | SSE: ",round(SSE,4)))


### Confusion matrices
mats <- list(); conf_mat <- list(); plots <- list()

### Confusion matrix for each expert
for(j in 1:m){
  mats[[j]] <- cbind(p_predict[,j],p_mat[,j])
  conf_mat[[j]] <- confusionMatrix(as.factor(mats[[j]][,1]), as.factor(mats[[j]][,2]))
  # print(conf_mat[[j]])
  table <- data.frame(confusionMatrix(as.factor(mats[[j]][,1]), as.factor(mats[[j]][,2]))$table)

  plotTable <- table %>%
    mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
    # group_by(Reference) %>%
    mutate(prop = Freq/sum(Freq))

  # Fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
  plots[[j]] <- ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
    ggtitle(dimnames(y0)[[2]][[j]]) +
    geom_tile() +
    geom_text(aes(label = round(prop,3)), vjust = .5, fontface  = "bold", alpha = 1, size=12) +
    scale_fill_manual(values = c(good = "green", bad = "red")) +
    theme_bw() +
    xlim(rev(levels(table$Reference))) + theme(text = element_text(size = 20),
                                               legend.position = 'none',
                                               plot.title = element_text(face = "bold"))
   # print(plots[[j]]) 
}

x <- mats[[1]]
for(j in 2:length(results$alpha)){
  x <- rbind(x,mats[[j]])
}



### Overall (all observations) confusion matrix 
conf_mat[[m+1]] <- confusionMatrix(as.factor(x[,1]), as.factor(x[,2]))
print(conf_mat[[m+1]])
table <- data.frame(confusionMatrix(as.factor(x[,1]), as.factor(x[,2]))$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  # group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# Fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
plots[[m+1]] <- ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  ggtitle("All Observations") +
  geom_tile() +
  geom_text(aes(label = round(prop,3)), vjust = .5, fontface  = "bold", alpha = 1, size=12) +
  scale_fill_manual(values = c(good = "green3", bad = "red4")) +
  theme_bw() +
  xlim(rev(levels(table$Reference))) + theme(text = element_text(size = 20),
                                             legend.position = 'none',
                                             plot.title = element_text(face = "bold"))

p_all <- plots[[m+1]]
print(p_all)

# Heat map of estimated p_ij values
phat00 <- p_hat[order(theta.init),]
colnames(phat00) <- colnames(y0)
phat00 <- phat00[,colnames(y00)]

x <- as.matrix(phat00)
rownames(x) <- nrow(y00):1
colnames(x) <- 1:ncol(y00)
par(mfrow=c(1,1))
heat_map2(x,y00,legend = NULL)

#########################################################################
# ------------------- End STAGE 5:  Diagnostics------------------------
#########################################################################



#########################################################################
### --------------------- Begin STAGE 6: --------------------------
### Evaluating Popularity alpha and Ideal Point phi (Experts)
#########################################################################

### Plotting both Popularity and Ideology in the Same Window


#------------ Begin popularity setup--------------------
df <- samples$alpha
colnames(df) <- colnames(y0)
type <- c(rep('Anti',mneg),rep('Pro',mpos))

df <- df[,order(results$alpha)] # Sort by increasing alpha 
type <- type[order(results$alpha)]

for (i in 1:m){
  y <- df[,i]
  df1 <- data.frame(
    Expert = colnames(df)[i],
    Type = type[i],
    Index = i,
    CI_Lower = quantile(y, 0.025),
    y25 = quantile(y, 0.25),
    y50 = median(y),
    y75 = quantile(y, 0.75),
    CI_Upper = quantile(y, 0.975)
  )
  if (i == 1){
    df2 <- df1
  } else{
    df2 <- rbind(df2,df1)
  }
}

df2$Expert <- factor(df2$Expert, levels = colnames(df))
df2$Ideology <- df2$Type

bp_horiz1 <- ggplot(df2, aes(x=Expert, col=Ideology)) +
  geom_hline(yintercept = 0, color="grey80", linetype = 'dotted') + 
  geom_boxplot(aes(ymin = CI_Lower, lower = y25, middle = y50, upper = y75, ymax = CI_Upper),
               stat = "identity",fill="grey98",outlier.shape = NA, width=.35) +
  theme_classic()+
  # labs(y=expression(hat(alpha[j])))+
  ylab(expression(paste("(a) Posterior Samples of ", alpha[j]))) +
  scale_y_continuous(limits=c(-6,6))+
  scale_color_manual(values=c("blue",'red'))+
  #scale_x_discrete()+
  coord_flip() + list(
    theme(plot.title = element_text(size = 14, face = "bold"),
          axis.title.x = element_text(size = 14, v = -10),
          axis.text.x = element_text(size = 12),
          #axis.ticks.x = element_blank(), #remove x axis ticks
          axis.title.y =  element_blank(),
          axis.text.y =  element_text(size = 10),  #remove y axis labels
          #axis.ticks.y = element_blank(), #remove y axis ticks
          legend.position = 'none',       # change to "none" if you want no legend
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12),
          legend.key.size = unit(2,'line'),
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=.5),
          plot.margin = margin(1,.5,1.5,.5, "cm")),
    guides(col = guide_legend(reverse = TRUE, override.aes = list(linetype=0,shape=16, fill = c('red',"blue")))))
#------------ end popularity setup --------------------------


#------------ Begin ideal point setup ----------------------
df <- samples$phi
colnames(df) <- colnames(y0)
type <- c(rep('Anti',mneg),rep('Pro',mpos))

df <- df[,order(results$phi)] # Increasing
type <- type[order(results$phi)]

for (i in 1:m){
  y <- df[,i]
  df1 <- data.frame(
    Expert = colnames(df)[i],
    Type = type[i],
    Index = i,
    CI_Lower = quantile(y, 0.025),
    y25 = quantile(y, 0.25),
    y50 = median(y),
    y75 = quantile(y, 0.75),
    CI_Upper = quantile(y, 0.975)
  )
  if (i == 1){
    df2 <- df1
  } else{
    df2 <- rbind(df2,df1)
  }
}

df2$Expert <- factor(df2$Expert, levels = colnames(df))
df2$Ideology <- df2$Type

bp_horiz2 <- ggplot(df2, aes(x=Expert, col=Ideology)) +
  geom_hline(yintercept = 0, color="grey80", linetype = 'dotted') + 
  geom_boxplot(aes(ymin = CI_Lower, lower = y25, middle = y50, upper = y75, ymax = CI_Upper),
               stat = "identity",fill="grey98",outlier.shape = NA, width=.35) +
  theme_classic()+
  # labs(y=expression(hat(alpha[j])))+
  ylab(expression(paste("(b) Posterior Samples of ", phi[j]))) +
  scale_y_continuous(limits=c(-1,1))+
  scale_color_manual(values=c("blue",'red'))+
  #scale_x_discrete()+
  coord_flip() + list(
    theme(plot.title = element_text(size = 14, face = "bold"),
          axis.title.x = element_text(size = 14, v = -10),
          axis.text.x = element_text(size = 12),
          #axis.ticks.x = element_blank(), #remove x axis ticks
          axis.title.y =  element_blank(),
          axis.text.y =  element_text(size = 10),  #remove y axis labels
          #axis.ticks.y = element_blank(), #remove y axis ticks
          legend.position = 'none',       # change to "none" if you want no legend
          legend.title = element_text(size = 14),
          legend.text = element_text(size = 12),
          legend.key.size = unit(2,'line'),
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=.5),
          plot.margin = margin(1,.5,1.5,.5, "cm")),
    guides(col = guide_legend(reverse = TRUE, override.aes = list(linetype=0,shape=16, fill = c('red',"blue")))))
#------------ Begin ideal point setup ----------------------


# Side-by-side boxplots popularity alpha, ideal point phi by expert
ggarrange(bp_horiz1, bp_horiz2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")

####################################################################
### Begin:  LateX - Building a Table of Statistics for alpha and phi
####################################################################
# Summary stats of alpha
df <- samples$alpha
colnames(df) <- colnames(y0)
type <- c(rep('Anti',21),rep('Pro',21))

df <- df[,order(results$alpha)] # Increasing
type <- type[order(results$alpha)]

for (i in 1:m){
  y <- df[,i]
  df1 <- data.frame(
    Expert = colnames(df)[i],
    Type = type[i],
    Index = i,
    CI_Lower = quantile(y, 0.025),
    y25 = quantile(y, 0.25),
    y50 = median(y),
    y_bar = mean(y),
    y_sd = sd(y),
    y75 = quantile(y, 0.75),
    CI_Upper = quantile(y, 0.975)
  )
  if (i == 1){
    df2 <- df1
  } else{
    df2 <- rbind(df2,df1)
  }
}

rownames(df2) <- df2$Index
df_pop <- df2


# Summary stats of phi
df <- samples$phi
colnames(df) <- colnames(y0)
type <- c(rep('Anti',mneg),rep('Pro',mpos))

df <- df[,order(results$phi)] # Increasing
type <- type[order(results$phi)]

for (i in 1:m){
  y <- df[,i]
  df1 <- data.frame(
    Expert = colnames(df)[i],
    Type = type[i],
    Index = i,
    CI_Lower = quantile(y, 0.025),
    y25 = quantile(y, 0.25),
    y50 = median(y),
    y_bar = mean(y),
    y_sd = sd(y),
    y75 = quantile(y, 0.75),
    CI_Upper = quantile(y, 0.975)
  )
  if (i == 1){
    df2 <- df1
  } else{
    df2 <- rbind(df2,df1)
  }
}

rownames(df2) <- df2$Index
df_inf <- df2

df_out <- rbind(df_pop[m,],df_pop[m-1,],df_pop[m-2,],df_pop[1,],df_pop[2,],df_pop[3,],
      df_inf[1,],df_inf[2,],df_inf[3,],df_inf[(m/2)-1,],df_inf[(m/2),],df_inf[(m/2)+1,],df_inf[(m/2)+2,],
      df_inf[m-2,],df_inf[m-1,],df_inf[m,])

df_out <- df_out[,c(1,2,7,6,8,4,10)]

colnames(df_out) <- c('Expert','Type','Mean','Median','SD','CI Lower','CI Upper')
rownames(df_out) <- NULL
print(xtable(df_out, type = "latex",digits = 3,scalebox='0.75'),include.rownames=FALSE)
##################################################################
### End:  LateX - Building a Table of Statistics for alpha and phi
##################################################################

#########################################################################
### --------------------- End STAGE 6: --------------------------
### Evaluating Popularity alpha and Ideal Point phi (experts)
#########################################################################




#########################################################################
### --------------------- Begin STAGE 7: --------------------------
### Evaluating Engagement beta and Ideal Point theta (Followers)
#########################################################################

#------------ Begin follower engagement histogram setup--------------------
annotations <- data.frame(
  value = c(round(mean(results_followers$beta), 3), 
        round(median(results_followers$beta), 3), 
        round(quantile(results_followers$beta, 0.25), 3), 
        round(quantile(results_followers$beta, 0.75), 3)),
  x = c(1,1,1,1),
  y = c(280, 265, 250, 235),
  label = c("Mean:", "Median:", "25th Percentile:", '75th Percentile:')
)

ggplot(results_followers, aes(beta)) +
  geom_histogram(color = "#000000", fill = "grey70", binwidth = .06) +
  theme_classic()+
  # labs(y=expression(hat(alpha[j])))+
  #xlab(expression(paste("Posterior Samples of ", beta[i]))) +
  labs(
    #title = "Histogram of Life Expectancy in Europe",
    #subtitle = "Made by Appsilon",
    x = expression(paste("Posterior Samples of ", beta[i])),
    y = "Frequency"
  ) +
  scale_x_continuous(limits=c(-5,5))+
  scale_y_continuous(limits=c(0,300))+
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.title.x = element_text(size = 14, v = -10),
        axis.text.x = element_text(size = 12),
        #axis.ticks.x = element_blank(), #remove x axis ticks
        axis.title.y =  element_text(size = 14, v = 10),
        axis.text.y =  element_text(size = 12),  #remove y axis labels
        #axis.ticks.y = element_blank(), #remove y axis ticks
        legend.position = 'none',       # change to "none" if you want no legend
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12),
        legend.key.size = unit(2,'line'),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=.5),
        plot.margin = margin(1,1.5,1.5,1.5, "cm")) +
        geom_text(data = annotations, aes(x = x, y = y,
                                          label = paste(label, value)), 
                  size = 5, fontface = "bold",hjust = 0)
#------------ End follower engagement histogram setup--------------------


#------------ Begin follower ideal points histogram setup---------------
annotations <- data.frame(
  value = c(round(mean(results_followers$theta), 3), 
            round(median(results_followers$theta), 3), 
            round(quantile(results_followers$theta, 0.25), 3), 
            round(quantile(results_followers$theta, 0.75), 3)),
  x = c(.3,.3,.3,.3),
  y = c(230, 215, 200, 185),
  label = c("Mean:", "Median:", "25th Percentile:", '75th Percentile:')
)

ggplot(results_followers, aes(theta)) +
  geom_histogram(color = "#000000", fill = "grey70", binwidth = .015) +
  theme_classic()+
  # labs(y=expression(hat(alpha[j])))+
  #xlab(expression(paste("Posterior Samples of ", beta[i]))) +
  labs(
    #title = "Histogram of Life Expectancy in Europe",
    #subtitle = "Made by Appsilon",
    x = expression(paste("Posterior Samples of ", theta[i])),
    y = "Frequency"
  ) +
  scale_x_continuous(limits=c(-1,1))+
  scale_y_continuous(limits=c(0,250))+
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.title.x = element_text(size = 14, v = -10),
        axis.text.x = element_text(size = 12),
        #axis.ticks.x = element_blank(), #remove x axis ticks
        axis.title.y =  element_text(size = 14, v = 10),
        axis.text.y =  element_text(size = 12),  #remove y axis labels
        #axis.ticks.y = element_blank(), #remove y axis ticks
        legend.position = 'none',       # change to "none" if you want no legend
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12),
        legend.key.size = unit(2,'line'),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=.5),
        plot.margin = margin(1,1.5,1.5,1.5, "cm")) +
  geom_text(data = annotations, aes(x = x, y = y,
                                    label = paste(label, value)), 
            size = 5, fontface = "bold",hjust = 0)
#------------ Begin follower engagement histogram setup---------------



#######################################################################
### Begin:  LateX - Building a Table of Statistics for beta, theta, mu,
###                 gamma, and lambda
#######################################################################

# Summary stats of beta
y <- results_followers$beta
df1 <- data.frame(
  CI_Lower = quantile(y, 0.025),
  y25 = quantile(y, 0.25),
  y50 = median(y),
  y_bar = mean(y),
  y_sd = sd(y),
  y75 = quantile(y, 0.75),
  CI_Upper = quantile(y, 0.975)
)

df_engage <- df1

# Summary stats of theta
y <- results_followers$theta
df1 <- data.frame(
  CI_Lower = quantile(y, 0.025),
  y25 = quantile(y, 0.25),
  y50 = median(y),
  y_bar = mean(y),
  y_sd = sd(y),
  y75 = quantile(y, 0.75),
  CI_Upper = quantile(y, 0.975)
)

df_ideo <- df1

# Summary stats of mu
y <- samples$mu
df1 <- data.frame(
  CI_Lower = quantile(y, 0.025),
  y25 = quantile(y, 0.25),
  y50 = median(y),
  y_bar = mean(y),
  y_sd = sd(y),
  y75 = quantile(y, 0.75),
  CI_Upper = quantile(y, 0.975)
)

df_mu <- df1

# Summary Stats of gamma
y <- samples$gam
df1 <- data.frame(
  CI_Lower = quantile(y, 0.025),
  y25 = quantile(y, 0.25),
  y50 = median(y),
  y_bar = mean(y),
  y_sd = sd(y),
  y75 = quantile(y, 0.75),
  CI_Upper = quantile(y, 0.975)
)

df_gam <- df1

# Summary stats of lambda
# y <- samples$lambda
# df1 <- data.frame(
#   CI_Lower = quantile(y, 0.025),
#   y25 = quantile(y, 0.25),
#   y50 = median(y),
#   y_bar = mean(y),
#   y_sd = sd(y),
#   y75 = quantile(y, 0.75),
#   CI_Upper = quantile(y, 0.975)
# )
# 
# df_lam <- df1

rownames(df2) <- df2$Index
df_inf <- df2

df_out <- rbind(df_engage,df_ideo,df_mu,df_gam)

df_out <- df_out[,c(4,3,5,1,7)]

colnames(df_out) <- c('Mean','Median','SD','CI Lower','CI Upper')
rownames(df_out) <- c('Beta','Theta','Mu','Gamma')
print(xtable(df_out, type = "latex",digits = 3,scalebox='0.75'),include.rownames=TRUE)
#######################################################################
### End:  LateX - Building a Table of Statistics for beta, theta, mu,
###               gamma, and lambda
#######################################################################
