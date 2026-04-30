# Inverse logit function
logitinv <- function(a){
  return(1/(1+exp(-a)))
}

# Heat map
heat_map2 <- function(x,y0,legend = NULL){
  x <- t(x)[,nrow(x):1]
  data_melt <- melt(x)  
  data_melt$Var1 <- as.factor(data_melt$Var1)
  levels(data_melt$Var1) <- colnames(y0)
  str(data_melt)
  names(data_melt)[3] <- 'Scale'
  
  ggp <- ggplot(data_melt, aes(Var1, Var2)) +                           # Create heatmap with ggplot2
    geom_tile(aes(fill = Scale)) + scale_fill_gradient(low = "white", high = "black", limits = c(0,1)) + scale_x_discrete(position = "top") +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(angle = 90, vjust = 0, hjust=.1),
          # axis.ticks.x=element_blank(), #remove x axis ticks
          axis.text.y=element_blank(),  #remove y axis labels
          axis.ticks.y=element_blank(), #remove y axis ticks
          legend.position = legend,       # change to "none" if you want no legend
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1)
    ) +
    scale_y_discrete(expand = expansion(mult = 0))
  ggp
}


############################################################
# ------------ Jeffreys Prior Plots -----------------------
############################################################

#------------------------------------------------------------------------------
# Plot of prior on mu
cauchy_mu_prior <- function(mu_pri=0, sigma_pri=1, init, label = "mu Prior", 
                            npts=100){
  ave <- mu_pri # mean
  SD <- pi*sigma_pri # sd
  t <- seq(ave-4*SD, ave + 4*SD, length=npts) # find wide range of values
  z <- (t-mu_pri)/sigma_pri
  y <- 1/(pi*sigma_pri)*((exp(-z/2)/(1+exp(-z))))
  plot(t,y,xlim=range(c(t,init)),ylim=c(0,max(y)),main = label, type="l", xlab=expression(mu))
  points(init,y=rep(0,length(init)),col=2)
  abline(h=0)
  abline(v=mu_pri, col=3)
  print(paste("Mean =", ave))
  print(paste("SD =", round(SD,4)))
}

#---------------------------------------

#------------------------------------------------------------------------------
# Plot of prior on alpha and beta
cauchy_type_prior <- function(mu_pri=0, sigma_pri=1, inits, label, npts=100){
  ave = mu_pri # mean
  SD <- pi*sigma_pri # sd
  t <- seq(ave-4*SD, ave + 4*SD, length=npts) # find wide range of values
  z <- (t-mu_pri)/sigma_pri
  y <- 1/(pi*sigma_pri)*(exp(-z/2)/(1+exp(-z)))
  if(label=="alpha") 
	xlab=expression(alpha)
  if(label=="beta") 
	xlab=expression(beta)
  plot(t,y,xlim=range(c(t,inits)),ylim=c(0,max(y)),xlab=xlab,main = paste(label,"Prior"),type="l")
  points(x=inits,y=rep(0,length(inits)),col=2)
  abline(h=0)
  abline(v=ave, col=3)
}

#---------------------------------------

#------------------------------------------------------------------------------
# Plot of prior on log(gamma)
lgamma_prior <- function(mu_pri=0, sigma_pri=1, init, label = "log(gamma) Prior", npts=100){
  ave <- mu_pri - 0.260443*sigma_pri
  SD <- sqrt(1.108939)*sigma_pri
  t <- seq(ave-4*SD, ave+4*SD,length=npts) # find wide range of values
  z <- (t-mu_pri)/sigma_pri
  y <- 4/(pi*sigma_pri)*(exp(z-exp(z)))/(1+exp(-2*exp(z)))
  plot(t,y,xlim=range(c(t,init)), ylim=c(0,max(y)), main = label, type="l", xlab=expression(log(gamma)))
  points(x=init,y=rep(0,length(init)),col=2, pch=)
  abline(v=ave,col=3)
  abline(h=0)
  print(paste("Mean =", round(ave,4)))
  print(paste("SD =", round(SD,4)))
}

#---------------------------------------

#------------------------------------------------------------------------------
# Plot of prior on phi
# mu_pri controls variance
# sigma_pri is the mode of the pdf
phi_pri <- function(cutphi, mu_pri, sigma_pri, inits, label="phi Priors",
                    npts=100, mneg = mneg){
# mu__pri>0; gam=1
# sigma_pri = mu_pri/gam
  normc <- 1/(sigma_pri*(atan(exp((1-mu_pri)/sigma_pri))-atan(exp((-cutphi-mu_pri)/sigma_pri))))
  # Positive ideal points
  x1 <- seq(-cutphi,1,length=npts)
  z1 <- (x1 - mu_pri)/sigma_pri
  y1 <- normc*exp(-z1)/(1+exp(-2*z1))
  #
  # Negative idea points
  x2 <- seq(-1,cutphi,length=npts)
  z2 <- (-x2 - mu_pri)/sigma_pri
  y2 <- normc*exp(-z2)/(1+exp(-2*z2))
  #
  plot(x1,y1,ylim=c(0,max(c(y1,y2))),xlim=c(-1,1),main = label,col=1, type="l",xlab=expression(phi))
  lines(x2,y2,col=2)
  abline(h=0)
  abline(v=c(-1,1), col=2:1)
  abline(v=c(-1,1)*mu_pri, col=3) # pdf max 
  abline(v=c(-1,1)*cutphi, col=1:2) # cutphi
  points(-inits[1:mneg], rep(0,mneg), col=2)
  points(inits[(mneg+1):length(inits)], rep(0,(length(inits)-mneg)), col=1)
}

#---------------------------------------

#------------------------------------------------------------------------------
# Plot of prior on theta
theta_pri <- function(mu_pri,sigma_pri,inits, label="theta Prior", npts=100){
  #This assumes lambda=0.5.
  t <- seq(-1,1,length=npts)
  normc = 1/(2*sigma_pri*(atan(exp((1-mu_pri)/sigma_pri))-atan(exp(-mu_pri/sigma_pri))))
  #
  z <- (abs(t)-mu_pri)/sigma_pri
  y <- normc*exp(-z)/(1+exp(-2*z))
  plot(t,y,ylim=c(0,max(y)),main=label,type="l", xlab=expression(theta))
  points(inits,y=rep(0,length(inits)),col=2)
  abline(h=0, v=c(-1,1))
  abline(v=mu_pri*c(-1,1), col=3) # pdf max
}

#---------------------------------------

#---------------------------------------
# Plot of prior on log(lambda)
llambda_pri <- function(mu_pri,sigma_pri,init, label="log(lambda) Prior", npts=100){
  ave <- mu_pri + (init - 0.361541)*sigma_pri # mean
  SD <- sqrt(1.388348)*sigma_pri # sd
  t <- seq(ave-4*SD, ave+4*SD, length=npts) # find wide range of values
  z <- (t-mu_pri)/sigma_pri
  y = (2/(pi*sigma_pri*exp(z)))*exp(-1/(2*exp(z)))/(1+exp(-1/(exp(z))));
  plot(t,y,type="l", xlim=range(c(t,init)), main=label, xlab=expression(log(lambda)))
  abline(h=0)
  abline(v=ave, col=3)
  points(init,0,col=2)
}