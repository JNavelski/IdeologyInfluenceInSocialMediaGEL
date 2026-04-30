stan.code <- '
functions {
// Jeffreys prior on mu
  real mu_prior_lpdf(real x, real mu_mu_pri,real sigma_mu_pri){
  // mean = mu_mu_pri, sd = pi*sigma_mu_pri 
    real y;
    real z;
    z = (x-mu_mu_pri)/sigma_mu_pri;
    y = 1/(sigma_mu_pri*pi())*(exp(-z/2)/(1+exp(-z)));
    return log(y);
  }
  //
// Jeffreys prior on alpha and beta
  real Cauchy_prior_lpdf(vector x, real mu_pri, real sigma_pri, int ss){
  // mean = mu_pri, sd = pi*sigma_pri
    vector[ss] y;
    real z;
    for(i in 1:ss){
      z = (x[i]-mu_pri)/sigma_pri;
      y[i] = 1/(sigma_pri*pi())*(exp(-z/2)/(1+exp(-z)));
    }
  return sum(log(y));
  }
  //
// Jeffreys prior on log(gamma)
  real lgam_pri_lpdf(real x, real mu_lgam_pri, real sigma_lgam_pri){
    // x is log(gamma)
    // mean = mu_lgam_pri - 0.260443*sigma_lgam_pri
    // sd = sqrt(1.108939)*sigma_lgam_pri
    real y;
    real z;
    z = (x-mu_lgam_pri)/sigma_lgam_pri;
    y = 4/(sigma_lgam_pri*pi())*exp(z-exp(z))/(1+exp(-2*exp(z)));
    return log(y);
  }
  //
// Jeffreys prior on tphi
  real tphi_pri_lpdf(vector x, real mu_phi_pri, real sigma_phi_pri, real tcutphi, int ss, int mcut){
    vector[ss] y;
    vector[ss] phi;
    real cutphi;
    real normc;
    real jacob;
    real z;
    cutphi = atan(tcutphi)*2/pi();
    normc = 1/(sigma_phi_pri*(atan(exp((1-mu_phi_pri)/sigma_phi_pri))-atan(exp((-cutphi-mu_phi_pri)/sigma_phi_pri))));
//
    for(j in 1:mcut){
    // make Anti (Fox) phi range from -1 to cutphi
      phi[j] = atan( tcutphi - exp(x[j]))*2/pi();
      jacob = (2*exp(x[j])/pi())*1/(1+square(-tcutphi+exp(x[j])));
      z = (-phi[j]-mu_phi_pri)/sigma_phi_pri;
      y[j] = jacob*normc*exp(-z)/(1+exp(-2*z));
    }
    //
    for(k in (mcut+1):ss){
    // make Pro (CNN) phi range from -cutphi to 1
      phi[k] = atan(-tcutphi + exp(x[k]))*2/pi();
      jacob = (2*exp(x[k])/pi())*1/(1+square(tcutphi-exp(x[k])));
      z = (phi[k]-mu_phi_pri)/sigma_phi_pri;
      y[k] = jacob*normc*exp(-z)/(1+exp(-2*z));
    }
    //
    return sum(log(y));
  }
//
// Jeffreys prior on theta
  real theta_pri_lpdf(vector x, real mu_theta_pri, real sigma_theta_pri, int ss){
    vector[ss] y;
    real normc;
    real z;
//
    normc = 1/(2*sigma_theta_pri*(atan(exp((1-mu_theta_pri)/sigma_theta_pri))-atan(exp(-mu_theta_pri/sigma_theta_pri))));
    for(i in 1:ss){
      z = (fabs(x[i])-mu_theta_pri)/sigma_theta_pri;
      y[i] = normc*exp(-z)/(1+exp(-2*z));
    }
    return sum(log(y));
  }
}
data {
  int<lower=1> n; // number of twitter users
  int<lower=1> m; // number of elite twitter accounts
  int<lower=1> mcut;
  int<lower=1> N; // N = m x n
  int<lower=1,upper=n> jj[N]; // twitter user for observation n
  int<lower=1,upper=m> kk[N]; // elite account for observation n
  int<lower=0,upper=1> y0[N]; // dummy if user i follows elite j
  real tcutphi;
  real lambda;
  real mu_mu_pri; 
  real sigma_mu_pri;
  real mu_alpha_pri;
  real sigma_alpha_pri;
  real mu_beta_pri;
  real sigma_beta_pri;
  real mu_phi_pri;
  real sigma_phi_pri;
  real mu_theta_pri;
  real sigma_theta_pri;
  real mu_lgam_pri;
  real sigma_lgam_pri;
}
parameters {
  real mu;
  real lgam;
  vector[m] alpha;
  vector[m] tphi;
  vector[n] beta;
  vector<lower=-1, upper=+1>[n] theta; 
}
transformed parameters {
  real mu0;
  real gam0;
  real alphabar;
  real betabar;
  real phibar;
  real phisd;
  real alpha0[m];
  real phi0[m];
  real beta0[n];
  real theta0[n];
  real<lower=0> gam;
  vector<lower=-1, upper=+1>[m] phi;
  //
  alphabar =  mean(alpha);
  for(j in 1:m){
    // constrain the alpha values to sum to 0
    alpha0[j] = alpha[j] - alphabar; //standardize
  }
  //
  betabar = mean(beta);
  for(j in 1:n){
    // constrain the beta values to sum to 0
    beta0[j] = beta[j] - betabar; //standardize
  }
  gam = exp(lgam);
  mu0 = mu + alphabar + betabar + gam/lambda;  //adjust mu
  //
  for(j in 1:mcut){
    // make Anti phi range from -1 to cutphi
    phi[j] = atan(+tcutphi - exp(tphi[j]))*2/pi();
  }
  for(k in (mcut+1):m){
    // make Pro phi range from -cutphi to 1
    phi[k] = atan(-tcutphi + exp(tphi[k]))*2/pi();
  }
  phibar = mean(phi);
  phisd = sd(phi);
  gam0 = (gam * (phisd)^(2 * lambda))/lambda; //adjust gamma
  for(j in 1:m){
	phi0[j] = (phi[j] - phibar)/phisd; //standardize phi
  }
  for(j in 1:n){
    theta0[j] = (theta[j] - phibar)/phisd; // standardize
  }
}
model {
  mu ~ mu_prior(mu_mu_pri,sigma_mu_pri);
  lgam ~ lgam_pri(mu_lgam_pri, sigma_lgam_pri);
  alpha ~ Cauchy_prior(mu_alpha_pri,sigma_alpha_pri,m);
  tphi ~ tphi_pri(mu_phi_pri, sigma_phi_pri, tcutphi, m, mcut);
  beta ~ Cauchy_prior(mu_beta_pri,sigma_beta_pri,n);
  theta ~ theta_pri(mu_theta_pri, sigma_theta_pri, n);
  for (j in 1:N){
    y0[j] ~ bernoulli_logit(mu0 + alpha0[kk[j]] + beta0[jj[j]] - 
            gam0*((square(theta0[jj[j]]-phi0[kk[j]]))^(lambda)));
  }
}
'