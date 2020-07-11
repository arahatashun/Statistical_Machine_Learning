data {
  int <lower=1> K; // rank
  int <lower=1> O; //n observations
  int <lower=1> J;
  int <lower=1> I;
  int <lower=0> users[O];
  int <lower=0> items[O];
  int <lower=0> scores[O];
}
parameters {
  real<lower=0> sigma;
  vector<lower=0>[K] rhou;
  vector<lower=0>[K] rhov;
  matrix[J, K] u;
  matrix[I, K] v;
}
model {
  for(j in 1:J){
    for (k in 1:K){
      u[j,k]  ~ normal(0.0, rhou[k]);
    }
  }
  for(i in 1:I){
    for (k in 1:K){
      v[i,k]  ~ normal(0.0, rhov[k]);
    }
  }
  for(o in 1:O){
      scores[o] ~  normal(dot_product(u[users[o]], v[items[o]]), sigma);
  }
}