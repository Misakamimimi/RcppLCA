#define ARMA_WARN_LEVEL 1

#include <memory>
#include "em_algorithm_array.h"
#include "RcLCA.c"
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
Rcpp::List EmAlgorithmRcpp(Rcpp::NumericMatrix features,
                           Rcpp::IntegerMatrix responses,
                           Rcpp::NumericVector initial_prob, int n_data,
                           int n_feature, int n_category,
                           Rcpp::IntegerVector n_outcomes, int n_cluster,
                           int n_rep, int n_thread, int max_iter,
                           double tolerance, Rcpp::IntegerVector seed) {
  int sum_outcomes = 0;  
  int* n_outcomes_array = n_outcomes.begin();
  for (int i = 0; i < n_category; ++i) {
    sum_outcomes += n_outcomes_array[i];
  }
  
  Rcpp::NumericMatrix posterior(n_data, n_cluster);
  Rcpp::NumericMatrix prior(n_data, n_cluster);
  Rcpp::NumericVector estimated_prob(sum_outcomes * n_cluster);
  Rcpp::NumericVector regress_coeff(n_feature * (n_cluster - 1));
  Rcpp::NumericVector ln_l_array(n_rep);
  Rcpp::NumericVector best_initial_prob(sum_outcomes * n_cluster);
  
  bool is_regress = n_feature > 1;
  RcppLCA::EmAlgorithmArray* fitter =
    new RcppLCA::EmAlgorithmArray(
        features.begin(), responses.begin(), initial_prob.begin(), n_data,
        n_feature, n_category, n_outcomes.begin(), sum_outcomes, n_cluster,
        n_rep, n_thread, max_iter, tolerance, posterior.begin(),
        prior.begin(), estimated_prob.begin(), regress_coeff.begin(),
        is_regress);
  
  std::seed_seq seed_seq(seed.begin(), seed.end());
  fitter->SetSeed(&seed_seq);
  fitter->set_best_initial_prob(best_initial_prob.begin());
  fitter->set_ln_l_array(ln_l_array.begin());
  
  fitter->Fit();
  
  int best_rep_index = fitter->get_best_rep_index();
  int n_iter = fitter->get_n_iter();
  bool has_restarted = fitter->get_has_restarted();
  
  Rcpp::List to_return;
  to_return.push_back(posterior);
  to_return.push_back(prior);
  to_return.push_back(estimated_prob);
  to_return.push_back(regress_coeff);
  to_return.push_back(ln_l_array);
  to_return.push_back(best_rep_index + 1);
  to_return.push_back(n_iter);
  to_return.push_back(best_initial_prob);
  to_return.push_back(has_restarted);
  return to_return;
}

// [[Rcpp::export]]
Rcpp::NumericVector ylik(Rcpp::NumericVector probs, Rcpp::IntegerVector y,
                         int obs, int items, Rcpp::IntegerVector numChoices,
                         int classes) {
  Rcpp::NumericVector lik(obs * classes);
  ylik(probs.begin(), y.begin(), &obs, &items, numChoices.begin(), &classes,
       lik.begin());
  return lik;
}