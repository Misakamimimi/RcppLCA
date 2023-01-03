#include "em_algorithm_regress.h"

RcppLCA::EmAlgorithmRegress::EmAlgorithmRegress(
  double* features, int* responses, double* initial_prob, int n_data,
  int n_feature, int n_category, int* n_outcomes, int sum_outcomes,
  int n_cluster, int max_iter, double tolerance, double* posterior,
  double* prior, double* estimated_prob, double* regress_coeff)
  : RcppLCA::EmAlgorithm(
      features, responses, initial_prob, n_data, n_feature, n_category,
      n_outcomes, sum_outcomes, n_cluster, max_iter, tolerance, posterior,
      prior, estimated_prob, regress_coeff) {
    this->n_parameters_ = n_feature * (n_cluster - 1);
    this->gradient_ = new double[this->n_parameters_];
    this->hessian_ = new double[this->n_parameters_ * this->n_parameters_];
    this->init_regress_coeff();
  }

RcppLCA::EmAlgorithmRegress::~EmAlgorithmRegress() {
  delete[] this->gradient_;
  delete[] this->hessian_;
}

void RcppLCA::EmAlgorithmRegress::Reset(
    std::uniform_real_distribution<double>* uniform) {
  this->RcppLCA::EmAlgorithm::Reset(uniform);
  this->init_regress_coeff();
}

void RcppLCA::EmAlgorithmRegress::InitPrior() {
  arma::Mat<double> features(this->features_, this->n_data_, this->n_feature_,
                             false);
  arma::Mat<double> regress_coeff(this->regress_coeff_, this->n_feature_,
                                  this->n_cluster_ - 1, false);
  arma::Mat<double> prior = features * regress_coeff;
  prior = exp(prior);
  
  for (int i = 0; i < this->n_data_; ++i) {
    this->prior_[i] = 1.0;
  }
  memcpy(this->prior_ + this->n_data_, prior.begin(),
         prior.size() * sizeof(*this->prior_));

  arma::Mat<double> prior_arma(this->prior_, this->n_data_, this->n_cluster_,
                               false);
  arma::Col<double> normaliser = sum(prior_arma, 1);
  double* prior_ptr = this->prior_;
  for (int m = 0; m < this->n_cluster_; ++m) {
    for (int i = 0; i < this->n_data_; ++i) {
      *prior_ptr /= normaliser[i];
      ++prior_ptr;
    }
  }
}

void RcppLCA::EmAlgorithmRegress::FinalPrior() {
  
}

double RcppLCA::EmAlgorithmRegress::GetPrior(int data_index,
                                                    int cluster_index) {
  return this->prior_[this->n_data_ * cluster_index + data_index];
}

bool RcppLCA::EmAlgorithmRegress::IsInvalidLikelihood(
    double ln_l_difference) {
  if (this->RcppLCA::EmAlgorithm::IsInvalidLikelihood(ln_l_difference)) {
    return true;
  } else {
    return ln_l_difference < -1e-7;
  }
}

bool RcppLCA::EmAlgorithmRegress::MStep() {
  this->EstimateProbability();
  
  this->CalcGrad();
  this->CalcHess();
  
  arma::Col<double> regress_coeff(this->regress_coeff_, this->n_parameters_,
                                  false);
  arma::Col<double> gradient(this->gradient_, this->n_parameters_, false);
  arma::Mat<double> hessian(this->hessian_, this->n_parameters_,
                            this->n_parameters_, false);
  try {
    regress_coeff -=
      arma::solve(hessian, gradient, arma::solve_opts::likely_sympd);
  } catch (const std::runtime_error) {
    return true;
  }
  
  this->InitPrior();
  
  return false;
}

void RcppLCA::EmAlgorithmRegress::NormalWeightedSumProb(
    int cluster_index) {
  arma::Col<double> posterior(this->posterior_ + cluster_index * this->n_data_,
                              this->n_data_, false);
  double normaliser = arma::sum(posterior);
  this->RcppLCA::EmAlgorithm::NormalWeightedSumProb(cluster_index,
                                                           normaliser);
}

void RcppLCA::EmAlgorithmRegress::init_regress_coeff() {
  for (int i = 0; i < this->n_parameters_; ++i) {
    this->regress_coeff_[i] = 0.0;
  }
}

void RcppLCA::EmAlgorithmRegress::CalcGrad() {
  double* gradient = this->gradient_;
  for (int m = 1; m < this->n_cluster_; ++m) {
    arma::Col<double> posterior_m(this->posterior_ + m * this->n_data_,
                                  this->n_data_, false);
    arma::Col<double> prior_m(this->prior_ + m * this->n_data_, this->n_data_,
                              false);
    arma::Col<double> post_minus_prior = posterior_m - prior_m;
    for (int p = 0; p < this->n_feature_; ++p) {
      arma::Col<double> x_p(this->features_ + p * this->n_data_, this->n_data_,
                            false);
      *gradient = arma::dot(x_p, post_minus_prior);
      ++gradient;
    }
  }
}

void RcppLCA::EmAlgorithmRegress::CalcHess() {
  for (int cluster_j = 0; cluster_j < this->n_cluster_ - 1; ++cluster_j) {
    for (int cluster_i = cluster_j; cluster_i < this->n_cluster_ - 1;
    ++cluster_i) {
      this->CalcHessSubBlock(cluster_i, cluster_j);
    }
  }
}

void RcppLCA::EmAlgorithmRegress::CalcHessSubBlock(int cluster_index_0,
                                                          int cluster_index_1) {
  arma::Col<double> posterior0(
      this->posterior_ + (cluster_index_0 + 1) * this->n_data_, this->n_data_,
      false);
  arma::Col<double> prior0(this->prior_ + (cluster_index_0 + 1) * this->n_data_,
                           this->n_data_, false);
  
  bool is_same_cluster = cluster_index_0 == cluster_index_1;
  arma::Col<double> posterior1(
      this->posterior_ + (cluster_index_1 + 1) * this->n_data_, this->n_data_,
      is_same_cluster);
  arma::Col<double> prior1(this->prior_ + (cluster_index_1 + 1) * this->n_data_,
                           this->n_data_, is_same_cluster);
  
  if (is_same_cluster) {
    posterior1 -= 1;
    prior1 -= 1;
  }
  arma::Col<double> prior_post_inter =
    prior0 % prior1 - posterior0 % posterior1;
  
  double hess_element;
  
  for (int j = 0; j < this->n_feature_; ++j) {
    for (int i = j; i < this->n_feature_; ++i) {
      hess_element = CalcHessElement(i, j, &prior_post_inter);
      *this->HessianAt(cluster_index_0, cluster_index_1, i, j) = hess_element;
      
      if (i != j) {
        *this->HessianAt(cluster_index_0, cluster_index_1, j, i) = hess_element;
      }
      
      if (cluster_index_0 != cluster_index_1) {
        *this->HessianAt(cluster_index_1, cluster_index_0, i, j) = hess_element;
        if (i != j) {
          *this->HessianAt(cluster_index_1, cluster_index_0, j, i) =
            hess_element;
        }
      }
    }
  }
}

double RcppLCA::EmAlgorithmRegress::CalcHessElement(
    int feature_index_0, int feature_index_1,
    arma::Col<double>* prior_post_inter) {
  arma::Col<double> feature0(this->features_ + feature_index_0 * this->n_data_,
                             this->n_data_, false);
  arma::Col<double> feature1(this->features_ + feature_index_1 * this->n_data_,
                             this->n_data_, false);
  return arma::sum(feature0 % feature1 % *prior_post_inter);
}

double* RcppLCA::EmAlgorithmRegress::HessianAt(int cluster_index_0,
                                                      int cluster_index_1,
                                                      int feature_index_0,
                                                      int feature_index_1) {
  return this->hessian_ +
    cluster_index_1 * this->n_parameters_ * this->n_feature_ +
    feature_index_1 * this->n_parameters_ +
    cluster_index_0 * this->n_feature_ + feature_index_0;
}