#include "em_algorithm.h"

RcppLCA::EmAlgorithm::EmAlgorithm(
  double* features, int* responses, double* initial_prob, int n_data,
  int n_feature, int n_category, int* n_outcomes, int sum_outcomes,
  int n_cluster, int max_iter, double tolerance, double* posterior,
  double* prior, double* estimated_prob, double* regress_coeff) {
  this->features_ = features;
  this->responses_ = responses;
  this->initial_prob_ = initial_prob;
  this->n_data_ = n_data;
  this->n_feature_ = n_feature;
  this->n_category_ = n_category;
  this->n_outcomes_ = n_outcomes;
  this->sum_outcomes_ = sum_outcomes;
  this->n_cluster_ = n_cluster;
  this->max_iter_ = max_iter;
  this->tolerance_ = tolerance;
  this->posterior_ = posterior;
  this->prior_ = prior;
  this->estimated_prob_ = estimated_prob;
  this->regress_coeff_ = regress_coeff;
  this->ln_l_array_ = new double[this->n_data_];
  
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937_64* rng = new std::mt19937_64(seed);
  this->rng_ = std::unique_ptr<std::mt19937_64>(rng);
}

RcppLCA::EmAlgorithm::~EmAlgorithm() { delete[] this->ln_l_array_; }

void RcppLCA::EmAlgorithm::Fit() {
  bool is_first_run = true;
  bool is_success = false;
  
  double ln_l_difference;
  double ln_l_before;
  
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  
  while (!is_success) {
    if (is_first_run) {
      std::memcpy(this->estimated_prob_, this->initial_prob_,
                  this->n_cluster_ * this->sum_outcomes_ *
                    sizeof(*this->estimated_prob_));
    } else {
      this->Reset(&uniform);
    }
    
    if (this->best_initial_prob_ != NULL) {
      std::memcpy(this->best_initial_prob_, this->estimated_prob_,
                  this->n_cluster_ * this->sum_outcomes_ *
                    sizeof(*this->best_initial_prob_));
    }
    
    ln_l_before = -INFINITY;
    
    this->InitPrior();
    
    is_success = true;
    for (this->n_iter_ = 0; this->n_iter_ <= this->max_iter_; ++this->n_iter_) {
      this->EStep();
      
      arma::Col<double> ln_l_array(this->ln_l_array_, this->n_data_, false);
      this->ln_l_ = sum(ln_l_array);
      
      ln_l_difference = this->ln_l_ - ln_l_before;
      if (this->IsInvalidLikelihood(ln_l_difference)) {
        is_success = false;
        break;
      }
      
      if (ln_l_difference < this->tolerance_) {
        break;
      }
      if (this->n_iter_ == this->max_iter_) {
        break;
      }
      ln_l_before = this->ln_l_;
      
      if (this->MStep()) {
        is_success = false;
        break;
      }
    }
    is_first_run = false;
  }
  
  this->FinalPrior();
}

void RcppLCA::EmAlgorithm::set_best_initial_prob(
    double* best_initial_prob) {
  this->best_initial_prob_ = best_initial_prob;
}

double RcppLCA::EmAlgorithm::get_ln_l() { return this->ln_l_; }

int RcppLCA::EmAlgorithm::get_n_iter() { return this->n_iter_; }

bool RcppLCA::EmAlgorithm::get_has_restarted() {
  return this->has_restarted_;
}

void RcppLCA::EmAlgorithm::set_seed(unsigned seed) {
  std::mt19937_64* rng = new std::mt19937_64(seed);
  this->rng_ = std::unique_ptr<std::mt19937_64>(rng);
}

void RcppLCA::EmAlgorithm::set_rng(
    std::unique_ptr<std::mt19937_64>* rng) {
  this->rng_ = std::move(*rng);
}

std::unique_ptr<std::mt19937_64> RcppLCA::EmAlgorithm::move_rng() {
  return std::move(this->rng_);
}

void RcppLCA::EmAlgorithm::Reset(
    std::uniform_real_distribution<double>* uniform) {
  this->has_restarted_ = true;
  RcppLCA::GenerateNewProb(this->rng_.get(), uniform, this->n_outcomes_,
                                  this->sum_outcomes_, this->n_category_,
                                  this->n_cluster_, this->estimated_prob_);
}

void RcppLCA::EmAlgorithm::InitPrior() {
  for (int i = 0; i < this->n_cluster_; ++i) {
    this->prior_[i] = 1.0 / static_cast<double>(this->n_cluster_);
  }
}

void RcppLCA::EmAlgorithm::FinalPrior() {
  double prior_copy[this->n_cluster_];
  std::memcpy(&prior_copy, this->prior_,
              this->n_cluster_ * sizeof(*this->prior_));
  for (int m = 0; m < this->n_cluster_; ++m) {
    for (int i = 0; i < this->n_data_; ++i) {
      this->prior_[m * this->n_data_ + i] = prior_copy[m];
    }
  }
}

double RcppLCA::EmAlgorithm::GetPrior(int data_index,
                                             int cluster_index) {
  return this->prior_[cluster_index];
}

void RcppLCA::EmAlgorithm::EStep() {
  double* estimated_prob;  
  int n_outcome;  
  double p;
  double normaliser;
  double posterior_iter;
  int y;  
  
  for (int i = 0; i < this->n_data_; ++i) {
    normaliser = 0.0;
    
    estimated_prob = this->estimated_prob_;
    for (int m = 0; m < this->n_cluster_; ++m) {
      p = 1.0;
      for (int j = 0; j < this->n_category_; ++j) {
        n_outcome = this->n_outcomes_[j];
        y = this->responses_[i * this->n_category_ + j];
        p *= estimated_prob[y - 1];
        estimated_prob += n_outcome;
      }
      posterior_iter = p * this->GetPrior(i, m);
      this->posterior_[m * this->n_data_ + i] = posterior_iter;
      normaliser += posterior_iter;
    }
    
    for (int m = 0; m < this->n_cluster_; ++m) {
      this->posterior_[m * this->n_data_ + i] /= normaliser;
    }
   
    this->ln_l_array_[i] = log(normaliser);
  }
}

bool RcppLCA::EmAlgorithm::IsInvalidLikelihood(double ln_l_difference) {
  return isnan(this->ln_l_);
}

bool RcppLCA::EmAlgorithm::MStep() {
  
  arma::Mat<double> posterior_arma(this->posterior_, this->n_data_,
                                   this->n_cluster_, false);
  arma::Row<double> prior = mean(posterior_arma, 0);
  std::memcpy(this->prior_, prior.begin(),
              this->n_cluster_ * sizeof(*this->prior_));
  
  
  this->EstimateProbability();
  
  return false;
}

void RcppLCA::EmAlgorithm::EstimateProbability() {
  
  for (int i = 0; i < this->n_cluster_ * this->sum_outcomes_; ++i) {
    this->estimated_prob_[i] = 0.0;
  }
  
  
  for (int m = 0; m < this->n_cluster_; ++m) {
    this->WeightedSumProb(m);
    this->NormalWeightedSumProb(m);
  }
}

void RcppLCA::EmAlgorithm::WeightedSumProb(int cluster_index) {
  int n_outcome;
  int y;
  double posterior_iter;
  double* estimated_prob_m =
    this->estimated_prob_ + cluster_index * this->sum_outcomes_;
  double* estimated_prob;
  for (int i = 0; i < this->n_data_; ++i) {
    estimated_prob = estimated_prob_m;
    for (int j = 0; j < this->n_category_; ++j) {
      n_outcome = this->n_outcomes_[j];
      y = this->responses_[i * this->n_category_ + j];
      posterior_iter = this->posterior_[cluster_index * this->n_data_ + i];
      estimated_prob[y - 1] += posterior_iter;
      estimated_prob += n_outcome;
    }
  }
}

void RcppLCA::EmAlgorithm::NormalWeightedSumProb(int cluster_index) {
  this->NormalWeightedSumProb(
      cluster_index,
      static_cast<double>(this->n_data_) * this->prior_[cluster_index]);
}

void RcppLCA::EmAlgorithm::NormalWeightedSumProb(int cluster_index,
                                                        double normaliser) {
  int n_outcome;
  double* estimated_prob =
    this->estimated_prob_ + cluster_index * this->sum_outcomes_;
  for (int j = 0; j < this->n_category_; ++j) {
    n_outcome = this->n_outcomes_[j];
    for (int k = 0; k < n_outcome; ++k) {
      estimated_prob[k] /= normaliser;
    }
    estimated_prob += n_outcome;
  }
}

void RcppLCA::GenerateNewProb(
    std::mt19937_64* rng, std::uniform_real_distribution<double>* uniform,
    int* n_outcomes, int sum_outcomes, int n_category, int n_cluster,
    double* prob) {
  for (double* ptr = prob; ptr < prob + n_cluster * sum_outcomes; ++ptr) {
    *ptr = (*uniform)(*rng);
  }
  int n_outcome;
  for (int m = 0; m < n_cluster; ++m) {
    for (int j = 0; j < n_category; ++j) {
      n_outcome = n_outcomes[j];
      arma::Col<double> prob_vector(prob, n_outcome, false);
      prob_vector /= sum(prob_vector);
      prob += n_outcome;
    }
  }
}