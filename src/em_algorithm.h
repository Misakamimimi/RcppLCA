#ifndef EM_ALGORITHM_H_
#define EM_ALGORITHM_H_

#include <math.h>

#include <chrono>
#include <random>

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

namespace RcppLCA {


class EmAlgorithm {
protected:
  double* features_;
  int* responses_;
  double* initial_prob_;
  int n_data_;
  int n_feature_;
  int n_category_;
  int* n_outcomes_;
  int sum_outcomes_;
  int n_cluster_;
  int max_iter_;
  double tolerance_;
  double* posterior_;
  double* prior_;
  double* estimated_prob_;
  double* regress_coeff_;
  double* best_initial_prob_ = NULL;
  
  double ln_l_ = -INFINITY;
  double* ln_l_array_;
  int n_iter_ = 0;
  bool has_restarted_ = false;
  std::unique_ptr<std::mt19937_64> rng_;
  
public:
  EmAlgorithm(double* features, int* responses, double* initial_prob,
              int n_data, int n_feature, int n_category, int* n_outcomes,
              int sum_outcomes, int n_cluster, int max_iter, double tolerance,
              double* posterior, double* prior, double* estimated_prob,
              double* regress_coeff);
  
  virtual ~EmAlgorithm();
  
  void Fit();
  
  void set_best_initial_prob(double* best_initial_prob);
  
  double get_ln_l();
  
  int get_n_iter();
  
  bool get_has_restarted();
  
  void set_seed(unsigned seed);
  
  void set_rng(std::unique_ptr<std::mt19937_64>* rng);
  
  std::unique_ptr<std::mt19937_64> move_rng();
  
protected:
  
  virtual void Reset(std::uniform_real_distribution<double>* uniform);
  
  
  virtual void InitPrior();
  
  
  virtual void FinalPrior();
  
  virtual double GetPrior(int data_index, int cluster_index);
  
  void EStep();
  
  virtual bool IsInvalidLikelihood(double ln_l_difference);
  
  virtual bool MStep();
  
  void EstimateProbability();
  
  void WeightedSumProb(int cluster_index);
  
  virtual void NormalWeightedSumProb(int cluster_index);
  
  void NormalWeightedSumProb(int cluster_index, double normaliser);
};

void GenerateNewProb(std::mt19937_64* rng,
                     std::uniform_real_distribution<double>* uniform,
                     int* n_outcomes, int sum_outcomes, int n_category,
                     int n_cluster, double* prob);

}  // namespace RcppLCA

#endif  // EM_ALGORITHM_H_