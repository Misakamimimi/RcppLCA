#ifndef EM_ALGORITHM_ARRAY_H_
#define EM_ALGORITHM_ARRAY_H_

#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "em_algorithm.h"
#include "em_algorithm_regress.h"

namespace RcppLCA {

class EmAlgorithmArray {
private:
  double* features_;
  int* responses_;
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
  bool is_regress_;
  double* best_initial_prob_ = NULL;
  
  int n_rep_;
  double optimal_ln_l_;
  int n_iter_;
  bool has_restarted_ = false;
  double* initial_prob_;
  int n_rep_done_;
  double* ln_l_array_ = NULL;
  int best_rep_index_;
  int n_thread_;
  
  std::mutex* n_rep_done_lock_;
  std::mutex* results_lock_;
  
protected:
  std::unique_ptr<unsigned[]> seed_array_ = NULL;
  
public:
  EmAlgorithmArray(double* features, int* responses, double* initial_prob,
                   int n_data, int n_feature, int n_category, int* n_outcomes,
                   int sum_outcomes, int n_cluster, int n_rep, int n_thread,
                   int max_iter, double tolerance, double* posterior,
                   double* prior, double* estimated_prob, double* regress_coeff,
                   bool is_regress);
  
  ~EmAlgorithmArray();
  
  void Fit();
  
  virtual void SetSeed(std::seed_seq* seed);
  
  void set_best_initial_prob(double* best_initial_prob);
  
  void set_ln_l_array(double* ln_l_array);
  
  int get_best_rep_index();
  
  double get_optimal_ln_l();
  
  int get_n_iter();
  
  bool get_has_restarted();
  
protected:
  virtual void SetFitterRng(RcppLCA::EmAlgorithm* fitter, int rep_index);
  
  virtual void MoveRngBackFromFitter(RcppLCA::EmAlgorithm* fitter);
  
private:
  void FitThread();
};

}  // namespace RcppLCA

#endif  // EM_ALGORITHM_ARRAY_H_