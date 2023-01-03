#ifndef EM_ALGORITHM_REGRESS_H_
#define EM_ALGORITHM_REGRESS_H_

#define ARMA_WARN_LEVEL 1

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
#include "em_algorithm.h"

namespace RcppLCA {


class EmAlgorithmRegress : public RcppLCA::EmAlgorithm {
private:
  int n_parameters_;
  double* gradient_;
  double* hessian_;
  
public:
  EmAlgorithmRegress(double* features, int* responses, double* initial_prob,
                     int n_data, int n_feature, int n_category, int* n_outcomes,
                     int sum_outcomes, int n_cluster, int max_iter,
                     double tolerance, double* posterior, double* prior,
                     double* estimated_prob, double* regress_coeff);
  
  ~EmAlgorithmRegress() override;
  
protected:
  void Reset(std::uniform_real_distribution<double>* uniform) override;
  
  void InitPrior() override;
  
  void FinalPrior() override;
  
  double GetPrior(int data_index, int cluster_index) override;
  
  bool IsInvalidLikelihood(double ln_l_difference) override;
  
  bool MStep() override;
  
  void NormalWeightedSumProb(int cluster_index) override;
  
private:
  void init_regress_coeff();
  
  void CalcGrad();
  
  void CalcHess();
  
  void CalcHessSubBlock(int cluster_index_0, int cluster_index_1);
  
  double CalcHessElement(int feature_index_0, int feature_index_1,
                         arma::Col<double>* prior_post_inter);
  
  double* HessianAt(int cluster_index_0, int cluster_index_1,
                    int feature_index_0, int feature_index_1);
};

}  // namespace RcppLCA

#endif  // EM_ALGORITHM_REGRESS_H_