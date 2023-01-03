# RcppLCA
## Latent Class Analysis using RcppArmadillo
该R包为对LCA算法进行的RcppArmadillo加速，在R中有\[[CRAN](https://cran.r-project.org/web/packages/poLCA/index.html),
[GitHub](https://github.com/dlinzer/poLCA)\]等包的实现。以LCA算法为基础，用相似的方式进行结果的计算，但在利用RcppArmadillo后计算效率有了大幅提升，同时增加多线程计算，使得计算更加快速。
## RcppLCA的环境需求
如果推荐的安装说明失败或有其他问题，请检查以下可能的环境是否已安装:
* R packages for installing and compiling:
  * [devtools](https://cran.r-project.org/web/packages/devtools/index.html)
  * [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
  * [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo)
* Dependent R packages:
  * [MASS](https://cran.r-project.org/web/packages/MASS/index.html)
  * [parallel](https://www.rdocumentation.org/packages/parallel/)
* A C++ compiler like [gcc](https://gcc.gnu.org/)
* [Armadillo](http://arma.sourceforge.net/)
* [LAPACK](http://www.netlib.org/lapack/)
* [OpenBLAS](https://www.openblas.net/)
## 利用R和devtools::install_github()进行安装
```
devtools::install_github("Misakamimimi/RcppLCA")
```
## 参考文献
* Dziak, J. J., Lanza, S. T., & Tan, X. (2014). Effect size, statistical power,
  and sample size requirements for the bootstrap likelihood ratio test in latent
  class analysis. *Structural Equation Modeling: A Multidisciplinary Journal*,
  21(4), 534-552.
  [[link]](https://www.tandfonline.com/doi/full/10.1080/10705511.2014.919819?casa_token=LgaSzKeeB8MAAAAA%3AB80XwZEIkLOIVsD4Gvp6O0gfktOnIqA6dOBBvUZIjjhs-7ilLIZJC_TmxCh8Umh45d0sWez4-em9)
* Linzer, D.A. & Lewis, J. (2013). poLCA: Polytomous Variable Latent
  Class Analysis. R package version 1.4.
  [[link]](https://github.com/dlinzer/poLCA)
* Linzer, D.A. & Lewis, J.B. (2011). poLCA: An R package for polytomous
  variable latent class analysis. *Journal of Statistical Software*,
  42(10): 1-29.
  [[link]](http://www.jstatsoft.org/v42/i10)
