# RcppLCA
## Latent Class Analysis using RcppArmadillo
该R包为对poLCA
\[[CRAN](https://cran.r-project.org/web/packages/poLCA/index.html),
[GitHub](https://github.com/dlinzer/poLCA)\]算法进行的RcppArmadillo加速。以poLCA算法为基础，用相似的方式进行结果的计算，但在利用RcppArmadillo后计算效率有了大幅提升，同时增加多线程计算，使得计算更加快速。
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
* [OpenBLAS](
## 利用R和devtools::install_github()进行安装
``
devtools::install_github("Misakamimimi/RcppLCA")
``
