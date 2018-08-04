//
// Created by jakub on 7/14/18.
//

#include "mtran.h"
#include <iostream>
using namespace std;

MomentTransform::MomentTransform() {}
MomentTransform::MomentTransform(unsigned int dim_in, unsigned int dim_out) {
  dim_in_ = dim_in;
  dim_out_ = dim_out;
}
MomentTransform::~MomentTransform() {}
// Moments MomentTransform::apply(std::function<VectorXd(const VectorXd &, float)> f,
//                                std::function<MatrixXd(const VectorXd &, float)> f_grad, const VectorXd &in_mean,
//                                const MatrixXd &in_cov, double dt) {
//     Moments out;
//     return out;
// }

/* 
  Linearization moment transform based on first-order Taylor expansion used in EKF.
*/
LinearizationTransform::LinearizationTransform(unsigned int dim_in, unsigned int dim_out) : MomentTransform(dim_in, dim_out) {}

LinearizationTransform::~LinearizationTransform() {}

Moments LinearizationTransform::apply(std::function<VectorXd(const VectorXd&, float)> f,
                                      std::function<MatrixXd(const VectorXd&, float)> f_grad,
                                      const VectorXd &in_mean, const MatrixXd &in_cov, double dt)
{
    VectorXd fm = f(in_mean, dt);
    MatrixXd Fm = f_grad(in_mean, dt);

    Moments out;
    out.mean = fm;
    out.cov = Fm*in_cov*Fm.transpose();
    out.ccov = in_cov*Fm.transpose();

    return out;
}

SigmaPointMomentTransform::SigmaPointMomentTransform(unsigned int dim_in, unsigned int dim_out) : MomentTransform(dim_in, dim_out) {}
SigmaPointMomentTransform::~SigmaPointMomentTransform() {}
void SigmaPointMomentTransform::set_weights() {}
void SigmaPointMomentTransform::set_sigma_points() {}

UnscentedTransform::UnscentedTransform(unsigned int dim_in, unsigned int dim_out, float kappa, float alpha, float beta) :
SigmaPointMomentTransform(dim_in, dim_out)
{
    num_points_ = 2*dim_in + 1;

    kappa_ = kappa;
    alpha_ = alpha;
    beta_ = beta;
    
    fcn_val_ = MatrixXd::Zero(dim_out_, num_points_);

    set_sigma_points();
    set_weights();
}

UnscentedTransform::~UnscentedTransform() {}

void UnscentedTransform::set_sigma_points() {
    points_ = MatrixXd::Zero(dim_in_, num_points_);
    double lambda = pow(alpha_, 2)*(dim_in_ + kappa_) - dim_in_;
    double c = sqrt(dim_in_ + lambda);
    MatrixXd I = MatrixXd::Identity(dim_in_, dim_in_);
    points_.col(0) = VectorXd::Zero(dim_in_);
    points_.block(0, 1, dim_in_, dim_in_) = c*I;
    points_.block(0, dim_in_+1, dim_in_, dim_in_) = -c*I;
}

void UnscentedTransform::set_weights() {
    double lambda = pow(alpha_, 2)*(dim_in_ + kappa_) - dim_in_;
    weights_mean_ = VectorXd::Ones(num_points_) / (2*(dim_in_ + lambda));
    weights_mean_[0] = lambda / (dim_in_ + lambda);

    weights_cov_ = VectorXd::Ones(num_points_) / (2*(dim_in_ + lambda));
    weights_cov_[0] = weights_mean_[0] + (1 - pow(alpha_, 2) + beta_);
}

Moments UnscentedTransform::apply(std::function<VectorXd(const VectorXd&, float)> f,
                                  std::function<MatrixXd(const VectorXd&, float)> f_grad,
                                  const VectorXd &in_mean, const MatrixXd &in_cov, double dt)
{
    // make sigma-points
    MatrixXd L = in_cov.llt().matrixL();
    MatrixXd x = in_mean.rowwise().replicate(num_points_) + L*points_;

    // function evaluations
    for (int i = 0; i < num_points_; ++i) {
        fcn_val_.col(i) = f(x.col(i), dt);
    }

    // output moments
    Moments out = {VectorXd(dim_out_), MatrixXd(dim_out_, dim_out_), MatrixXd(dim_in_, dim_out_)};
    // transformed mean
    out.mean = fcn_val_*weights_mean_;
    // cout << "fcn_val_.cols(): " << fcn_val_.cols() << endl;
    // cout << "fcn_val_.rows(): " << fcn_val_.rows() << endl;
    // cout << "out.mean.cols(): " << out.mean.rowwise().replicate(num_points_).cols() << endl;
    // cout << "out.mean.rows(): " << out.mean.rowwise().replicate(num_points_).rows() << endl;

    // transformed covariance and input-output covariance
    MatrixXd df = fcn_val_ - out.mean.rowwise().replicate(num_points_);
    MatrixXd dx = points_ - in_mean.rowwise().replicate(num_points_);
    for (int i = 0; i < num_points_; ++i) {
        out.cov += weights_cov_(i) * (df.col(i) * df.col(i).transpose());
        out.ccov += weights_mean_(i) * (dx.col(i) * df.col(i).transpose());
    }

    return out;
}