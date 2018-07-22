//
// Created by jakub on 7/14/18.
//

#include "mtran.h"

LinearizationTransform::LinearizationTransform(int dim_in, int dim_out) {
    dim_in_ = dim_in;
    dim_out_ = dim_out;
}

LinearizationTransform::~LinearizationTransform() {}

Moments LinearizationTransform::apply(VectorXd (*f)(VectorXd, float), MatrixXd (*f_grad)(VectorXd, float),
                                      const VectorXd &in_mean, const MatrixXd &in_cov, float dt) {
    VectorXd fm = f(in_mean, dt);
    MatrixXd Fm = f_grad(in_mean, dt);

    Moments out;
    out.mean = fm;
    out.cov = Fm*in_cov*Fm.transpose();
    out.ccov = in_cov*Fm.transpose();

    return out;
}


UnscentedTransform::UnscentedTransform(int dim_in, int dim_out, float kappa, float alpha=1.0F, float beta=2.0F) {
    dim_in_ = dim_in;
    dim_out_ = dim_out;
    num_points_ = 2*dim_in + 1;

    kappa_ = kappa;
    alpha_ = alpha;
    beta_ = beta;

    fcn_val_ = MatrixXd::Zero(dim_out_, num_points_);

    set_sigma_points();
    set_weights();
}

UnscentedTransform::~UnscentedTransform() {}

MatrixXd UnscentedTransform::set_sigma_points() {
    points_ = MatrixXd::Zero(dim_in_, num_points_);
    double lambda = pow(alpha_, 2)*(dim_in_ + kappa_) - dim_in_;
    double c = sqrt(dim_in_ + lambda);
    for (int i = 1; i <= dim_in_; ++i) {
        points_[i, i] = c;
        points_[i, 2*i] = -c;
    }
}

VectorXd UnscentedTransform::set_weights() {
    double lambda = pow(alpha_, 2)*(dim_in_ + kappa_) - dim_in_;
    weights_mean_ = VectorXd::Ones(num_points_) / (2*(dim_in_ + lambda));
    weights_mean_[0] = lambda / (dim_in_ + lambda);

    weights_cov_ = VectorXd::Ones(num_points_) / (2*(dim_in_ + lambda));
    weights_cov_[0] = weights_mean_[0] + (1 - pow(alpha_, 2) + beta_);
}

Moments UnscentedTransform::apply(VectorXd (*f)(VectorXd, float), MatrixXd (*f_grad)(VectorXd, float),
                                  const VectorXd &in_mean, const MatrixXd &in_cov, float dt) {
    // make sigma-points
    MatrixXd L = in_cov.llt().matrixL();
    MatrixXd x = in_mean.rowwise().replicate(num_points_) + L*points_;

    // function evaluations
    for (int i = 0; i < num_points_; ++i) {
        fcn_val_.col(i) = f(x.col(i), dt);
    }

    // output moments
    Moments out;
    // transformed mean
    out.mean = fcn_val_*weights_mean_;
    // transformed covariance and input-output covariance
    VectorXd df = fcn_val_ - out.mean.rowwise().replicate(num_points_);
    VectorXd dx = points_ - in_mean.rowwise().replicate(num_points_);
    for (int i = 0; i < num_points_; ++i) {
        out.cov += weights_cov_(i) * (df.col(i) * df.col(i).transpose());
        out.ccov += weights_mean_(i) * (dx.col(i) * df.col(i).transpose());
    }

    return out;
}