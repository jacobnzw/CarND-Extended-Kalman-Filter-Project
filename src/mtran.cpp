//
// Created by jakub on 7/14/18.
//

#include "mtran.h"


UnscentedTransform::UnscentedTransform(int dim, float kappa, float alpha=1.0F, float beta=2.0F) {
    dim_ = dim;

    kappa_ = kappa;
    alpha_ = alpha;
    beta_ = beta;

    set_sigma_points();
    set_weights();
}

UnscentedTransform::~UnscentedTransform() {}

MatrixXd UnscentedTransform::set_sigma_points() {
    points_ = MatrixXd::Zero(dim_, 2*dim_+1);
    double lambda = pow(alpha_, 2)*(dim_ + kappa_) - dim_;
    double c = sqrt(dim_ + lambda);
    for (int i = 1; i <= dim_; ++i) {
        points_[i, i] = c;
        points_[i, 2*i] = -c;
    }
}

VectorXd UnscentedTransform::set_weights() {
    double lambda = pow(alpha_, 2)*(dim_ + kappa_) - dim_;
    weights_mean_ = VectorXd::Ones(2*dim_ + 1) / (2*(dim_ + lambda));
    weights_mean_[0] = lambda / (dim_ + lambda);

    weights_cov_ = VectorXd::Ones(2*dim_ + 1) / (2*(dim_ + lambda));
    weights_cov_[0] = weights_mean_[0] + (1 - pow(alpha_, 2) + beta_);
}

Moments UnscentedTransform::apply(VectorXd (*f)(VectorXd), const VectorXd &in_mean, const MatrixXd &in_cov) {
    Eigen::LLT<MatrixXd> chol(in_cov);
    MatrixXd L = chol.matrixL();
    MatrixXd x = in_mean + L*points_;

    // TODO: calculate mean, cov and ccov
    // TODO: maybe move into sigma-point transform
}