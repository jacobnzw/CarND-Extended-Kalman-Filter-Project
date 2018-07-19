//
// Created by jakub on 7/14/18.
//

#ifndef EXTENDEDKF_MTRAN_H
#define EXTENDEDKF_MTRAN_H
#include "Eigen/Dense"

using Eigen::VectorXd
using Eigen::MatrixXd

// struct for bundling output of any resulting moments of any moment transform
struct Moments {
    VectorXd mean;
    MatrixXd cov;
    MatrixXd ccov;
};


class MomentTransform {
public:
    int dim_;

    // arguments: input mean and covariance, function handle, function parameters
    virtual Moments apply(VectorXd (*f)(VectorXd), const VectorXd &in_mean, const MatrixXd &in_cov) = 0;
};


class LinearizationTransform : public MomentTransform {
    /**
     * Linearization based on first-order Taylor expansion.
     */

public:
    LinearizationTransform();
    virtual ~LinearizationTransform();
};


class SigmaPointMomentTransform : public MomentTransform {
public:
    /**
     * Sigma-points of the moment transform.
     */
    MatrixXd points_;

    /**
     * Transform weights.
     */
    VectorXd weights_mean_;
    VectorXd weights_cov_;

private:
    virtual MatrixXd set_sigma_points() = 0;

    virtual VectorXd set_weights() = 0;
};


class UnscentedTransform : public SigmaPointMomentTransform {
public:
    float kappa_;
    float alpha_;
    float beta_;

    UnscentedTransform(int dim, float kappa, float alpha, float beta);
    virtual ~UnscentedTransform();

private:
    /**
     * Set UT sigma-points, which can later be accessed via public member
     */
    virtual MatrixXd set_sigma_points();

    /**
     * Set UT weights, which can later be accessed via public member
     */
    virtual VectorXd set_weights();
};


#endif //EXTENDEDKF_MTRAN_H
