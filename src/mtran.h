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
    // arguments: input mean and covariance, function handle, function parameters
    virtual Moments apply(const VectorXd &in_mean, const MatrixXd &in_cov);
};


class SigmaPointMomentTransform : public MomentTransform {
public:
    /**
     * Sigma-points of the moment transform.
     */
    MatrixXd points;

    /**
     * Transform weights.
     */
    VectorXd weights;

    virtual MatrixXd set_sigma_points();

    virtual VectorXd set_weights();
};


class UnscentedTransform : public SigmaPointMomentTransform {
public:
    UnscentedTransform();
    virtual ~UnscentedTransform();

    /**
     * Set UT sigma-points, which can later be accessed via public member
     */
    virtual MatrixXd set_sigma_points(float kappa, float alpha, float beta);

    /**
     * Set UT weights, which can later be accessed via public member
     */
    virtual VectorXd set_weights(float kappa, float alpha, float beta);
};


#endif //EXTENDEDKF_MTRAN_H
