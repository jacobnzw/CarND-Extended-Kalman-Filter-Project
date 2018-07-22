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
    int dim_in_;
    int dim_out_;

    // arguments: input mean and covariance, function handle, function parameters
    virtual Moments apply(VectorXd (*f)(VectorXd), const VectorXd &in_mean, const MatrixXd &in_cov) = 0;
};


class LinearizationTransform : public MomentTransform {
    /**
     * Linearization based on first-order Taylor expansion.
     */

public:
    LinearizationTransform(int dim_in, int dim_out);
    virtual ~LinearizationTransform();
    Moments apply(VectorXd (*f) (VectorXd), const VectorXd &in_mean, const MatrixXd &in_cov);
};


class SigmaPointMomentTransform : public MomentTransform {
public:
    /**
     * Sigma-points of the moment transform.
     */
    MatrixXd points_;

    /**
     * Function values.
     */
     MatrixXd fcn_val_;

    /**
     * Transform weights.
     */
    VectorXd weights_mean_;
    VectorXd weights_cov_;

    int num_points_;

private:
    virtual MatrixXd set_sigma_points() = 0;
    virtual VectorXd set_weights() = 0;
};


class UnscentedTransform : public SigmaPointMomentTransform {
public:
    float kappa_;
    float alpha_;
    float beta_;

    UnscentedTransform(int dim_in, int dim_out, float kappa, float alpha, float beta);
    virtual ~UnscentedTransform();
    Moments apply(VectorXd (*f) (VectorXd), const VectorXd &in_mean, const MatrixXd &in_cov);

private:
    /**
     * Set UT sigma-points, which can later be accessed via public member
     */
    MatrixXd set_sigma_points();

    /**
     * Set UT weights, which can later be accessed via public member
     */
    VectorXd set_weights();
};


#endif //EXTENDEDKF_MTRAN_H
