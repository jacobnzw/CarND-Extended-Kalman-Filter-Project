//
// Created by jakub on 7/14/18.
//

#ifndef EXTENDEDKF_MTRAN_H
#define EXTENDEDKF_MTRAN_H
#include "Eigen/Dense"
#include <functional>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// struct for bundling output of any resulting moments of any moment transform
struct Moments {
    VectorXd mean;
    MatrixXd cov;
    MatrixXd ccov;
};


class MomentTransform {
public:
    unsigned int dim_in_;
    unsigned int dim_out_;

    MomentTransform();
    MomentTransform(unsigned int dim_in, unsigned int dim_out);
    virtual ~MomentTransform();

    virtual Moments apply(std::function<VectorXd(const VectorXd&, float)> f,
                          std::function<MatrixXd(const VectorXd&, float)> f_grad,
                          const VectorXd &in_mean, const MatrixXd &in_cov, double dt);
};


class LinearizationTransform : public MomentTransform {
    /**
     * Linearization based on first-order Taylor expansion.
     */

public:
    LinearizationTransform(unsigned int dim_in, unsigned int dim_out);
    virtual ~LinearizationTransform();
    Moments apply(std::function<VectorXd(const VectorXd&, float)> f,
                  std::function<MatrixXd(const VectorXd&, float)> f_grad,
                  const VectorXd &in_mean, const MatrixXd &in_cov, double dt);
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

    unsigned int num_points_;

    SigmaPointMomentTransform(unsigned int dim_in, unsigned int dim_out);
    virtual ~SigmaPointMomentTransform();
    
private:
    virtual void set_sigma_points();
    virtual void set_weights();
};


class UnscentedTransform : public SigmaPointMomentTransform {
public:
    float kappa_;
    float alpha_;
    float beta_;

    UnscentedTransform(unsigned int dim_in, unsigned int dim_out, float kappa, float alpha=1.0F, float beta=2.0F);
    virtual ~UnscentedTransform();
    Moments apply(std::function<VectorXd(const VectorXd&, float)> f,
                  std::function<MatrixXd(const VectorXd&, float)> f_grad,
                  const VectorXd &in_mean, const MatrixXd &in_cov, double dt);

private:
    /**
     * Set UT sigma-points, which can later be accessed via public member
     */
    void set_sigma_points();

    /**
     * Set UT weights, which can later be accessed via public member
     */
    void set_weights();
};


#endif //EXTENDEDKF_MTRAN_H
