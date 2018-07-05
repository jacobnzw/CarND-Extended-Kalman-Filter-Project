#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
     * Calculates RMSE of the estimated trajectory.
     */
    auto num_steps = estimations[0].size();
    VectorXd sum, diff;
    sum.setZero(num_steps);
    for (unsigned int i = 0; i < estimations.size(); i++) {
        diff = estimations[i] - ground_truth[i];
        sum += diff.cwiseProduct(diff);
    }
    return (sum / num_steps).cwiseSqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    /**
     * TODO: Calculates Jacobian evaluated at point given by x_state
     */
    MatrixXd a;
    return a;
}
