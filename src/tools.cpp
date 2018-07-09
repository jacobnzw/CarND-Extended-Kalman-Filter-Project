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
     * Calculates Jacobian evaluated at point given by x_state.
     */
    MatrixXd jacobian;
    double p_x = x_state[0];
    double p_y = x_state[1];
    double v_x = x_state[2];
    double v_y = x_state[3];
    double norm2 = pow(p_x, 2) + pow(p_y, 2);
    jacobian << p_x/(sqrt(norm2)), p_y/(sqrt(norm2)), 0, 0,
                -p_y/norm2, p_x/norm2, 0, 0,
                p_y * (v_x*p_y - v_y*p_x)/pow(norm2, 1.5), p_x * (v_y*p_x - v_x*p_y)/pow(norm2, 1.5),
                p_x/sqrt(norm2), p_y/sqrt(norm2);
    return jacobian;
}
