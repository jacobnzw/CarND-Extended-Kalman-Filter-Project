#include "kalman_filter.h"
#include "tools.h"
#include <iostream>
#include <cfloat>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
    I_ = MatrixXd::Identity(x_in.size(), x_in.size());
    Tools tools_ = Tools();
}

void KalmanFilter::Predict() {
    /**
     * Predict state.
     */
    x_ = F_*x_;
    P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
     * Kalman filter update.
     */
    MatrixXd Pz = H_*P_*H_.transpose() + R_;
    MatrixXd Pzx = H_*P_;
    MatrixXd K = Pz.ldlt().solve(Pzx).transpose();
    VectorXd e = z - H_*x_;
    x_ = x_ + K*e;
    P_ = (I_ - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
     * Extended Kalman filter update with linearized measurement model.
     */
    VectorXd mz = VectorXd(3U);
    // compute predicted measurement and safeguard against division by zero
    double norm = sqrt(pow(x_[0], 2) + pow(x_[1], 2)) + DBL_EPSILON;
    mz << norm,
          atan2(x_[1], x_[0]),
          (x_[0]*x_[2] + x_[1]*x_[3])/norm;

    VectorXd e = z - mz;
    // keep difference in angles between -pi and pi
    double temp = e[1] / (2*M_PI);
    if (abs(temp) > 1) {
        unsigned int pi_count = floor(temp);
        e[1] -= 2*pi_count*M_PI;
    }
    MatrixXd Pz = H_*P_*H_.transpose() + R_;
    MatrixXd Pzx = H_*P_;
    MatrixXd K = Pz.ldlt().solve(Pzx).transpose();
    x_ = x_ + K*e;
    P_ = (I_ - K*H_)*P_;
}
