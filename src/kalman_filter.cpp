#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
    H_ = tools_.CalculateJacobian(z);
    MatrixXd Pz = H_*P_*H_.transpose() + R_;
    MatrixXd Pzx = H_*P_;
    MatrixXd K = Pz.ldlt().solve(Pzx).transpose();
    VectorXd e = z - H_*x_;
    x_ = x_ + K*e;
    P_ = (I_ - K*H_)*P_;
}
