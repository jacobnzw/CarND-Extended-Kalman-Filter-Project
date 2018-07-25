#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cfloat>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    MatrixXd eye_4 = MatrixXd::Identity(4, 4);
    VectorXd ones_4 = VectorXd::Ones(4);

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

VectorXd FusionEKF::processFunction(VectorXd &x, float dt) {
  MatrixXd F;
  F <<  1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;
  return F * x;
}

VectorXd FusionEKF::radarFunction(VectorXd &x, float dt) {
  VectorXd out = VectorXd(3U);
  // compute predicted measurement and safeguard against division by zero
  double norm = sqrt(pow(x[0], 2) + pow(x[1], 2)) + DBL_EPSILON;
  out << norm, atan2(x[1], x[0]), (x[0]*x[2] + x[1]*x[3])/norm;
}

VectorXd FusionEKF::laserFunction(VectorXd &x, float dt) {
  MatrixXd H = MatrixXd::Ones(2, 4);
  return H * x;
}

MatrixXd FusionEKF::processFunctionGrad(VectorXd &x, float dt) {
  MatrixXd F = MatrixXd::Ones(4, 4);
  F[0, 2] = dt;
  F[1, 3] = dt;
  F.transposeInPlace();
  return F;
}

MatrixXd FusionEKF::radarFunctionGrad(VectorXd &x, float dt) {
  MatrixXd jacobian = MatrixXd(3, 4);
  double p_x = x[0];
  double p_y = x[1];
  double v_x = x[2];
  double v_y = x[3];
  double norm2 = pow(p_x, 2) + pow(p_y, 2) + DBL_EPSILON;
  jacobian << p_x/(sqrt(norm2)), p_y/(sqrt(norm2)), 0, 0,
              -p_y/norm2, p_x/norm2, 0, 0,
              p_y * (v_x*p_y - v_y*p_x)/pow(norm2, 1.5), p_x * (v_y*p_x - v_x*p_y)/pow(norm2, 1.5),
              p_x/sqrt(norm2), p_y/sqrt(norm2);
  return jacobian;
}

MatrixXd FusionEKF::laserFunctionGrad(VectorXd &x, float dt) {
  return MatrixXd::Ones(2, 4);
}

void FusionEKF::measurementUpdate(const VectorXd &z) {
    MatrixXd K = Pz_.llt().solve(Pxz_.transpose());
    mx_ = mx_ + K.transpose() * (z - mz_);
    Px_ = Px_ - K.transpose() * Pz_ * K;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // first measurement
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.

            This seemingly ad-hocy conversion happens because we are initializing state with measurements.
            Hence the need for inverse transform from polar (measurement space) to cartesian (state space).
            */
            double rho = measurement_pack.raw_measurements_[0];
            double theta = measurement_pack.raw_measurements_[1];
            ekf_.x_[0] = rho * cos(theta);
            ekf_.x_[1] = rho * sin(theta);
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
//            ekf_.x_.segment(0, 2) = measurement_pack.raw_measurements_.segment(0, 2);
            ekf_.x_[0] = measurement_pack.raw_measurements_[0];
            ekf_.x_[1] = measurement_pack.raw_measurements_[1];
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1e6;
    previous_timestamp_ = measurement_pack.timestamp_;

    ekf_.F_ <<  1, 0, dt, 0,
                0, 1, 0, dt,
                0, 0, 1, 0,
                0, 0, 0, 1;
    double noise_ax = 9;
    double noise_ay = 9;
    ekf_.Q_ << noise_ax*pow(dt, 4)/4, 0, noise_ax*pow(dt, 3)/2, 0,
               0, noise_ay*pow(dt, 4)/4, 0, noise_ay*pow(dt, 3)/2,
               noise_ax*pow(dt, 3)/2, 0, noise_ax*pow(dt, 2), 0,
               0, noise_ay*pow(dt, 3)/2, 0, noise_ay*pow(dt, 2);
    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     TODO:
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        Tools tools;
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
//    cout << "x_ = " << ekf_.x_ << endl;
//    cout << "P_ = " << ekf_.P_ << endl;
}
