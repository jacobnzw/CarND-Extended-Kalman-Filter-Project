#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

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

void measurementUpdate(const VectorXd &z) {
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

    ekf_.F_ << 1, 0, dt, 0,
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
