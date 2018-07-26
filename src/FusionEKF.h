#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "mtran.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class FusionEKF
{
  public:
    VectorXd mx_, mz_;
    MatrixXd Px_, Pz_, Pxz_;

    MomentTransform mt_dyn_;
    MomentTransform mt_radar_;
    MomentTransform mt_laser_;

    /**
      * Constructor.
      */
    FusionEKF();

    /**
      * Destructor.
      */
    virtual ~FusionEKF();

    static VectorXd processFunction(const VectorXd &x, double dt);

    static VectorXd radarFunction(const VectorXd &x, double dt);

    static VectorXd laserFunction(const VectorXd &x, double dt);

    static MatrixXd processFunctionGrad(const VectorXd &x, double dt);

    static MatrixXd radarFunctionGrad(const VectorXd &x, double dt);

    static MatrixXd laserFunctionGrad(const VectorXd &x, double dt);

    static MatrixXd processCovariance(double dt);

    virtual void measurementUpdate(const VectorXd &z);

    /**
      * Run the whole flow of the Kalman Filter from here.
      */
    void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  private:
    // check whether the tracking toolbox was initialized or not (first measurement)
    bool is_initialized_;

    // previous timestamp
    long long previous_timestamp_;

    // tool object used to compute RMSE
    Tools tools;

    MatrixXd R_laser_;
    MatrixXd R_radar_;
};

#endif /* FusionEKF_H_ */
