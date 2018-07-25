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

  MomentTransform mt_dyn_, mt_radar_, mt_laser_;

  /**
    * Constructor.
    */
  FusionEKF();

  /**
    * Destructor.
    */
  virtual ~FusionEKF();

  virtual VectorXd processFunction(VectorXd &x, float dt);

  virtual VectorXd radarFunction(VectorXd &x, float dt);

  virtual VectorXd laserFunction(VectorXd &x, float dt);

  virtual MatrixXd processFunctionGrad(VectorXd &x, float dt);

  virtual MatrixXd radarFunctionGrad(VectorXd &x, float dt);

  virtual MatrixXd laserFunctionGrad(VectorXd &x, float dt);

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
  MatrixXd H_laser_;
};

#endif /* FusionEKF_H_ */
