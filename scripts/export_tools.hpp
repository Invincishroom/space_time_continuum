#ifndef EXPORT_TOOLS_H
#define EXPORT_TOOLS_H

#include "spacetime/estimators/Estimator.hpp"

namespace Spacetime
{
    void printError(Estimator *estimator, const SystemState<DTYPE> &state, const std::vector<SensorMeasurement> &gt); 
    void exportErrorData(Estimator *estimator, const SystemState<DTYPE> &state, const std::vector<SensorMeasurement> &gt, std::string output_path);
    void exportTrajData(const SystemState<DTYPE> &state, const std::vector<SensorMeasurement> &gt, const std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &meas, std::string output_path);
    void exportRuntimeData(const Estimator::Results &result, std::string output_path);
    void exportMapData(std::shared_ptr<steam_icp::Map> map, std::string output_path);
    void exportPointCloudData(Estimator *estimator, const SystemState<DTYPE> &state, const std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &meas, std::string output_path);
}

#endif // EXPORT_TOOLS_H