#ifndef MEASUREMENTFACTOR_H
#define MEASUREMENTFACTOR_H

#include <vector>

#include <Eigen/Dense>

#include "spacetime/factors/Factor.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class MeasurementFactor : public Factor
    {
    public:
        MeasurementFactor(Eigen::MatrixX<double> weight,
                          const Spacetime::SensorMeasurement &meas)
            : Factor(weight),
              m_meas(meas),
              m_meas_value(meas.value.cast<DTYPE>()) {}

        const Spacetime::SensorMeasurement &getMeas() const { return m_meas; }
        const Node &getOperatingPoint() const { return operating_point; }
        void setOperatingPoint(const Node &node) const { operating_point = node; }
        int getRoutedNodeIndex() const { return routed_node_index; }
        void setRoutedNodeIndex(int node_index) const { routed_node_index = node_index; }

    protected:
        const Spacetime::SensorMeasurement m_meas;
        const Eigen::MatrixX<DTYPE> m_meas_value;

    private: 
        mutable SystemState<DTYPE>::Node operating_point;
        mutable int routed_node_index = -1;
    };
}
#endif // MEASUREMENTFACTOR_H