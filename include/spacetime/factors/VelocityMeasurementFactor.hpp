#ifndef VELOCITYMEASUREMENTFACTOR_H
#define VELOCITYMEASUREMENTFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/MeasurementFactor.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class VelocityMeasurementFactor : public MeasurementFactor
    {
    public:
        VelocityMeasurementFactor(Eigen::Matrix<double, 3, 3> weight,
                                  const Spacetime::SensorMeasurement &meas)
            : MeasurementFactor(weight, meas) {}

        ~VelocityMeasurementFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 1);

            Eigen::Vector<DTYPE, 3> e;
            e << this->m_meas_value + nodes[0].varpi.block<3, 1>(0, 0);

            for (unsigned int i = 0; i < 3; i++)
            {
                if (this->m_meas.mask(i, 0) == 0)
                {
                    e(i, 0) = 0;
                }
            }
            return e;
        };

        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
        {
            assert(nodes.size() == 1);
            Eigen::Matrix<double, 3, 18> H = Eigen::Matrix<double, 3, 18>::Zero();
            H.block<3, 3>(0, 12) = Eigen::Matrix<double, 3, 3>::Identity();
            e = getError(nodes);

            for (unsigned int i = 0; i < 3; i++)
            {
                if (this->m_meas.mask(i, 0) == 0)
                {
                    H.row(i).setZero();
                }
            }
            return H;
        };
    };
}
#endif // VELOCITYMEASUREMENTFACTOR_H