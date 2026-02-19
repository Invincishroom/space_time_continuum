#ifndef FBGSMEASUREMENTFACTOR_H
#define FBGSMEASUREMENTFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/MeasurementFactor.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class FBGSMeasurementFactor : public MeasurementFactor
    {
    public:
        FBGSMeasurementFactor(Eigen::Matrix<double, 6, 6> weight,
                                const Spacetime::SensorMeasurement &meas,
                                double theta_offset = 0.0)
            : MeasurementFactor(weight, meas),
              m_theta_offset(theta_offset) {}

        ~FBGSMeasurementFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 1);

            Eigen::Vector<DTYPE, 6> e;
            e << this->m_meas_value + nodes[0].epsilon;

            for (unsigned int i = 0; i < 6; i++)
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
            Eigen::Matrix<double, 6, 18> H = Eigen::Matrix<double, 6, 18>::Zero();
            H.block<6, 6>(0, 6) = Eigen::Matrix<double, 6, 6>::Identity();
            e = getError(nodes);

            for (unsigned int i = 0; i < 6; i++)
            {
                if (this->m_meas.mask(i, 0) == 0)
                {
                    H.row(i).setZero();
                }
            }
            return H;
        };

    private:
        // Precomputed values
        double m_theta_offset; // TODO: Should this be estimated?
    };
}
#endif // FBGSMEASUREMENTFACTOR_H