#ifndef POSEMEASUREMENTFACTOR_H
#define POSEMEASUREMENTFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/MeasurementFactor.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class PoseMeasurementFactor : public MeasurementFactor
    {
    public:
        PoseMeasurementFactor(Eigen::Matrix<double, 6, 6> weight,
                              const Spacetime::SensorMeasurement &meas)
            : MeasurementFactor(weight, meas) {}

        ~PoseMeasurementFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 1);

            Eigen::Vector<DTYPE, 6> e;
            e << se3::tran2vec(invertTransformation(nodes[0].pose * this->m_meas_value));
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

            Spacetime::SystemState<double>::Node node = nodes[0].cast<double>();

            Eigen::Matrix<double, 4, 4> tran = invertTransformation(node.pose * this->m_meas_value.cast<double>());
            Eigen::Matrix<double, 6, 1> xi = se3::tran2vec(tran);

            // Compute the error vector
            e = getError(nodes); // 5% slower, keep only for debug
            // e = xi.cast<DTYPE>();

            Eigen::Matrix<double, 6, 18> S;
            S << -se3::vec2jacinv(xi) * se3::tranAd(tran), ZERO(6), ZERO(6);

            for (unsigned int i = 0; i < 6; i++)
            {
                if (this->m_meas.mask(i, 0) == 0)
                {
                    S.row(i).setZero();
                }
            }

            return S;
        };
    };
}
#endif // POSEMEASUREMENTFACTOR_H