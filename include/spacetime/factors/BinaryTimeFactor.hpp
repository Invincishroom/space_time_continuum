#ifndef BINARYTIMEFACTOR_H
#define BINARYTIMEFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/Factor.hpp"
#include "spacetime/static_funcs.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class BinaryTimeFactor : public Factor
    {
    public:
        BinaryTimeFactor(Eigen::Matrix<double, 6, 6> Q1, Eigen::Matrix<double, 6, 6> Q3) : Factor(), m_Q1(Q1), m_Q3(Q3) {}
        ~BinaryTimeFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 2);

            double dt = nodes[1].time - nodes[0].time;
            Eigen::Matrix<DTYPE, 6, 1> xi = se3::tran2vec(nodes[1].pose * invertTransformation(nodes[0].pose));

            Eigen::Matrix<DTYPE, 6, 1> e1 = nodes[1].varpi - nodes[0].varpi;
            Eigen::Matrix<DTYPE, 6, 1> e3 = nodes[1].epsilon - nodes[0].epsilon;
            Eigen::Matrix<DTYPE, 6, 1> e5 = xi - dt * nodes[0].varpi;

            Eigen::Vector<DTYPE, 18> e;
            e << e5, e3, e1;
            return e;
        };

        Spacetime::SystemState<DTYPE>::Node getZeroErorNode(const Spacetime::SystemState<DTYPE>::Node &node0, double dt) const
        {
            Spacetime::SystemState<DTYPE>::Node node1;
            node1.arclength = node0.arclength;
            node1.time = node0.time + dt;

            node1.epsilon = node0.epsilon;
            node1.varpi = node0.varpi;
            auto xi = dt * node0.varpi;
            node1.pose = se3::vec2tran(xi) * node0.pose;

            return node1;
        };

        Eigen::Matrix<DTYPE, 18, 18> getWeight(const std::vector<Node> &nodes) const
        {
            double dt = nodes[1].time - nodes[0].time;
            const DTYPE dt2 = dt * dt / 2.0;
            const DTYPE dt3 = dt * dt * dt / 3.0;

            Eigen::Matrix<DTYPE, 18, 18> Qt;
            Qt << dt3 * m_Q1, ZERO<DTYPE>(6), dt2 * m_Q1,
                ZERO<DTYPE>(6), dt * m_Q3, ZERO<DTYPE>(6),
                dt2 * m_Q1, ZERO<DTYPE>(6), dt * m_Q1;
            return Qt.inverse();
        }

        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
        {
            assert(nodes.size() == 2);

            Spacetime::SystemState<double>::Node node0 = nodes[0].cast<double>();
            Spacetime::SystemState<double>::Node node1 = nodes[1].cast<double>();
            double dt = nodes[1].time - nodes[0].time;

            Eigen::Matrix<double, 4, 4> tran = node1.pose * invertTransformation(node0.pose);
            Eigen::Matrix<double, 6, 1> xi = se3::tran2vec(tran);

            // compute e
            e = getError(nodes); // 5% slower, keep only for debug
            // Eigen::Vector<double, 18> _e;
            // Eigen::Matrix<double, 6, 1> e1 = node1.varpi - node0.varpi;
            // Eigen::Matrix<double, 6, 1> e3 = node1.epsilon - node0.epsilon;
            // Eigen::Matrix<double, 6, 1> e5 = xi - dt * node0.varpi;
            // _e << e5, e3, e1;
            // e = _e.cast<DTYPE>();

            // Compute Jacobian
            Eigen::Matrix<double, 18, 18> dedx0, dedx1;
            dedx0 << -se3::vec2jacinv(xi) * se3::tranAd(tran), ZERO(6), -dt * EYE(6),
                ZERO(6), -EYE(6), ZERO(6),
                ZERO(6), ZERO(6), -EYE(6); // F

            dedx1 << se3::vec2jacinv(xi), ZERO(6), ZERO(6),
                ZERO(6), EYE(6), ZERO(6),
                ZERO(6), ZERO(6), EYE(6); // -E

            // Assemble terms into correct spot in matrix
            Eigen::Matrix<double, 18, 36> H;
            H << dedx0, dedx1;

            return H;
        };

    private:
        const Eigen::Matrix<double, 6, 6> m_Q1, m_Q3;
    };
}
#endif // BINARYTIMEFACTOR_H
