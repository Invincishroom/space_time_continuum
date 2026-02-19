#ifndef BINARYSPACEFACTOR_H
#define BINARYSPACEFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/Factor.hpp"
#include "spacetime/static_funcs.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class BinarySpaceFactor : public Factor
    {
    public:
        BinarySpaceFactor(Eigen::Matrix<double, 6, 6> Q2, Eigen::Matrix<double, 6, 6> Q3) : Factor(), m_Q2(Q2), m_Q3(Q3) {}
        ~BinarySpaceFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 2);
            
            double ds = nodes[1].arclength - nodes[0].arclength;
            Eigen::Matrix<DTYPE, 6, 1> xi = se3::tran2vec(nodes[1].pose * invertTransformation(nodes[0].pose));

            // Eigen::Matrix<DTYPE, 6, 1> e2 = nodes[1].varpi - nodes[0].varpi;
            Eigen::Matrix<DTYPE, 6, 1> e2 = nodes[1].varpi - nodes[0].varpi - ds * se3::curlyhat(nodes[0].epsilon) * nodes[0].varpi;
            Eigen::Matrix<DTYPE, 6, 1> e4 = nodes[1].epsilon - nodes[0].epsilon;
            Eigen::Matrix<DTYPE, 6, 1> e6 = xi - ds * nodes[0].epsilon;

            Eigen::Vector<DTYPE, 18> e;
            e << e6, e4, e2;
            return e;
        };

        Spacetime::SystemState<DTYPE>::Node getZeroErorNode(const Spacetime::SystemState<DTYPE>::Node &node0, double ds) const
        {
            Spacetime::SystemState<DTYPE>::Node node1;
            node1.arclength = node0.arclength + ds;
            node1.time = node0.time;

            node1.epsilon = node0.epsilon;
            node1.varpi = node0.varpi + ds * (se3::curlyhat(node0.epsilon) * node0.varpi);
            // node1.varpi = node0.varpi;
            auto xi = ds * node0.epsilon;
            node1.pose = se3::vec2tran(xi) * node0.pose;

            return node1;
        };

        Eigen::Matrix<DTYPE, 18, 18> getWeight(const std::vector<Node> &nodes) const
        {
            double ds = nodes[1].arclength - nodes[0].arclength;
            const DTYPE ds2 = ds * ds / 2.0;
            const DTYPE ds3 = ds * ds * ds / 3.0;

            Eigen::Matrix<DTYPE, 18, 18> Qs;
            // Qs << ds3 * m_Q2, ds2 * m_Q2, ZERO<DTYPE>(6),
            //     ds2 * m_Q2, ds * m_Q2, ZERO<DTYPE>(6),
            //     ZERO<DTYPE>(6), ZERO<DTYPE>(6), ds * m_Q3;

            Eigen::Matrix<DTYPE, 6, 6> varpihat = se3::curlyhat(nodes[0].varpi);
            Qs << ds3 * m_Q2,
                ds2 * m_Q2,
                -ds3 * m_Q2 * varpihat.transpose(),
                ds2 * m_Q2,
                ds * m_Q2,
                -ds2 * m_Q2 * varpihat.transpose(),
                -ds3 * varpihat * m_Q2,
                -ds2 * varpihat * m_Q2,
                ds * m_Q3 + ds3 * varpihat * m_Q2 * varpihat.transpose();
            return Qs.inverse();
        }

        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
        {
            assert(nodes.size() == 2);

            Spacetime::SystemState<double>::Node node0 = nodes[0].cast<double>();
            Spacetime::SystemState<double>::Node node1 = nodes[1].cast<double>();
            double ds = nodes[1].arclength - nodes[0].arclength;

            Eigen::Matrix<double, 4, 4> tran = node1.pose * invertTransformation(node0.pose);
            Eigen::Matrix<double, 6, 1> xi = se3::tran2vec(tran);

            // compute e
            e = getError(nodes); // 5% slower, keep only for debug

            // Compute Jacobian
            Eigen::Matrix<double, 18, 18> dedx0, dedx1;
            dedx0 << -se3::vec2jacinv(xi) * se3::tranAd(tran), -ds * EYE(6), ZERO(6),                  // e2
                ZERO(6), -EYE(6), ZERO(6),                                                             // e4
                ZERO(6), ds * se3::curlyhat(node0.varpi), -EYE(6) - ds * se3::curlyhat(node0.epsilon); // e6
            // dedx0 << -se3::vec2jacinv(xi) * se3::tranAd(tran), -ds * EYE(6), ZERO(6),                  // e2
            //     ZERO(6), -EYE(6), ZERO(6),                                                             // e4
            //     ZERO(6), ZERO(6), -EYE(6); // e6

            dedx1 << se3::vec2jacinv(xi), ZERO(6), ZERO(6),
                ZERO(6), EYE(6), ZERO(6), // e4
                ZERO(6), ZERO(6), EYE(6); // e2; // e6

            // Assemble terms into correct spot in matrix
            Eigen::Matrix<double, 18, 36> H;
            H << dedx0, dedx1;

            return H;
        };

    private:
        const Eigen::Matrix<double, 6, 6> m_Q2, m_Q3;
    };
}
#endif // BINARYSPACEFACTOR_H
