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
        BinarySpaceFactor(Eigen::Matrix<double, 6, 6> Q2, Eigen::Matrix<double, 6, 6> Q4)
            : Factor(), m_Q2_inv(Q2.inverse()), m_Q4_inv(Q4.inverse()) {}
        ~BinarySpaceFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 2);
            
            double ds = nodes[1].arclength - nodes[0].arclength;
            Eigen::Matrix<DTYPE, 6, 1> xi = se3::tran2vec(nodes[1].pose * invertTransformation(nodes[0].pose));

            Eigen::Matrix<DTYPE, 6, 1> e2 = nodes[1].varpi - nodes[0].varpi + se3::curlyhat(nodes[0].varpi) * xi;
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
            Eigen::Matrix<DTYPE, 18, 18> weight = Eigen::Matrix<DTYPE, 18, 18>::Zero();
            weight.template block<6, 6>(0, 0) = (12.0 / (ds * ds * ds)) * m_Q2_inv.template cast<DTYPE>();
            weight.template block<6, 6>(0, 6) = (-6.0 / (ds * ds)) * m_Q2_inv.template cast<DTYPE>();
            weight.template block<6, 6>(6, 0) = weight.template block<6, 6>(0, 6);
            weight.template block<6, 6>(6, 6) = (4.0 / ds) * m_Q2_inv.template cast<DTYPE>();
            weight.template block<6, 6>(12, 12) = (1.0 / ds) * m_Q4_inv.template cast<DTYPE>();
            return weight;
        }

        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
        {
            assert(nodes.size() == 2);

            Spacetime::SystemState<double>::Node node0 = nodes[0].cast<double>();
            Spacetime::SystemState<double>::Node node1 = nodes[1].cast<double>();
            double ds = nodes[1].arclength - nodes[0].arclength;

            Eigen::Matrix<double, 4, 4> tran = node1.pose * invertTransformation(node0.pose);
            Eigen::Matrix<double, 6, 1> xi = se3::tran2vec(tran);

            e = getError(nodes);

            const Eigen::Matrix<double, 6, 6> pose_jacobian0 = -se3::vec2jacinv(xi) * se3::tranAd(tran);
            const Eigen::Matrix<double, 6, 6> pose_jacobian1 = se3::vec2jacinv(xi);
            const Eigen::Matrix<double, 6, 6> varpihat = se3::curlyhat(node0.varpi);
            Eigen::Matrix<double, 18, 18> dedx0, dedx1;
            dedx0 << pose_jacobian0, -ds * EYE(6), ZERO(6),
                ZERO(6), -EYE(6), ZERO(6),
                varpihat * pose_jacobian0, ZERO(6), -EYE(6) - se3::curlyhat(xi);

            dedx1 << pose_jacobian1, ZERO(6), ZERO(6),
                ZERO(6), EYE(6), ZERO(6),
                varpihat * pose_jacobian1, ZERO(6), EYE(6);

            // Assemble terms into correct spot in matrix
            Eigen::Matrix<double, 18, 36> H;
            H << dedx0, dedx1;

            return H;
        };

    private:
        const Eigen::Matrix<double, 6, 6> m_Q2_inv, m_Q4_inv;
    };
}
#endif // BINARYSPACEFACTOR_H
