#ifndef UNARYFACTOR_H
#define UNARYFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/Factor.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"
#include "spacetime/static_funcs.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class UnaryFactor : public Factor
    {
    public:
        UnaryFactor(Eigen::Matrix<double, 18, 18> weight, const Node &node0) : Factor(weight), m_intial_node(node0) {}
        ~UnaryFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            assert(nodes.size() == 1);

            Eigen::Vector<DTYPE, 18> e;
            e << se3::tran2vec(m_intial_node.pose * invertTransformation(nodes[0].pose)),
                m_intial_node.epsilon - nodes[0].epsilon,
                m_intial_node.varpi - nodes[0].varpi;
            return e;
        };

        Spacetime::SystemState<DTYPE>::Node getZeroErorNode() const
        {
            Spacetime::SystemState<DTYPE>::Node node1 = m_intial_node;
            return node1;
        };

        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
        {
            assert(nodes.size() == 1);

            Spacetime::SystemState<double>::Node node = nodes[0].cast<double>();

            Eigen::Matrix<double, 4, 4> tran = m_intial_node.pose.cast<double>() * invertTransformation(node.pose);
            Eigen::Matrix<double, 6, 1> xi = se3::tran2vec(tran);

            e = getError(nodes); // 5% slower, keep only for debug
            // Eigen::Vector<DTYPE, 18> _e;
            // _e << xi.cast<DTYPE>(),
            //     m_intial_node.epsilon - nodes[0].epsilon,
            //     m_intial_node.varpi - nodes[0].varpi;
            // e = _e;

            Eigen::Matrix<double, 6, 6> Jinv = se3::vec2jacinv(xi);
            Eigen::Matrix<double, 6, 6> Ad = se3::tranAd(tran);

            Eigen::Matrix<double, 18, 18> S;
            S << -Jinv * Ad, ZERO<double>(6), ZERO<double>(6),
                ZERO<double>(6), -EYE<double>(6), ZERO<double>(6),
                ZERO<double>(6), ZERO<double>(6), -EYE<double>(6);

            return S;
        };

    private:
        const Node m_intial_node;
    };
}
#endif // UNARYFACTOR_H