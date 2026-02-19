#ifndef FACTOR_H
#define FACTOR_H

#include <vector>

#include <Eigen/Dense>
#ifdef USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#endif

#include "spacetime/types.hpp"
#include "spacetime/static_funcs.hpp"
#include "spacetime/utilities.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;

namespace Spacetime::Factors
{
    class Factor
    {
    public:
        Factor() {}
        Factor(Eigen::MatrixX<double> weight) : m_weight(invertDiagonal(weight)) {}
        virtual ~Factor() = default;

        virtual Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const = 0;

        virtual std::function<Eigen::VectorX<DTYPE>(const Eigen::VectorX<DTYPE> &)> getErrorFunc(const std::vector<Node> &nodes) const
        {
            return [&](const Eigen::VectorX<DTYPE> &perturb)
            {
                std::vector<Node> _nodes;
                for (size_t i = 0; i < nodes.size(); i++)
                {
                    Node _node;
                    const Eigen::Vector<DTYPE, 18> &_perturb = perturb.block<18, 1>(i * 18, 0);
                    perturbNode(nodes[i], _perturb, _node);
                    _nodes.push_back(_node);
                }
                return getError(_nodes);
            };
        }

        virtual DTYPE getCost(const Eigen::VectorX<DTYPE> &e) const
        {
            // Computes a squared error cost as default implementation
            DTYPE cost = 0.5 * e.transpose() * m_weight.cast<DTYPE>() * e;
            return cost;
        }
        virtual DTYPE getCost(const std::vector<Node> &nodes) const
        {
            const Eigen::VectorX<DTYPE> e = getError(nodes);
            return getCost(e);
        }

        virtual Eigen::MatrixX<double> getJacobian([[maybe_unused]] const std::vector<Node> &nodes, [[maybe_unused]] Eigen::VectorX<DTYPE> &e) const
        {
#ifndef USE_AUTODIFF
            throw std::runtime_error("Autodiff is not enabled. Please enable it or define an analytic override to use this function.");
#else
            // warning("Using autodiff to compute Jacobian. This may be slow.");

            // Provides a default implementation of the Jacobian using autodiff
            // This should be overridden in derived classes if an analytical Jacobian is available
            Eigen::MatrixX<double> S;
            Eigen::VectorX<DTYPE> vec = Eigen::VectorX<DTYPE>::Zero(nodes.size() * 18);
            autodiff::jacobian(getErrorFunc(nodes), autodiff::wrt(vec), autodiff::at(vec), e, S);
            return S;
#endif
        }
        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes) const
        {
            Eigen::VectorX<DTYPE> e;
            Eigen::MatrixX<double> S = getJacobian(nodes, e);
            return S;
        }

        virtual Eigen::MatrixX<DTYPE> getWeight() const { return m_weight; }

    protected:
        mutable Eigen::MatrixX<DTYPE> m_weight;
    };
}
#endif // FACTOR_H