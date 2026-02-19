#ifndef SPACETIME_STATIC_H
#define SPACETIME_STATIC_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#ifdef USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#endif

#include "spacetime/types.hpp"
#include "spacetime/utilities.hpp"

namespace Spacetime
{
    template <typename U = DTYPE, typename T = double>
    static Eigen::Matrix<U, 18, 18> getQt(T dt, const Eigen::Matrix<U, 6, 6> &Q1, const Eigen::Matrix<U, 6, 6> &Q3)
    {
        assert(dt > 0 && "Time must be positive.");
        const U _dt = U(dt);
        Eigen::Matrix<U, 6, 6> _Q1 = Q1.template cast<U>();
        Eigen::Matrix<U, 6, 6> _Q3 = Q3.template cast<U>();

        Eigen::Matrix<U, 18, 18> Qt;
        Qt << (_dt * _dt * _dt) * _Q1 / 3.0, ZERO<U>(6), (_dt * _dt) * _Q1 / 2.0,
            ZERO<U>(6), _dt * _Q3, ZERO<U>(6),
            (_dt * _dt) * _Q1 / 2.0, ZERO<U>(6), _dt * _Q1;

        return Qt;
    };

    template <typename U = DTYPE, typename T = double>
    static Eigen::Matrix<U, 18, 18> getQs(T ds, const Eigen::Matrix<T, 6, 6> &Q2, const Eigen::Matrix<T, 6, 6> &Q3, const typename SystemState<U>::Node &node)
    {
        assert(ds > 0 && "Arclength must be positive.");
        const U _ds = U(ds);
        const U _ds2 = _ds * _ds / 2.0;
        const U _ds3 = _ds * _ds * _ds / 3.0;
        Eigen::Matrix<U, 6, 6> _Q2 = Q2.template cast<U>();
        Eigen::Matrix<U, 6, 6> _Q3 = Q3.template cast<U>();
        Eigen::Matrix<U, 6, 6> varpihat = se3::curlyhat(node.varpi);

        Eigen::Matrix<U, 18, 18> Qs;
        // Qs << _ds3 * _Q2, _ds2 * _Q2, ZERO<U>(6),
        //     _ds2 * _Q2, _ds * _Q2, ZERO<U>(6),
        //     ZERO<U>(6), ZERO<U>(6), _ds * _Q3; // Original 1D interp 

        Qs << _ds3 * _Q2,
            _ds2 * _Q2,
            -_ds3 * _Q2 * varpihat.transpose(),
            _ds2 * _Q2,
            _ds * _Q2,
            -_ds2 * _Q2 * varpihat.transpose(),
            -_ds3 * varpihat * _Q2,
            -_ds2 * varpihat * _Q2,
            _ds * _Q3 + _ds3 * varpihat * _Q2 * varpihat.transpose();

        return Qs;
    }; // TODO: commit to removing node from this function if not used

    template <typename U = DTYPE, typename T = double>
    static Eigen::Matrix<U, 18, 18> getQtInv(T dt, const Eigen::Matrix<T, 6, 6> &Q1, const Eigen::Matrix<T, 6, 6> &Q3)
    {
        assert(dt > 0 && "Time must be positive.");
        const U _dt = U(dt);
        Eigen::Matrix<U, 6, 6> Q1_inv = invertDiagonal(Q1).template cast<U>();
        Eigen::Matrix<U, 6, 6> Q3_inv = invertDiagonal(Q3).template cast<U>();

        Eigen::Matrix<U, 18, 18> Qt_inv;
        Qt_inv << 12.0 * Q1_inv / (_dt * _dt * _dt), ZERO<U>(6), -6.0 * Q1_inv / (_dt * _dt),
            ZERO<U>(6), 1.0 * Q3_inv / _dt, ZERO<U>(6),
            -6.0 * Q1_inv / (_dt * _dt), ZERO<U>(6), 4.0 * Q1_inv / _dt;

        return Qt_inv;
    };

    template <typename U = DTYPE, typename T = double>
    static Eigen::Matrix<U, 18, 18> getQsInv(T ds, const Eigen::Matrix<T, 6, 6> &Q2, const Eigen::Matrix<T, 6, 6> &Q3, const typename SystemState<U>::Node &node)
    {
        assert(ds > 0 && "Arclength must be positive.");
        return getQs(ds, Q2, Q3, node).inverse();
        // const U _ds = U(ds);
        // Eigen::Matrix<U, 6, 6> Q3_inv = invertDiagonal(Q3);
        // Eigen::Matrix<U, 6, 6> Q2_inv = invertDiagonal(Q2);

        // Eigen::Matrix<U, 18, 18> Qs_inv;
        // Qs_inv << 12.0 * Q2_inv / (_ds * _ds * _ds), -6.0 * Q2_inv / (_ds * _ds), ZERO<U>(6),
        //     -6.0 * Q2_inv / (_ds * _ds), 4.0 * Q2_inv / _ds, ZERO<U>(6),
        //     ZERO<U>(6), ZERO<U>(6), 1.0 * Q3_inv / _ds;

        // return Qs_inv;
    };

    template <typename U = DTYPE, typename T = double>
    static Eigen::Matrix<U, 18, 18> getPhiT(T dt)
    {
        const U _dt = U(dt);
        Eigen::Matrix<U, 18, 18> Phi_t;
        Phi_t << EYE<U>(6), ZERO<U>(6), _dt * EYE<U>(6),
            ZERO<U>(6), EYE<U>(6), ZERO<U>(6),
            ZERO<U>(6), ZERO<U>(6), EYE<U>(6);
        return Phi_t;
    };

    template <typename U = DTYPE, typename T = double>
    static Eigen::Matrix<U, 18, 18> getPhiS(T ds, const SystemState<DTYPE>::Node &node)
    {
        const U _ds = U(ds);
        Eigen::Matrix<U, 18, 18> Phi_s;
        Phi_s << EYE<T>(6), _ds * EYE<T>(6), ZERO<U>(6),
            ZERO<U>(6), EYE<T>(6), ZERO<U>(6),
            ZERO<U>(6), ZERO<U>(6), EYE<T>(6);
        // Phi_s << EYE<T>(6), _ds * EYE<T>(6), ZERO<U>(6),
        //     ZERO<U>(6), EYE<T>(6), ZERO<U>(6),
        //     ZERO<U>(6), -_ds * se3::curlyhat(node.varpi), EYE<T>(6) + _ds * se3::curlyhat(node.epsilon);
        return Phi_s;
    }

    template <typename T = DTYPE>
    static void globalToLocal(const typename SystemState<T>::Node &node, const typename SystemState<T>::Node &node_nk, Eigen::Matrix<T, 18, 1> &gamma, Eigen::Matrix<double, 18, 18> &dgdx0, Eigen::Matrix<double, 18, 18> &dgdx1, bool compute_jacobian = true, bool use_autodiff = false)
    {
        if (use_autodiff)
        {
#ifndef USE_AUTODIFF
            throw std::runtime_error("Autodiff not enabled.");
#else
            auto _globalToLocal = [&](const Eigen::Vector<DTYPE, 36> &perturb) -> Eigen::Vector<DTYPE, 18>
            {
                typename SystemState<T>::Node _node0, _node1;
                const Eigen::Vector<DTYPE, 18> &perturb_nk = perturb.block<18, 1>(0, 0);
                perturbNode(node_nk, perturb_nk, _node0);

                Eigen::Matrix<T, 18, 1> _gamma;
                if (&node_nk == &node)
                {
                    _gamma.template block<6, 1>(0, 0).setZero();
                    _gamma.template block<6, 1>(6, 0) = _node0.epsilon;
                    _gamma.template block<6, 1>(12, 0) = _node0.varpi;
                    return _gamma;
                }
                const Eigen::Vector<DTYPE, 18> &perturb1 = perturb.block<18, 1>(18, 0);
                perturbNode(node, perturb1, _node1);

                Eigen::Matrix<T, 6, 1> _xi = se3::tran2vec(_node1.pose * invertTransformation(_node0.pose));
                Eigen::Matrix<T, 6, 6> _jacinv = se3::vec2jacinv(_xi);
                _gamma.template block<6, 1>(0, 0) = _xi;
                _gamma.template block<6, 1>(6, 0) = _jacinv * _node1.epsilon;
                _gamma.template block<6, 1>(12, 0) = _jacinv * _node1.varpi;
                return _gamma;
            };

            Eigen::Matrix<double, 18, 36> J;
            Eigen::Vector<T, 36> perturb = Eigen::Vector<T, 36>::Zero();

            if (compute_jacobian)
            {
                autodiff::jacobian(_globalToLocal, autodiff::wrt(perturb), autodiff::at(perturb), gamma, J);
                dgdx0 = J.block<18, 18>(0, 0);
                dgdx1 = J.block<18, 18>(0, 18);
            }
            else
            {
                gamma = _globalToLocal(perturb);
                dgdx0.setZero();
                dgdx1.setZero();
            }
#endif
            return;
        }

        Eigen::Matrix<T, 4, 4> tran = node.pose * invertTransformation(node_nk.pose);
        Eigen::Matrix<T, 6, 1> xi = se3::tran2vec(tran);
        Eigen::Matrix<T, 6, 6> jacinv = se3::vec2jacinv(xi);
        gamma.template block<6, 1>(0, 0) = xi;
        gamma.template block<6, 1>(6, 0) = jacinv * node.epsilon;
        gamma.template block<6, 1>(12, 0) = jacinv * node.varpi;

        if (compute_jacobian)
        {
            if (&node_nk == &node)
            {
                // Case where node and node_nk are the same point in memory, by construction xi = 0 at this node.
                // Note: This is different than a node that has the same time and arclength.
                dgdx0 << ZERO<double>(6), ZERO<double>(6), ZERO<double>(6),
                    ZERO<double>(6), EYE<double>(6), ZERO<double>(6),
                    ZERO<double>(6), ZERO<double>(6), EYE<double>(6);
                dgdx1.setZero();
                return;
            }
            Eigen::Matrix<double, 6, 6> jacinv_d = jacinv.template cast<double>();
            Eigen::Matrix<double, 6, 6> JinvAd = jacinv_d * se3::tranAd(tran.template cast<double>());
            Eigen::Matrix<double, 6, 6> epsilonhat = se3::curlyhat(node.epsilon.template cast<double>());
            Eigen::Matrix<double, 6, 6> varpihat = se3::curlyhat(node.varpi.template cast<double>());

            dgdx0 << -JinvAd, ZERO<double>(6), ZERO<double>(6),
                -0.5 * epsilonhat * JinvAd, ZERO<double>(6), ZERO<double>(6),
                -0.5 * varpihat * JinvAd, ZERO<double>(6), ZERO<double>(6);

            dgdx1 << jacinv_d, ZERO<double>(6), ZERO<double>(6),
                0.5 * epsilonhat * jacinv_d, jacinv_d, ZERO<double>(6),
                0.5 * varpihat * jacinv_d, ZERO<double>(6), jacinv_d;
        }
    }

    template <typename T = DTYPE>
    static void globalToLocal(const typename SystemState<T>::Node &node, const typename SystemState<T>::Node &node_nk, Eigen::Matrix<T, 18, 1> &gamma, Eigen::Matrix<double, 18, 18> &dgdx0, bool compute_jacobian = true)
    {
        Eigen::Matrix<double, 18, 18> dump = Eigen::Matrix<double, 18, 18>::Zero();
        globalToLocal(node, node_nk, gamma, dgdx0, dump, compute_jacobian);
    }

    template <typename T = DTYPE>
    static void globalToLocal(const typename SystemState<T>::Node &node, const typename SystemState<T>::Node &node_nk, Eigen::Matrix<T, 18, 1> &gamma)
    {
        Eigen::Matrix<double, 18, 18> dump = Eigen::Matrix<double, 18, 18>::Zero();
        globalToLocal(node, node_nk, gamma, dump, dump, false);
    }

    template <typename T = DTYPE>
    static void perturbNode(const typename SystemState<T>::Node &node, const Eigen::Vector<T, 18> &perturb, typename SystemState<T>::Node &new_node)
    {
        new_node = node;
        new_node.pose = se3::vec2tran(perturb.template block<6, 1>(0, 0)) * node.pose;
        new_node.epsilon = perturb.template block<6, 1>(6, 0) + node.epsilon;
        new_node.varpi = perturb.template block<6, 1>(12, 0) + node.varpi;
    }

    template <typename T = DTYPE>
    static void getPerturbation(const typename SystemState<T>::Node &node, typename SystemState<T>::Node &node0, Eigen::Vector<T, 18> &perturb)
    {
        perturb.template block<6, 1>(0, 0) = se3::tran2vec(node.pose * invertTransformation(node0.pose));
        perturb.template block<6, 1>(6, 0) = node.epsilon - node0.epsilon;
        perturb.template block<6, 1>(12, 0) = node.varpi - node0.varpi;
    }

    template <typename T = DTYPE>
    static void localToGlobal(const Eigen::Matrix<T, 18, 1> &gamma, const typename SystemState<T>::Node &node_nk, typename SystemState<T>::Node &node)
    {
        // throw std::runtime_error("localToGlobal not implemented.");
        Eigen::Matrix<T, 6, 1> xi = gamma.template block<6, 1>(0, 0);
        Eigen::Matrix<T, 6, 6> jac = se3::vec2jac(xi);
        Eigen::Matrix<T, 4, 4> tran = se3::vec2tran(xi) * node_nk.pose;
        node.pose = tran;
        node.epsilon = jac * gamma.template block<6, 1>(6, 0);
        node.varpi = jac * gamma.template block<6, 1>(12, 0);
    }

    template <typename EstimatorT>
    static typename SystemState<DTYPE>::Node getClosestNode(const EstimatorT &estimator, const Spacetime::SystemState<DTYPE> &state, Eigen::Vector3d position, double time, int iterations = 0)
    {
        /*
        Given a point in 3D space and a time, find the closest point on the robot estimate trajectory to that point. Uses a binary search approach with a set number of iterations. 
        */
        using Node = typename SystemState<DTYPE>::Node;
        if (!state.ib)
        {
            throw std::runtime_error("getClosestNode: State must be in inertial body frame.");
        }

        // 1. Identify relevant time blocks
        auto it_lb = std::lower_bound(state.estimation_nodes.begin(), state.estimation_nodes.end(), time,
                                      [](const Node &a, double val)
                                      { return a.time < val; });

        size_t start1 = 0, end1 = 0;
        size_t start2 = 0, end2 = 0;

        bool exact_match = (it_lb != state.estimation_nodes.end() && std::abs(it_lb->time - time) < 1e-6);

        if (exact_match)
        {
            // Range 1 is the exact match block
            start1 = std::distance(state.estimation_nodes.begin(), it_lb);
            auto it = it_lb;
            while (it != state.estimation_nodes.end() && std::abs(it->time - time) < 1e-6)
            {
                ++it;
            }
            end1 = std::distance(state.estimation_nodes.begin(), it);
        }
        else
        {
            // Range 2 (t_next)
            if (it_lb != state.estimation_nodes.end())
            {
                start2 = std::distance(state.estimation_nodes.begin(), it_lb);
                double t_next = it_lb->time;
                auto it = it_lb;
                while (it != state.estimation_nodes.end() && std::abs(it->time - t_next) < 1e-6)
                {
                    ++it;
                }
                end2 = std::distance(state.estimation_nodes.begin(), it);
            }

            // Range 1 (t_prev)
            if (it_lb != state.estimation_nodes.begin())
            {
                auto it = it_lb; // Points to first element > time or end
                // Find end of t_prev block (which is it_lb)
                end1 = std::distance(state.estimation_nodes.begin(), it);

                double t_prev = (it_lb - 1)->time;
                // Scan backwards for start
                auto it_prev = it_lb - 1;
                while (true)
                {
                    if (std::abs(it_prev->time - t_prev) >= 1e-6)
                    {
                        // it_prev is now the element BEFORE the block
                        start1 = std::distance(state.estimation_nodes.begin(), it_prev) + 1;
                        break;
                    }
                    if (it_prev == state.estimation_nodes.begin())
                    {
                        start1 = 0;
                        break;
                    }
                    --it_prev;
                }
            }
        }

        if (start1 == end1 && start2 == end2)
        {
            throw std::runtime_error("getClosestNode: No nodes found.");
        }

        // 2. Find closest estimation node (using raw stored poses as hints)
        size_t best_global_idx = 0;
        double best_raw_dist = std::numeric_limits<double>::max();

        auto update_best = [&](size_t idx)
        {
            double d = (state.estimation_nodes[idx].pose.template block<3, 1>(0, 3) - position).norm();
            if (d < best_raw_dist)
            {
                best_raw_dist = d;
                best_global_idx = idx;
            }
        };

        for (size_t i = start1; i < end1; ++i)
            update_best(i);
        for (size_t i = start2; i < end2; ++i)
            update_best(i);

        double s_best = state.estimation_nodes[best_global_idx].arclength;

        // Refine best_dist using the actual time via queryState
        Node current_best_node;
        current_best_node.time = time;
        current_best_node.arclength = s_best;
        estimator->interpolateMean2D(current_best_node, state);
        double best_dist = (current_best_node.pose.template block<3, 1>(0, 3) - position).norm();

        if (iterations < 1)
            return current_best_node;

        // 3. Setup Binary Search
        // Find s_prev and s_next
        double s_prev = -1.0;
        double s_next = -1.0;
        double min_diff_prev = std::numeric_limits<double>::max();
        double min_diff_next = std::numeric_limits<double>::max();

        auto check_neighbors = [&](size_t idx)
        {
            double s = state.estimation_nodes[idx].arclength;
            if (s < s_best)
            {
                if ((s_best - s) < min_diff_prev)
                {
                    min_diff_prev = s_best - s;
                    s_prev = s;
                }
            }
            else if (s > s_best)
            {
                if ((s - s_best) < min_diff_next)
                {
                    min_diff_next = s - s_best;
                    s_next = s;
                }
            }
        };

        for (size_t i = start1; i < end1; ++i)
            check_neighbors(i);
        for (size_t i = start2; i < end2; ++i)
            check_neighbors(i);

        double s_start, s_end;
        double d_start, d_end;

        // Evaluate neighbors at INTERPOLATED time to decide direction
        double dist_prev = std::numeric_limits<double>::max();
        if (s_prev >= 0)
        {
            Node n;
            n.time = time;
            n.arclength = s_prev;
            estimator->interpolateMean2D(n, state);
            dist_prev = (n.pose.template block<3, 1>(0, 3) - position).norm();
        }

        double dist_next = std::numeric_limits<double>::max();
        if (s_next >= 0)
        {
            Node n;
            n.time = time;
            n.arclength = s_next;
            estimator->interpolateMean2D(n, state);
            dist_next = (n.pose.template block<3, 1>(0, 3) - position).norm();
        }

        // Logic for bounds
        if (dist_prev < dist_next)
        {
            // Move towards prev
            if (s_prev < 0)
            { // Can't move left
                return current_best_node;
            }
            s_start = s_prev;
            s_end = s_best;
            d_start = dist_prev;
            d_end = best_dist;
        }
        else
        {
            // Move towards next
            if (s_next < 0)
            { // Can't move right
                return current_best_node;
            }
            s_start = s_best;
            s_end = s_next;
            d_start = best_dist;
            d_end = dist_next;
        }

        for (int iter = 0; iter < iterations; ++iter)
        {
            double s_mid = (s_start + s_end) / 2.0;
            Node query_node;
            query_node.time = time;
            query_node.arclength = s_mid;

            estimator->interpolateMean2D(query_node, state);

            double d_mid = (query_node.pose.template block<3, 1>(0, 3) - position).norm();

            if (d_mid < best_dist)
            {
                best_dist = d_mid;
                current_best_node = query_node;
            }

            if (d_start < d_end)
            {
                s_end = s_mid;
                d_end = d_mid;
            }
            else
            {
                s_start = s_mid;
                d_start = d_mid;
            }
        }

        return current_best_node;
    }
}

#endif // SPACETIME_STATIC_H