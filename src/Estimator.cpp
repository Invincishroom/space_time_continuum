#include <numeric>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <random>
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#ifdef USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <lgmath/CommonMath.hpp>
#endif

#include "spacetime/estimators/Estimator.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/Solver.hpp"
#include "spacetime/static_funcs.hpp"
#include "spacetime/factors.hpp"

namespace Spacetime
{
    void Estimator::initializeEstimator(RobotTopology topology, Hyperparameters parameters, Options options)
    {
        setOptions(options);
        setRobotTopology(topology);
        setHyperparameters(parameters);

        m_unary_factor = new Factors::UnaryFactor(m_hyperparameters.P0, m_node0);
        m_binary_time_factor = new Factors::BinaryTimeFactor(m_hyperparameters.Q1, m_hyperparameters.Q3);
        m_binary_space_factor = new Factors::BinarySpaceFactor(m_hyperparameters.Q2, m_hyperparameters.Q3);
        if (!m_robot_topology.time_nodes_on_measurements)
            m_state = constructInitialGuess(m_options.init_guess_type, {}, topology.t0);
    }

    void Estimator::setRobotTopology(RobotTopology topology)
    {
        // Asserts for inputs to make sure they are valid
        validateRobotTopology(topology);

        m_robot_topology = topology;

        // Constrcut projection matrix
        if (!m_robot_topology.time_nodes_on_measurements)
            m_P = constructProjectionMatrix();

        m_node0 = SystemState<DTYPE>::Node();
        m_node0.pose = invertTransformation(m_robot_topology.T0);
        m_node0.epsilon = -m_robot_topology.epsilon0;
        m_node0.varpi = -m_robot_topology.varpi0;
        m_node0.arclength = 0.0;
        m_node0.time = 0.0;
    }

    void Estimator::setHyperparameters(Hyperparameters parameters)
    {
        // Asserts for inputs to make sure they are valid
        validateHyperparameters(parameters);
        m_hyperparameters = parameters;
    }

    void Estimator::setOptions(Options options)
    {
        // Asserts for inputs to make sure they are valid
        validateOptions(options);

        m_options = options;
    }

    void Estimator::validateRobotTopology(RobotTopology topology)
    {
        // Go through every variable in structure and make sure their values are valid
        assert((topology.N > 0) && "Topology needs to feature at least one point in space");

        if (!topology.time_nodes_on_measurements)
            assert((topology.K > 0) && "Topology needs to feature at least one point in time");

        assert((topology.epsilon0.rows() == 6) && ((topology.epsilon0.cols() == 1)) && "Incorrect epsilon0 shape");
        assert((topology.varpi0.rows() == 6) && ((topology.varpi0.cols() == 1)) && "Incorrect varpi0 shape");

        validateTransformationMatrix(topology.T0);

        if (topology.spatial_node_positions.size() > 0)
        {
            assert((topology.spatial_node_positions.size() == topology.N) && "Spatial node positions must match number of spatial nodes");
            for (unsigned int i = 0; i < topology.spatial_node_positions.size(); i++)
            {
                assert((topology.spatial_node_positions(i) >= 0) && (topology.spatial_node_positions(i) <= topology.L) && "Spatial node positions must be within [0, L]");
            }
        }
    }

    void Estimator::validateHyperparameters(Hyperparameters parameters)
    {
        // Go through every variable in structure and make sure their values are valid
        validateDiagonalMatrix(parameters.P0);
        validateDiagonalMatrix(parameters.Q1);
        validateDiagonalMatrix(parameters.Q2);
        validateDiagonalMatrix(parameters.Q3);
    }

    void Estimator::validateOptions(Options options)
    {
        // Go through every variable in structure and make sure their values are valid
        if (!(options.init_guess_type == Options::InitialGuessType::Custom))
            return;

        assert((options.custom_guess.estimation_nodes.size() == m_robot_topology.K * m_robot_topology.N) && "Number of estimation nodes must match number of estimation nodes in topology");

        assert((options.custom_guess.estimation_nodes[0].pose - m_robot_topology.T0).isZero() && "Pose of first node of each robot in initial guess must be identical to its base frame");
        for (unsigned int j = 0; j < options.custom_guess.estimation_nodes.size(); j++)
        {
            validateTransformationMatrix(options.custom_guess.estimation_nodes[j].pose);
        }
    }

    SystemState<DTYPE> Estimator::constructInitialGuess(Options::InitialGuessType type, std::vector<double> ts, double start_t) const
    {
        // Returns initial guess construction in T_bi convention
        SystemState<DTYPE> initial_guess;

        if (type == Options::InitialGuessType::Straight)
        {
            SystemState<DTYPE> robot_state;
            robot_state.estimation_nodes = {};
            robot_state.interpolation_nodes = {};

            SystemState<DTYPE>::Node node = m_unary_factor->getZeroErorNode();
            Eigen::Matrix<DTYPE, 6, 1> epsilon, varpi;
            epsilon << node.epsilon.block<3, 1>(0, 0), 0, 0, 0; // Use linear strain component for straight initialization
            varpi << 0, 0, 0, 0, 0, 0;

            for (unsigned int i = 0; i < m_robot_topology.K * m_robot_topology.N; i++)
            {
                SystemState<DTYPE>::Node node;

                int k = std::floor(i / m_robot_topology.N);
                int n = i - k * m_robot_topology.N;

                // Compute arc-length at node ksel
                double s = 0.0, t = (ts.size() > 0) ? ts[0] : start_t;
                if (m_robot_topology.N > 1)
                {
                    if (m_robot_topology.spatial_node_positions.size() > 0)
                    {
                        assert(m_robot_topology.spatial_node_positions.size() == m_robot_topology.N && "Spatial node positions must match the number of nodes in the robot topology");
                        s = m_robot_topology.spatial_node_positions.coeff(n);
                    }
                    else
                    {
                        s = (double)n / ((double)(m_robot_topology.N - 1)) * m_robot_topology.L;
                    }
                }

                if (m_robot_topology.K > 1)
                {
                    if (ts.size() == 0)
                    {
                        t = start_t + (double)k / ((double)(m_robot_topology.K - 1)) * m_robot_topology.T;
                    }
                    else
                    {
                        assert(ts.size() == m_robot_topology.K && "Time steps must match the number of time nodes in the robot topology");
                        t = ts[k];
                    }
                }

                node.arclength = s;
                node.time = t;

                // Define pose of node k in base frame of robot
                Eigen::Matrix<DTYPE, 6, 1> xi = DTYPE(s) * epsilon;
                Eigen::Matrix<DTYPE, 4, 4> T_ik = se3::vec2tran(xi) * m_node0.pose;
                node.pose = T_ik;
                node.epsilon = epsilon;
                node.varpi = varpi;

                robot_state.estimation_nodes.push_back(node);
            }

            initial_guess = robot_state;
        }
        else if (type == Options::InitialGuessType::Custom)
        {
            initial_guess = m_options.custom_guess;
        }
        else if (type == Options::InitialGuessType::ZeroErrorPrior)
        {
            // throw std::runtime_error("ZeroErrorPrior not implemented yet");
            SystemState<DTYPE> robot_state;
            robot_state.estimation_nodes = {};
            robot_state.interpolation_nodes = {};

            double dt = m_robot_topology.T / (m_robot_topology.K - 1);
            // double ds = m_robot_topology.L / (m_robot_topology.N - 1);

            // Add first node with zero unary factor error
            debug() << "Start time for zero error prior initialization: " << start_t;
            SystemState<DTYPE>::Node node = m_unary_factor->getZeroErorNode();
            node.time = start_t;
            robot_state.estimation_nodes.push_back(node);

            // Space first, then time
            for (unsigned int n = 1; n < m_robot_topology.N; n++)
            {
                double ds = (m_robot_topology.spatial_node_positions.size() > 0) ? (m_robot_topology.spatial_node_positions(n) - m_robot_topology.spatial_node_positions(n - 1)) : (m_robot_topology.L / (m_robot_topology.N - 1));

                SystemState<DTYPE>::Node node = m_binary_space_factor->getZeroErorNode(robot_state.estimation_nodes[n - 1], ds);
                robot_state.estimation_nodes.push_back(node);
            }
            for (unsigned int k = 1; k < m_robot_topology.K; k++)
            {
                if (ts.size() > 0)
                    dt = ts[k] - ts[k - 1];
                SystemState<DTYPE>::Node node = m_binary_time_factor->getZeroErorNode(robot_state.estimation_nodes[(k - 1) * m_robot_topology.N], dt);
                robot_state.estimation_nodes.push_back(node);
                for (unsigned int n = 1; n < m_robot_topology.N; n++)
                {
                    double ds = (m_robot_topology.spatial_node_positions.size() > 0) ? (m_robot_topology.spatial_node_positions(n) - m_robot_topology.spatial_node_positions(n - 1)) : (m_robot_topology.L / (m_robot_topology.N - 1));

                    SystemState<DTYPE>::Node node = m_binary_space_factor->getZeroErorNode(robot_state.estimation_nodes[k * m_robot_topology.N + n - 1], ds);
                    robot_state.estimation_nodes.push_back(node);
                }
            }

            initial_guess = robot_state;
        }
        else if (type == Options::InitialGuessType::Last)
        {
            initial_guess = m_state;
        }

        // info() << "Applying perturbation: " << m_options.initial_perturbation;
        // perturbState(initial_guess, m_options.initial_perturbation);

        // Write out prior error of initial guess
        // std::vector<Eigen::Triplet<double>> A_tripletList, b_tripletList;
        // DTYPE cost_p;
        // assemblePriorTerms(A_tripletList, b_tripletList, cost_p, initial_guess, m_robot_topology, true);
        // info() << "Initial prior error: " << cost_p;

        return initial_guess;
    }

    void Estimator::assemblePriorTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only) const
    {
        cost = 0;

        unsigned int total_unary = topology.K;
        unsigned int total_binary_space = topology.K * (topology.N - 1);
        unsigned int total_binary_time = (topology.K - 1) * topology.N;
        unsigned int total_A = total_unary * 18 * 18 + total_binary_space * 36 * 36 + total_binary_time * 36 * 36;
        unsigned int total_b = total_unary * 18 + total_binary_space * 36 + total_binary_time * 36;
        A_tripletList.reserve(total_A);
        b_tripletList.reserve(total_b);

#pragma omp declare reduction(dtype_add:DTYPE : omp_out += omp_in) initializer(omp_priv = DTYPE(0.0))
#pragma omp declare reduction(merge_triplet_list : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<Eigen::Triplet<double>>())
#pragma omp parallel for reduction(dtype_add : cost) reduction(merge_triplet_list : A_tripletList, b_tripletList)
        for (unsigned int i = 0; i < topology.K * topology.N; i++)
        {
            unsigned int n = i % topology.N;
            unsigned int k = i / topology.N;

            if (n == 0) // Add at base for each timestep
            {
                assembleUnaryTerm(A_tripletList, b_tripletList, cost, state, n, k, topology, cost_only);
            }
            if (n != 0)
            {
                assembleBinarySpaceTerm(A_tripletList, b_tripletList, cost, state, n, k, topology, cost_only);
            }
            if (k > 0 && !topology.use_1D_estimator)
            {
                assembleBinaryTimeTerm(A_tripletList, b_tripletList, cost, state, n, k, topology, cost_only);
            }
        }
    }

    void Estimator::assembleUnaryTerm(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, int n, int k, const RobotTopology &topology, bool cost_only) const
    {
        const auto &node = state.estimation_nodes[k * topology.N + n];

        if (cost_only)
        {
            cost = cost + m_unary_factor->getCost(std::vector({node}));
            return;
        }

        Eigen::VectorX<DTYPE> e;
        Eigen::Matrix<double, 18, 18> S;
        if (m_options.use_autodiff)
            S = m_unary_factor->Factor::getJacobian({node}, e);
        else
            S = m_unary_factor->getJacobian({node}, e);
        cost = cost + m_unary_factor->getCost(e);
        Eigen::Matrix<double, 18, 18> weight = m_unary_factor->getWeight().cast<double>();

        Eigen::MatrixX<double> A_block = S.transpose() * weight * S;
        Eigen::MatrixX<double> b_block = -S.transpose() * weight * e.cast<double>();

        // Save the coefficients and entries in A and b
        std::array<unsigned int, 1> row_indices = {
            getOptimizationIndex(n, k, topology)};
        for (int row_block = 0; row_block < 18; row_block++)
        {
            unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);
            for (int col_block = 0; col_block < 18; col_block++)
            {
                unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
            }

            b_tripletList.emplace_back(idx_row, 0, b_block(row_block, 0));
        }
    }

    void Estimator::assembleBinarySpaceTerm(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, int n, int k, const RobotTopology &topology, bool cost_only) const
    {
        assert(n > 0 && "Binary space term can only be computed for n > 0");
        const auto &state1 = state.estimation_nodes[k * topology.N + n];
        const auto &state0 = state.estimation_nodes[k * topology.N + n - 1];

        Eigen::VectorX<DTYPE> e;
        Eigen::Matrix<DTYPE, 18, 18> weight = m_binary_space_factor->getWeight({state0, state1});

        if (cost_only)
        {
            e = m_binary_space_factor->getError({state0, state1});
            cost = cost + DTYPE(0.5 * e.transpose() * weight * e);
            return;
        }
        Eigen::Matrix<double, 18, 36> S;
        if (m_options.use_autodiff)
            S = m_binary_space_factor->Factor::getJacobian({state0, state1}, e);
        else
            S = m_binary_space_factor->getJacobian({state0, state1}, e);
        cost = cost + DTYPE(0.5 * e.transpose() * weight * e);

        Eigen::Matrix<double, 18, 18> weightd = weight.cast<double>();
        Eigen::MatrixX<double> A_block = S.transpose() * weightd * S;
        Eigen::MatrixX<double> b_block = -S.transpose() * weightd * e.cast<double>();

        // Precompute the indices for row and column blocks
        std::array<unsigned int, 2> row_indices = {
            getOptimizationIndex(n - 1, k, topology),
            getOptimizationIndex(n, k, topology)};

        // Save the coefficients and entries in A and b
        for (int row_block = 0; row_block < 36; row_block++)
        {
            unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);

            for (int col_block = 0; col_block < 36; col_block++)
            {
                unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
            }

            b_tripletList.emplace_back(idx_row, 0, b_block(row_block, 0));
        }
    }

    void Estimator::assembleBinaryTimeTerm(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, int n, int k, const RobotTopology &topology, bool cost_only) const
    {
        assert(k > 0 && "Binary time term can only be computed for k > 0");
        const auto &state0 = state.estimation_nodes[(k - 1) * topology.N + n];
        const auto &state1 = state.estimation_nodes[k * topology.N + n];

        Eigen::VectorX<DTYPE> e;
        Eigen::Matrix<DTYPE, 18, 18> weight = m_binary_time_factor->getWeight({state0, state1});

        if (cost_only)
        {
            e = m_binary_time_factor->getError({state0, state1});
            cost = cost + DTYPE(0.5 * e.transpose() * weight * e);
            return;
        }

        Eigen::Matrix<double, 18, 36> S;
        if (m_options.use_autodiff)
            S = m_binary_time_factor->Factor::getJacobian({state0, state1}, e);
        else
            S = m_binary_time_factor->getJacobian({state0, state1}, e);
        cost = cost + DTYPE(0.5 * e.transpose() * weight * e);

        Eigen::Matrix<double, 18, 18> weightd = weight.cast<double>();
        Eigen::MatrixX<double> A_block = S.transpose() * weightd * S;
        Eigen::MatrixX<double> b_block = -S.transpose() * weightd * e.cast<double>();

        // // Precompute the indices for row and column blocks
        std::array<unsigned int, 2> row_indices = {
            getOptimizationIndex(n, k - 1, topology),
            getOptimizationIndex(n, k, topology)};

        // Save the coefficients and entries in A and b
        for (int row_block = 0; row_block < 36; row_block++)
        {
            unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);

            for (int col_block = 0; col_block < 36; col_block++)
            {
                unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
            }

            b_tripletList.emplace_back(idx_row, 0, b_block(row_block, 0));
        }
    }

    bool Estimator::assembleMeasurementTerm(std::shared_ptr<Factors::MeasurementFactor> p_factor, std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only) const
    {
        // Returns true if measurement factor was included, false if it was skipped (e.g., out of time bounds)
        const auto &measurement = p_factor->getMeas();

        if (measurement.t < state.estimation_nodes.front().time - TOLERANCE || measurement.t >= state.estimation_nodes.back().time - TOLERANCE)
            return false; // Measurement is outside of time bounds of estimation. Range is [first state time, last state time)
        if (!m_options.interpolate_measurements)
            throw std::runtime_error("Snapping measurements is broken");

        SystemState<DTYPE>::Node node_ts;
        node_ts.time = measurement.t;
        node_ts.arclength = measurement.s;

        // Update cost
        if (cost_only)
        {
            interpolateMean2D(node_ts, state);
            cost += p_factor->getCost(std::vector({node_ts}));
            return true;
        }

        unsigned int n, k, n1, k1;
        getNKIndeces(measurement.s, measurement.t, n, k, state, topology);

        if (n == (unsigned int)(-1) || k == (unsigned int)(-1))
        {
            warning("Something broke with measurement factor. Ignoring this one.");
            return false;
        }

        SystemState<DTYPE>::Node node0 = state.estimation_nodes[k * topology.N + n];

        k1 = (std::abs(node_ts.time - node0.time) < TOLERANCE) ? -1 : k + 1;
        n1 = (std::abs(node_ts.arclength - node0.arclength) < TOLERANCE) ? -1 : n + 1;
        // std::cout << "Measurement at t=" << measurement.t << ", s=" << measurement.s << " uses nodes k=" << k << ", n=" << n << " and k1=" << k1 << ", n1=" << n1 << std::endl;
        if (topology.K <= 1)
            k1 = (unsigned int)(-1);
        if (topology.N <= 1)
            n1 = (unsigned int)(-1);

        SystemState<DTYPE>::Node node1;
        Eigen::VectorX<DTYPE> e;
        Eigen::MatrixX<double> G, S; // G: measurement jacobian, S: integrated measurement and interpolation jacobian

        if (std::abs(node_ts.arclength - node0.arclength) < TOLERANCE && std::abs(node_ts.time - node0.time) < TOLERANCE)
        {
            // No interpolation required
            if (m_options.use_autodiff)
                S = p_factor->Factor::getJacobian({node0}, e);
            else
                S = p_factor->getJacobian({node0}, e);
            G = S;
            node_ts = node0;
        }
        else
        {
            // Interpolation required
            //             if (m_options.use_autodiff)
            //             {
            // #ifdef USE_AUTODIFF
            //                 // Option 1: interpolate fully from error to nodes
            //                 node1 = state.estimation_nodes[k1 * topology.N + n1];
            //                 auto _getError = [&](const Eigen::Vector<DTYPE, 36> &perturb)
            //                 {
            //                     SystemState<DTYPE>::Node _node0, _node1;
            //                     const Eigen::Vector<DTYPE, 18> &perturb0 = perturb.block<18, 1>(0, 0);
            //                     const Eigen::Vector<DTYPE, 18> &perturb1 = perturb.block<18, 1>(18, 0);
            //                     perturbNode(node0, perturb0, _node0);
            //                     perturbNode(node1, perturb1, _node1);

            //                     interpolateMean1D(node_ts, _node0, _node1);
            //                     return p_factor->getError({node_ts});
            //                 };

            //                 Eigen::Vector<DTYPE, 36> perturb = Eigen::Vector<DTYPE, 36>::Zero();
            //                 autodiff::jacobian(_getError, autodiff::wrt(perturb), autodiff::at(perturb), e, S);
            // #else
            //                 throw std::runtime_error("Autodiff not enabled");
            // #endif
            //             }
            //             else
            //             {
            // Option 2: interpolate in two parts, using analytical p_factor jacobians
            Eigen::MatrixXd jac;
            interpolateMean2D(node_ts, state, jac, true);
            if (m_options.use_autodiff)
                G = p_factor->Factor::getJacobian({node_ts}, e);
            else
                G = p_factor->getJacobian({node_ts}, e);
            // G = p_factor->getJacobian({node_ts}, e);
            S = G * jac;
            // }
        }
        p_factor->setOperatingPoint(node_ts);

        cost += p_factor->getCost(e);
        Eigen::MatrixX<double> Rinv = p_factor->getWeight().cast<double>();

        Eigen::MatrixX<double> weight;
        // double dt_k_tau = node_ts.time - node0.time;
        // double ds_n_sigma = node_ts.arclength - node0.arclength;
        // if (dt_k_tau > TOLERANCE && ds_n_sigma < TOLERANCE)
        // {
        //     // Account for the uncertainty of the interpolation
        // throw std::runtime_error("Not implemented. Node1 undefined.");
        //     double dt_tau_k1 = node1.time - node_ts.time;
        //     Eigen::Matrix<double, 18, 18> Q_k_tau_inv = getQtInv<double>(dt_k_tau, m_hyperparameters.Q1, m_hyperparameters.Q3);
        //     Eigen::Matrix<double, 18, 18> Q_tau_k1_inv = getQtInv<double>(dt_tau_k1, m_hyperparameters.Q1, m_hyperparameters.Q3);
        //     Eigen::Matrix<double, 18, 18> Phi_tau = getPhiT<double>(dt_k_tau);
        //     Eigen::Matrix<double, 18, 18> Sigma_tau_inv = Q_k_tau_inv + Phi_tau.transpose() * Q_tau_k1_inv * Phi_tau;

        //     // Update the weight matrix
        //     Eigen::MatrixX<double> update = G * Sigma_tau_inv.inverse() * G.transpose();

        //     weight = (invertDiagonal(Rinv) + update).inverse();
        // }
        // else if (ds_n_sigma > TOLERANCE && dt_k_tau < TOLERANCE)
        // {
        //     // Account for the uncertainty of the interpolation
        //     double ds_sigma_n1 = node1.arclength - node_ts.arclength;
        // throw std::runtime_error("Not implemented. Node1 undefined.");
        //     Eigen::Matrix<double, 18, 18> Q_n_sigma_inv = getQsInv<double>(ds_n_sigma, m_hyperparameters.Q2, m_hyperparameters.Q3, node0);
        //     Eigen::Matrix<double, 18, 18> Q_sigma_n1_inv = getQsInv<double>(ds_sigma_n1, m_hyperparameters.Q2, m_hyperparameters.Q3, node0);
        //     Eigen::Matrix<double, 18, 18> Phi_sigma = getPhiS<double>(ds_n_sigma, node0);
        //     Eigen::Matrix<double, 18, 18> Sigma_sigma_inv = Q_n_sigma_inv + Phi_sigma.transpose() * Q_sigma_n1_inv * Phi_sigma;

        //     // Update the weight matrix
        //     Eigen::MatrixX<double> update = G * Sigma_sigma_inv.inverse() * G.transpose();

        //     weight = (invertDiagonal(Rinv) + update).inverse();
        // }
        // else if (dt_k_tau > TOLERANCE && ds_n_sigma > TOLERANCE)
        // {
        //     throw std::runtime_error("Spatiotemporal measurement interpolation not implemented yet");
        // }
        // else
        {
            weight = Rinv;
        }

        // Define indexing depending on robot and node ID
        const Eigen::MatrixX<double> A_block = S.transpose() * weight * S;
        const Eigen::MatrixX<double> b_block = -S.transpose() * weight * e.cast<double>();

        // Save the coefficients and entries in A and b
        // Precompute the indices for row and column blocks
        std::vector<unsigned int> row_indices;
        row_indices.push_back(getOptimizationIndex(n, k, topology));
        if (n1 != (unsigned int)(-1)) // If interpolation in space is required
            row_indices.push_back(getOptimizationIndex(n1, k, topology));
        if (k1 != (unsigned int)(-1)) // If interpolation in time is required
            row_indices.push_back(getOptimizationIndex(n, k1, topology));
        if (n1 != (unsigned int)(-1) && k1 != (unsigned int)(-1)) // If interpolation in space and time is required
            row_indices.push_back(getOptimizationIndex(n1, k1, topology));

        // Save the coefficients and entries in A and b
        for (int row_block = 0; row_block < A_block.rows(); row_block++)
        {
            unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);

            for (int col_block = 0; col_block < A_block.cols(); col_block++)
            {
                unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
            }

            b_tripletList.emplace_back(idx_row, 0, b_block(row_block, 0));
        }

        return true;
    }
    void sparseAddIdentity(std::vector<Eigen::Triplet<double>> &tripletList, unsigned int row, unsigned int col, unsigned int size)
    {
        for (unsigned int i = 0; i < size; i++)
        {
            tripletList.emplace_back(row + i, col + i, 1.0);
        }
    }

    Eigen::SparseMatrix<double> Estimator::constructProjectionMatrix()
    {
        int total_nodes = m_robot_topology.N * m_robot_topology.K;
        int total_terms = m_robot_topology.N * (m_robot_topology.K);
        int num_cols = 18 * total_nodes;
        int num_rows = 18 * total_terms;

        // info() << "Number of rows in projection matrix: " << num_rows;
        // info() << "Number of columns in projection matrix: " << num_cols;

        // Check if first and/or last pose are locked (thus, won't be updated)
        if (m_robot_topology.lock_first_pose)
        {
            num_rows = num_rows - 15 * (m_robot_topology.K); // Substract initial pose of robot i (xi, varpi, and linear strain)
        }
        else if (m_robot_topology.lock_first_position)
        {
            num_rows = num_rows - 9 * (m_robot_topology.K); // Substract initial position of robot i (xi, varpi, and linear strain)
        }

        // If we consider Kirchhoff rods, the translational strain variables of each node are locked won't be updated
        if (m_options.kirchhoff_rods)
        {
            num_rows = num_rows - 3 * total_terms;
        }

        if (m_robot_topology.lock_last_strain)
        {
            // Locks epsilon and eta at n=0
            if (m_options.kirchhoff_rods)
                num_rows = num_rows - 3 * (m_robot_topology.K); // Substract initial strain of robot i
            else
                num_rows = num_rows - 6 * (m_robot_topology.K); // Substract initial strain of robot i
        }

        // info() << "Number of rows in projection matrix after locking: " << num_rows;
        // info() << "Number of columns in projection matrix after locking: " << num_cols;

        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(num_rows);

        // Fill in the projection matrix
        int P_idx = 0;
        for (unsigned int i = 0; i < m_robot_topology.N * m_robot_topology.K; i++)
        {
            // Fill in the projection matrix accordingly
            unsigned int p_offset = 0;
            int Id_idx = i * 18;

            bool lock_strain = ((i % m_robot_topology.N == (m_robot_topology.N - 1)) && m_robot_topology.lock_last_strain);
            bool lock_translational_strain = m_options.kirchhoff_rods;
            bool lock_pose = ((i % m_robot_topology.N == 0) && m_robot_topology.lock_first_pose);
            bool lock_position = ((i % m_robot_topology.N == 0) && m_robot_topology.lock_first_position);

            if (lock_pose)
            {
                info() << "Locking pose at node: " << i;
            }
            else if (lock_position)
            {
                info() << "Locking position at node: " << i;
                // Update rot part of pose
                sparseAddIdentity(tripletList, P_idx + p_offset, Id_idx + 3, 3);
                p_offset += 3;
            }
            else
            {
                // Update pose
                sparseAddIdentity(tripletList, P_idx + p_offset, Id_idx, 6);
                p_offset += 6;
            }

            if (lock_strain)
            {
                info() << "Locking strain at node: " << i;
                // Update nothing
            }
            else if (lock_translational_strain || lock_pose || lock_position)
            {
                info() << "Locking translational strain at node: " << i;
                // Update only rot strain
                sparseAddIdentity(tripletList, P_idx + p_offset, Id_idx + 9, 3);
                p_offset += 3;
            }
            else
            {
                // Update full strain
                sparseAddIdentity(tripletList, P_idx + p_offset, Id_idx + 6, 6);
                p_offset += 6;
            }

            if (lock_pose)
            {
                info() << "Locking velocity at node: " << i;
                // Update nothing
            }
            else if (lock_position)
            {
                info() << "Locking translational velocity at node: " << i;
                // Update velocity
                sparseAddIdentity(tripletList, P_idx + p_offset, Id_idx + 15, 3);
                p_offset += 3;
            }
            else
            {
                // Update velocity
                sparseAddIdentity(tripletList, P_idx + p_offset, Id_idx + 12, 6);
                p_offset += 6;
            }

            P_idx += p_offset;
        }

        Eigen::SparseMatrix<double> P(num_rows, num_cols);
        P.setFromTriplets(tripletList.begin(), tripletList.end());

        return P;
    }

    Eigen::VectorXd Estimator::solveLinearSystem(const Eigen::SparseMatrix<double> &A, const Eigen::SparseMatrix<double> &b, bool initialize)
    {
        int size = m_robot_topology.N * m_robot_topology.K * 18;
        Eigen::MatrixX<double> P(size, size);
        P.setIdentity();
        return solveLinearSystem(A, b, P.sparseView(), initialize);
    }
    Eigen::VectorXd Estimator::solveLinearSystem(const Eigen::SparseMatrix<double> &A, const Eigen::SparseMatrix<double> &b, const Eigen::SparseMatrix<double> &P, bool initialize)
    {
        // Apply projection matrix
        Eigen::SparseMatrix<double> A_projected = P * A * P.transpose();
        const Eigen::MatrixX<double> b_projected = P * b;

        if (m_options.LM_damping > 0.0)
        {
            Eigen::SparseMatrix<double> I(A_projected.rows(), A_projected.rows());
            I.setIdentity();

            A_projected = A_projected + m_options.LM_damping * I;
        }

        if (initialize)
        {
            m_solver.analyzePattern(A_projected);
            m_solver.factorize(A_projected);

            // assert(m_solver.vectorD().minCoeff() > 1e-5 && "System matrix is not invertible, state estimation problem not well defined (i.e. not enough constraints or measurements)");
        }
        else
        {
            m_solver.factorize(A_projected);
        }
        if (m_solver.info() != Eigen::Success)
        {
            // printSparsity(A_projected.toDense(), "A_projected");
            error("Solver factorization failed. Analyzing matrix properties...");

            // Check if the matrix is square
            if (A_projected.rows() != A_projected.cols())
            {
                std::cerr << "Matrix is not square. Rows: " << A_projected.rows() << ", Cols: " << A_projected.cols();
            }

            // Check if the matrix is symmetric
            if (!A_projected.isApprox(A_projected.transpose()))
            {
                std::cerr << "Matrix is not symmetric.";
            }

            // Check for sparsity pattern issues
            info() << "Non-zero elements in matrix: " << A_projected.nonZeros();

            // Check for positive definiteness
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<double>> eigen_solver(A_projected);
            if (eigen_solver.info() != Eigen::Success)
            {
                std::cerr << "Eigenvalue computation failed.";
            }

            auto eigenvalues = eigen_solver.eigenvalues();
            if ((eigenvalues.array() <= 0).any())
            {
                std::cerr << "Matrix is not positive definite. Smallest eigenvalue: " << eigenvalues.minCoeff();
            }
            else
            {
                info() << "Matrix is positive definite. Smallest eigenvalue: " << eigenvalues.minCoeff();
            }

            // Print sparsity pattern for debugging
            printSparsity(A_projected.toDense(), "A_projected", 6);
            throw std::runtime_error("Solver factorization failed.");
        }

        Eigen::VectorXd dx = m_solver.solve(b_projected);
        if (m_solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Solver solve failed.");
        }

        // Undo projection matrix
        dx = P.transpose() * dx;

        return dx;
    }

    void Estimator::perturbState(SystemState<DTYPE> &state, double scale) //unused?
    {
        std::normal_distribution<double> dist(0, 1);
        auto gaussian = [&](double)
        { return dist(m_rng); };

        error("TOFIX: Perturbation should be independent of topology.K");
        int rows = 18 * m_robot_topology.K * m_robot_topology.N;
        Eigen::VectorX<double> random_vector = Eigen::VectorX<double>::NullaryExpr(rows, 1, gaussian);
        Eigen::VectorX<double> random_vector_projected = m_P.transpose() * m_P * random_vector;

        Eigen::VectorX<double> dx = random_vector_projected * scale;

        updateStateVariables(state, dx);
    }

    void Estimator::updateStateVariables(SystemState<DTYPE> &state, Eigen::VectorXd &dx)
    {
        assert(dx.rows() == 18 * m_robot_topology.K * m_robot_topology.N && "dx has incorrect dimensions.");
#pragma omp parallel for
        for (unsigned int n = 0; n < m_robot_topology.N; n++)
            for (unsigned int k = 0; k < m_robot_topology.K; k++)
            {
                unsigned int i_state = k * m_robot_topology.N + n;
                unsigned int i_update = getOptimizationIndex(n, k, m_robot_topology);
                auto &node = state.estimation_nodes[i_state];

                Eigen::VectorX<DTYPE> dx_block = dx.block<18, 1>(i_update, 0).cast<DTYPE>();
                node.pose = se3::vec2tran(dx_block.block<6, 1>(0, 0)) * node.pose;
                node.epsilon += dx_block.block<6, 1>(6, 0);
                node.varpi += dx_block.block<6, 1>(12, 0);
            }
    }

    Eigen::Matrix<DTYPE, 18, 36> Estimator::computeLambdaPsiS(double s_i, double s_n, double s_n1, const SystemState<DTYPE>::Node &node) const
    {
        // Space based components (arclength)
        const double ds_i = s_i - s_n;
        const double ds = s_n1 - s_n;
        Eigen::Matrix<DTYPE, 18, 18> lambda_s, psi_s;

        if (ds_i == 0 || ds == 0)
        {
            lambda_s = EYE<DTYPE>(18);
            psi_s = ZERO<DTYPE>(18);
        }
        else if (ds - ds_i == 0)
        {
            lambda_s = ZERO<DTYPE>(18);
            psi_s = EYE<DTYPE>(18);
        }
        else
        {
            Eigen::Matrix<DTYPE, 18, 18> Qs_inv = getQsInv(ds, m_hyperparameters.Q2, m_hyperparameters.Q3, node), Qs_i = getQs(ds_i, m_hyperparameters.Q2, m_hyperparameters.Q3, node);
            Eigen::Matrix<DTYPE, 18, 18> phis_n_i, phis_i_n1, phis_n_n1;
            phis_n_i << getPhiS(s_i - s_n, node);
            phis_i_n1 << getPhiS(s_n1 - s_i, node);
            phis_n_n1 << getPhiS(ds, node);
            psi_s = Qs_i * phis_i_n1.transpose() * Qs_inv;
            lambda_s = phis_n_i - psi_s * phis_n_n1;
        }

        // Space-time based components
        Eigen::Matrix<DTYPE, 18, 36> lambda_psi;
        lambda_psi << lambda_s, psi_s;

        return lambda_psi;
    }

    Eigen::Matrix<DTYPE, 18, 36> Estimator::computeLambdaPsiT(double t_i, double t_k, double t_k1) const
    {
        // throw std::runtime_error("computeLambdaPsi not implemented yet.");
        // Time based components
        const double dt_i = t_i - t_k;
        const double dt = t_k1 - t_k;
        Eigen::Matrix<DTYPE, 18, 18> lambda_t, psi_t;

        if (dt_i == 0 || dt == 0) // Node is on the edge of the estimation window or only operating over space
        {
            lambda_t = EYE<DTYPE>(18);
            psi_t = ZERO<DTYPE>(18);
        }
        else if (dt - dt_i == 0) // Node is on the edge of the estimation window, k1 == ki
        {
            lambda_t = ZERO<DTYPE>(18);
            psi_t = EYE<DTYPE>(18);
        }
        else
        {
            Eigen::Matrix<DTYPE, 18, 18> Qt_inv = getQtInv(dt, m_hyperparameters.Q1, m_hyperparameters.Q3), Qt_i = getQt(dt_i, m_hyperparameters.Q1, m_hyperparameters.Q3);
            Eigen::Matrix<DTYPE, 18, 18> phit_k_i, phit_i_k1, phit_k_k1;
            phit_k_i << getPhiT(t_i - t_k);
            phit_i_k1 << getPhiT(t_k1 - t_i);
            phit_k_k1 << getPhiT(dt);
            psi_t = Qt_i * phit_i_k1.transpose() * Qt_inv;
            lambda_t = phit_k_i - psi_t * phit_k_k1;
        }

        // Space-time based components
        Eigen::Matrix<DTYPE, 18, 36> lambda_psi;
        lambda_psi << lambda_t, psi_t;

        return lambda_psi;
    }

    void Estimator::interpolateMean2D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE> &state, Eigen::MatrixXd &jac, bool compute_jac) const
    {
        unsigned int k, n;
        getNKIndeces(query_node.arclength, query_node.time, n, k, state, m_robot_topology);

        assert((n != (unsigned int)-1) && "ERR: Query arclength not on robot.");
        assert((k != (unsigned int)-1) && "ERR: Query time not in estimation window.");

        const unsigned int max_K = state.getK(m_robot_topology);

        // Avoid invalid neighbor access on degenerate grids and time/space boundaries.
        if (m_robot_topology.N <= 1 || max_K <= 1)
        {
            const auto &node00 = state.estimation_nodes[k * m_robot_topology.N + n];

            if (m_robot_topology.N <= 1 && max_K <= 1)
            {
                query_node = node00;
                if (compute_jac)
                {
                    jac.resize(18, 18);
                    jac.setZero();
                    jac.block<18, 18>(0, 0).setIdentity();
                }
                return;
            }

            if (m_robot_topology.N <= 1)
            {
                if (std::abs(query_node.time - node00.time) < TOLERANCE || (k + 1) >= max_K)
                {
                    query_node = node00;
                    if (compute_jac)
                    {
                        jac.resize(18, 18);
                        jac.setZero();
                        jac.block<18, 18>(0, 0).setIdentity();
                    }
                    return;
                }

                const auto &node01 = state.estimation_nodes[(k + 1) * m_robot_topology.N + n];
                interpolateMean1D(query_node, node00, node01, jac, compute_jac);
                return;
            }

            if (std::abs(query_node.arclength - node00.arclength) < TOLERANCE || (n + 1) >= m_robot_topology.N)
            {
                query_node = node00;
                if (compute_jac)
                {
                    jac.resize(18, 18);
                    jac.setZero();
                    jac.block<18, 18>(0, 0).setIdentity();
                }
                return;
            }

            const auto &node10 = state.estimation_nodes[k * m_robot_topology.N + (n + 1)];
            interpolateMean1D(query_node, node00, node10, jac, compute_jac);
            return;
        }

        const unsigned int k1 = k + 1;
        const unsigned int n1 = n + 1;

        interpolateMean2D(query_node, state.estimation_nodes[k * m_robot_topology.N + n], state.estimation_nodes[k1 * m_robot_topology.N + n], state.estimation_nodes[k * m_robot_topology.N + n1], state.estimation_nodes[k1 * m_robot_topology.N + n1], jac, compute_jac);
    }

    void Estimator::interpolateMean2D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE>::Node &node00, const SystemState<DTYPE>::Node &node01, const SystemState<DTYPE>::Node &node10, const SystemState<DTYPE>::Node &node11, Eigen::MatrixXd &jac, bool compute_jac) const
    {
        /*
        node00: (s_n, t_k)
        node01: (s_n, t_k1)
        node10: (s_n1, t_k)
        node11: (s_n1, t_k1)
        */
        if (std::abs(query_node.arclength - node00.arclength) < TOLERANCE && std::abs(query_node.time - node00.time) < TOLERANCE)
        {
            // No interpolation required
            query_node = node00;
            if (compute_jac)
            {
                jac.resize(18, 36);
                jac.setZero();
                jac.block<18, 18>(0, 0).setIdentity();
            }
            return;
        }
        else if (std::abs(query_node.arclength - node00.arclength) < TOLERANCE)
        {
            // Interpolation over time
            interpolateMean1D(query_node, node00, node01, jac, compute_jac);
            return;
        }
        else if (std::abs(query_node.time - node00.time) < TOLERANCE)
        {
            // Interpolation over space
            interpolateMean1D(query_node, node00, node10, jac, compute_jac);
            return;
        }

        // Interpolation over space and time
        // First, interpolate over space
        SystemState<DTYPE>::Node node_time_interp_t0, node_time_interp_t1;
        node_time_interp_t0.arclength = query_node.arclength;
        node_time_interp_t0.time = node00.time;
        node_time_interp_t1.arclength = query_node.arclength;
        node_time_interp_t1.time = node11.time;

        Eigen::MatrixXd dt0_d0010, dt1_d0111;
        interpolateMean1D(node_time_interp_t0, node00, node10, dt0_d0010, compute_jac);
        interpolateMean1D(node_time_interp_t1, node01, node11, dt1_d0111, compute_jac);

        // Finally, interpolate over time between the two space-interpolated nodes
        Eigen::MatrixXd dq_dt0t1;
        interpolateMean1D(query_node, node_time_interp_t0, node_time_interp_t1, dq_dt0t1, compute_jac);

        if (compute_jac)
        {
            jac.resize(18, 72);
            // Chain rule to compute full jacobian
            // dq / d[node00, node10, node01, node11]
            jac.block<18, 36>(0, 0) = dq_dt0t1.block<18, 18>(0, 0) * dt0_d0010;
            jac.block<18, 36>(0, 36) = dq_dt0t1.block<18, 18>(0, 18) * dt1_d0111;
        }
    }

    void Estimator::interpolateMean1D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE>::Node &node0, const SystemState<DTYPE>::Node &node1, Eigen::MatrixXd &jac, bool compute_jac) const
    {
        if (std::abs(query_node.arclength - node0.arclength) < TOLERANCE && std::abs(query_node.time - node0.time) < TOLERANCE)
        {
            // No interpolation required
            query_node = node0;
            return;
        }

        Eigen::Matrix<DTYPE, 18, 36> lambda_psi;
        if (std::abs(query_node.arclength - node0.arclength) < TOLERANCE)
        {
            // Interpolation over time
            const double t_k = node0.time;
            const double t_k1 = node1.time;
            const double t_i = query_node.time;

            lambda_psi = computeLambdaPsiT(t_i, t_k, t_k1);
        }
        else if (std::abs(query_node.time - node0.time) < TOLERANCE)
        {
            // Interpolation over space
            const double s_n = node0.arclength;
            const double s_n1 = node1.arclength;
            const double s_i = query_node.arclength;

            lambda_psi = computeLambdaPsiS(s_i, s_n, s_n1, node0);
        }
        else
        {
            throw std::runtime_error("interpolateMean1D called for spatiotemporal interpolation.");
        }

        Eigen::Vector<DTYPE, 18> g_hat0, g_hat1;
        Eigen::Matrix<double, 18, 18> dg0dx0, dg0dx1, dg1dx0, dg1dx1;
        globalToLocal(node0, node0, g_hat0, dg0dx0, dg0dx1, compute_jac, m_options.use_autodiff);
        globalToLocal(node1, node0, g_hat1, dg1dx0, dg1dx1, compute_jac, m_options.use_autodiff);

        Eigen::Vector<DTYPE, 36> g_hat;
        g_hat << g_hat0, g_hat1;

        const Eigen::Vector<DTYPE, 18> g_hat_st = lambda_psi * g_hat;
        localToGlobal(g_hat_st, node0, query_node);

        if (!compute_jac)
            return;

        // Compute the Jacobian
        if (m_options.use_autodiff)
        {
#ifdef USE_AUTODIFF
            auto _interp = [&](const Eigen::Vector<DTYPE, 36> &perturb)
            {
                SystemState<DTYPE>::Node _node0, _node1;
                const Eigen::Vector<DTYPE, 18> &perturb0 = perturb.block<18, 1>(0, 0);
                const Eigen::Vector<DTYPE, 18> &perturb1 = perturb.block<18, 1>(18, 0);
                perturbNode(node0, perturb0, _node0);
                perturbNode(node1, perturb1, _node1);

                SystemState<DTYPE>::Node _query_node = query_node;
                interpolateMean1D(_query_node, _node0, _node1);

                Eigen::Vector<DTYPE, 18> ret_perturb;
                getPerturbation(_query_node, query_node, ret_perturb);
                return ret_perturb;
            };

            Eigen::VectorX<DTYPE> e;
            Eigen::Vector<DTYPE, 36> perturb = Eigen::Vector<DTYPE, 36>::Zero();
            autodiff::jacobian(_interp, autodiff::wrt(perturb), autodiff::at(perturb), e, jac); // Get interp jacobian
#else
            throw std::runtime_error("Autodiff not enabled");
#endif
        }
        else
        {
            // Compute Jacobian
            Eigen::Matrix<double, 36, 36> dgdx;
            dgdx << dg0dx0, dg0dx1,
                dg1dx0, dg1dx1;
            const Eigen::Matrix<double, 18, 36> lambda_dgdx = lambda_psi.cast<double>() * dgdx;

            Eigen::Matrix<double, 6, 18> P1 = Eigen::Matrix<double, 6, 18>::Zero();
            Eigen::Matrix<double, 6, 18> P2 = Eigen::Matrix<double, 6, 18>::Zero();
            Eigen::Matrix<double, 6, 18> P3 = Eigen::Matrix<double, 6, 18>::Zero();
            P1.block<6, 6>(0, 0).setIdentity();
            P2.block<6, 6>(0, 6).setIdentity();
            P3.block<6, 6>(0, 12).setIdentity();

            const Eigen::Vector<double, 6> xi_st = g_hat_st.block<6, 1>(0, 0).cast<double>();
            const Eigen::Matrix<double, 4, 4> tran_st = se3::vec2tran(xi_st);
            const Eigen::Matrix<double, 6, 6> jac_st = se3::vec2jac(xi_st);

            Eigen::Matrix<double, 6, 36> dxst_dx1, dxst_dx2, dxst_dx3;
            dxst_dx1 << jac_st * P1 * lambda_dgdx;
            dxst_dx1.block<6, 6>(0, 0) = dxst_dx1.block<6, 6>(0, 0) + se3::tranAd(tran_st);
            dxst_dx2 << (jac_st * P2 - 0.5 * se3::curlyhat(P2 * g_hat_st.cast<double>()) * P1) * lambda_dgdx;
            dxst_dx3 << (jac_st * P3 - 0.5 * se3::curlyhat(P3 * g_hat_st.cast<double>()) * P1) * lambda_dgdx;

            jac.resize(18, 36);
            jac << dxst_dx1, dxst_dx2, dxst_dx3;
        }
    }

    void Estimator::queryState(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE> &state) const
    {
        // New method
        unsigned int k, n;
        getNKIndeces(query_node.arclength, query_node.time, n, k, state, m_robot_topology);

        assert((n != (unsigned int)-1) && "ERR: Query arclength not on robot.");
        assert((k != (unsigned int)-1) && "ERR: Query time not in estimation window.");

        const unsigned int k1 = k + 1;
        const unsigned int n1 = n + 1;

        const SystemState<DTYPE>::Node &node00 = state.estimation_nodes[k * m_robot_topology.N + n];
        const SystemState<DTYPE>::Node &node01 = state.estimation_nodes[k1 * m_robot_topology.N + n];
        const SystemState<DTYPE>::Node &node10 = state.estimation_nodes[k * m_robot_topology.N + n1];
        const SystemState<DTYPE>::Node &node11 = state.estimation_nodes[k1 * m_robot_topology.N + n1];

        if (std::abs(query_node.arclength - node00.arclength) < TOLERANCE && std::abs(query_node.time - node00.time) < TOLERANCE)
        {
            query_node = node00;
            return;
        }

        Eigen::MatrixXd jac_k_tau, jac_tau_k1, covariance_square;
        Eigen::VectorX<DTYPE> e;
        Eigen::Matrix<double, 18, 18> F_k_tau, F_tau_k1, E_k_tau, E_tau_k1, Q_k_tau_inv, Q_tau_k1_inv;

        if (std::abs(query_node.arclength - node00.arclength) < TOLERANCE)
        {
            // Interpolation over time
            interpolateMean1D(query_node, node00, node01);

            jac_k_tau = m_binary_time_factor->getJacobian({node00, query_node}, e);
            jac_tau_k1 = m_binary_time_factor->getJacobian({query_node, node01}, e);
            Q_k_tau_inv = getQtInv(query_node.time - node00.time, m_hyperparameters.Q1, m_hyperparameters.Q3).cast<double>();
            Q_tau_k1_inv = getQtInv(node01.time - query_node.time, m_hyperparameters.Q1, m_hyperparameters.Q3).cast<double>();
            covariance_square = node00.getCovarianceSquareTime();
        }
        else if (std::abs(query_node.time - node00.time) < TOLERANCE)
        {
            // Interpolation over space
            interpolateMean1D(query_node, node00, node10);

            jac_k_tau = m_binary_space_factor->getJacobian({node00, query_node}, e);
            jac_tau_k1 = m_binary_space_factor->getJacobian({query_node, node10}, e);
            Q_k_tau_inv = getQsInv(query_node.arclength - node00.arclength, m_hyperparameters.Q2, m_hyperparameters.Q3, node00).cast<double>();
            Q_tau_k1_inv = getQsInv(node10.arclength - query_node.arclength, m_hyperparameters.Q2, m_hyperparameters.Q3, node00).cast<double>();
            covariance_square = node00.getCovarianceSquareSpace();
        }
        else
        {
            // Interpolation over space and time

            // Interpolate intermediate nodes over space first
            SystemState<DTYPE>::Node node_time_interp_t0, node_time_interp_t1;
            node_time_interp_t0.arclength = query_node.arclength;
            node_time_interp_t0.time = node00.time;
            node_time_interp_t1.arclength = query_node.arclength;
            node_time_interp_t1.time = node11.time;
            interpolateMean1D(node_time_interp_t0, node00, node10);
            interpolateMean1D(node_time_interp_t1, node01, node11);

            Eigen::MatrixXd jac_k_n_sigma, jac_k_sigma_n1;
            Eigen::MatrixXd jac_k1_n_sigma, jac_k1_sigma_n1;

            jac_k_n_sigma = m_binary_space_factor->getJacobian({node00, node_time_interp_t0}, e);
            jac_k_sigma_n1 = m_binary_space_factor->getJacobian({node_time_interp_t0, node10}, e);
            jac_k1_n_sigma = m_binary_space_factor->getJacobian({node01, node_time_interp_t1}, e);
            jac_k1_sigma_n1 = m_binary_space_factor->getJacobian({node_time_interp_t1, node11}, e);

            /*
            Here I need to create a stacked system of F, Es, and Qs to compute the covariance square at the interpolated nodes

            Sigma : 36 x 36
            covariance_square : 72 x 72
            D : 36 x 72
            E : 36 x 36
            F : 36 x 36
            Q_inv : 36 x 36
            */

            Eigen::Matrix<double, 36, 36> F_n_sigma_stacked, F_sigma_n1_stacked, E_n_sigma_stacked, E_sigma_n1_stacked, Q_n_sigma_inv_stacked, Q_sigma_n1_inv_stacked;

            F_n_sigma_stacked << jac_k_n_sigma.block<18, 18>(0, 0), Eigen::Matrix<double, 18, 18>::Zero(),
                Eigen::Matrix<double, 18, 18>::Zero(), jac_k1_n_sigma.block<18, 18>(0, 0);
            F_sigma_n1_stacked << jac_k_sigma_n1.block<18, 18>(0, 0), Eigen::Matrix<double, 18, 18>::Zero(),
                Eigen::Matrix<double, 18, 18>::Zero(), jac_k1_sigma_n1.block<18, 18>(0, 0);
            E_n_sigma_stacked << -jac_k_n_sigma.block<18, 18>(0, 18), Eigen::Matrix<double, 18, 18>::Zero(),
                Eigen::Matrix<double, 18, 18>::Zero(), -jac_k1_n_sigma.block<18, 18>(0, 18);
            E_sigma_n1_stacked << -jac_k_sigma_n1.block<18, 18>(0, 18), Eigen::Matrix<double, 18, 18>::Zero(),
                Eigen::Matrix<double, 18, 18>::Zero(), -jac_k1_sigma_n1.block<18, 18>(0, 18);

            Q_n_sigma_inv_stacked << getQsInv(node_time_interp_t0.arclength - node00.arclength, m_hyperparameters.Q2, m_hyperparameters.Q3, node00).cast<double>(), Eigen::Matrix<double, 18, 18>::Zero(),
                Eigen::Matrix<double, 18, 18>::Zero(), getQsInv(node_time_interp_t1.arclength - node01.arclength, m_hyperparameters.Q2, m_hyperparameters.Q3, node00).cast<double>();
            Q_sigma_n1_inv_stacked << getQsInv(node10.arclength - node_time_interp_t0.arclength, m_hyperparameters.Q2, m_hyperparameters.Q3, node00).cast<double>(), Eigen::Matrix<double, 18, 18>::Zero(),
                Eigen::Matrix<double, 18, 18>::Zero(), getQsInv(node11.arclength - node_time_interp_t1.arclength, m_hyperparameters.Q2, m_hyperparameters.Q3, node00).cast<double>();

            Eigen::Matrix<double, 36, 36> Sigma;
            Sigma << (E_n_sigma_stacked.transpose() * Q_n_sigma_inv_stacked * E_n_sigma_stacked + F_sigma_n1_stacked.transpose() * Q_sigma_n1_inv_stacked * F_sigma_n1_stacked).inverse();
            Eigen::Matrix<double, 36, 72> D;
            D << E_n_sigma_stacked.transpose() * Q_n_sigma_inv_stacked * F_n_sigma_stacked, F_sigma_n1_stacked.transpose() * Q_sigma_n1_inv_stacked * E_sigma_n1_stacked;

            Eigen::Matrix<double, 36, 36> P_sigma;
            covariance_square = Sigma + Sigma * D * node00.covariance_square * D.transpose() * Sigma;

            // Interpolate over time between the two space-interpolated nodes
            interpolateMean1D(query_node, node_time_interp_t0, node_time_interp_t1);
            jac_k_tau = m_binary_time_factor->getJacobian({node_time_interp_t0, query_node}, e);
            jac_tau_k1 = m_binary_time_factor->getJacobian({query_node, node_time_interp_t1}, e);
            Q_k_tau_inv = getQtInv(query_node.time - node_time_interp_t0.time, m_hyperparameters.Q1, m_hyperparameters.Q3).cast<double>();
            Q_tau_k1_inv = getQtInv(node_time_interp_t1.time - query_node.time, m_hyperparameters.Q1, m_hyperparameters.Q3).cast<double>();
        }

        F_k_tau = jac_k_tau.block<18, 18>(0, 0);
        F_tau_k1 = jac_tau_k1.block<18, 18>(0, 0);

        E_k_tau = -jac_k_tau.block<18, 18>(0, 18);
        E_tau_k1 = -jac_tau_k1.block<18, 18>(0, 18);

        Eigen::Matrix<double, 18, 18> Sigma;
        Sigma << (E_k_tau.transpose() * Q_k_tau_inv * E_k_tau + F_tau_k1.transpose() * Q_tau_k1_inv * F_tau_k1).inverse();
        Eigen::Matrix<double, 18, 36> D;
        D << E_k_tau.transpose() * Q_k_tau_inv * F_k_tau, F_tau_k1.transpose() * Q_tau_k1_inv * E_tau_k1;

        Eigen::Matrix<double, 18, 18> P_tau;
        P_tau = Sigma + Sigma * D * covariance_square * D.transpose() * Sigma;

        query_node.covariance_square = P_tau;
    }

    void Estimator::interpolateStates(SystemState<DTYPE> &state)
    {
        // throw std::runtime_error("interpolateStates not implemented yet.");
        state.interpolation_nodes.clear();
        std::vector<SystemState<DTYPE>::Node> _interpolation_nodes;

// Iterate through all but last time step
#pragma omp declare reduction(merge_node_list : std::vector<SystemState<DTYPE>::Node> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<SystemState<DTYPE>::Node>())
#pragma omp parallel for reduction(merge_node_list : _interpolation_nodes)
        for (unsigned int i = 0; i < state.estimation_nodes.size(); i++)
        {
            if (m_robot_topology.Mt == 1 && m_robot_topology.Ms == 1)
            {
                // Saves some compute if no interpolation required
                SystemState<DTYPE>::Node node = state.estimation_nodes[i];
                _interpolation_nodes.push_back(node);
                continue;
            }

            // throw std::runtime_error("interpolation not implemented yet.");

            const unsigned int k = std::floor(i / m_robot_topology.N);
            const unsigned int n = i - k * m_robot_topology.N;

            // Clips +1 terms to ensure that outer edge is added to interpolation nodes
            const unsigned int k1 = ((k + 1) * m_robot_topology.N >= (unsigned int)state.estimation_nodes.size()) ? k : k + 1;
            const unsigned int n1 = (n + 1 >= m_robot_topology.N) ? n : n + 1;

            const double tk = state.estimation_nodes[k * m_robot_topology.N + n].time;
            const double tk1 = state.estimation_nodes[k1 * m_robot_topology.N + n].time;
            const double dt = tk1 - tk; //(tk1 - tk > 0) ? tk1 - tk : 1; // To prevent nan. Term cancels in case where tk1 == tk, interpolating in just one dimension

            const double sn = state.estimation_nodes[k * m_robot_topology.N + n].arclength;
            const double sn1 = state.estimation_nodes[k * m_robot_topology.N + n1].arclength;
            const double ds = sn1 - sn; //(sn1 - sn > 0) ? sn1 - sn : 1;

            for (unsigned int mt = 0; mt < ((k1 != k) ? m_robot_topology.Mt : 1); mt++)
            {
                const double dt_i = (double)mt / ((double)m_robot_topology.Mt) * dt;
                const double t_i = tk + dt_i;

                for (unsigned int ms = 0; ms < ((n1 != n) ? m_robot_topology.Ms : 1); ms++)
                {
                    const double ds_i = (double)ms / ((double)m_robot_topology.Ms) * ds;
                    const double s_i = sn + ds_i;

                    SystemState<DTYPE>::Node query_node;
                    query_node.arclength = s_i;
                    query_node.time = t_i;

                    if (m_options.compute_covariances)
                    {
                        assert(ms * mt == 0 && "Covariance interpolation in time and space not yet supported.");
                        queryState(query_node, state);
                    }
                    else
                        interpolateMean2D(query_node, state);

                    _interpolation_nodes.push_back(query_node);
                }
            }
        }

        state.interpolation_nodes = _interpolation_nodes;
        std::sort(state.interpolation_nodes.begin(), state.interpolation_nodes.end(), [](const SystemState<DTYPE>::Node &a, const SystemState<DTYPE>::Node &b)
                  { return (a.time < b.time) || ((a.time == b.time) && (a.arclength < b.arclength)); }); // TODO: This is inefficient. Should bake into construction
    }

    void Estimator::buildCovarianceSquare(SystemState<DTYPE>::Node &node, const Eigen::MatrixX<double> &covariance, const SystemState<DTYPE> &state, const RobotTopology &topology)
    {
        // order is: nk 00, 01, 10, 11
        unsigned int n, k;
        getNKIndeces(node.arclength, node.time, n, k, state, topology);
        unsigned int max_K = state.getK(topology);
        unsigned int n1 = n + 1;
        unsigned int k1 = k + 1;
        const int diag_offset = 18 * topology.N;

        Eigen::MatrixX<double> P_square = Eigen::MatrixX<double>::Zero(72, 72);

        /*
        Variance of nk:nk
        X O O O
        O O O O
        O O O O
        O O O O
        */
        P_square.block<18, 18>(0, 0) = covariance.block((k * topology.N + n) * 18, diag_offset + n * 18, 18, 18); // nk:nk
        if (k1 == max_K && n1 == topology.N)
        {
            node.covariance_square = P_square;
            return;
        }
        /*
        Covariance of nk1:nk, nk1:nk1, only if time step available
        O O O O
        X X O O
        O O O O
        O O O O
        */
        if (k1 < max_K)
        {
            P_square.block<18, 18>(18, 0) = covariance.block((k1 * topology.N + n) * 18, n * 18, 18, 18).eval();                // nk1:nk
            P_square.block<18, 18>(18, 18) = covariance.block((k1 * topology.N + n) * 18, diag_offset + n * 18, 18, 18).eval(); // nk1:nk1
        }
        /*
        Covariance of n1k:nk, n1k:n1k, only if space step available
        O O O O
        O O O O
        X O X O
        O O O O
        */
        if (n1 < topology.N)
        {
            P_square.block<18, 18>(36, 0) = covariance.block((k * topology.N + n1) * 18, diag_offset + n * 18, 18, 18).eval();   // n1k:nk
            P_square.block<18, 18>(36, 36) = covariance.block((k * topology.N + n1) * 18, diag_offset + n1 * 18, 18, 18).eval(); // n1k:n1k
        }
        /*
        Covariance of n1k1:nk, n1k1:nk1, n1k1:n1k, n1k1:n1k1, n1k:nk1, only if space and time step available
        O O O O
        O O O O
        O X O O
        X X X X
        */
        if (k1 < max_K && n1 < topology.N)
        {
            P_square.block<18, 18>(54, 0) = covariance.block((k1 * topology.N + n1) * 18, n * 18, 18, 18).eval();                 // n1k1:nk
            P_square.block<18, 18>(54, 18) = covariance.block((k1 * topology.N + n1) * 18, diag_offset + n * 18, 18, 18).eval();  // n1k1:nk1
            P_square.block<18, 18>(54, 36) = covariance.block((k1 * topology.N + n1) * 18, n1 * 18, 18, 18).eval();               // n1k1:n1k
            P_square.block<18, 18>(54, 54) = covariance.block((k1 * topology.N + n1) * 18, diag_offset + n1 * 18, 18, 18).eval(); // n1k1:n1k1

            P_square.block<18, 18>(36, 18) = covariance.block((k1 * topology.N + n) * 18, n1 * 18, 18, 18).transpose().eval(); // n1k:nk1 = nk1:n1k.transpose();
        }
        P_square.triangularView<Eigen::Upper>() = P_square.transpose().triangularView<Eigen::Upper>();

        node.covariance_square = P_square;
    }

    Estimator::Results Estimator::computeStateEstimate(const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements, bool verbose_mode)
    {
        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());

        m_measurement_factors = measurements;

        // Calculates size and relevant values when K is not predefined
        if (m_robot_topology.time_nodes_on_measurements)
        {
            std::vector<double> ts;
            for (const auto &measurement : m_measurement_factors)
            {
                if (!ts.empty())
                    if (measurement->getMeas().t == ts.back())
                        continue;
                ts.push_back(measurement->getMeas().t);
            }
            m_robot_topology.K = ts.size();
            m_robot_topology.T = ts.back();
            info() << "Number of time nodes: " << m_robot_topology.K;
            m_P = constructProjectionMatrix();
            m_state = constructInitialGuess(m_options.init_guess_type, ts);
        }
        // Setup the initial guess depending on chosen option
        Results results;
        results.state = m_state;

        m_sampler_initialized = false;

        // Create and resize matrices according to robot topology
        int total_nodes = m_robot_topology.K * m_robot_topology.N;
        assert(total_nodes == (int)results.state.estimation_nodes.size() && "Unexpected number of estimation nodes present in state.");

        // Create the triplet lists
        std::vector<Eigen::Triplet<double>> A_tripletList;
        std::vector<Eigen::Triplet<double>> b_tripletList;
        Eigen::SparseMatrix<double> A;
        Eigen::SparseMatrix<double> b;

        // Solve the system iteratively
        for (unsigned int iter = 0; iter < m_options.max_optimization_iterations; iter++)
        {
            std::cout << "Iteration: " << iter << std::endl;
            DTYPE cost = 0.0;
            
            // Clear triplet lists
            A_tripletList.clear();
            b_tripletList.clear();
            A.resize(18 * total_nodes, 18 * total_nodes);
            b.resize(18 * total_nodes, 1);
            /*
            if(verbose_mode)
            {
                std::cout << "A size: " << A.rows() << " x " << A.cols() << std::endl;
                std::cout << "b size: " << b.rows() << " x " << b.cols() << std::endl;
                //Print A and b
                for (int i = 0; i < A.rows(); i++)
                {
                    for (int j = 0; j < A.cols(); j++)
                    {
                        if (A.coeff(i, j) != 0)
                        {
                            std::cout << "A(" << i << "," << j << ") = " << A.coeff(i, j) << std::endl;
                        }
                    }
                }
                for (int i = 0; i < b.rows(); i++)
                {
                    if (b.coeff(i, 0) != 0)
                    {
                        std::cout << "b(" << i << ",0) = " << b.coeff(i, 0) << std::endl;
                    }
                }
            }
            */
            // Assemble prior terms
            DTYPE cost_p = 0.0;
            assemblePriorTerms(A_tripletList, b_tripletList, cost_p, results.state, m_robot_topology, false);

            // Assemble measurement terms
            DTYPE cost_m = 0.0;

            int end_meas = m_measurement_factors.size();
            for (int i = m_start_meas_idx; i < end_meas; i++)
            {
                const auto &measurement = m_measurement_factors[i]->getMeas();
                if (measurement.t < results.state.estimation_nodes.front().time - TOLERANCE)
                    m_start_meas_idx++;
                if (measurement.t > results.state.estimation_nodes.back().time + TOLERANCE)
                {
                    end_meas = i;
                    break;
                }
            }

            // Run through all of the measurements
#pragma omp declare reduction(dtype_add:DTYPE : omp_out += omp_in) initializer(omp_priv = DTYPE(0.0))
#pragma omp declare reduction(merge_triplet_list : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<Eigen::Triplet<double>>())
#pragma omp parallel for reduction(dtype_add : cost_m) reduction(merge_triplet_list : A_tripletList, b_tripletList)
            for (int i = m_start_meas_idx; i < end_meas; i++)
            {
                assembleMeasurementTerm(m_measurement_factors[i], A_tripletList, b_tripletList, cost_m, results.state, m_robot_topology, false);
            }

            DTYPE cost_ext = 0.0;
            if (m_external_terms_available) // Add external terms if applicable
                addChildFactors(A_tripletList, b_tripletList, cost_ext, results.state);

            // Add terms together
            A.setFromTriplets(A_tripletList.begin(), A_tripletList.end());
            b.setFromTriplets(b_tripletList.begin(), b_tripletList.end());

            // Save cost for current iteration
            cost = cost_p + cost_m;

            // Print information about current iteration
            if (verbose_mode)
            {
                if (iter == 0)
                {
                    // printSparsity(A, "A");
                }
                infoStart();
                std::cout << "Iteration: " << iter << std::endl;
                std::cout << "Cost: " << double(cost) << std::endl;
                std::cout << "Cost Prior: " << double(cost_p) << std::endl;
                std::cout << "Cost Measurements: " << double(cost_m) << std::endl;
                std::cout << "Cost External: " << double(cost_ext) << std::endl;
                logReset();
            }

            // Solve for dx
            bool initialize = false;
            if (iter == 0)
            {
                initialize = true;
            }

            Eigen::VectorX<double> dx = solveLinearSystem(A, b, m_P, initialize);

            Eigen::VectorX<double> p = dx;
            double m;
            m = (-dx.transpose() * p)(0, 0);
            double alpha = 1;

            // Update the state variables
            if (m_options.solver == Options::Solver::NewtonLineSearch) // Do linesearch if selected
            {
                SystemState<DTYPE> state_check;
                DTYPE new_cost;

                Eigen::VectorX<double> step;
                double c = 0.5;
                double tau = 0.5;
                int ls_iter = 0;
                int max_ls_iter = 10;

                do // As long as the cost of the next step are larger than the current cost
                {
                    // Reset state_check to current state
                    state_check = results.state;

                    // Define step
                    step = alpha * p;

                    // Apply current step to state_check
                    updateStateVariables(state_check, step);

                    // Compute the cost for the new state
                    assemblePriorTerms(A_tripletList, b_tripletList, cost_p, state_check, m_robot_topology, true);
#pragma omp declare reduction(dtype_add:DTYPE : omp_out += omp_in) initializer(omp_priv = DTYPE(0.0))
#pragma omp declare reduction(merge_triplet_list : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<Eigen::Triplet<double>>())
#pragma omp parallel for reduction(dtype_add : cost_m) reduction(merge_triplet_list : A_tripletList, b_tripletList)
                    for (auto factor : m_measurement_factors)
                    {
                        assembleMeasurementTerm(factor, A_tripletList, b_tripletList, cost_m, state_check, m_robot_topology, true);
                    }
                    new_cost = cost_m + cost_p;
                    debug() << "Line search: " << double(new_cost) << ", " << cost_m << ", " << cost_p;

                    alpha = tau * alpha; // Half the step to make for next iteration

                    ls_iter++;
                } while ((new_cost > (double(cost) + m * c * alpha) || std::isnan(double(new_cost))) && ls_iter < max_ls_iter);

                if (new_cost < (double(cost) + m * c * alpha))
                {
                    // Set the current state to the state that minimzed the cost
                    results.state = state_check;
                }
                else
                {
                    if (!m_options.LM_damping)
                    {
                        results.message << "Line search failed.";
                        return results;
                    }
                    else
                    {
                        warning() << "Line search failed. Increasing LM damping without taking a step.";
                    }
                }
            }
            else // Otherwise just apply the whole step
            {
                updateStateVariables(results.state, dx);
            }
            /*
            if(verbose_mode)
            {
                for (unsigned int i = 0; i < dx.size(); i++)
                {
                    std::cout << "dx[" << i << "] = " << dx[i] << std::endl;
                }                
            }*/

            // Check if the cost during the last couple iterations are still changing
            bool detect_convergence = false;
            if (verbose_mode)
                info() << "Convergence: " << m;
            if (m > -1 * m_options.convergence_threshold)
            {
                detect_convergence = true;
            }

            if (detect_convergence)
            {
                results.message << "Detected convergence of cost function.";
                break;
            }

            if (iter == m_options.max_optimization_iterations - 1)
            {
                results.message << "Maximum number of iterations reached.";
                return results;
            }
        }
        std::cout << "State estimation complete." << std::endl;
        /*
        if(verbose_mode)
        {
            for (unsigned int i = 0; i < results.state.estimation_nodes.size(); i++)
            {
                std::cout << "Node " << i << ": s = " << results.state.estimation_nodes[i].arclength << ", t = " << results.state.estimation_nodes[i].time << std::endl;
            }
        }
        */
        // Extract covariances and uncertainties
        if (m_options.compute_covariances)
        {
            Eigen::MatrixX<double> results_covariance;
            m_spacetime_solver.extractCovariances(m_P.transpose() * m_solver.matrixL() * m_P, m_robot_topology, results_covariance, m_covariance_timesteps_to_extract, m_options.reverse_order);

// Build covariance squares for start of window nodes
#pragma omp parallel for
            for (unsigned int i = 0; i < m_robot_topology.K * m_robot_topology.N; i++)
            {
                buildCovarianceSquare(results.state.estimation_nodes[i], results_covariance, results.state, m_robot_topology);
            }
        }

        results.success = true;

        if (verbose_mode)
            info() << "Time taken: " << (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_time).count() / 1e6 << "s";

        results.runtimes.push_back((std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_time).count());
        return results;
    }

    std::vector<SystemState<DTYPE>> Estimator::samplePrior(const RobotTopology &topology, int n)
    {
        return samplePrior(constructInitialGuess(Options::InitialGuessType::Straight), topology, n);
    }
    std::vector<SystemState<DTYPE>> Estimator::samplePrior(const SystemState<DTYPE> &state, const RobotTopology &topology, int n)
    {
        SystemState<DTYPE> _state = state;

        if (_state.ib) // Converts to T_bi
            _state.convertStateMeanBodyInertial();

        if (!m_sampler_initialized)
        { // Create the triplet lists
            std::vector<Eigen::Triplet<double>> A_tripletList;
            std::vector<Eigen::Triplet<double>> b_tripletList;

            // Assemble prior terms
            DTYPE cost_p = 0.0;
            assemblePriorTerms(A_tripletList, b_tripletList, cost_p, _state, topology, false);

            // Assemble measurement terms
            DTYPE cost_m = 0.0;
#pragma omp declare reduction(dtype_add:DTYPE : omp_out += omp_in) initializer(omp_priv = DTYPE(0.0))
#pragma omp declare reduction(merge_triplet_list : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<Eigen::Triplet<double>>())
#pragma omp parallel for reduction(dtype_add : cost_m) reduction(merge_triplet_list : A_tripletList, b_tripletList)
            for (auto factor : m_measurement_factors)
            {
                assembleMeasurementTerm(factor, A_tripletList, b_tripletList, cost_m, _state, m_robot_topology, false);
            }

            Eigen::SparseMatrix<double> A;
            A.resize(18 * topology.K * topology.N, 18 * topology.K * topology.N);
            A.setFromTriplets(A_tripletList.begin(), A_tripletList.end());

            Eigen::SparseMatrix<double> I(m_P.rows(), m_P.rows());
            I.setIdentity();

            Eigen::SparseMatrix<double> A_projected = m_P * A * m_P.transpose();
            Eigen::SparseMatrix<double> A_projected_LM = A_projected + m_options.LM_damping * I;

            m_sampling_solver.compute(A_projected_LM);
            if (m_sampling_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Failed to factorize A_projected_LM.");
            }

            if (m_options.compute_covariances)
            {
                Eigen::SparseMatrix<double> L_projected = m_sampling_solver.matrixL();
                Eigen::SparseMatrix<double> L = m_P.transpose() * L_projected * m_P;
                m_spacetime_solver.extractCovariances(L.cast<double>(), topology, m_sampler_covariances);
            }

            Eigen::SparseMatrix<double> P = m_sampling_solver.solve(I);

            m_sampling_solver.compute(P);
            if (m_sampling_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Failed to factorize P.");
            }
            info("Sampler initialized.");
            m_sampler_initialized = true;
        }

        std::normal_distribution<double> dist(0, 1);
        auto gaussian = [&](double)
        { return dist(m_rng); };

        Eigen::MatrixX<double> random_vector = Eigen::MatrixX<double>::NullaryExpr(m_sampling_solver.matrixL().cols(), n, gaussian);
        Eigen::MatrixX<double> x_projected = m_sampling_solver.matrixL() * random_vector.cast<double>();

        Eigen::MatrixX<double> x = m_P.transpose() * x_projected;

        std::vector<SystemState<DTYPE>> samples;
        for (int i = 0; i < n; i++)
        {
            SystemState<DTYPE> sample = _state;
            Eigen::VectorX<double> dx = x.col(i);
            updateStateVariables(sample, dx);

            if (m_options.compute_covariances)
            {
                for (unsigned int i = 0; i < sample.estimation_nodes.size(); i++)
                {
                    buildCovarianceSquare(sample.estimation_nodes[i], m_sampler_covariances, sample, topology);
                }
            }
            if (_state.ib != state.ib) // Converts back to original frame
                sample.convertStateMeanBodyInertial();
            samples.push_back(sample);
        }

        samples[0].print();

        return samples;
    }

    void Estimator::extractJacobianHessian(const SystemState<DTYPE> &state,
                                           Eigen::SparseMatrix<double> &H,
                                           std::vector<Eigen::VectorXd> &factor_errors,
                                           std::vector<Eigen::MatrixXd> &factor_jacobians,
                                           std::vector<Eigen::MatrixXd> &factor_Qs,
                                           std::vector<std::vector<int>> &factor_node_indices) const
    {
        // Assemble prior and measurement terms similarly to computeStateEstimate,
        // but capture per-factor linearisation data (e, E, Q) and build H.
        unsigned int total_nodes = m_robot_topology.K * m_robot_topology.N;

        std::cout << "Extracting Jacobian and Hessian for " << total_nodes << " nodes." << std::endl;
        std::vector<Eigen::Triplet<double>> A_tripletList;
        A_tripletList.reserve(1024);

        factor_errors.clear();
        factor_jacobians.clear();
        factor_Qs.clear();
        factor_node_indices.clear();

        // PRIOR TERMS
        std::cout << "Assembling prior terms..." << std::endl;
        for (unsigned int i = 0; i < total_nodes; i++)
        {
            unsigned int n = i % m_robot_topology.N;
            unsigned int k = i / m_robot_topology.N;

            if (n == 0)
            {
                // Unary prior at base node
                const auto &node = state.estimation_nodes[k * m_robot_topology.N + n];
                Eigen::VectorX<DTYPE> e;
                Eigen::Matrix<double, 18, 18> S;
                if (m_options.use_autodiff)
                    S = m_unary_factor->Factor::getJacobian({node}, e);
                else
                    S = m_unary_factor->getJacobian({node}, e);

                Eigen::Matrix<double, 18, 18> weight = m_unary_factor->getWeight().cast<double>();

                Eigen::MatrixX<double> A_block = S.transpose() * weight * S;

                // store linearization
                factor_errors.push_back(e.template cast<double>());
                factor_jacobians.push_back(S);
                factor_Qs.push_back(weight);
                factor_node_indices.push_back({(int)getOptimizationIndex(n, k, m_robot_topology)});

                // push triplets
                std::array<unsigned int, 1> row_indices = {getOptimizationIndex(n, k, m_robot_topology)};
                for (int row_block = 0; row_block < 18; row_block++)
                {
                    unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);
                    for (int col_block = 0; col_block < 18; col_block++)
                    {
                        unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                        A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
                    }
                }
            }

            if (n != 0)
            {
                // Binary space term between n-1 and n
                const auto &state1 = state.estimation_nodes[k * m_robot_topology.N + n];
                const auto &state0 = state.estimation_nodes[k * m_robot_topology.N + n - 1];

                Eigen::VectorX<DTYPE> e;
                Eigen::Matrix<DTYPE, 18, 18> weightDT = m_binary_space_factor->getWeight({state0, state1});
                Eigen::Matrix<double, 18, 36> S;
                if (m_options.use_autodiff)
                    S = m_binary_space_factor->Factor::getJacobian({state0, state1}, e);
                else
                    S = m_binary_space_factor->getJacobian({state0, state1}, e);

                Eigen::Matrix<double, 18, 18> weight = weightDT.cast<double>();
                Eigen::MatrixX<double> A_block = S.transpose() * weight * S;

                // store linearization
                factor_errors.push_back(e.template cast<double>());
                factor_jacobians.push_back(S);
                factor_Qs.push_back(weight);
                factor_node_indices.push_back({(int)getOptimizationIndex(n - 1, k, m_robot_topology), (int)getOptimizationIndex(n, k, m_robot_topology)});

                std::array<unsigned int, 2> row_indices = {getOptimizationIndex(n - 1, k, m_robot_topology), getOptimizationIndex(n, k, m_robot_topology)};
                for (int row_block = 0; row_block < 36; row_block++)
                {
                    unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);
                    for (int col_block = 0; col_block < 36; col_block++)
                    {
                        unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                        A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
                    }
                }
            }

            if (k > 0 && !m_robot_topology.use_1D_estimator)
            {
                // Binary time term between k-1 and k at same spatial index n
                const auto &state0 = state.estimation_nodes[(k - 1) * m_robot_topology.N + n];
                const auto &state1 = state.estimation_nodes[k * m_robot_topology.N + n];

                Eigen::VectorX<DTYPE> e;
                Eigen::Matrix<DTYPE, 18, 18> weightDT = m_binary_time_factor->getWeight({state0, state1});
                Eigen::Matrix<double, 18, 36> S;
                if (m_options.use_autodiff)
                    S = m_binary_time_factor->Factor::getJacobian({state0, state1}, e);
                else
                    S = m_binary_time_factor->getJacobian({state0, state1}, e);

                Eigen::Matrix<double, 18, 18> weight = weightDT.cast<double>();
                Eigen::MatrixX<double> A_block = S.transpose() * weight * S;

                // store linearization
                factor_errors.push_back(e.template cast<double>());
                factor_jacobians.push_back(S);
                factor_Qs.push_back(weight);
                factor_node_indices.push_back({(int)getOptimizationIndex(n, k - 1, m_robot_topology), (int)getOptimizationIndex(n, k, m_robot_topology)});

                std::array<unsigned int, 2> row_indices = {getOptimizationIndex(n, k - 1, m_robot_topology), getOptimizationIndex(n, k, m_robot_topology)};
                for (int row_block = 0; row_block < 36; row_block++)
                {
                    unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);
                    for (int col_block = 0; col_block < 36; col_block++)
                    {
                        unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                        A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
                    }
                }
            }
        }

        std::cout << "Assembling measurement terms..." << std::endl;
        // MEASUREMENT TERMS
        int end_meas = m_measurement_factors.size();
        for (int i = 0; i < end_meas; i++)
        {
            const auto &p_factor = m_measurement_factors[i];
            const auto &measurement = p_factor->getMeas();

            if (measurement.t < state.estimation_nodes.front().time - TOLERANCE)
                continue;
            if (measurement.t > state.estimation_nodes.back().time + TOLERANCE)
                continue;

            SystemState<DTYPE>::Node node_ts;
            node_ts.time = measurement.t;
            node_ts.arclength = measurement.s;

            Eigen::VectorX<DTYPE> e;
            Eigen::MatrixX<double> G, S;

            unsigned int n, k, n1, k1;
            getNKIndeces(measurement.s, measurement.t, n, k, state, m_robot_topology);
            if (n == (unsigned int)(-1) || k == (unsigned int)(-1))
                continue;

            SystemState<DTYPE>::Node node0 = state.estimation_nodes[k * m_robot_topology.N + n];

            k1 = (std::abs(node_ts.time - node0.time) < TOLERANCE) ? -1 : k + 1;
            n1 = (std::abs(node_ts.arclength - node0.arclength) < TOLERANCE) ? -1 : n + 1;
            if (m_robot_topology.K <= 1)
                k1 = (unsigned int)(-1);
            if (m_robot_topology.N <= 1)
                n1 = (unsigned int)(-1);

            Eigen::MatrixX<double> jac;
            interpolateMean2D(node_ts, state, jac, true);
            if (m_options.use_autodiff)
                G = p_factor->Factor::getJacobian({node_ts}, e);
            else
                G = p_factor->getJacobian({node_ts}, e);
            S = G * jac;

            p_factor->setOperatingPoint(node_ts);

            Eigen::MatrixX<double> weight = p_factor->getWeight().cast<double>();

            Eigen::MatrixX<double> A_block = S.transpose() * weight * S;

            // determine row indices (nodes involved)
            std::vector<unsigned int> row_indices;
            row_indices.push_back(getOptimizationIndex(n, k, m_robot_topology));
            if (n1 != (unsigned int)(-1))
                row_indices.push_back(getOptimizationIndex(n1, k, m_robot_topology));
            if (k1 != (unsigned int)(-1))
                row_indices.push_back(getOptimizationIndex(n, k1, m_robot_topology));
            if (n1 != (unsigned int)(-1) && k1 != (unsigned int)(-1))
                row_indices.push_back(getOptimizationIndex(n1, k1, m_robot_topology));

            // store linearization
            factor_errors.push_back(e.template cast<double>());
            factor_jacobians.push_back(S);
            factor_Qs.push_back(weight);
            std::vector<int> nodes_int;
            for (auto ri : row_indices)
                nodes_int.push_back((int)ri);
            factor_node_indices.push_back(nodes_int);

            // write triplets
            for (int row_block = 0; row_block < (int)A_block.rows(); row_block++)
            {
                unsigned int idx_row = row_indices[row_block / 18] + (row_block % 18);
                for (int col_block = 0; col_block < (int)A_block.cols(); col_block++)
                {
                    unsigned int idx_col = row_indices[col_block / 18] + (col_block % 18);
                    A_tripletList.emplace_back(idx_row, idx_col, A_block(row_block, col_block));
                }
            }
        }
        std::cout << "Finished assembling Jacobian and Hessian." << std::endl;
        std::cout << "Number of factors: " << factor_errors.size() << std::endl;
        std::cout << "Number of triplets: " << A_tripletList.size() << std::endl;
        std::cout << "Number of nodes: " << total_nodes << std::endl;
        // Build sparse H
        int size = 18 * total_nodes;
        H.resize(size, size);
        H.setFromTriplets(A_tripletList.begin(), A_tripletList.end());
        std::cout << "Hessian size: " << H.rows() << " x " << H.cols() << std::endl;
    }

} // namespace Spacetime
