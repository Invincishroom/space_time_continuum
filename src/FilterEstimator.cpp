#include <numeric>
#include <algorithm>
#include <chrono>
#include <limits>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <lgmath.hpp>

#include "spacetime/estimators/FilterEstimator.hpp"
#include "spacetime/estimators/Estimator.hpp"
#include "spacetime/utilities.hpp"


namespace Spacetime
{
    Estimator::Results FilterEstimator::computeStateEstimate(const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements, bool verbose_mode)
    {
        // Info / Parameter checks
        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
        info("Starting IEKF estimation.");

        assert(!m_robot_topology.time_nodes_on_measurements && "FilterEstimator does not support time_nodes_on_measurements = true");

        Results result, window_result;

        window_result.state = Estimator::m_state; // Initialize window state
        m_sizeN = m_robot_topology.N * 18;
        m_A_static.resize(m_sizeN, m_sizeN); // Initialize to zero matrix if first time
        m_b_static.resize(m_sizeN, 1);
        m_A_marginal.resize(m_sizeN, m_sizeN);
        m_b_marginal.resize(m_sizeN, 1);
        m_last_state.estimation_nodes.reserve(m_robot_topology.N); // Reserve space for last state

        int total_K;
        std::vector<double> ts;
        int ts_index = 0;

        if (!m_time_nodes_on_measurements)
        {
            total_K = std::max(1, (int)((measurements.back()->getMeas().t - m_robot_topology.t0) / m_robot_topology.T + 2));
        }
        else
        {
            for (const auto &measurement : measurements)
            {
                // debug() << "Measurement time: " << measurement->getMeas().t;
                if (!ts.empty())
                    if (abs(measurement->getMeas().t - ts.back()) < TOLERANCE)
                        continue;
                ts.push_back(measurement->getMeas().t);
            }
            total_K = ts.size();

            std::vector<double> ts_window;
            for (int i = 0; i < m_robot_topology.K && i < ts.size(); i++)
            {
                ts_window.push_back(ts[i]);
                ts_index++;
            }
            m_state = constructInitialGuess(m_options.init_guess_type, ts_window);
        }

        result.runtimes.reserve(total_K);
        result.state.estimation_nodes.reserve(m_robot_topology.N * total_K);

        for (int k = 0; k < total_K; k++)
        {
            auto window_start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
            // Initialize state given new window
            if (k != 0)
            {
                // Shift state back by one time step
                SystemState<DTYPE> new_state;

                // Add new time step at end
                double dt = m_robot_topology.T;
                if (m_time_nodes_on_measurements)
                {
                    if (ts_index < ts.size())
                        dt = ts[ts_index] - ts[ts_index - 1];
                    ts_index++;
                }

                double ds = m_robot_topology.L / (m_robot_topology.N - 1);

                int start_index = result.state.estimation_nodes.size() - m_robot_topology.N;

                SystemState<DTYPE>::Node node = m_binary_time_factor->getZeroErorNode(result.state.estimation_nodes[start_index], dt);
                new_state.estimation_nodes.push_back(node);
                for (unsigned int n = 1; n < m_robot_topology.N; n++)
                {
                    SystemState<DTYPE>::Node nodeSpace = m_binary_space_factor->getZeroErorNode(new_state.estimation_nodes[n - 1], ds);
                    SystemState<DTYPE>::Node nodeTime = m_binary_time_factor->getZeroErorNode(result.state.estimation_nodes[start_index + n], dt);
                    nodeSpace.average(nodeTime); // Average the two predictions
                    new_state.estimation_nodes.push_back(nodeSpace);
                }

                Estimator::m_state = new_state;
            }
            toggleExternalTerms(k != 0);

            // Compute state estimate for current window
            window_result = Estimator::computeStateEstimate(measurements, verbose_mode);
            result.cost += window_result.cost;
            if (verbose_mode)
            {
                info() << "New window between times " << Estimator::m_state.estimation_nodes[0].time << "s - " << Estimator::m_state.estimation_nodes.back().time << "s.";
            }
            
            if (!window_result.success)
            {
                result.message << "IEKF estimation failed.";
                // Add failed window state to result
                for (unsigned int n = 0; n < m_robot_topology.N; n++)
                {
                    result.state.estimation_nodes.push_back(window_result.state.estimation_nodes[n]);
                }

                return result;
            }

            // Add first window state to result
            m_last_state.estimation_nodes.clear();
            for (unsigned int n = 0; n < m_robot_topology.N; n++)
            {
                result.state.estimation_nodes.push_back(window_result.state.estimation_nodes[n]);
                m_last_state.estimation_nodes.push_back(window_result.state.estimation_nodes[n]); // Save last state
            }

            // Save static results
            if (k != 0)
            {
                m_A_static = m_A_marginal;
                m_b_static = m_b_marginal;
            }

            result.runtimes.push_back((std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()) - window_start_time).count());
        }

        info() << "Sliding window estimation complete. Time elapsed: " << (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_time).count() / 1e6 << "s.";
        result.success = true;

        return result;
    }

    void FilterEstimator::addChildFactors(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, bool cost_only) const
    {
        // Adds a prior factor for the previously marginalized time steps

        /*
        Estimator A matrix:
        [
        A11_e, A12_e
        A21_e, A22_e
        ] where A11_e is the component of A11_sw added by the nodes and measurements on and after time 1

        FilterEstimator A matrix:
        [
        A00_sw - A_static, A01_sw
        A10_sw           , A11_sw
        ] where A11_sw is the component of A11_sw added by the nodes and measurements occuring before time 1
        A_static is any static component saved from previous marginalized steps

        Total A matrix
        [
        A00_sw - A_static, A01_sw            , 0
        A10_sw           , A11_swe + A11_e, A12_e
        0                , A21_e          , A22_e
        ]

        Marginalized matrix
        [
        A11_e + A11_swe - A10_sw (A00_sw - A_static)^{-1} A01_sw, A12
        A21                                                     , A22
        ]

        This function takes in A_tripletList that represents the Estimator block and adds the
        A11_swe - A10_sw (A00_sw - A_static)^{-1} A01_sw block.

        Similarly, this function takes in b_tripletList that represents the Estimator block and adds the
        b1_swe - A10_sw (A00_sw - A_static)^{-1} (b0 - b_static)  block.

        Estimator b matrix:
        [
        b1_e
        b2
        ]

        FilterEstimator b matrix
        [
        b0
        b1_swe
        ]

        Total b matrix
        [
        b0
        b1_e + b1_swe
        b2
        ]

        Marginalized b matrix
        [
        b1_2 + b1_swe - A10_sw (A00_sw - A_static)^{-1} (b0 - b_static)
        b2
        ]
        */
        if (!m_marginalize)
            return; // No additional terms added if not marginalizing

        SystemState<DTYPE> joint_state = m_last_state;
        // Add current window first time step
        for (unsigned int n = 0; n < m_robot_topology.N; n++)
            joint_state.estimation_nodes.push_back(state.estimation_nodes[n]);

        RobotTopology topology = m_robot_topology;
        topology.K = 2; // Only consider first two time steps for prior terms
        std::vector<Eigen::Triplet<double>> A_marginal_tripletList, b_marginal_tripletList;
        constructPriorTerms(A_marginal_tripletList, b_marginal_tripletList, cost, joint_state, topology, cost_only);
        constructMeasurementTerms(A_marginal_tripletList, b_marginal_tripletList, cost, joint_state, topology, cost_only);

        Eigen::SparseMatrix<double> A_sw, b_sw;
        A_sw.resize(m_sizeN * 2, m_sizeN * 2);
        b_sw.resize(m_sizeN * 2, 1);

        A_sw.setFromTriplets(A_marginal_tripletList.begin(), A_marginal_tripletList.end());
        b_sw.setFromTriplets(b_marginal_tripletList.begin(), b_marginal_tripletList.end());

        // A11_swe - A10_sw (A00_sw - A_static)^{-1} A01_sw
        Eigen::SparseMatrix<double> A00_sw, A01_sw, A10_sw, A11_sw;
        // Order has been reversed, hence this indexing
        if (m_options.reverse_order)
        {
            A00_sw = A_sw.block(m_sizeN, m_sizeN, m_sizeN, m_sizeN);
            A01_sw = A_sw.block(m_sizeN, 0, m_sizeN, m_sizeN);
            A10_sw = A_sw.block(0, m_sizeN, m_sizeN, m_sizeN);
            A11_sw = A_sw.block(0, 0, m_sizeN, m_sizeN);
        }
        else
        {
            A00_sw = A_sw.block(0, 0, m_sizeN, m_sizeN);
            A01_sw = A_sw.block(0, m_sizeN, m_sizeN, m_sizeN);
            A10_sw = A_sw.block(m_sizeN, 0, m_sizeN, m_sizeN);
            A11_sw = A_sw.block(m_sizeN, m_sizeN, m_sizeN, m_sizeN);
        }

        m_solver_marginalization.compute(A00_sw + m_A_static);
        m_A00_sw_inv = m_solver_marginalization.solve(Eigen::MatrixXd::Identity(m_sizeN, m_sizeN)).sparseView();

        if (m_solver_marginalization.info() != Eigen::Success)
        {
            error() << "FilterEstimator marginalization failed. No marginalization will be performed.";

            // Print detailed failure information
            debug() << "Eigen solver info: ";
            switch (m_solver_marginalization.info())
            {
            case Eigen::NumericalIssue:
                debug() << "NumericalIssue - Matrix is not invertible or numerically unstable";
                break;
            case Eigen::NoConvergence:
                debug() << "NoConvergence - Iterative solver failed to converge";
                break;
            case Eigen::InvalidInput:
                debug() << "InvalidInput - Input matrix has invalid properties";
                break;
            default:
                debug() << "Unknown error code: " << static_cast<int>(m_solver_marginalization.info());
            }

            // Check matrix properties (assuming you have access to the matrix being factorized)
            // Replace 'matrix_to_factorize' with your actual matrix variable
            const auto &H = (A00_sw - m_A_static).toDense(); // Replace with your actual matrix

            debugStart();
            std::cout << "Matrix properties:" << std::endl;
            std::cout << "  - Size: " << H.rows() << "x" << H.cols() << std::endl;
            std::cout << "  - Determinant: " << H.determinant() << std::endl;
            std::cout << "  - Rank: " << Eigen::FullPivLU<Eigen::MatrixXd>(Eigen::MatrixXd(H)).rank() << std::endl;
            std::cout << "  - Condition number estimate: " << H.norm() * H.inverse().norm() << std::endl;
            std::cout << "  - Has NaN: " << !H.allFinite() << std::endl;
            std::cout << "  - Min eigenvalue: " << H.eigenvalues().real().minCoeff() << std::endl;
            std::cout << "  - Max eigenvalue: " << H.eigenvalues().real().maxCoeff() << std::endl;
            logReset();

            throw std::runtime_error("FilterEstimator marginalization failed.");
        }

        m_A_marginal = A11_sw - A10_sw * m_A00_sw_inv * A01_sw;

        // b1_swe - A10_sw (A00_sw - A_static)^{-1} (b0 - b_static)
        Eigen::SparseMatrix<double> b0, b1;
        if (m_options.reverse_order)
        {
            b0 = b_sw.block(m_sizeN, 0, m_sizeN, 1);
            b1 = b_sw.block(0, 0, m_sizeN, 1);
        }
        else
        {
            b0 = b_sw.block(0, 0, m_sizeN, 1);
            b1 = b_sw.block(m_sizeN, 0, m_sizeN, 1);
        }

        m_b_marginal = b1 - A10_sw * m_A00_sw_inv * (b0 + m_b_static);

        // Add m_A_marginal to A_tripletList
        for (int k = 0; k < m_A_marginal.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(m_A_marginal, k); it; ++it)
            {
                if (it.col() >= m_sizeN || it.row() >= m_sizeN)
                    throw std::runtime_error("Index out of bounds in FilterEstimator marginalization.");
                A_tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
            }
        }

        // Add m_b_marginal to b_tripletList
        for (int k = 0; k < m_b_marginal.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(m_b_marginal, k); it; ++it)
            {
                b_tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
            }
        }
    }
    void FilterEstimator::constructPriorTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only) const
    {
#pragma omp declare reduction(dtype_add:DTYPE : omp_out += omp_in) initializer(omp_priv = DTYPE(0.0))
#pragma omp declare reduction(merge_triplet_list : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<Eigen::Triplet<double>>())
#pragma omp parallel for reduction(dtype_add : cost) reduction(merge_triplet_list : A_tripletList, b_tripletList)
        for (unsigned int i = 0; i < 2 * topology.N; i++)
        {
            unsigned int n = i % topology.N;
            unsigned int k = i / topology.N;

            if (n == 0 && k == 0) // Add at base for first timestep (second timestep is added in Estimator)
            {
                assembleUnaryTerm(A_tripletList, b_tripletList, cost, state, n, k, topology, cost_only);
            }
            if (n != 0 && k == 0) // Add space terms for first time step (others are added in Estimator)
            {
                // TODO: Possible runtime improvement here as the only states involved here are locked. Can precompute this once.
                assembleBinarySpaceTerm(A_tripletList, b_tripletList, cost, state, n, k, topology, cost_only);
            }
            if (k != 0 && !topology.use_1D_estimator)
            {
                assembleBinaryTimeTerm(A_tripletList, b_tripletList, cost, state, n, k, topology, cost_only);
            }
        }
    }
    void FilterEstimator::constructMeasurementTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only) const
    {
        // Run through all of the measurements
#pragma omp declare reduction(dtype_add:DTYPE : omp_out += omp_in) initializer(omp_priv = DTYPE(0.0))
#pragma omp declare reduction(merge_triplet_list : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = std::vector<Eigen::Triplet<double>>())
#pragma omp parallel for reduction(dtype_add : cost) reduction(merge_triplet_list : A_tripletList, b_tripletList)
        for (const auto &factor : m_measurement_factors)
        {
            assembleMeasurementTerm(factor, A_tripletList, b_tripletList, cost, state, topology, cost_only);
        }
    }
} // namespace Spacetime