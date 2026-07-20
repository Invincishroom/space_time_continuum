#ifndef SPACETIMECONTINUUM_H
#define SPACETIMECONTINUUM_H

#include <utility>
#include <vector>
#include <random>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <Eigen/SparseLU>

#include "spacetime/Solver.hpp"
#include "spacetime/factors.hpp"
#include "spacetime/types.hpp"

namespace Spacetime
{
    class Estimator
    {
        // Gives tests access to private functions and variables
        friend class FactorTests_TestUnaryFactor_Test;
        friend class FactorTests_TestSpaceBinaryFactor_Test;
        friend class FactorTests_TestTimeBinaryFactor_Test;
        friend class FactorTests_TestQuaternaryFactor_Test;
        friend class FactorTests_TestAssembleFactors_Test;
        friend class FactorTests_TestPoseFactor_Test;
        friend class FactorTests_TestStrainRegularizationFactor_Test;

        friend class FactorTestsAutodiff_TestUnaryFactor_Test;
        friend class FactorTestsAutodiff_TestSpaceBinaryFactor_Test;

        friend class FunctionTests_TestNKIndeces_Test;

        friend class AutodiffJacTests_TestUnaryFactor_Test;
        friend class AutodiffJacTests_TestBinarySpaceFactor_Test;
        friend class AutodiffJacTests_TestBinaryTimeFactor_Test;
        friend class AutodiffJacTests_TestPoseFactor_Test;
        friend class AutodiffJacTests_TestGyroFactor_Test;

    private:
        std::default_random_engine m_rng;

        Eigen::SparseMatrix<double> m_P; // Projection matrix

        Eigen::MatrixX<double> m_covariance; // Covariance matrix
        SystemState<DTYPE>::Node m_node0;

        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> m_solver;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> m_sampling_solver;
        Solver m_spacetime_solver;
        Eigen::MatrixX<double> m_sampler_covariances;
        bool m_sampler_initialized = false;
        bool m_external_terms_available = false;
        int m_start_meas_idx = 0; // Index of the first measurement factor in the current time window

        // Validates the parameters within each structure and ensures they are set correctly
        // TODO: Need to check newly added topology and options parameters
        // TODO: Need to check FBG strain measurements
        void validateRobotTopology(RobotTopology topology);
        void validateHyperparameters(Hyperparameters parameters);
        void validateOptions(Options options);
        void validateMeasurements(const std::vector<SensorMeasurement> &measurements);

        Eigen::Matrix<DTYPE, 18, 36> computeLambdaPsiS(double s_i, double s_n, double s_n1, const SystemState<DTYPE>::Node &node) const;
        Eigen::Matrix<DTYPE, 18, 36> computeLambdaPsiT(double t_i, double t_k, double t_k1) const;

        // Constructs the projection matrix of the system based on robot toplogy
        Eigen::SparseMatrix<double> constructProjectionMatrix();

        // Solves the linear system in each iteration Ax=b, while considering the projection matrix M and the chosen solving method
        Eigen::VectorXd solveLinearSystem(const Eigen::SparseMatrix<double> &A, const Eigen::SparseMatrix<double> &b, bool initialize);
        Eigen::VectorXd solveLinearSystem(const Eigen::SparseMatrix<double> &A, const Eigen::SparseMatrix<double> &b, const Eigen::SparseMatrix<double> &P, bool initialize);

        void perturbState(SystemState<DTYPE> &state, double scale = 1e-3);
        // Updates the State Variables based on dx
        void updateStateVariables(SystemState<DTYPE> &state, Eigen::VectorXd &dx);

    protected:
        SystemState<DTYPE> m_state; // Expressed with T_ib frames
        RobotTopology m_robot_topology;
        Hyperparameters m_hyperparameters;
        Options m_options;

        Factors::UnaryFactor *m_unary_factor;
        Factors::BinaryTimeFactor *m_binary_time_factor;
        Factors::BinarySpaceFactor *m_binary_space_factor;
        std::vector<std::shared_ptr<Factors::MeasurementFactor>> m_measurement_factors;
        int m_covariance_timesteps_to_extract = -1;

        void toggleExternalTerms(bool terms)
        {
            m_external_terms_available = terms;
        };
        virtual void addChildFactors([[maybe_unused]] std::vector<Eigen::Triplet<double>> &A_tripletList, [[maybe_unused]] std::vector<Eigen::Triplet<double>> &b_tripletList, [[maybe_unused]] DTYPE &cost, [[maybe_unused]] const SystemState<DTYPE> &state, [[maybe_unused]] bool cost_only = false) const {};

        // Returns a system state to be used as an initial guess in the optimization based on the chosen type
        // Expressed with T_ib frames
        SystemState<DTYPE> constructInitialGuess(Options::InitialGuessType type, std::vector<double> ts = {}, double start_t = 0.0) const;

        // Returns matrix A, vector b and the cost for the prior, coupling and measurement terms based on robot topology, current state and measurements
        void assemblePriorTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only = false) const;
        void assembleUnaryTerm(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, int n, int k, const RobotTopology &topology, bool cost_only = false) const;
        void assembleBinarySpaceTerm(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, int n, int k, const RobotTopology &topology, bool cost_only = false) const;
        void assembleBinaryTimeTerm(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, int n, int k, const RobotTopology &topology, bool cost_only = false) const;
        bool assembleMeasurementTerm(std::shared_ptr<Factors::MeasurementFactor> factor, std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only = false) const;

    public:
        struct Results
        {
            bool success = false;
            double cost = 0.0;
            Spacetime::SystemState<DTYPE> state;
            std::vector<double> runtimes;
            std::ostringstream message;
            std::shared_ptr<steam_icp::Map> map;
        };

        struct FactorGraphLinearization
        {
            Eigen::SparseMatrix<double> H;
            Eigen::VectorXd rhs;
            std::vector<Eigen::VectorXd> factor_errors;
            std::vector<Eigen::MatrixXd> factor_jacobians;
            std::vector<Eigen::MatrixXd> factor_Qs;
            std::vector<std::vector<int>> factor_node_indices;
        };

        Estimator() = default;
        Estimator(RobotTopology topology, Hyperparameters parameters, Options options) { initializeEstimator(topology, parameters, options); }
        void initializeEstimator(RobotTopology topology, Hyperparameters parameters, Options options);

        void setRobotTopology(RobotTopology topology);
        RobotTopology getRobotTopology() { return m_robot_topology; }

        void setHyperparameters(Hyperparameters parameters);
        Hyperparameters getHyperparameters() { return m_hyperparameters; }

        void setOptions(Options options);
        Options getOptions() { return m_options; }

        // Get system state expressed with T_ib frames
        SystemState<DTYPE> getSystemState() { return m_state; }

        // Computes the and returns state estimate given a set of sensor measurements
        void buildMeasurementFactors(const std::vector<SensorMeasurement> &measurements, std::shared_ptr<steam_icp::Map> environment_map = nullptr);
        virtual Results computeStateEstimate(const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements, bool verbose_mode = false);

        // Interpolate between the estimation nodes
        void interpolateStates(SystemState<DTYPE> &state);
        void buildCovarianceSquare(SystemState<DTYPE>::Node &node, const Eigen::MatrixX<double> &covariance, const SystemState<DTYPE> &state, const RobotTopology &topology);

        // Extract per-factor linearisation data (errors, jacobians, information matrices)
        // and assemble the full information matrix H for a given state.
        void extractJacobianHessian(const SystemState<DTYPE> &state,
                        Eigen::SparseMatrix<double> &H,
                        std::vector<Eigen::VectorXd> &factor_errors,
                        std::vector<Eigen::MatrixXd> &factor_jacobians,
                        std::vector<Eigen::MatrixXd> &factor_Qs,
                        std::vector<std::vector<int>> &factor_node_indices) const;

        static FactorGraphLinearization assembleFactorGraph(const std::vector<SystemState<DTYPE>::Node> &nodes,
                                    const std::vector<std::shared_ptr<Factors::Factor>> &factors,
                                    const std::vector<std::vector<int>> &factor_node_indices = {});

        // Returns the state estimate of the last estimation computation with additional queried nodes
        // Careful: Will use the last known state (i.e. the last state computed and returned from computeStateEstimate)
        void interpolateMean1D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE>::Node &node0, const SystemState<DTYPE>::Node &node1) const
        {
            Eigen::MatrixXd dummy_jac;
            interpolateMean1D(query_node, node0, node1, dummy_jac, false);
        };
        void interpolateMean1D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE>::Node &node0, const SystemState<DTYPE>::Node &node1, Eigen::MatrixXd &jac, bool compute_jac = true) const;
        void interpolateMean2D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE> &state) const
        {
            Eigen::MatrixXd dummy_jac;
            interpolateMean2D(query_node, state, dummy_jac, false);
        };
        void interpolateMean2D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE> &state, Eigen::MatrixXd &jac, bool compute_jac = true) const;
        void interpolateMean2D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE>::Node &node00, const SystemState<DTYPE>::Node &node01, const SystemState<DTYPE>::Node &node10, const SystemState<DTYPE>::Node &node11) const
        {
            Eigen::MatrixXd dummy_jac;
            interpolateMean2D(query_node, node00, node01, node10, node11, dummy_jac, false);
        }
        void interpolateMean2D(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE>::Node &node00, const SystemState<DTYPE>::Node &node01, const SystemState<DTYPE>::Node &node10, const SystemState<DTYPE>::Node &node11, Eigen::MatrixXd &jac, bool compute_jac) const;
        void queryState(SystemState<DTYPE>::Node &query_node, const SystemState<DTYPE> &state) const;

        std::vector<SystemState<DTYPE>> samplePrior(const RobotTopology &topology, int n);
        std::vector<SystemState<DTYPE>> samplePrior(const SystemState<DTYPE> &state, const RobotTopology &topology, int n);
        unsigned int getOptimizationIndex(unsigned int n, unsigned int k, const RobotTopology &topology) const
        {
            // Given an n and k index, returns the corresponding index in the optimization problem
            if (!m_options.reverse_order)
                return 18 * (k * topology.N + n);
            return 18 * ((topology.K - 1 - k) * topology.N + n);
        }

        template <typename T = double>
        static void getNKIndeces(double s, double t, unsigned int &n, unsigned int &k, const SystemState<T> &state, RobotTopology topology)
        {
            /*
            Specific function behaviour:
            - If s and t exist inside the grid, n, k, are returned such that each is the lower bounding node on the grid
            - If the requested point is invalid, n and k are set to -1
            */
            n = -1;
            k = -1;

            if (s > topology.L + TOLERANCE || s < 0)
            {
                error() << "s (" << s << ") is outside of allowable range [0," << topology.L << "]";
                throw std::runtime_error("Invalid arclength");
            }
            if (t < state.estimation_nodes.front().time - TOLERANCE || t > state.estimation_nodes.back().time + TOLERANCE)
            {
                error() << "t (" << t << ") is outside of allowable range [" << state.estimation_nodes.front().time << ", " << state.estimation_nodes.back().time << "]";
                throw std::runtime_error("Invalid time");
            }

            if (topology.N == 1)
                n = 0;
            else
                for (unsigned int _n = 0; _n < topology.N; _n++)
                {
                    const auto &node = state.estimation_nodes[_n];
                    if (s < node.arclength - TOLERANCE)
                    {
                        n = _n - (_n > 0);
                        break;
                    }
                    else if (std::abs(s - node.arclength) < TOLERANCE) // Node is on the tip node
                    {
                        n = _n;
                        break;
                    }
                }

            unsigned int max_K = state.getK(topology);
            if (max_K == 1)
                k = 0;
            else
                for (unsigned int _k = 0; _k < max_K; _k++)
                {
                    const auto &node = state.estimation_nodes[_k * topology.N];
                    if (t < node.time - TOLERANCE) // Iterated to node that is larger than time step
                    {
                        k = _k - (_k > 0);
                        break;
                    }
                    else if (std::abs(t - node.time) < TOLERANCE) // Node is on the current timestep
                    {
                        k = _k;
                        break;
                    }
                }
        }
    };
} // namespace Spacetime
#endif // SPACETIMECONTINUUM_H
