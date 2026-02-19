#ifndef EKFESTIMATOR_H
#define EKFESTIMATOR_H

#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>

#include "spacetime/types.hpp"
#include "spacetime/estimators/Estimator.hpp"

namespace Spacetime
{
    class FilterEstimator : public Estimator
    {
    public:
        FilterEstimator(RobotTopology topology, Hyperparameters parameters, Options options) : m_marginalize(options.marginalize), m_time_nodes_on_measurements(topology.time_nodes_on_measurements)
        {
            assert(topology.K == 1 && "FilterEstimator only supports K=1 (single time step) topology.");
            warning("FilterEstimator does not account for cost only operation. Use of fully autodiff system may not work as expected.");
            options.reverse_order = false; // Ensure reverse order is always true for FilterEstimator, required for covariance extraction
            m_covariance_timesteps_to_extract = 1; // Only extract variance of first timestep and covariance between first and second timestep if not running a filter

            topology.time_nodes_on_measurements = false; // SWEstimator child class handles this logic internally
            initializeEstimator(topology, parameters, options);
        }

        // Computes the and returns state estimate given a set of sensor measurements
        Results computeStateEstimate(const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements, bool verbose_mode = false) override;

    private:
        SystemState<DTYPE> m_last_state; // Expressed with T_ib frames
        Eigen::MatrixXd m_covariance;    // Covariance matrix
        Eigen::SparseMatrix<double> m_A_static, m_b_static;
        mutable Eigen::SparseMatrix<double> m_A_marginal, m_b_marginal, m_A00_sw_inv;
        bool m_marginalize, m_time_nodes_on_measurements;
        int m_sizeN;

        void addChildFactors(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, bool cost_only = false) const override;
        void constructPriorTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only = false) const;
        void constructMeasurementTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only = false) const;
        mutable Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> m_solver_marginalization;
    };
} // namespace Spacetime
#endif // EKFESTIMATOR_H