#ifndef SWESTIMATOR_H
#define SWESTIMATOR_H

#include <utility>
#include <vector>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>

#include "spacetime/types.hpp"
#include "spacetime/estimators/Estimator.hpp"

namespace Spacetime
{
    class SWEstimator : public Estimator
    {
    public:
        SWEstimator(RobotTopology topology, Hyperparameters parameters, Options options) : m_marginalize(options.marginalize), m_time_nodes_on_measurements(topology.time_nodes_on_measurements)
        {
            assert(topology.K >= 2 && "SWEstimator requires at least 2 time nodes.");
            warning("SWEstimator does not account for cost only operation. Use of fully autodiff system may not work as expected.");
            options.reverse_order = !options.extract_from_front; // Ensure reverse order is always true for SWEstimator, required for covariance extraction
            m_covariance_timesteps_to_extract = 2; // Default to extracting only the first timestep covariance

            if (options.extract_from_front)
                topology.t0 -= topology.T; // Adjust t0 so that the first extracted state is at time 0

            topology.time_nodes_on_measurements = false; // SWEstimator child class handles this logic internally
            initializeEstimator(topology, parameters, options);
        }

        // Computes the and returns state estimate given a set of sensor measurements
        Results computeStateEstimate(const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements, bool verbose_mode = false) override;

    private:
        SystemState<DTYPE> m_last_state; // Expressed with T_ib frames
        Eigen::MatrixXd m_covariance;    // Covariance matrix
        Eigen::SparseMatrix<double> m_A_static, m_b_static;
        mutable Eigen::SparseMatrix<double> m_A_marginal, m_b_marginal, m_A00_sw_inv, m_diagonal_A_block, m_diagonal_A_block_static;
        bool m_marginalize, m_time_nodes_on_measurements;
        int m_sizeN;

        void addChildFactors(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, bool cost_only = false) const override;
        void constructPriorTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only = false) const;
        void constructMeasurementTerms(std::vector<Eigen::Triplet<double>> &A_tripletList, std::vector<Eigen::Triplet<double>> &b_tripletList, DTYPE &cost, const SystemState<DTYPE> &state, const RobotTopology &topology, bool cost_only = false) const;
        void updateMap(Results &result) const; 
        mutable Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> m_solver_marginalization;
        mutable std::vector<std::shared_ptr<Factors::MeasurementFactor>> m_marginalized_measurement_factors;
    };
} // namespace Spacetime
#endif // SWESTIMATOR_H