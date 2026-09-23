#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "spacetime/types.hpp"
#include "spacetime/factors.hpp"

namespace spacetime {

// Per-factor linearisation information produced during linearisation.
struct LinearizationData {
    Eigen::VectorXd e;   // factor error
    Eigen::MatrixXd E;   // factor Jacobian w.r.t. involved nodes
    Eigen::MatrixXd Q;   // factor information matrix (weight)
    std::vector<int> node_indices; // global node indices for this factor
};

// Container for data produced by a lower-level solve.
struct LowerLevelSolution {
    Eigen::VectorXd x;            // stacked perturbation vector x*
    Eigen::SparseMatrix<double> H; // information matrix (H)
    // Per-factor linearisation data: e_f, E_f, Q_f (copies)
    std::vector<LinearizationData> linearizations;
};

struct ValidationTarget {
    int node_index = -1;
    int sample_group_id = 0;
    Spacetime::SystemState<DTYPE>::Node ground_truth;
    Eigen::Matrix<int, 18, 1> observed_mask = Eigen::Matrix<int, 18, 1>::Ones();
};

struct ValidationLossData {
    Eigen::VectorXd nees_per_node;
    Eigen::VectorXd node_errors_stacked;
    double mean_nees = 0.0;
    double mahalanobis_sum = 0.0;
    double logdet_sum = 0.0;
    double mean_logdet = 0.0;
    double delta = 0.0;
    double loss = 0.0;
};

// Represents the lower-level estimation problem (factor graph and Q construction).
class OptimizationProblem {
public:
    struct RunSummary {
        int validation_sample_count = 0;
        Eigen::Matrix<double, 6, 1> R_pose = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 3, 1> R_gyro = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 18, 1> validation_variance = Eigen::Matrix<double, 18, 1>::Zero();
    };

    using Node = Spacetime::SystemState<DTYPE>::Node;

    OptimizationProblem();

    // Set the noise parameter vector theta in the order [diag(P0), diag(Q1), diag(Q2), diag(Q3)].
    void setTheta(const Eigen::VectorXd& theta);

    // Set the current factor graph and the node indices touched by each factor.
    void setProblemData(const std::vector<Node>& nodes,
                       const std::vector<std::shared_ptr<Spacetime::Factors::Factor>>& factors,
                       const std::vector<std::vector<int>>& factor_node_indices);

    // Set estimator options loaded externally (e.g. from config/estimator/batch.json).
    void setEstimatorOptions(const Spacetime::Options &options);

    // Set robot topology loaded externally (e.g. from config/robot/minimal.json and trial initial conditions).
    void setRobotTopology(const Spacetime::RobotTopology &topology);

    // Assemble and cache P0/Q1/Q2/Q3 from the current theta.
    void assembleFromTheta();

    // Solve the lower-level Gauss-Newton problem. Returns x*, H and linearisation data.
    LowerLevelSolution solveLowerLevel(bool verbose = false);

    // Set validation nodes and their corresponding ground-truth states.
    void setValidationTargets(const std::vector<ValidationTarget> &targets);

    void setRunSummary(const RunSummary &summary);

    // Compute validation NLL diagnostics and upper-level loss using the latest lower-level state.
    ValidationLossData computeValidationLoss(const LowerLevelSolution &solution) const;

    // Assemble the adjoint right-hand side b = (2*delta/V) * sum_v J_v^T * P_v^{-1} * e_v.
    Eigen::VectorXd computeAdjointRhs(const LowerLevelSolution &solution,
                                      const ValidationLossData &validation) const;

    // Compute geometric validation errors for validation nodes.
    Eigen::VectorXd computeValidationErrors() const;

    // Accessors for factor list and indexing helpers.
    const std::vector<std::shared_ptr<Spacetime::Factors::Factor>>& factors() const;
    const Eigen::VectorXd& theta() const;
    const std::vector<Node>& solvedNodes() const;
    const std::vector<ValidationTarget>& validationTargets() const;
    const Spacetime::RobotTopology& robotTopology() const;
    const RunSummary& runSummary() const;

private:
    Eigen::VectorXd theta_; // constrained parameters
    Spacetime::Hyperparameters hyperparameters_;
    std::vector<Node> nodes_;
    std::vector<std::shared_ptr<Spacetime::Factors::Factor>> factors_;
    std::vector<std::vector<int>> factor_node_indices_;
    std::vector<Node> solved_nodes_;
    std::vector<ValidationTarget> validation_targets_;
    RunSummary run_summary_;
    bool run_summary_set_ = false;
    Spacetime::Options estimator_options_;
    bool estimator_options_set_ = false;
    Spacetime::RobotTopology robot_topology_;
    bool robot_topology_set_ = false;
};

} // namespace spacetime
