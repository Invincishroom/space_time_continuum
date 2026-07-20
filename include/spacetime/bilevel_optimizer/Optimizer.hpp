#pragma once

#include <memory>
#include <vector>
#include "OptimizationProblem.hpp"

namespace spacetime {

// High-level bilevel optimizer implementing the algorithm from proposal.tex.
class Optimizer {
public:
    struct Result {
        Eigen::VectorXd phi; // unconstrained params
        Eigen::VectorXd theta; // constrained (exp(phi))
        double loss = 0.0;
        int iterations = 0;
        bool converged = false;
    };
    struct OptimizerConfig {
        int max_iterations = 100;
        double learning_rate = 0.01;
        double tol_grad = 1e-5;
        double tol_loss = 1e-5;
        bool use_exponential_param = true; // if true, theta = exp(phi)
        bool verbose = false; // if true, print debug info
    } optimizer_config;
    std::shared_ptr<OptimizationProblem> problem_;
    Optimizer(const OptimizationProblem& problem);
    void setConfig(const OptimizerConfig& cfg);
    // Run the outer optimization loop; returns final state.
    Result optimize();

private:

    // Single outer iteration: assemble Q, solve lower level, compute adjoint,
    // accumulate gradients and perform phi update.
    Result step(const Eigen::VectorXd& phi);
};

} // namespace spacetime
