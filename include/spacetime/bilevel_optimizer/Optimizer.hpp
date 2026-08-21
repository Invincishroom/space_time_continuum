#pragma once

#include <memory>
#include <vector>
#include "OptimizationProblem.hpp"

namespace spacetime {

// High-level bilevel optimizer implementing the algorithm from proposal.tex.
class Optimizer {
public:
    struct Result {
        Eigen::VectorXd psi;    // unconstrained precision params
        Eigen::VectorXd lambda;  // positive precision params
        Eigen::VectorXd theta;   // positive variance params (1 / lambda)
        double loss = 0.0;
        int iterations = 0;
        bool converged = false;
    };
    struct OptimizerConfig {
        int max_iterations = 100;
        double learning_rate = 0.01;
        bool use_adam = true;
        double adam_beta1 = 0.9;
        double adam_beta2 = 0.999;
        double adam_epsilon = 1e-8;
        double tol_grad = 1e-5;
        double tol_loss = 1e-5;
        bool use_exponential_param = true; // if true, lambda = exp(psi)
        bool verbose = false; // if true, print debug info
        bool enable_gradient_fd_check = false; // if true, run factor-kernel finite-difference checks on iteration 0
        bool enable_unary_factor_fd_check = true; // if true, run unary factor finite-difference check
        bool enable_time_factor_fd_check = true; // if true, run time factor finite-difference check
        bool enable_space_factor_fd_check = true; // if true, run space factor finite-difference check
        double gradient_fd_epsilon = 1e-6; // central-difference epsilon for gradient checks
        bool gradient_fd_use_lambda_perturbation = true; // if true, perturb lambda and chain-rule back to theta for FD checks
        bool freeze_p0_non_pose = false; // if true, freeze lambda[6..17] so only pose-related P0 terms are updated
        double max_gradient_update_norm = 1e4; // clip update direction norm before applying optimizer step
        double max_psi_step_norm = 0.5; // clip per-iteration psi update norm
        double min_theta = 1e-8; // lower clamp for theta after inversion
        double max_theta = 1e3; // upper clamp for theta after inversion
    } optimizer_config;
    std::shared_ptr<OptimizationProblem> problem_;
    Optimizer(const OptimizationProblem& problem);
    void setConfig(const OptimizerConfig& cfg);
    // Run the outer optimization loop; returns final state.
    Result optimize();

private:

    // Single outer iteration: assemble Q, solve lower level, compute adjoint,
    // accumulate gradients and perform psi update.
    Result step(const Eigen::VectorXd& psi);
};

} // namespace spacetime
