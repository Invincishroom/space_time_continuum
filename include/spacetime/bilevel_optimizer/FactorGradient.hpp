#pragma once

#include <Eigen/Dense>
#include <vector>

namespace spacetime {

// Helper utilities for factor-wise gradient contributions described in proposal.tex.
struct FactorGradientContrib {
    Eigen::VectorXd dL_dtheta_state; // state-sensitivity contribution for this factor (per-diagonal)
    Eigen::VectorXd dL_dtheta_info;  // information-term contribution for this factor (per-diagonal)
};

class FactorGradient
{
public:
    virtual ~FactorGradient() = default;

    // Compute v_f = Q_f^{-1} e_f, u_f = Q_f^{-1} E_f lambda_f, w_f = Q_f^{-1} E_f e_nodes
    // and return the two contributions (state and info) for the diagonal parameters of Q.
    virtual FactorGradientContrib compute(const Eigen::VectorXd &e_f,
                                          const Eigen::MatrixXd &Q_f,
                                          const Eigen::MatrixXd &E_f,
                                          const Eigen::VectorXd &lambda_f,
                                          const Eigen::VectorXd &e_nodes,
                                          double scale = 1.0) const = 0;
};

} // namespace spacetime
