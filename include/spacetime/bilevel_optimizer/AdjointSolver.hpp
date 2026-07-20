#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace spacetime {

// Abstract interface for solving linear systems involving the information matrix H.
class AdjointSolver {
public:
    virtual ~AdjointSolver() = default;

    // Solve H * lambda = b. The implementation may reuse factorization from the
    // lower-level solve when available.
    virtual Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& H,
                                  const Eigen::VectorXd& b) = 0;
};

} // namespace spacetime
