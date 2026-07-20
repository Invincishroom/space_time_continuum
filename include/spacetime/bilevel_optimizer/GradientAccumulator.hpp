#pragma once

#include <Eigen/Dense>
#include <vector>
#include "FactorGradient.hpp"

namespace spacetime {

// Accumulates gradients over all factors using a provided FactorGradient strategy.
class GradientAccumulator {
public:
    GradientAccumulator() = default;

    // Accumulate into dL_dtheta (same ordering as theta vector).
    void accumulate(const std::vector<FactorGradientContrib> &per_factor_contribs,
                    Eigen::VectorXd &dL_dtheta);
};

} // namespace spacetime
