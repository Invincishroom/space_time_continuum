#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "spacetime/types.hpp"

namespace Spacetime
{
    class Solver
    {
    public:
        Solver() {};

        void extractCovariances(const Eigen::SparseMatrix<double> &L, const RobotTopology &topology, Eigen::MatrixX<double> &covariances, int timesteps_to_extract = -1, bool reverse_order = false);

    private:
        Eigen::SparseMatrix<DTYPE>
            m_L, m_LT, m_D;
    };
} // namespace Spacetime
#endif // SOLVER_H