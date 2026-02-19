#include <iostream>

#include <Eigen/SparseLU>

#include "spacetime/Solver.hpp"
#include "spacetime/utilities.hpp"

namespace Spacetime
{
    void Solver::extractCovariances(const Eigen::SparseMatrix<double> &L, const RobotTopology &topology, Eigen::MatrixX<double> &covariances, int timesteps_to_extract, bool reverse_order)
    {
        const int K = (int)topology.K;
        const int N = (int)topology.N;
        const int timesteps = (timesteps_to_extract == -1) ? K : timesteps_to_extract;

        // Covariance: (K * N * 18) x (2 * N * 18)
        // Initialize
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        const int size = 18 * N;
        covariances.resize(K * size, 2 * size);
        covariances.setZero();

        Eigen::MatrixX<double> Linv_r = Eigen::MatrixX<double>::Zero(size, size);

        Eigen::SparseMatrix<double> EYE_r;
        EYE_r.reserve(Eigen::VectorXi::Constant(size, 1));

        const Eigen::SparseMatrix<double> EYE = [size]
        {
            Eigen::SparseMatrix<double> mat(size, size);
            mat.setIdentity();
            return mat;
        }();

        // Implementation of Eq. 3.73 in book
        for (int k = K - 1; k >= K - timesteps; k--)
        {
            // Set up
            const Eigen::SparseMatrix<double> Lk = L.block(size * k, size * k, size, size);

            const Eigen::SparseMatrix<double> S = getFullRankProjection(Lk);
            EYE_r.resize(S.cols(), S.cols());
            EYE_r.setIdentity();

            // Invert non-zero portion and project back

            // Lk_r = S.transpose() * Lk * S;
            const Eigen::SparseMatrix<double> temp1 = S.transpose() * Lk;
            const Eigen::SparseMatrix<double> Lk_r = temp1 * S;

            if (Lk_r.rows() * Lk_r.cols() == 0)
                // Handles case where Lk is zero
                Linv_r = Lk_r;
            else
            {
                solver.compute(Lk_r);
                Linv_r = solver.solve(EYE_r);
            }

            // Linv = S * Linv_r * S.transpose();
            const Eigen::MatrixX<double> temp2 = S * Linv_r;
            const Eigen::MatrixX<double> S_trans = S.transpose();
            const Eigen::MatrixX<double> Linv = temp2 * S_trans;
            const Eigen::MatrixX<double> Linv_trans = Linv.transpose();

            if (reverse_order)
            {
                if (k == K - 1)
                {
                    // Eq. 3.74
                    covariances.block((K - 1 - k) * size, size, size, size) = Linv_trans * Linv; // Cov_k_k
                    continue;
                }

                const Eigen::SparseMatrix<double> L1 = L.block(size * (k + 1), size * k, size, size);

                // Eq. 3.73a
                // diag =Linv_trans * (EYE + L1.transpose() * covariances.block((K - 2 - k) * size, size, size, size) * L1) * Linv;
                const Eigen::MatrixX<double> temp3 = covariances.block((K - 2 - k) * size, size, size, size) * L1;
                const Eigen::MatrixX<double> L1_trans = L1.transpose();
                const Eigen::MatrixX<double> temp4 = L1_trans * temp3;
                const Eigen::MatrixX<double> temp5 = EYE + temp4;
                const Eigen::MatrixX<double> temp6 = Linv_trans * temp5;
                const Eigen::MatrixX<double> diag = temp6 * Linv;
                covariances.block((K - 1 - k) * size, size, size, size) = diag; // Diagonal, Cov_k_k

                // Eq. 3.73b
                // offdiag = -covariances.block((K - 2 - k) * size, size, size, size) * L1 * Linv;
                const Eigen::MatrixX<double> temp7 = L1 * Linv;
                const Eigen::MatrixX<double> offdiag = -covariances.block((K - 2 - k) * size, size, size, size) * temp7;
                covariances.block((K - 1 - k) * size, 0, size, size) = offdiag.transpose(); // Off-diagonal, Cov_k_k+1
            }
            else
            {
                if (k == K - 1)
                {
                    // Eq. 3.74
                    covariances.block(k * size, size, size, size) =Linv_trans * Linv; // Cov_k_k
                    continue;
                }

                const Eigen::SparseMatrix<double> L1 = L.block(size * (k + 1), size * k, size, size);

                // Eq. 3.73a
                // diag =Linv_trans * (EYE + L1.transpose() * covariances.block((k + 1) * size, size, size, size) * L1) * Linv;
                const Eigen::MatrixX<double> temp3 = covariances.block((k + 1) * size, size, size, size) * L1;
                const Eigen::MatrixX<double> L1_trans = L1.transpose();
                const Eigen::MatrixX<double> temp4 = L1_trans * temp3;
                const Eigen::MatrixX<double> temp5 = EYE + temp4;
                const Eigen::MatrixX<double> temp6 = Linv_trans * temp5;
                const Eigen::MatrixX<double> diag = temp6 * Linv;
                covariances.block(k * size, size, size, size) = diag; // Diagonal, Cov_k_k

                // Eq. 3.73b
                // offdiag = -covariances.block((K - 2 - k) * size, size, size, size) * L1 * Linv;
                const Eigen::MatrixX<double> temp7 = L1 * Linv;
                const Eigen::MatrixX<double> offdiag = -covariances.block((k + 1) * size, size, size, size) * temp7;
                covariances.block((k + 1) * size, 0, size, size) = offdiag; // Off-diagonal, Cov_k_k+1
            }
        }
    }
}
