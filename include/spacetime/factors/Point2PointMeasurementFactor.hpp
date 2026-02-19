#ifndef POINT2POINTMEASUREMENTFACTOR_HPP
#define POINT2POINTMEASUREMENTFACTOR_HPP

#include <Eigen/Dense>

#include "spacetime/factors/MeasurementFactor.hpp"
#include "spacetime/factors/map.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;
using ArrayVector3d = std::vector<steam_icp::Point3D>;

namespace Spacetime::Factors
{
    class Point2PointMeasurementFactor : public MeasurementFactor
    {

    public:
        Point2PointMeasurementFactor(Eigen::Matrix<double, 1, 1> weight,
                             const Spacetime::SensorMeasurement &meas, std::shared_ptr<steam_icp::Map> environment_map, Spacetime::Options::ICPOptions options)
            : MeasurementFactor(weight, meas), mp_environment_map(environment_map), m_options(options), m_scalar_weight(1.0 / weight(0,0))
        {
            this->m_meas.mask = Eigen::VectorXi::Ones(meas.value.rows());
        }

        ~Point2PointMeasurementFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            throw std::runtime_error("Point2PointMeasurementFactor::getError is not ready yet, do not use");
            // This error implementation is NOT differentiable - use of autodiff will not work.
            assert(nodes.size() == 1);

            Eigen::Vector<DTYPE, 1> e = Eigen::Vector<DTYPE, 1>::Zero();
            Eigen::Matrix<DTYPE, 4, 4> T_ib = invertTransformation(nodes[0].pose); // Inertial to body transformation

            Eigen::VectorXi valid_points = Eigen::VectorXi::Zero(this->m_meas_value.rows());
            // #pragma omp declare reduction(merge : Eigen::Vector<DTYPE, 1> : omp_out += omp_in)
            // #pragma omp parallel for reduction(merge : e)
            for (int i = 0; i < this->m_meas_value.rows(); i++)
            {
                Eigen::Vector<DTYPE, 3> p_b = this->m_meas_value.row(i).transpose();                 // Point in body frame
                Eigen::Vector<DTYPE, 3> q_i = T_ib.block<3, 3>(0, 0) * p_b + T_ib.block<3, 1>(0, 3); // Transform point to world frame

                ArrayVector3d vector_neighbors = mp_environment_map->searchNeighbors(q_i.cast<double>(), 1, m_options.VOXEL_SIZE, m_options.MAX_NEIGHBORS);
                if (vector_neighbors.size() < (size_t)m_options.MIN_NEIGHBORS) // Minimum number of neighbors
                {
                    // debug() << "Not enough neighbors: " << vector_neighbors.size();
                    continue;
                }
                auto neighborhood = steam_icp::Map::computeNeighborhoodDistribution(vector_neighbors);

                const double weight = std::pow(neighborhood.a2D, m_options.POWER_PLANARITY);

                Eigen::Vector<DTYPE, 3> p_i = vector_neighbors[0].pt.cast<DTYPE>();
                Eigen::Vector<DTYPE, 3> normal = neighborhood.normal.cast<DTYPE>();

                const double dist_to_plane = std::abs(((q_i - p_i).transpose() * normal).cast<double>()(0));
                if (dist_to_plane > m_options.MAX_DIST_TO_PLANE)
                {
                    // debug() << "Too far from plane: " << dist_to_plane;
                    continue;
                }

                // Compute point-to-plane error
                e = e + weight * normal.transpose() * (p_i - q_i);
                valid_points(i) = 1;
            }
            this->m_meas.mask = valid_points;

            return e;
        };

        Eigen::MatrixX<double> getJacobian(const std::vector<Node> &nodes, Eigen::VectorX<DTYPE> &e) const override
        {
            // debug() << "Using analytic jacobian.";
            // This error implementation is NOT differentiable - use of autodiff will not work.
            assert(nodes.size() == 1);

            Eigen::Matrix<DTYPE, 4, 4> T_ib = invertTransformation(nodes[0].pose); // Inertial to body transformation
            Eigen::Matrix<DTYPE, 6, 6> TranAd = se3::tranAd(T_ib);

            int zero_counter = 0;
            int min_neighbor_counter = 0;
            int dist_counter = 0;
            int count_valid_points = 0;
            int M = this->m_meas_value.rows() * 3;
            Eigen::VectorXi valid_points = Eigen::VectorXi::Zero(M);

            Eigen::VectorX<DTYPE> _e = Eigen::VectorX<DTYPE>::Zero(M);
            Eigen::MatrixXd _G = Eigen::MatrixXd::Zero(M, 6);
            Eigen::MatrixX<DTYPE> _W = Eigen::MatrixX<DTYPE>::Identity(M, M) * m_scalar_weight;
            // #pragma omp declare reduction(merge_error : Eigen::VectorX<DTYPE> : omp_out += omp_in)
            // #pragma omp declare reduction(merge_jac : Eigen::Matrix<double, 1, 6> : omp_out += omp_in)
            // #pragma omp parallel for reduction(merge_error : e) reduction(merge_jac : _G)
            for (int i = 0; i < M / 3; i++)
            {
                Eigen::Vector<DTYPE, 3> q_b = this->m_meas_value.row(i).transpose(); // Point in body frame
                // If the point is at the origin, skip it
                if (q_b.norm() < 1e-6)
                {
                    zero_counter++;
                    continue;
                }

                Eigen::Vector<DTYPE, 3> q_i = T_ib.block<3, 3>(0, 0) * q_b + T_ib.block<3, 1>(0, 3); // Transform point to world frame

                ArrayVector3d vector_neighbors = mp_environment_map->searchNeighbors(q_i.cast<double>(), m_options.VOXEL_SEARCH_RADIUS, m_options.VOXEL_SIZE, m_options.MAX_NEIGHBORS);
                if (vector_neighbors.size() < 1) // Must find at least one match
                {
                    min_neighbor_counter++;
                    // debug() << "Not enough neighbors: " << vector_neighbors.size();
                    continue;
                }


                Eigen::Vector<DTYPE, 3> p_i = vector_neighbors[0].pt.cast<DTYPE>();

                const Eigen::Vector<DTYPE, 3> difference = p_i - q_i;
                if (difference.cast<double>().transpose() * difference.cast<double>() > m_options.MAX_DIST_TO_PLANE * m_options.MAX_DIST_TO_PLANE)
                {
                    dist_counter++;
                    // debug() << "Too far from plane: " << dist_to_plane;
                    continue;
                }
                valid_points(i) = 1;

                _e.block<3, 1>(count_valid_points * 3, 0) = difference;
                _G.block<3, 6>(count_valid_points * 3, 0) = ((se3::point2fs(q_i.block<3, 1>(0, 0)) * TranAd).block<3, 6>(0, 0)).cast<double>();
                count_valid_points++;
            }
            this->m_meas.mask = valid_points;

            if (this->m_meas_value.rows() > 0) // Discard all points if less than 50% are valid
                if ((double)count_valid_points / (double)this->m_meas_value.rows() < m_options.rejection_threshold)
                {
                    warning() << "Point2PointMeasurementFactor: Only " << count_valid_points << " / " << this->m_meas_value.rows() << " points were valid.";
                    count_valid_points = 0;
                    this->m_meas.mask = Eigen::VectorXi::Zero(this->m_meas_value.rows());
                }

            // debug() << "Point2PointMeasurementFactor: total points: " << this->m_meas_value.rows()
            //           << ", success: " << count_valid_points
            //           << ", zero points: " << zero_counter
            //           << ", min_neighbors fails: " << min_neighbor_counter
            //           << ", behind_plane fails: " << behind_plane_counter
            //           << ", dist_to_plane fails: " << dist_counter
            //          ;

            m_weight.resize(count_valid_points * 3, count_valid_points * 3);
            m_weight = _W.block(0, 0, count_valid_points * 3, count_valid_points * 3);
            e.resize(count_valid_points * 3);
            e = _e.block(0, 0, count_valid_points * 3, 1);
            Eigen::MatrixXd S(count_valid_points * 3, 18);
            S << _G.block(0, 0, count_valid_points * 3, 6), Eigen::MatrixXd::Zero(count_valid_points * 3, 12);

            return S;
        };

    private:
        const std::shared_ptr<steam_icp::Map> mp_environment_map;
        const Spacetime::Options::ICPOptions m_options;
        const double  m_scalar_weight;
    };
}
#endif // POINT2POINTMEASUREMENTFACTOR_HPP