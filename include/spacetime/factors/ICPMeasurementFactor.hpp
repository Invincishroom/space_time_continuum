#ifndef ICPMEASUREMENTFACTOR_H
#define ICPMEASUREMENTFACTOR_H

#include <Eigen/Dense>

#include "spacetime/factors/MeasurementFactor.hpp"
#include "spacetime/factors/map.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/types.hpp"

using Node = Spacetime::SystemState<DTYPE>::Node;
using ArrayVector3d = std::vector<steam_icp::Point3D>;

namespace Spacetime::Factors
{
    class ICPMeasurementFactor : public MeasurementFactor
    {

    public:
        ICPMeasurementFactor(Eigen::Matrix<double, 1, 1> weight,
                             const Spacetime::SensorMeasurement &meas, std::shared_ptr<steam_icp::Map> environment_map, Spacetime::Options::ICPOptions options)
            : MeasurementFactor(weight, meas), mp_environment_map(environment_map), m_options(options), m_scalar_uncertainty(weight(0, 0))
        {
            this->m_meas.mask = Eigen::VectorXi::Ones(meas.value.rows());
        }

        ~ICPMeasurementFactor() = default;

        Eigen::VectorX<DTYPE> getError(const std::vector<Node> &nodes) const override
        {
            throw std::runtime_error("ICPMeasurementFactor::getError is not ready yet, do not use");
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
            int M = this->m_meas_value.rows();
            Eigen::VectorXi valid_points = Eigen::VectorXi::Zero(M);

            Eigen::VectorX<DTYPE> _e = Eigen::VectorX<DTYPE>::Zero(M);
            Eigen::MatrixXd _G = Eigen::MatrixXd::Zero(M, 6);
            Eigen::MatrixX<DTYPE> _W = Eigen::MatrixX<DTYPE>::Zero(M, M);
            // #pragma omp declare reduction(merge_error : Eigen::VectorX<DTYPE> : omp_out += omp_in)
            // #pragma omp declare reduction(merge_jac : Eigen::Matrix<double, 1, 6> : omp_out += omp_in)
            // #pragma omp parallel for reduction(merge_error : e) reduction(merge_jac : _G)
            for (int i = 0; i < M; i++)
            {
                Eigen::Vector<DTYPE, 3> q_b = this->m_meas_value.row(i).transpose(); // Point in body frame

                double point_norm = static_cast<double>(q_b.norm());
                // If the point is at the origin, skip it
                if (point_norm < 1e-6)
                {
                    zero_counter++;
                    continue;
                }

                Eigen::Vector<DTYPE, 3> q_i = T_ib.block<3, 3>(0, 0) * q_b + T_ib.block<3, 1>(0, 3); // Transform point to world frame

                ArrayVector3d vector_neighbors = mp_environment_map->searchNeighbors(q_i.cast<double>(), m_options.VOXEL_SEARCH_RADIUS, m_options.VOXEL_SIZE, m_options.MAX_NEIGHBORS);
                if (vector_neighbors.size() < (size_t)m_options.MIN_NEIGHBORS) // Minimum number of neighbors
                {
                    min_neighbor_counter++;
                    // debug() << "Not enough neighbors: " << vector_neighbors.size();
                    continue;
                }

                auto neighborhood = steam_icp::Map::computeNeighborhoodDistribution(vector_neighbors);

                const double weight = std::pow(neighborhood.a2D, m_options.POWER_PLANARITY);

                Eigen::Vector<DTYPE, 3> p_i = vector_neighbors[0].pt.cast<DTYPE>();
                Eigen::Vector<DTYPE, 3> normal = neighborhood.normal.cast<DTYPE>();

                // const DTYPE dist_to_plane = normal.transpose() * (p_i - q_i);
                // if (std::abs(static_cast<double>(dist_to_plane)) > m_options.MAX_DIST_TO_PLANE)
                // {
                //     dist_counter++;
                //     // debug() << "Too far from plane: " << dist_to_plane;
                //     continue;
                // }
                valid_points(i) = 1;
                _e(count_valid_points) = weight * normal.transpose() * (p_i - q_i);

                double scalar_uncertainty = get_distance_weight(point_norm) * weight * weight + m_scalar_uncertainty;
                double scalar_weight = 1.0 / (scalar_uncertainty);

                DTYPE eT_W_e = scalar_weight * _e(count_valid_points) * _e(count_valid_points);
                // DTYPE robust_factor = 1.0; // Squared error 
                DTYPE robust_factor = 1.0 / (1.0 + eT_W_e); // Cauchy robust cost
                // DTYPE robust_factor = 1.0 / ((1.0 + eT_W_e) * (1.0 + eT_W_e)); // Geman McClure robust cost

                _W(count_valid_points, count_valid_points) = scalar_weight * robust_factor;
                _G.block<1, 6>(count_valid_points, 0) = (weight * normal.transpose() * (se3::point2fs(q_i.block<3, 1>(0, 0)) * TranAd).block<3, 6>(0, 0)).cast<double>();
                count_valid_points++;
            }
            this->m_meas.mask = valid_points;

            if (this->m_meas_value.rows() > 0) // Discard all points if less than 50% are valid
                if ((double)count_valid_points / (double)this->m_meas_value.rows() < m_options.rejection_threshold)
                {
                    warning() << "ICPMeasurementFactor: Only " << count_valid_points << " / " << this->m_meas_value.rows() << " points were valid.";
                    count_valid_points = 0;
                    this->m_meas.mask = Eigen::VectorXi::Zero(this->m_meas_value.rows());
                }

            // debug() << "ICPMeasurementFactor: total points: " << this->m_meas_value.rows()
            //           << ", success: " << count_valid_points
            //           << ", zero points: " << zero_counter
            //           << ", min_neighbors fails: " << min_neighbor_counter
            //           << ", dist_to_plane fails: " << dist_counter
            //          ;

            m_weight.resize(count_valid_points, count_valid_points);
            m_weight = _W.block(0, 0, count_valid_points, count_valid_points);
            e.resize(count_valid_points);
            e = _e.block(0, 0, count_valid_points, 1);
            Eigen::MatrixXd S(count_valid_points, 18);
            S << _G.block(0, 0, count_valid_points, 6), Eigen::MatrixXd::Zero(count_valid_points, 12);

            return S;
        };

        std::shared_ptr<steam_icp::Map> getMap() const { return mp_environment_map; }

        double get_distance_weight(double distance) const 
        {
            double std = 0.0; 

            if (distance < 0.025) // 2.5 cm
                throw std::runtime_error("Points closer than 2.5 cm should have been filtered out earlier."); 
            else if (distance < 0.6) // 60 cm
                std = ((distance - 0.025) / (0.6 - 0.025) * (0.012 - 0.014) + 0.014) * distance; // Linear from 0.014 to 0.012
            else if (distance < 1.2)
                std = ((distance - 0.6) / (1.2 - 0.6) * (0.006 - 0.012) + 0.012) * distance; // Linear from 0.012 to 0.01
            else 
                std = 0.006 * distance; 

            return std * std;
        }; 

        double get_scalar_uncertainty() const { return m_scalar_uncertainty; }

    private:
        const std::shared_ptr<steam_icp::Map> mp_environment_map;
        const Spacetime::Options::ICPOptions m_options;
        const double m_scalar_uncertainty;
    };
}
#endif // ICPMEASUREMENTFACTOR_H