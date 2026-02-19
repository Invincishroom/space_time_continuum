#include <filesystem>

#include "export_tools.hpp"
#include "spacetime/static_funcs.hpp"
#include "spacetime/estimators/Estimator.hpp"

namespace Spacetime
{
    void printError(Estimator *estimator, const SystemState<DTYPE> &state, const std::vector<SensorMeasurement> &gt)
    {
        SystemState<DTYPE> _state = state;
        if (_state.ib)
            _state.convertStateMeanBodyInertial(); // Switches to bi form if needed

        std::map<double, std::vector<double>> total_error, tran_error, rot_error;

        double start_time = _state.estimation_nodes.front().time;
        double end_time = _state.estimation_nodes.back().time;

        // Interpolate the states at the ground truth measurements
        _state.interpolation_nodes.clear();
        for (const auto &measurement : gt)
        {
            assert(measurement.type == SensorMeasurement::Pose && "Ground truth measurements must be of type Pose.");
            if (measurement.t < start_time - TOLERANCE || measurement.t > end_time + TOLERANCE)
                continue;

            SystemState<DTYPE>::Node node;
            node.arclength = measurement.s;
            if (node.arclength <= 0.0)
                continue;
            node.time = measurement.t;

            estimator->interpolateMean2D(node, _state);

            Eigen::Matrix<DTYPE, 4, 4> error_tran = node.pose * measurement.value;
            Eigen::Matrix<DTYPE, 6, 1> error_vec = se3::tran2vec(error_tran);

            Eigen::Matrix<DTYPE, 3, 1> error_rot_vec = error_vec.block<3, 1>(3, 0);
            Eigen::Matrix<DTYPE, 3, 1> error_tran_vec = error_tran.block<3, 1>(0, 3);

            if (tran_error.find(measurement.s) == tran_error.end())
            {
                tran_error[measurement.s] = std::vector<double>{static_cast<double>(error_tran_vec.norm())};
                rot_error[measurement.s] = std::vector<double>{static_cast<double>(error_rot_vec.norm())};
                total_error[measurement.s] = std::vector<double>{static_cast<double>(error_vec.norm())};
            }
            else
            {
                tran_error[measurement.s].push_back(static_cast<double>(error_tran_vec.norm()));
                rot_error[measurement.s].push_back(static_cast<double>(error_rot_vec.norm()));
                total_error[measurement.s].push_back(static_cast<double>(error_vec.norm()));
            }
        }

        std::cout << "Error Analysis\n===============================" << std::endl;
        std::cout << "-------------------------------" << std::endl << std::endl;
        for (const auto &item : tran_error)
        {
            double sq_sum = 0.0;
            double sum = 0.0; 
            for (const auto &val : item.second)
            {
                sq_sum += val * val;
                sum += val;
            }
            double rmse = std::sqrt(sq_sum / item.second.size());
            double mean = sum / item.second.size();
            std::cout << "Arclength: " << item.first << ", Translation RMSE: " << rmse << ", Percentage: " << (rmse / item.first) * 100 << " %" << std::endl;
            std::cout << "Arclength: " << item.first << ", Translation Mean Error: " << mean << ", Percentage: " << (mean / item.first) * 100 << " %" << std::endl;
        }
        std::cout << "-------------------------------" << std::endl << std::endl;
        for (const auto &item : rot_error)
        {
            double sq_sum = 0.0;
            double sum = 0.0; 
            for (const auto &val : item.second)
            {
                sq_sum += val * val;
                sum += val;
            }
            double rmse = std::sqrt(sq_sum / item.second.size());
            double mean = sum / item.second.size();
            std::cout << "Arclength: " << item.first << ", Rotation RMSE: " << rmse << " rad" << std::endl;
            std::cout << "Arclength: " << item.first << ", Rotation Mean Error: " << mean << " rad" << std::endl;
        }
        std::cout << "-------------------------------" << std::endl << std::endl;
        for (const auto &item : total_error)
        {
            double sq_sum = 0.0;
            double sum = 0.0; 
            for (const auto &val : item.second)
            {
                sq_sum += val * val;
                sum += val;
            }
            double rmse = std::sqrt(sq_sum / item.second.size());
            double mean = sum / item.second.size();
            std::cout << "Arclength: " << item.first << ", Total RMSE: " << rmse << std::endl;
            std::cout << "Arclength: " << item.first << ", Total Mean Error: " << mean << std::endl;
        }
        std::cout << "-------------------------------" << std::endl << std::endl;
    }

    void exportErrorData(Estimator *estimator, const SystemState<DTYPE> &state, const std::vector<SensorMeasurement> &gt, std::string output_path)
    {
        SystemState<DTYPE> _state = state;
        if (_state.ib)
            _state.convertStateMeanBodyInertial(); // Switches to bi form if needed

        // Interpolate the states at the ground truth measurements
        _state.interpolation_nodes.clear();
        for (const auto &measurement : gt)
        {
            assert(measurement.type == SensorMeasurement::Pose && "Ground truth measurements must be of type Pose.");

            SystemState<DTYPE>::Node node;
            node.arclength = measurement.s;
            node.time = measurement.t;

            if (_state.estimation_nodes[0].covarianceAvailable())
                estimator->queryState(node, _state);
            else
                estimator->interpolateMean2D(node, _state);
            _state.interpolation_nodes.push_back(node);
        }

        _state.convertStateMeanBodyInertial(); // Switches to ib form

        { // Save predicted state
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/error_pred.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            size_t index = 0;
            *log_file_stream << "time,arclength,x1,x2,x3,x4,x5,x6,e1,e2,e3,e4,e5,e6,v1,v2,v3,v4,v5,v6," << std::endl;
            for (auto &measurement : gt)
            {
                assert(index < _state.interpolation_nodes.size() && "Ground truth measurements must match the interpolation nodes size.");
                assert(std::abs(measurement.t - _state.interpolation_nodes[index].time) < TOLERANCE && "Ground truth measurements must match the interpolation nodes time within the tolerance.");

                *log_file_stream << _state.interpolation_nodes[index].time << "," << _state.interpolation_nodes[index].arclength << ",";
                Eigen::Matrix<DTYPE, 6, 1> measurement_value = se3::tran2vec(_state.interpolation_nodes[index].pose);
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << _state.interpolation_nodes[index].epsilon(n) << ",";
                }
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << _state.interpolation_nodes[index].varpi(n) << ",";
                }
                *log_file_stream << std::endl;

                index++;
            }
            log_file_stream->close();
        }

        { // Save covariances if available
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/error_pred_cov.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            size_t index = 0;
            *log_file_stream << "time,arclength," << std::endl;
            for (auto &measurement : gt)
            {
                assert(index < _state.interpolation_nodes.size() && "Ground truth measurements must match the interpolation nodes size.");
                assert(std::abs(measurement.t - _state.interpolation_nodes[index].time) < TOLERANCE && "Ground truth measurements must match the interpolation nodes time within the tolerance.");

                if (_state.interpolation_nodes[index].covarianceAvailable())
                {
                    *log_file_stream << _state.interpolation_nodes[index].time << "," << _state.interpolation_nodes[index].arclength << ",";
                    Eigen::Matrix<double, 18, 18> covariance = _state.interpolation_nodes[index].getCovariance();
                    for (int row = 0; row < 18; row++)
                    {
                        for (int col = 0; col < 18; col++)
                        {
                            *log_file_stream << covariance(row, col);
                            *log_file_stream << ",";
                        }
                    }
                    *log_file_stream << std::endl;
                }

                index++;
            }
            log_file_stream->close();
        }

        { // Save gt data
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/error_gt.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            *log_file_stream << "time,arclength,x1,x2,x3,x4,x5,x6," << std::endl;
            for (auto &measurement : gt)
            {
                *log_file_stream << measurement.t << "," << measurement.s << ",";
                Eigen::Matrix<double, 6, 1> measurement_value = se3::tran2vec(measurement.value);
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                *log_file_stream << std::endl;
            }
            log_file_stream->close();
        }
    }

    void exportTrajData(const SystemState<DTYPE> &state, const std::vector<SensorMeasurement> &gt, const std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &meas, std::string output_path)
    {
        SystemState<DTYPE> _state = state;
        if (!_state.ib)
            _state.convertStateMeanBodyInertial();
        { // Save predicted state
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_pred.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            int index = 0;
            *log_file_stream << "time,arclength,x1,x2,x3,x4,x5,x6,e1,e2,e3,e4,e5,e6,v1,v2,v3,v4,v5,v6," << std::endl;
            for (SystemState<DTYPE>::Node node : _state.interpolation_nodes)
            {
                *log_file_stream << node.time << "," << node.arclength << ",";
                Eigen::Matrix<DTYPE, 6, 1> measurement_value = se3::tran2vec(node.pose);
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << node.epsilon(n) << ",";
                }
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << node.varpi(n) << ",";
                }
                *log_file_stream << std::endl;

                index++;
            }
            log_file_stream->close();
        }

        { // Save covariances if available
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_pred_cov.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            int index = 0;
            *log_file_stream << "time,arclength," << std::endl;
            for (SystemState<DTYPE>::Node node : _state.interpolation_nodes)
            {
                if (node.covarianceAvailable())
                {
                    *log_file_stream << node.time << "," << node.arclength << ",";
                    Eigen::Matrix<double, 18, 18> covariance = node.getCovariance();
                    for (int row = 0; row < 18; row++)
                    {
                        for (int col = 0; col < 18; col++)
                        {
                            *log_file_stream << covariance(row, col);
                            *log_file_stream << ",";
                        }
                    }
                    *log_file_stream << std::endl;
                }

                index++;
            }
            log_file_stream->close();
        }

        { // Save predicted _state
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_pred_est.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            int index = 0;
            *log_file_stream << "time,arclength,x1,x2,x3,x4,x5,x6,e1,e2,e3,e4,e5,e6,v1,v2,v3,v4,v5,v6," << std::endl;
            for (SystemState<DTYPE>::Node node : _state.estimation_nodes)
            {
                *log_file_stream << node.time << "," << node.arclength << ",";
                Eigen::Matrix<DTYPE, 6, 1> measurement_value = se3::tran2vec(node.pose);
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << node.epsilon(n) << ",";
                }
                for (int n = 0; n < 6; n++)
                {
                    *log_file_stream << node.varpi(n) << ",";
                }
                *log_file_stream << std::endl;

                index++;
            }
            log_file_stream->close();
        }

        { // Save covariances
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_pred_cov_est.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            int index = 0;
            *log_file_stream << "time,arclength," << std::endl;
            for (SystemState<DTYPE>::Node node : _state.estimation_nodes)
            {
                if (node.covarianceAvailable())
                {
                    *log_file_stream << node.time << "," << node.arclength << ",";
                    Eigen::Matrix<double, 18, 18> covariance = node.getCovariance();
                    for (int row = 0; row < 18; row++)
                    {
                        for (int col = 0; col < 18; col++)
                        {
                            *log_file_stream << covariance(row, col);
                            *log_file_stream << ",";
                        }
                    }
                    *log_file_stream << std::endl;
                }

                index++;
            }
            log_file_stream->close();
        }

        { // Save gt data
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_gt.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            *log_file_stream << "time,arclength,x1,x2,x3,x4,x5,x6," << std::endl;
            for (auto &measurement : gt)
            {
                *log_file_stream << measurement.t << "," << measurement.s << ",";
                Eigen::Matrix<double, 6, 1> measurement_value = se3::tran2vec(measurement.value);
                for (int n = 0; n < measurement_value.size(); n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                *log_file_stream << std::endl;
            }
            log_file_stream->close();
        }

        { // Save pose meas data
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_meas_pose.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            *log_file_stream << "time,arclength,x1,x2,x3,x4,x5,x6," << std::endl;
            for (auto &measurement : meas)
            {
                if (measurement->getMeas().type != SensorMeasurement::Type::Pose)
                    continue; // Only save pose measurements
                *log_file_stream << measurement->getMeas().t << "," << measurement->getMeas().s << ",";
                Eigen::Matrix<double, 6, 1> measurement_value = se3::tran2vec(measurement->getMeas().value);
                for (int n = 0; n < measurement_value.size(); n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                *log_file_stream << std::endl;
            }
            log_file_stream->close();
        }

        { // Save gyro meas data
            std::ofstream *log_file_stream = new std::ofstream();
            std::filesystem::path log_file_path = output_path + "/logs/traj_meas_gyro.txt";
            log_file_stream->open(log_file_path, std::ios::out);

            *log_file_stream << "time,arclength,vr,vp,vy," << std::endl;
            for (auto &measurement : meas)
            {
                if (measurement->getMeas().type != SensorMeasurement::Type::Gyro)
                    continue; // Only save pose measurements
                *log_file_stream << measurement->getMeas().t << "," << measurement->getMeas().s << ",";
                Eigen::Matrix<double, 3, 1> measurement_value = measurement->getMeas().value;
                for (int n = 0; n < measurement_value.size(); n++)
                {
                    *log_file_stream << measurement_value(n) << ",";
                }
                *log_file_stream << std::endl;
            }
            log_file_stream->close();
        }
    }

    void exportRuntimeData(const Estimator::Results &result, std::string output_path)
    {
        std::ofstream *log_file_stream = new std::ofstream();
        std::filesystem::path log_file_path = output_path + "/logs/results.txt";
        log_file_stream->open(log_file_path, std::ios::out);

        *log_file_stream << "Success: " << result.success << std::endl;
        *log_file_stream << "Message: " << result.message.str() << std::endl;
        *log_file_stream << "Cost: " << result.cost << std::endl;
        *log_file_stream << "Runtimes (us): ";
        for (const auto &runtime : result.runtimes)
        {
            *log_file_stream << runtime << ", ";
        }
        *log_file_stream << std::endl;

        log_file_stream->close();
    }

    void exportMapData(std::shared_ptr<steam_icp::Map> map, std::string output_path)
    {
        std::string map_path = output_path + "/map_estimate.csv";
        map->save(map_path);
    }
    void getRPYFromPose(const Eigen::Matrix4d &pose, double &roll, double &pitch, double &yaw)
    {
        Eigen::Vector3d rpy = pose.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
        yaw = rpy(0);
        pitch = rpy(1);
        roll = rpy(2);
    }
    void exportPointCloudData(Estimator *estimator, const SystemState<DTYPE> &state, const std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &meas, std::string output_path)
    {
        /*
        NODE x y z roll pitch yaw
        x y z
        x y z
        [...]
        */
        auto _state = state;
        if (!_state.ib)
            _state.convertStateMeanBodyInertial();
        std::string pc_path = output_path + "/point_cloud.log";
        std::ofstream *pc_file_stream = new std::ofstream();
        pc_file_stream->open(pc_path, std::ios::out);

        for (const auto &measurement_factor : meas)
        {
            if (measurement_factor->getMeas().type != SensorMeasurement::Type::Lidar)
                continue;
            std::shared_ptr<Spacetime::Factors::ICPMeasurementFactor> factor = std::dynamic_pointer_cast<Spacetime::Factors::ICPMeasurementFactor>(measurement_factor);
            SystemState<DTYPE>::Node operating_point = measurement_factor->getOperatingPoint();

            Eigen::MatrixXd ps_b = measurement_factor->getMeas().value;
            Eigen::Matrix<double, 4, 4> T_bi = operating_point.pose.cast<double>();
            Eigen::Matrix<double, 4, 4> T_ib = invertTransformation(T_bi);
            double roll, pitch, yaw;
            getRPYFromPose(T_ib, roll, pitch, yaw);
            *pc_file_stream << "NODE " << T_ib(0, 3) << " " << T_ib(1, 3) << " " << T_ib(2, 3) << " " << roll << " " << pitch << " " << yaw << " " << measurement_factor->getMeas().t << " " << measurement_factor->getMeas().s << std::endl;

            for (int i = 0; i < ps_b.rows(); i++)
            {
                Eigen::Vector3d p_b = ps_b.row(i).transpose();
                Eigen::Vector3d p_i = T_ib.block<3, 3>(0, 0) * p_b + T_ib.block<3, 1>(0, 3);

                auto closest_node = getClosestNode(estimator, _state, p_i, operating_point.time);
                if (state.estimation_nodes[0].covarianceAvailable())
                {
                    estimator->queryState(closest_node, _state);
                }

                double distance_to_robot = (closest_node.pose.block<3, 1>(0, 3).cast<double>() - p_i).norm();
                if (distance_to_robot < estimator->getRobotTopology().radius + 0.025) // Robot radius + sensor low distance_to_robot dropout of 0.25cm
                {
                    warning() << "Point too close to robot surface: " << distance_to_robot << " m. Skipping point export.";
                    continue;
                }

                *pc_file_stream << p_b(0, 0) << " " << p_b(1, 0) << " " << p_b(2, 0); 
                if (closest_node.covarianceAvailable())
                {
                    double distance = p_b.norm(); 
                    double scalar_weight = factor->get_distance_weight(distance) + factor->get_scalar_uncertainty(); 

                    Eigen::Matrix<double, 4, 6> p_dot = se3::point2fs(p_i);
                    Eigen::Matrix<double, 6, 6> Tad = se3::tranAd(T_ib);
                    Eigen::Matrix<double, 4, 4> temp = p_dot * Tad * closest_node.getCovariance().block<6, 6>(0, 0) * Tad.transpose() * p_dot.transpose();

                    Eigen::Matrix3d cov_point = temp.block<3, 3>(0, 0) + scalar_weight * Eigen::Matrix3d::Identity();

                    *pc_file_stream << " " << cov_point(0, 0) << " " << cov_point(0, 1) << " " << cov_point(0, 2)
                                    << " " << cov_point(1, 0) << " " << cov_point(1, 1) << " " << cov_point(1, 2)
                                    << " " << cov_point(2, 0) << " " << cov_point(2, 1) << " " << cov_point(2, 2);
                }
                *pc_file_stream << std::endl;
            }
        }

        pc_file_stream->close();
    }

} // namespace Spacetime
