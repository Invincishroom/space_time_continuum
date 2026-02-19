#ifndef DATALOADER_H
#define DATALOADER_H

#include <utility>
#include <vector>
#include <glob.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <json/json.h>
#include <json/value.h>

#include "spacetime/types.hpp"
#include "spacetime/factors.hpp"
#include "spacetime/Visualizer.hpp"

struct DataLoaderOptions
{
    std::string data_dir;
    double start_time = 0.0;
    double end_time = std::numeric_limits<double>::max();
    std::string map_file = ""; // File path to a PCD map to load for ICP

    struct ExperimentLoaderOptions
    {
        // Optional parameters for Experiment data
        bool include_aurora = false;
        bool include_velocity = false;
        bool include_gyro = false;
        bool include_fbgs = false;
        bool include_lidar = false;
    };
    ExperimentLoaderOptions experiment_options;

    struct Weights
    {
        Eigen::Matrix<double, 6, 6> R_pose;
        Eigen::Matrix<double, 3, 3> R_gyro, R_velocity;
        Eigen::Matrix<double, 1, 1> R_lidar;
        Eigen::Matrix<double, 6, 6> R_fbgs;
    };
    Weights weights;
};

void filterMeasurements(std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &measurements, double start_k, double end_k)
{
    if (measurements.size() == 0)
        return;

    if (end_k < 0)
        end_k = measurements.back()->getMeas().t;

    // Returns measurements such that measurement.k \in [start_k, end_k]
    auto start_it = std::find_if(measurements.begin(), measurements.end(), [start_k](const auto &m)
                                 { return m->getMeas().t >= start_k; });
    auto end_it = std::find_if(measurements.begin(), measurements.end(), [end_k](const auto &m)
                               { return m->getMeas().t > end_k; });

    measurements.erase(end_it, measurements.end());
    measurements.erase(measurements.begin(), start_it);
}
void filterMeasurements(std::vector<Spacetime::SensorMeasurement> &measurements, double start_k, double end_k)
{
    if (measurements.size() == 0)
        return;

    if (end_k < 0)
        end_k = measurements.back().t;

    // Returns measurements such that measurement.k \in [start_k, end_k]
    auto start_it = std::find_if(measurements.begin(), measurements.end(), [start_k](const auto &m)
                                 { return m.t >= start_k; });
    auto end_it = std::find_if(measurements.begin(), measurements.end(), [end_k](const auto &m)
                               { return m.t > end_k; });

    measurements.erase(end_it, measurements.end());
    measurements.erase(measurements.begin(), start_it);
}

bool loadConfig(std::string filename, Spacetime::RobotTopology &topology, Spacetime::Hyperparameters &params, Spacetime::Options &options, DataLoaderOptions &data_options, Spacetime::VisualizerOptions &vis_options)
{
    std::ifstream config_file(filename, std::ifstream::binary);
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;

    Json::Value root;

    if (!parseFromStream(builder, config_file, &root, &errs))
    {
        error() << errs;
        return false;
    }
    // setLogLevel(root["log"].asInt());
    info() << "Configuration file: " << filename;

    // Topology
    Json::Value topology_root;
    if (safeRead(root, "topology", topology_root))
    {
        // topology.N = root["topology"]["N"].asInt();
        safeRead(topology_root, "N", topology.N);
        safeRead(topology_root, "K_per_second", topology.K);

        // Number of interpolated states between estimation nodes per robot
        // M=1 results in no interpolation and the interpolation nodes will be equal to the estimation nodes
        // M=2 results in one additional interpolated node between each estimation node etc
        safeRead(topology_root, "Ms", topology.Ms);
        safeRead(topology_root, "Mt", topology.Mt);

        // Load the length and time of the robot
        safeRead(topology_root, "length", topology.L);
        safeRead(topology_root, "radius", topology.radius);
        safeRead(topology_root, "time", topology.T);

        // Define if we lock the pose of the robots' ends
        safeRead(topology_root, "lock_first_position", topology.lock_first_position);
        safeRead(topology_root, "lock_first_pose", topology.lock_first_pose);
        safeRead(topology_root, "lock_last_strain", topology.lock_last_strain);

        safeRead(topology_root, "fbg_theta_offset", topology.fbg_theta_offset);
        loadArrayData(topology_root, "spatial_node_positions", topology.spatial_node_positions);
        if (topology.spatial_node_positions.size() > 0)
        {
            info() << "Loaded spatial node positions. N will be ignored.";
            topology.N = topology.spatial_node_positions.size();
        }
        safeRead(topology_root, "time_nodes_on_measurements", topology.time_nodes_on_measurements);
        safeRead(topology_root, "use_1D_estimator", topology.use_1D_estimator);
    }

    Json::Value initial_condition_root;
    if (safeRead(root, "initial_condition", initial_condition_root))
    {
        // Load the initial condition for the robot
        loadMatrix4Data(initial_condition_root, "T0", topology.T0);
        loadArrayData(initial_condition_root, "epsilon0", topology.epsilon0);
        loadArrayData(initial_condition_root, "varpi0", topology.varpi0);
        safeRead(initial_condition_root, "start_time", topology.t0);

        bool initialize_straight = options.init_guess_type == Spacetime::Options::InitialGuessType::Straight;
        safeRead(initial_condition_root, "initialize_straight", initialize_straight);
        if (initialize_straight)
            options.init_guess_type = Spacetime::Options::InitialGuessType::Straight;
        else
            options.init_guess_type = Spacetime::Options::InitialGuessType::ZeroErrorPrior;

        safeRead(initial_condition_root, "initial_perturbation", options.initial_perturbation);
    }

    // Measurement noise
    Json::Value weights_root;
    if (safeRead(root, "weights", weights_root))
    {
        double measurement_factor = 1.0;
        safeRead(weights_root, "measurement_factor", measurement_factor);

        Eigen::Matrix<double, 6, 1> R_pose;
        loadArrayData(weights_root, "R_pose", R_pose);
        data_options.weights.R_pose = measurement_factor * R_pose.asDiagonal();

        Eigen::Matrix<double, 3, 1> R_velocity;
        loadArrayData(weights_root, "R_velocity", R_velocity);
        data_options.weights.R_velocity = measurement_factor * R_velocity.asDiagonal();

        Eigen::Matrix<double, 3, 1> R_gyro;
        loadArrayData(weights_root, "R_gyro", R_gyro);
        data_options.weights.R_gyro = measurement_factor * R_gyro.asDiagonal();

        Eigen::Matrix<double, 1, 1> R_lidar;
        loadArrayData(weights_root, "R_lidar", R_lidar);
        data_options.weights.R_lidar = measurement_factor * R_lidar.asDiagonal();

        Eigen::Matrix<double, 6, 1> R_fbgs;
        loadArrayData(weights_root, "R_fbgs", R_fbgs);
        data_options.weights.R_fbgs = measurement_factor * R_fbgs.asDiagonal();

        // Prior noise
        double prior_factor = 1.0;
        safeRead(weights_root, "prior_factor", prior_factor);

        Eigen::Matrix<double, 6, 1> Q1, Q2, Q3;
        loadArrayData(weights_root, "Q1", Q1);
        loadArrayData(weights_root, "Q2", Q2);
        loadArrayData(weights_root, "Q3", Q3);
        params.Q1 = prior_factor * Q1.asDiagonal();
        params.Q2 = prior_factor * Q2.asDiagonal();
        params.Q3 = prior_factor * Q3.asDiagonal();

        Eigen::Matrix<double, 18, 1> P0;
        loadArrayData(weights_root, "P0", P0);
        params.P0 = prior_factor * P0.asDiagonal();
    }

    Json::Value optimizer_root;
    if (safeRead(root, "optimizer", optimizer_root))
    {
        options.solver = (optimizer_root["line_search"].asBool())
                             ? Spacetime::Options::Solver::NewtonLineSearch
                             : Spacetime::Options::Solver::Newton;

        safeRead(optimizer_root, "max_iterations", options.max_optimization_iterations);
        safeRead(optimizer_root, "kirchoff", options.kirchhoff_rods);
        safeRead(optimizer_root, "convergence_threshold", options.convergence_threshold);
        safeRead(optimizer_root, "LM_damping", options.LM_damping);
        safeRead(optimizer_root, "use_autodiff", options.use_autodiff);
        safeRead(optimizer_root, "marginalize", options.marginalize);
    }

    Json::Value options_root;
    if (safeRead(root, "options", options_root))
    {
        std::string estimator_type;
        safeRead(options_root, "estimator", estimator_type);
        if (estimator_type == "batch")
        {
            options.estimator_type = Spacetime::Options::EstimatorType::Batch;
        }
        else if (estimator_type == "filter")
        {
            options.estimator_type = Spacetime::Options::EstimatorType::Filter;
        }
        else if (estimator_type == "window")
        {
            options.estimator_type = Spacetime::Options::EstimatorType::Window;
        }
        else
        {
            throw std::runtime_error("Unknown estimator type: " + estimator_type);
        }
        safeRead(options_root, "interpolate_measurements", options.interpolate_measurements);
        safeRead(options_root, "compute_covariances", options.compute_covariances);
        safeRead(options_root, "extract_from_front", options.extract_from_front);
        safeRead(options_root, "dt", options.estimator_dt);
    }

    Json::Value icp_root;
    if (safeRead(root, "icp_options", icp_root))
    {
        safeRead(icp_root, "min_neighbors", options.icp_options.MIN_NEIGHBORS);
        safeRead(icp_root, "max_neighbors", options.icp_options.MAX_NEIGHBORS);
        safeRead(icp_root, "voxel_search_radius", options.icp_options.VOXEL_SEARCH_RADIUS);
        safeRead(icp_root, "power_planarity", options.icp_options.POWER_PLANARITY);
        safeRead(icp_root, "voxel_size", options.icp_options.VOXEL_SIZE);
        safeRead(icp_root, "max_dist_to_plane", options.icp_options.MAX_DIST_TO_PLANE);
        safeRead(icp_root, "voxel_lifetime", options.icp_options.VOXEL_LIFETIME);
        safeRead(icp_root, "min_distance_points", options.icp_options.MIN_DISTANCE_POINTS);
        safeRead(icp_root, "max_num_points_in_voxel", options.icp_options.MAX_NUM_POINTS_IN_VOXEL);
        safeRead(icp_root, "point2point", options.icp_options.point2point);
        safeRead(icp_root, "rejection_threshold", options.icp_options.rejection_threshold);
    }

    // Dataloader
    Json::Value data_root;
    if (safeRead(root, "data", data_root))
    {
        safeRead(data_root, "start_time", data_options.start_time);
        safeRead(data_root, "end_time", data_options.end_time);

        safeRead(data_root, "folder_path", data_options.data_dir);
        safeRead(data_root, "map_file", data_options.map_file);
        safeRead(data_root, "aurora", data_options.experiment_options.include_aurora);
        safeRead(data_root, "velocity", data_options.experiment_options.include_velocity);
        safeRead(data_root, "gyro", data_options.experiment_options.include_gyro);
        safeRead(data_root, "lidar", data_options.experiment_options.include_lidar);
        safeRead(data_root, "fbgs", data_options.experiment_options.include_fbgs);
        safeRead(data_root, "start_time", data_options.start_time);
        safeRead(data_root, "end_time", data_options.end_time);
    }

    // Visualizer options
    Json::Value vis_root;
    if (safeRead(root, "visualizer", vis_root))
    {
        Json::Value robot_root;
        if (safeRead(vis_root, "robot", robot_root))
        {
            safeRead(robot_root, "show_interpolation", vis_options.robot.show_interpolation);
            safeRead(robot_root, "show_covariances", vis_options.robot.show_covariances);
            safeRead(robot_root, "show_frames", vis_options.robot.show_frames);
        }

        Json::Value measurements_root;
        if (safeRead(vis_root, "measurements", measurements_root))
        {
            Json::Value vicon_root;
            if (safeRead(measurements_root, "vicon", vicon_root))
            {
                safeRead(vicon_root, "show", vis_options.vicon.show);
                safeRead(vicon_root, "count", vis_options.vicon.count);
                safeRead(vicon_root, "size", vis_options.vicon.size);
            }

            Json::Value lidar_root;
            if (safeRead(measurements_root, "lidar", lidar_root))
            {
                safeRead(lidar_root, "show", vis_options.lidar.show);
                safeRead(lidar_root, "count", vis_options.lidar.count);
                safeRead(lidar_root, "size", vis_options.lidar.size);
            }

            Json::Value aurora_root;
            if (safeRead(measurements_root, "aurora", aurora_root))
            {
                safeRead(aurora_root, "show", vis_options.aurora.show);
                safeRead(aurora_root, "count", vis_options.aurora.count);
                safeRead(aurora_root, "size", vis_options.aurora.size);
            }
        }

        safeRead(vis_root, "show_ground_plane", vis_options.show_ground_plane);

        Json::Value environment_root;
        if (safeRead(vis_root, "environment", environment_root))
        {
            safeRead(environment_root, "show", vis_options.environment.show);
            safeRead(environment_root, "scale", vis_options.environment.scale);
            safeRead(environment_root, "opacity", vis_options.environment.opacity);
            safeRead(environment_root, "path", vis_options.environment.path);
        }

        Json::Value camera_root;
        if (safeRead(vis_root, "cameras", camera_root))
        {
            for (unsigned int i = 0; i < camera_root.size(); i++)
            {
                Spacetime::VisualizerOptions::CameraOptions cam_options;
                loadArrayData(camera_root[i], "position", cam_options.position);
                loadArrayData(camera_root[i], "look_at", cam_options.focal_point);
                loadArrayData(camera_root[i], "up", cam_options.view_up);
                vis_options.cameras.push_back(cam_options);
            }
        }
    }

    return true;
}

std::shared_ptr<steam_icp::Map> loadMap(const DataLoaderOptions &options, const Spacetime::Options::ICPOptions &icp_options)
{
    auto map = std::make_shared<steam_icp::Map>(icp_options.VOXEL_LIFETIME);

    if (options.map_file == "")
    {
        info() << "No map file provided. Map will be built online.";
        return map;
    }

    CSVReader reader(options.map_file);
    info() << "Opening: " << options.map_file;
    std::vector<double> values;
    while (reader.readLine(values))
    {
        if ((values.size() == 3))
        {
            Eigen::Vector3d pt(values[0], values[1], values[2]);
            map->add(pt, icp_options.VOXEL_SIZE, icp_options.MAX_NUM_POINTS_IN_VOXEL, icp_options.MIN_DISTANCE_POINTS); // Voxel size, max points per voxel, min distance between points
        }
        else if (values.size() == 7)
        {
            // Normals and planarity coefficient provided
            Eigen::Vector3d pt(values[0], values[1], values[2]);
            Eigen::Vector3d normal(values[3], values[4], values[5]);
            double planarity = values[6];
            map->add(pt, icp_options.VOXEL_SIZE, icp_options.MAX_NUM_POINTS_IN_VOXEL, icp_options.MIN_DISTANCE_POINTS, 0, normal, planarity); // Voxel size, max points per voxel, min distance between points
        }
        else
        {
            std::cerr << "ERROR: Map file should contain Nx3 or Nx7 points. Found row with " << values.size() << " entries.";
            return nullptr;
        }
    }
    info() << "Loaded map from file: " << options.map_file << " with " << map->pointcloud().size() << " points.";
    return map;
}

bool loadData(const DataLoaderOptions &data_options, Spacetime::Options &estimator_options, const Spacetime::Hyperparameters &params, std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &measurements, std::vector<Spacetime::SensorMeasurement> &ground_truth, std::shared_ptr<steam_icp::Map> &map)
{
    map = loadMap(data_options, estimator_options.icp_options);

    // Cache for lidar measurements grouped by time and arclength
    std::map<std::pair<double, double>, std::vector<Eigen::Vector3d>> lidar_cache;

    std::string glob_path = data_options.data_dir + "/*.csv";

    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    int return_value = glob(glob_path.c_str(), GLOB_TILDE, NULL, &glob_result);

    if (return_value != 0)
    {
        std::cerr << "Glob failed with error code: " << return_value;
        return false;
    }

    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        std::string file_path = std::string(glob_result.gl_pathv[i]);

        CSVReader reader(file_path);
        info() << "Opening: " << file_path;
        std::vector<std::string> text;
        reader.readLine(text);
        std::string sensor_type = text[1];
        sensor_type.erase(std::remove(sensor_type.begin(), sensor_type.end(), '\r'), sensor_type.end());
        reader.burnLine(); // header line

        double timestep = -1.0;
        double arclength = -1.0;
        while (reader.read(timestep))
        {
            reader.read(arclength);
            std::vector<double> values;
            reader.readLine(values);
            std::shared_ptr<Spacetime::Factors::MeasurementFactor> factor = nullptr;
            Spacetime::SensorMeasurement measurement;
            measurement.s = arclength;
            measurement.t = timestep; // Convert to seconds

            if (data_options.start_time > timestep)
                // Skip any data generated before the first vicon reading
                continue;
            if (sensor_type == std::string("vicon"))
            {
                measurement.type = Spacetime::SensorMeasurement::Pose;
                measurement.mask = Eigen::Matrix<int, 6, 1>::Ones(); // Assuming all components are valid
                Eigen::Matrix<double, 6, 1> value = Eigen::Map<Eigen::Matrix<double, 6, 1>>(std::vector<double>(values.begin(), values.end()).data());
                measurement.value = se3::vec2tran(value);
                factor = std::make_shared<Spacetime::Factors::PoseMeasurementFactor>(data_options.weights.R_pose, measurement);
            }
            else if (sensor_type == std::string("aurora") && data_options.experiment_options.include_aurora)
            {
                measurement.type = Spacetime::SensorMeasurement::Pose;
                measurement.mask = Eigen::Matrix<int, 6, 1>::Ones(); // Assuming all components are valid
                Eigen::Matrix<double, 6, 1> value = Eigen::Map<Eigen::Matrix<double, 6, 1>>(std::vector<double>(values.begin(), values.end()).data());
                measurement.value = se3::vec2tran(value);
                factor = std::make_shared<Spacetime::Factors::PoseMeasurementFactor>(data_options.weights.R_pose, measurement);
            }
            else if (sensor_type == std::string("velocity") && (data_options.experiment_options.include_velocity || data_options.experiment_options.include_gyro))
            {
                measurement.mask = Eigen::Matrix<int, 3, 1>::Ones(); // Assuming all components are valid
                if (data_options.experiment_options.include_velocity)
                {
                    Spacetime::SensorMeasurement _measurement = measurement;
                    _measurement.type = Spacetime::SensorMeasurement::Velocity;
                    Eigen::Matrix<double, 6, 1> value = Eigen::Map<Eigen::Matrix<double, 6, 1>>(std::vector<double>(values.begin(), values.end()).data());
                    _measurement.value = value.block<3, 1>(0, 0); // Linear velocity part
                    factor = std::make_shared<Spacetime::Factors::VelocityMeasurementFactor>(data_options.weights.R_velocity, _measurement);
                    measurements.push_back(factor);
                }
                if (data_options.experiment_options.include_gyro)
                {
                    Spacetime::SensorMeasurement _measurement = measurement;
                    _measurement.type = Spacetime::SensorMeasurement::Gyro;
                    Eigen::Matrix<double, 6, 1> value = Eigen::Map<Eigen::Matrix<double, 6, 1>>(std::vector<double>(values.begin(), values.end()).data());
                    _measurement.value = value.block<3, 1>(3, 0); // Angular velocity part
                    factor = std::make_shared<Spacetime::Factors::GyroMeasurementFactor>(data_options.weights.R_gyro, _measurement);
                    measurements.push_back(factor);
                }
                continue;
            }
            else if (sensor_type == std::string("gyro") && data_options.experiment_options.include_gyro)
            {
                measurement.type = Spacetime::SensorMeasurement::Gyro;
                measurement.mask = Eigen::Matrix<int, 3, 1>::Ones(); // Assuming all components are valid
                Eigen::Matrix<double, 3, 1> value = Eigen::Map<Eigen::Matrix<double, 3, 1>>(std::vector<double>(values.begin(), values.end()).data());
                measurement.value = value;
                factor = std::make_shared<Spacetime::Factors::GyroMeasurementFactor>(data_options.weights.R_gyro, measurement);
            }
            else if (sensor_type == std::string("lidar") && data_options.experiment_options.include_lidar)
            {
                // Cache lidar points instead of creating factors immediately
                std::pair<double, double> time_arc_key = {timestep, arclength};

                // Parse the lidar points
                for (size_t j = 0; j < values.size(); j += 3)
                {
                    if (j + 2 < values.size())
                    {
                        Eigen::Vector3d point(values[j], values[j + 1], values[j + 2]);
                        lidar_cache[time_arc_key].push_back(point);
                    }
                }
                continue; // Don't create factor yet, skip to next iteration
            }
            else if (sensor_type == std::string("fbgs") && data_options.experiment_options.include_fbgs)
            {
                measurement.type = Spacetime::SensorMeasurement::Strain;
                measurement.mask = Eigen::Matrix<int, 6, 1>::Ones(); // Assuming all components are valid
                Eigen::Matrix<double, 6, 1> value = Eigen::Map<Eigen::Matrix<double, 6, 1>>(std::vector<double>(values.begin(), values.end()).data());
                measurement.value = value;
                factor = std::make_shared<Spacetime::Factors::FBGSMeasurementFactor>(data_options.weights.R_fbgs, measurement);
            }
            else
            {
                continue;
            }

            if (sensor_type == std::string("vicon"))
            {
                ground_truth.push_back(measurement);
                // measurements.push_back(factor); // Uncomment to use vicon as pose measuremnet - simultates ground truth
            }
            else if (factor != nullptr)
                measurements.push_back(factor);
        }
    }

    globfree(&glob_result);

    // Create lidar factors from cached data
    for (const auto &[time_arc_pair, points] : lidar_cache)
    {
        double timestep = time_arc_pair.first;
        double arclength = time_arc_pair.second;

        // Create measurement with all points for this time/arclength
        Spacetime::SensorMeasurement measurement;
        measurement.s = arclength;
        measurement.t = timestep;
        measurement.type = Spacetime::SensorMeasurement::Lidar;

        // Convert vector of points to Nx3 matrix
        Eigen::MatrixXd point_matrix(points.size(), 3);
        for (size_t i = 0; i < points.size(); ++i)
        {
            point_matrix.row(i) = points[i].transpose();
        }
        measurement.value = point_matrix;

        // Create the appropriate factor
        std::shared_ptr<Spacetime::Factors::MeasurementFactor> factor;
        if (estimator_options.icp_options.point2point)
            factor = std::make_shared<Spacetime::Factors::Point2PointMeasurementFactor>(data_options.weights.R_lidar, measurement, map, estimator_options.icp_options);
        else
            factor = std::make_shared<Spacetime::Factors::ICPMeasurementFactor>(data_options.weights.R_lidar, measurement, map, estimator_options.icp_options);

        measurements.push_back(factor);
    }

    // Enforce measurements are in chronological order
    std::sort(measurements.begin(), measurements.end(), [](const std::shared_ptr<Spacetime::Factors::MeasurementFactor> &a, const std::shared_ptr<Spacetime::Factors::MeasurementFactor> &b)
              { return (a->getMeas().t < b->getMeas().t) || ((a->getMeas().t == b->getMeas().t) && (a->getMeas().s < b->getMeas().s)); });
    std::sort(ground_truth.begin(), ground_truth.end(), [](const Spacetime::SensorMeasurement &a, const Spacetime::SensorMeasurement &b)
              { return (a.t < b.t) || ((a.t == b.t) && (a.s < b.s)); });

    filterMeasurements(measurements, data_options.start_time, data_options.end_time);
    filterMeasurements(ground_truth, data_options.start_time, data_options.end_time);

    info() << "Loaded " << measurements.size() << " measurements between t=" << data_options.start_time << " and t=" << data_options.end_time;

    warning("Removed measurement validation step. This should probably be added back in.");

    return true;
}
#endif // DATALOADER_H
