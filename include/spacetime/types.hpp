#ifndef TYPES_H
#define TYPES_H

#include <iostream>

#include <Eigen/Core>
#include <lgmath.hpp>
#ifdef USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using DTYPE = autodiff::real1st;
#else
using DTYPE = double;
#endif

#include "spacetime/logger.hpp"
#include "spacetime/utilities.hpp"

namespace se3 = lgmath::se3;

namespace Spacetime
{
    struct RobotTopology
    {
        unsigned int N;                      // Number of estimation node states along robot
        unsigned int K;                      // Number of estimation node states along time
        unsigned int Ms;                     // Number of interpolated notes between estimation nodes
        unsigned int Mt;                     // Number of interpolated notes between estimation nodes
        double L;                            // Total arclength of robot
        double T;                            // Total time of the estimator batch (s)
        Eigen::Matrix<DTYPE, 4, 4> T0;       // Base frame of robot at time 0 in ib frame
        Eigen::Matrix<DTYPE, 6, 1> epsilon0; // Initial strain of robot at time 0, arclength 0 in ib frame
        Eigen::Matrix<DTYPE, 6, 1> varpi0;   // Initial velocity of robot at time 0, arclength 0 in ib frame
        double t0;                           // Start time of the estimator (s)
        double radius;                       // Robot radius (m)

        bool lock_first_position;
        bool lock_first_pose;
        bool lock_last_strain;
        double fbg_theta_offset; // TODO: Is this the best spot to store this?

        Eigen::VectorXd spatial_node_positions;  // Manual placement of the spatial nodes along the robot in meters. In this case, N is ignored.
        bool time_nodes_on_measurements = false; // If true, the time nodes are placed on the measurement times. In this case, K is ignored.
        bool use_1D_estimator = false;           // If true, all time-based factors will be removed. To use, ensure time_nodes_on_measurements = true and include_gyro = false in the data loader options.
    
        void print() const
        {
            std::cout << "Robot Topology:" << std::endl;
            std::cout << "N: " << N << std::endl;
            std::cout << "K: " << K << std::endl;
            std::cout << "Ms: " << Ms << std::endl;
            std::cout << "Mt: " << Mt << std::endl;
            std::cout << "L: " << L << std::endl;
            std::cout << "T: " << T << std::endl;
            std::cout << "T0: \n" << T0 << std::endl;
            std::cout << "epsilon0: \n" << epsilon0.transpose() << std::endl;
            std::cout << "varpi0: \n" << varpi0.transpose() << std::endl;
            std::cout << "t0: " << t0 << std::endl;
            std::cout << "radius: " << radius << std::endl;
        }
    };

    struct Hyperparameters
    {
        Eigen::Matrix<double, 18, 18> P0;
        Eigen::Matrix<double, 6, 6> Q1;
        Eigen::Matrix<double, 6, 6> Q2;
        Eigen::Matrix<double, 6, 6> Q3;

        void print() const
        {
            std::cout << "Hyperparameters:" << std::endl;
            std::cout << "P0: \n" << P0 << std::endl;
            std::cout << "Q1: \n" << Q1 << std::endl;
            std::cout << "Q2: \n" << Q2 << std::endl;
            std::cout << "Q3: \n" << Q3 << std::endl;
        }
    };

    template <typename T = DTYPE>
    struct SystemState
    {
        struct Node
        {
            Eigen::Matrix<T, 4, 4> pose; // Expressed in inertial frame T_ik
            Eigen::Matrix<T, 6, 1> epsilon;
            Eigen::Matrix<T, 6, 1> varpi;

            Eigen::MatrixXd covariance_square; // order is: nk 00, 01, 10, 11
            bool covarianceAvailable() const 
            {
                return covariance_square.rows() * covariance_square.cols() > 0; 
            }
            Eigen::MatrixXd getCovariance() const
            {
                assert(covariance_square.rows() >= 18 && covariance_square.cols() >= 18 && "Covariance square is not large enough. Are you sure it's been computed?");
                Eigen::MatrixX<double> cov = covariance_square.block<18, 18>(0, 0);
                return cov;
            }
            Eigen::MatrixXd getCovarianceSquareTime() const
            {
                assert(covariance_square.rows() >= 36 && covariance_square.cols() >= 36 && "Covariance square is not large enough. Are you sure it's been computed?");
                Eigen::MatrixX<double> cov_time(36, 36);
                cov_time << covariance_square.block<18, 18>(0, 0), covariance_square.block<18, 18>(0, 18),
                    covariance_square.block<18, 18>(18, 0), covariance_square.block<18, 18>(18, 18);
                return cov_time;
            };
            Eigen::MatrixXd getCovarianceSquareSpace() const
            {
                assert(covariance_square.rows() >= 54 && covariance_square.cols() >= 54 && "Covariance square is not large enough. Are you sure it's been computed?");
                Eigen::MatrixX<double> cov_space(36, 36);
                cov_space << covariance_square.block<18, 18>(0, 0), covariance_square.block<18, 18>(0, 36),
                    covariance_square.block<18, 18>(36, 0), covariance_square.block<18, 18>(36, 36);
                return cov_space;
            };

            double arclength;
            double time;

            template <typename CastType>
            typename SystemState<CastType>::Node cast() const
            {
                typename SystemState<CastType>::Node newNode;
                newNode.pose = pose.template cast<CastType>();
                newNode.epsilon = epsilon.template cast<CastType>();
                newNode.varpi = varpi.template cast<CastType>();
                newNode.covariance_square = covariance_square;
                newNode.arclength = arclength;
                newNode.time = time;
                return newNode;
            }

            void setRandom()
            {
                Eigen::Matrix<T, 18, 1> rand = Eigen::Matrix<T, 18, 1>::Random();
                pose = se3::vec2tran(rand.template block<6, 1>(0, 0));
                epsilon = rand.template block<6, 1>(6, 0);
                varpi = rand.template block<6, 1>(12, 0);
            }

            void average(SystemState<T>::Node other)
            {
                /*
                T_1i, T_2i
                T_2i * T_1i^-1 = T_21
                T_1.5i = exp(0.5 * log(T_21)) * T_1i
                */
                Eigen::Vector<T, 6> xi_21 = se3::tran2vec(other.pose * invertTransformation(pose));
                pose = se3::vec2tran(0.5 * xi_21) * pose;
                epsilon = 0.5 * (epsilon + other.epsilon);
                varpi = 0.5 * (varpi + other.varpi);
                time = 0.5 * (time + other.time);
                arclength = 0.5 * (arclength + other.arclength);
            }

            void print() const
            {
                std::cout << "-------------------------------" << std::endl;
                std::cout << "Time:" << time << std::endl;
                std::cout << "Arclength:" << arclength << std::endl;

                std::cout << "Pose:" << std::endl;
                std::cout << pose << std::endl;
                std::cout << "Epsilon:" << epsilon.transpose() << std::endl;
                std::cout << "Varpi:" << varpi.transpose() << std::endl;
                std::cout << "-------------------------------" << std::endl;
            }
        };

        std::vector<Node> estimation_nodes;
        std::vector<Node> interpolation_nodes;
        bool ib = false; // bool dictating representation of state. True = T_ib, False = T_bi

        template <typename CastType>
        SystemState<CastType> cast() const
        {
            SystemState<CastType> result;
            std::transform(estimation_nodes.begin(), estimation_nodes.end(), std::back_inserter(result.estimation_nodes), [](const Node &node)
                           { return node.template cast<CastType>(); });
            std::transform(interpolation_nodes.begin(), interpolation_nodes.end(), std::back_inserter(result.interpolation_nodes), [](const Node &node)
                           { return node.template cast<CastType>(); });
            return result;
        }

        void convertStateMeanBodyInertial()
        {
            for (unsigned int i = 0; i < estimation_nodes.size(); i++)
            {
                estimation_nodes[i].pose = invertTransformation(estimation_nodes[i].pose);
                estimation_nodes[i].epsilon = -estimation_nodes[i].epsilon;
                estimation_nodes[i].varpi = -estimation_nodes[i].varpi;
            }
            for (unsigned int m = 0; m < interpolation_nodes.size(); m++)
            {
                interpolation_nodes[m].pose = invertTransformation(interpolation_nodes[m].pose);
                interpolation_nodes[m].epsilon = -interpolation_nodes[m].epsilon;
                interpolation_nodes[m].varpi = -interpolation_nodes[m].varpi;
            }
            ib = !ib;
        }

        void print()
        {
            infoStart();
            std::cout << "System State: " << std::endl;
            for (unsigned int i = 0; i < estimation_nodes.size(); i++)
                estimation_nodes[i].print();
            logReset();
        }

        int getK(const RobotTopology &topology) const
        {
            return (int)(estimation_nodes.size() / topology.N);
        }
    };

    struct Options
    {
        enum InitialGuessType
        {
            Straight,
            Last,
            ZeroErrorPrior,
            Custom
        }; // Choice of the initial guess of the optimization problem (straight robots, last known system state - if applicable, custom guess - e.g. we can use this to get our guess from strain measurements)
        enum Solver
        {
            Newton,
            NewtonLineSearch
        }; // Solver for the optimization problem
        struct ICPOptions
        {
            enum Type
            {
                POINT_TO_PLANE = 0
            };
            Type type = POINT_TO_PLANE;
            int MIN_NEIGHBORS = 5;
            int MAX_NEIGHBORS = 20;
            int VOXEL_SEARCH_RADIUS = 0;
            double POWER_PLANARITY = 2.0;
            double VOXEL_SIZE = 0.01;
            double MAX_DIST_TO_PLANE = 0.1;
            int VOXEL_LIFETIME = 10;
            double MIN_DISTANCE_POINTS = 0.005;
            int MAX_NUM_POINTS_IN_VOXEL = 20;
            bool point2point = false;
            double rejection_threshold = 0.0;
        };
        enum EstimatorType
        {
            Batch,
            Filter,
            Window
        };

        InitialGuessType init_guess_type = Straight;
        Solver solver = Newton;
        EstimatorType estimator_type = Batch;
        ICPOptions icp_options;
        SystemState<DTYPE> custom_guess; // Custom initial guess for solving the system (Expressed with T_ib frames)
        unsigned int max_optimization_iterations;
        double convergence_threshold;
        double initial_perturbation = 0;
        double estimator_dt = 1.0 / 30.0; 
        double LM_damping = 0.0;
        bool use_autodiff = false;
        bool kirchhoff_rods;
        bool interpolate_measurements;
        bool compute_covariances;
        bool marginalize = false;
        bool reverse_order = false;
        bool extract_from_front = false; // For sliding window estimators, selects which end of the window to extract the state from
    
        void print() const
        {
            std::cout << "Options:" << std::endl;
            std::cout << "Initial Guess Type: " << init_guess_type << std::endl;
            std::cout << "Solver: " << solver << std::endl;
            std::cout << "Estimator Type: " << estimator_type << std::endl;
            std::cout << "Max Optimization Iterations: " << max_optimization_iterations << std::endl;
            std::cout << "Convergence Threshold: " << convergence_threshold << std::endl;
            std::cout << "Initial Perturbation: " << initial_perturbation << std::endl;
            std::cout << "Estimator Time Step (dt): " << estimator_dt << std::endl;
            std::cout << "LM Damping: " << LM_damping << std::endl;
            std::cout << "Use Autodiff: " << use_autodiff << std::endl;
            std::cout << "Kirchhoff Rods: " << kirchhoff_rods << std::endl;
            std::cout << "Interpolate Measurements: " << interpolate_measurements << std::endl;
            std::cout << "Compute Covariances: " << compute_covariances << std::endl;
            std::cout << "Marginalize: " << marginalize << std::endl;
            std::cout << "Reverse Order: " << reverse_order << std::endl;
            std::cout << "Extract From Front: " << extract_from_front << std::endl;
        }
    };

    struct SensorMeasurement // Expressed with T_ib frames
    {
        enum Type
        {
            Pose,
            Strain,
            Velocity,
            Gyro,
            Lidar,
            Custom = -1
        };

        Type type;
        Eigen::MatrixXd value;        // 4x4 Transformation matrix (Pose), 6x1 vector (Strain, Velocity), 3x1 vector (Gyro), Nx3 matrix (Lidar)
        mutable Eigen::VectorXi mask; // Mask for valid measurement components. Mutable as some factors may modify it at runtime (e.g. ICP may reject some points)

        double s; // Arclength of measurement (node index)
        double t; // Timestep of measurement (s)

        void print() const
        {
            std::cout << "Measurement\n------------------------------"
                      << "\nType: " << type
                      << "\nS: " << s
                      << "\nT: " << t
                      << "\n------------------------------\n";

            for (int row = 0; row < value.rows(); row++)
            {
                for (int col = 0; col < value.cols(); col++)
                {
                    std::cout << value(row, col) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n------------------------------\n";
        }
    };

    struct VisualizerOptions
    {
        struct RobotOptions
        {
            bool show_interpolation = true;
            bool show_covariances = false;
            bool show_frames = true;
            double radius = 0.01;
        };
        RobotOptions robot;

        struct MeasurementOptions
        {
            bool show = false;
            unsigned int count = 0;
            double size = 0.05;
        };
        MeasurementOptions vicon;
        MeasurementOptions lidar;
        MeasurementOptions aurora;

        bool show_ground_plane = true;
        bool show_gt = true;

        struct EnvironmentOptions
        {
            bool show = false;
            double scale = 1.0;
            double opacity = 1.0;
            std::string path = "";
        };
        EnvironmentOptions environment;

        struct CameraOptions
        {
            Eigen::Vector3d position = Eigen::Vector3d(0, -1, 0.5);
            Eigen::Vector3d focal_point = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d view_up = Eigen::Vector3d(0, 0, 1);
        };
        std::vector<CameraOptions> cameras;

        RobotTopology topology;
        Options estimator_options;

        void print() const
        {
            std::cout << "Visualizer Options:" << std::endl;
            std::cout << "Robot Options:" << std::endl;
            std::cout << "Show Interpolation: " << robot.show_interpolation << std::endl;
            std::cout << "Show Covariances: " << robot.show_covariances << std::endl;
            std::cout << "Show Frames: " << robot.show_frames << std::endl;
            std::cout << "Radius: " << robot.radius << std::endl;

            std::cout << "Vicon Measurement Options:" << std::endl;
            std::cout << "Show: " << vicon.show << std::endl;
            std::cout << "Count: " << vicon.count << std::endl;
            std::cout << "Size: " << vicon.size << std::endl;

            std::cout << "Lidar Measurement Options:" << std::endl;
            std::cout << "Show: " << lidar.show << std::endl;
            std::cout << "Count: " << lidar.count << std::endl;
            std::cout << "Size: " << lidar.size << std::endl;

            std::cout << "Aurora Measurement Options:" << std::endl;
            std::cout << "Show: " << aurora.show << std::endl;
            std::cout << "Count: " << aurora.count << std::endl;
            std::cout << "Size: " << aurora.size << std::endl;

            std::cout << "Ground Plane: " << show_ground_plane << std::endl;
            std::cout << "Show GT: " << show_gt<<std::endl;

            if (environment.show)
            {
                std::cout << "Environment Options:" << std::endl;
                std::cout << "Scale: " << environment.scale<<std::endl;
                std::cout << "Opacity: " 	<< environment.opacity<<std::endl;
                if (!environment.path.empty())
                    std::cout<<"Path: "<<environment.path<<std::endl;
            }

            for (size_t i = 0; i < cameras.size(); i++)
            {
                const auto &cam = cameras[i];
                std::cout<<"Camera "<<i<<" Options:"<<std::endl;
                std::cout<<"Position: "<<cam.position.transpose()<<std::endl
                            <<"Focal Point: "<<cam.focal_point.transpose()<<std::endl
                            <<"View Up: "<<cam.view_up.transpose()<<std::endl;
            }
        }
    };
} // namespace Spacetime
#endif // TYPES_H