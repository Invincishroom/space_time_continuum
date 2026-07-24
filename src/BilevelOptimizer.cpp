#include "spacetime/bilevel_optimizer.hpp"
#include "spacetime/bilevel_optimizer/FactorGradient.hpp"
#include "spacetime/bilevel_optimizer/GradientAccumulator.hpp"
#include "spacetime/estimators/Estimator.hpp"
#include "spacetime/factors.hpp"
#include "spacetime/utilities.hpp"
#include "spacetime/Solver.hpp"

#include <Eigen/Dense>
#include <Eigen/LU>
#ifdef USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <lgmath/CommonMath.hpp>
#endif

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include<iostream>
#include<random>
#include<algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include<json/json.h>

namespace // anonymous
{
    using Matrix18d = Eigen::Matrix<double, 18, 18>;
    using Vector18d = Eigen::Matrix<double, 18, 1>;

    struct MarginalCacheData
    {
        std::unordered_map<int, Matrix18d> node_information_blocks;
        double factorization_ms = 0.0;
        double marginal_solve_ms = 0.0;
    };

    struct FactorKernelSample
    {
        bool available = false;
        std::vector<Spacetime::SystemState<DTYPE>::Node> local_nodes;
        Eigen::VectorXd e_f;
        Eigen::MatrixXd info_f;
        Eigen::MatrixXd E_f;
        Eigen::VectorXd lambda_f;
        Eigen::VectorXd e_nodes;
        double info_scale = 0.0;
    };

    constexpr int kThetaSize = 36;
    constexpr int kP0Offset = 0;
    constexpr int kQ1Offset = 18;
    constexpr int kQ2Offset = 24;
    constexpr int kQ3Offset = 30;

    Eigen::Matrix<double, 18, 1> computeNodeValidationError(const Spacetime::SystemState<DTYPE>::Node &estimate,
                                                            const Spacetime::SystemState<DTYPE>::Node &ground_truth);

    //load JSON file from path and return as Json::Value
    std::filesystem::path resolveExistingPath(const std::initializer_list<std::filesystem::path>& candidates)
    {
        for (const auto& candidate : candidates)
        {
            if (std::filesystem::exists(candidate))
            {
                return candidate;
            }
        }

        throw std::runtime_error("Unable to locate required minimal config/data file.");
    }

    Json::Value loadJsonFile(const std::filesystem::path& path)
    {
        std::ifstream input(path);
        if (!input.is_open())
        {
            throw std::runtime_error("Failed to open JSON file: " + path.string());
        }

        Json::CharReaderBuilder builder;
        JSONCPP_STRING errors;
        Json::Value root;
        if (!parseFromStream(builder, input, &root, &errors))
        {
            throw std::runtime_error("Failed to parse JSON file: " + path.string() + " - " + errors);
        }

        return root;
    }

    std::vector<std::vector<std::string>> readCsvRows(const std::filesystem::path& path)
    {
        std::ifstream input(path);
        if (!input.is_open())
        {
            throw std::runtime_error("Failed to open CSV file: " + path.string());
        }

        std::vector<std::vector<std::string>> rows;
        std::string line;
        while (std::getline(input, line))
        {
            if (line.empty())
            {
                continue;
            }

            std::vector<std::string> cells;
            std::stringstream line_stream(line);
            std::string cell;
            while (std::getline(line_stream, cell, ','))
            {
                cells.push_back(cell);
            }
            rows.push_back(std::move(cells));
        }

        return rows;
    }

    // Parse a CSV row into a 6-DoF pose vector.
    Eigen::Matrix<double, 6, 1> parsePoseRow(const std::vector<std::string>& row)
    {
        if (row.size() < 8)
        {
            throw std::runtime_error("Minimal CSV row does not contain a 6-DoF pose measurement.");
        }

        Eigen::Matrix<double, 6, 1> pose;
        for (int i = 0; i < 6; ++i)
        {
            pose(i) = std::stod(row[2 + i]);
        }
        return pose;
    }

    void loadEstimatorOptionsFromJson(const Json::Value &root, Spacetime::Options &options)
    {
        options.init_guess_type = Spacetime::Options::InitialGuessType::Custom;
        options.solver = Spacetime::Options::Solver::Newton;
        options.estimator_type = Spacetime::Options::EstimatorType::Batch;
        options.max_optimization_iterations = 10;
        options.convergence_threshold = 1e-8;
        options.initial_perturbation = 0.0;
        options.LM_damping = 0.0;
        options.use_autodiff = false;
        options.kirchhoff_rods = false;
        options.interpolate_measurements = false;
        options.compute_covariances = false;
        options.marginalize = false;
        options.reverse_order = false;
        options.extract_from_front = false;

        const Json::Value optimizer_root = root["optimizer"];
        if (!optimizer_root.isNull())
        {
            if (optimizer_root.isMember("line_search") && optimizer_root["line_search"].isBool())
            {
                options.solver = optimizer_root["line_search"].asBool()
                                     ? Spacetime::Options::Solver::NewtonLineSearch
                                     : Spacetime::Options::Solver::Newton;
            }
            if (optimizer_root.isMember("max_iterations"))
            {
                options.max_optimization_iterations = optimizer_root["max_iterations"].asUInt();
            }
            if (optimizer_root.isMember("convergence_threshold"))
            {
                options.convergence_threshold = optimizer_root["convergence_threshold"].asDouble();
            }
            if (optimizer_root.isMember("LM_damping"))
            {
                options.LM_damping = optimizer_root["LM_damping"].asDouble();
            }
            if (optimizer_root.isMember("use_autodiff"))
            {
                options.use_autodiff = optimizer_root["use_autodiff"].asBool();
            }
            if (optimizer_root.isMember("kirchoff"))
            {
                options.kirchhoff_rods = optimizer_root["kirchoff"].asBool();
            }
            if (optimizer_root.isMember("marginalize"))
            {
                options.marginalize = optimizer_root["marginalize"].asBool();
            }
        }

        const Json::Value options_root = root["options"];
        if (!options_root.isNull())
        {
            if (options_root.isMember("estimator") && options_root["estimator"].isString())
            {
                const std::string estimator_type = options_root["estimator"].asString();
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
            }

            if (options_root.isMember("interpolate_measurements"))
            {
                options.interpolate_measurements = options_root["interpolate_measurements"].asBool();
            }
            if (options_root.isMember("compute_covariances"))
            {
                options.compute_covariances = options_root["compute_covariances"].asBool();
            }
            if (options_root.isMember("extract_from_front"))
            {
                options.extract_from_front = options_root["extract_from_front"].asBool();
            }
            if (options_root.isMember("dt"))
            {
                options.estimator_dt = options_root["dt"].asDouble();
            }
        }
    }

    void loadBilevelOptimizerConfigFromJson(const Json::Value &root, spacetime::Optimizer::OptimizerConfig &config)
    {
        if (root.isMember("max_iterations"))
        {
            config.max_iterations = root["max_iterations"].asInt();
        }
        if (root.isMember("learning_rate"))
        {
            config.learning_rate = root["learning_rate"].asDouble();
        }
        if (root.isMember("use_adam"))
        {
            config.use_adam = root["use_adam"].asBool();
        }
        if (root.isMember("adam_beta1"))
        {
            config.adam_beta1 = root["adam_beta1"].asDouble();
        }
        if (root.isMember("adam_beta2"))
        {
            config.adam_beta2 = root["adam_beta2"].asDouble();
        }
        if (root.isMember("adam_epsilon"))
        {
            config.adam_epsilon = root["adam_epsilon"].asDouble();
        }
        if (root.isMember("tol_grad"))
        {
            config.tol_grad = root["tol_grad"].asDouble();
        }
        if (root.isMember("tol_loss"))
        {
            config.tol_loss = root["tol_loss"].asDouble();
        }
        if (root.isMember("use_exponential_param"))
        {
            config.use_exponential_param = root["use_exponential_param"].asBool();
        }
        if (root.isMember("verbose"))
        {
            config.verbose = root["verbose"].asBool();
        }
        if (root.isMember("enable_gradient_fd_check"))
        {
            config.enable_gradient_fd_check = root["enable_gradient_fd_check"].asBool();
        }
        if (root.isMember("gradient_fd_epsilon"))
        {
            config.gradient_fd_epsilon = root["gradient_fd_epsilon"].asDouble();
        }
    }

    Eigen::VectorXd loadInitialThetaFromRobotConfig(const Json::Value &robot_config)
    {
        const Json::Value weights = robot_config["weights"];
        if (weights.isNull())
        {
            throw std::runtime_error("Robot config is missing the 'weights' section for initial theta.");
        }

        auto readWeights = [&](const char *name, int expected_size) {
            const Json::Value values = weights[name];
            if (!values.isArray() || static_cast<int>(values.size()) != expected_size)
            {
                throw std::runtime_error(std::string("Robot config weights '") + name + "' must be an array of size " + std::to_string(expected_size) + ".");
            }

            Eigen::VectorXd out(expected_size);
            for (int i = 0; i < expected_size; ++i)
            {
                const double value = values[i].asDouble();
                if (!(value > 0.0))
                {
                    throw std::runtime_error(std::string("Initial theta weight '") + name + "' must be strictly positive.");
                }
                out(i) = value;
            }
            return out;
        };

        Eigen::VectorXd theta = Eigen::VectorXd::Zero(kThetaSize);
        theta.segment<18>(kP0Offset) = readWeights("P0", 18);
        theta.segment<6>(kQ1Offset) = readWeights("Q1", 6);
        theta.segment<6>(kQ2Offset) = readWeights("Q2", 6);
        theta.segment<6>(kQ3Offset) = readWeights("Q3", 6);
        return theta;
    }

    void loadRobotTopologyFromJson(const Json::Value &root, Spacetime::RobotTopology &topology)
    {
        topology.N = root["topology"].isMember("N") ? root["topology"]["N"].asUInt() : 1;
        topology.K = root["topology"].isMember("K_per_second") ? root["topology"]["K_per_second"].asUInt() : 1;
        topology.Ms = root["topology"].isMember("Ms") ? root["topology"]["Ms"].asUInt() : 1;
        topology.Mt = root["topology"].isMember("Mt") ? root["topology"]["Mt"].asUInt() : 1;
        topology.L = root["topology"].isMember("length") ? root["topology"]["length"].asDouble() : 0.0;
        topology.radius = root["topology"].isMember("radius") ? root["topology"]["radius"].asDouble() : 0.01;
        topology.lock_first_position = root["topology"].isMember("lock_first_position") ? root["topology"]["lock_first_position"].asBool() : false;
        topology.lock_first_pose = root["topology"].isMember("lock_first_pose") ? root["topology"]["lock_first_pose"].asBool() : false;
        topology.lock_last_strain = root["topology"].isMember("lock_last_strain") ? root["topology"]["lock_last_strain"].asBool() : false;
        topology.time_nodes_on_measurements = root["topology"].isMember("time_nodes_on_measurements") ? root["topology"]["time_nodes_on_measurements"].asBool() : false;
        topology.use_1D_estimator = root["topology"].isMember("use_1D_estimator") ? root["topology"]["use_1D_estimator"].asBool() : false;
        topology.fbg_theta_offset = 0.0;
        topology.T = 0.0;
        topology.t0 = 0.0;
        topology.T0 = Eigen::Matrix<double, 4, 4>::Identity();
        topology.epsilon0 = Eigen::Matrix<double, 6, 1>::Zero();
        topology.varpi0 = Eigen::Matrix<double, 6, 1>::Zero();
    }

    void loadTrialInitialConditionFromJson(const Json::Value &root, Spacetime::RobotTopology &topology)
    {
        const Json::Value initial_condition = root["initial_condition"];
        if (!initial_condition.isNull())
        {
            topology.t0 = initial_condition.isMember("start_time") ? initial_condition["start_time"].asDouble() : 0.0;

            if (initial_condition.isMember("T0") && initial_condition["T0"].isArray() && initial_condition["T0"].size() == 16)
            {
                for (int row = 0; row < 4; ++row)
                {
                    for (int col = 0; col < 4; ++col)
                    {
                        topology.T0(row, col) = initial_condition["T0"][col * 4 + row].asDouble();
                    }
                }
            }

            if (initial_condition.isMember("epsilon0") && initial_condition["epsilon0"].isArray() && initial_condition["epsilon0"].size() == 6)
            {
                for (int i = 0; i < 6; ++i)
                {
                    topology.epsilon0(i) = initial_condition["epsilon0"][i].asDouble();
                }
            }

            if (initial_condition.isMember("varpi0") && initial_condition["varpi0"].isArray() && initial_condition["varpi0"].size() == 6)
            {
                for (int i = 0; i < 6; ++i)
                {
                    topology.varpi0(i) = initial_condition["varpi0"][i].asDouble();
                }
            }
        }
    }

    std::pair<double, double> loadTrialTimeWindowFromJson(const Json::Value &root)
    {
        const Json::Value data = root["data"];
        const double start_time = data.isMember("start_time") ? data["start_time"].asDouble() : 0.0;
        const double end_time = data.isMember("end_time") ? data["end_time"].asDouble() : std::numeric_limits<double>::infinity();
        return {start_time, end_time};
    }

    template <typename MatrixType>
    void fillDiagonalFromTheta(MatrixType &matrix, const Eigen::VectorXd &theta, int offset, const char *name)
    {
        const int diagonal_size = static_cast<int>(matrix.rows());
        if (matrix.rows() != matrix.cols())
        {
            throw std::runtime_error(std::string("Expected diagonal square matrix for ") + name + ".");
        }
        if (theta.size() != kThetaSize)
        {
            throw std::runtime_error("Theta must contain exactly 36 diagonal parameters for P0, Q1, Q2, and Q3.");
        }
        if (offset + diagonal_size > theta.size())
        {
            throw std::runtime_error(std::string("Theta layout is inconsistent while assembling ") + name + ".");
        }

        matrix.setIdentity();
        for (int i = 0; i < diagonal_size; ++i)
        {
            const double value = theta(offset + i);
            if (!(value > 0.0))
            {
                throw std::runtime_error(std::string(name) + " must be strictly positive on the diagonal.");
            }
            matrix(i, i) = value;
        }
    }

    Eigen::VectorXd solvePositiveDefiniteSystem(const Eigen::MatrixXd& matrix, const Eigen::VectorXd& rhs)
    {
        if (matrix.rows() == 0)
        {
            return Eigen::VectorXd();
        }

        Eigen::LDLT<Eigen::MatrixXd> solver(matrix);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to factorize information matrix while computing factor gradients.");
        }

        return solver.solve(rhs);
    }

    Eigen::MatrixXd solvePositiveDefiniteSystem(const Eigen::MatrixXd& matrix, const Eigen::MatrixXd& rhs)
    {
        if (matrix.rows() == 0)
        {
            return Eigen::MatrixXd();
        }

        Eigen::LDLT<Eigen::MatrixXd> solver(matrix);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to factorize dense matrix while solving for node marginals.");
        }

        return solver.solve(rhs);
    }

    //u_f = Q_f^{-1} e_f lambda_f;
    //v_f = Q_f^{-1} e_f;
    //w_f = Q_f^{-1} E_f e_nodes;
    Eigen::VectorXd applyFactorInformation(const Eigen::MatrixXd& info_f, const Eigen::VectorXd& rhs)
    {
        if (info_f.cols() != rhs.size())
        {
            throw std::runtime_error("Factor information matrix and vector size mismatch.");
        }

        return info_f * rhs;
    }

    Eigen::VectorXd blockSegment(const Eigen::VectorXd& vector, int block_index, int block_size)
    {
        return vector.segment(block_index * block_size, block_size);
    }

    // For 
    spacetime::FactorGradientContrib computeDiagonalFactorGradient(const Eigen::VectorXd& e_f,
                                                                    const Eigen::MatrixXd& info_f,
                                                                    const Eigen::MatrixXd& E_f,
                                                                    const Eigen::VectorXd& lambda_f,
                                                                    const Eigen::VectorXd& e_nodes,
                                                                    double info_scale)
    {
        const Eigen::VectorXd v = applyFactorInformation(info_f, e_f);
        const Eigen::VectorXd u = applyFactorInformation(info_f, E_f * lambda_f);
        const Eigen::VectorXd w = applyFactorInformation(info_f, E_f * e_nodes);

        spacetime::FactorGradientContrib contrib;
        contrib.dL_dtheta_state = -(u.array() * v.array()).matrix();
        contrib.dL_dtheta_info = info_scale * w.array().square().matrix();
        return contrib;
    }

    spacetime::FactorGradientContrib computeBinaryTimeFactorGradient(const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                                                      const Eigen::VectorXd& e_f,
                                                                      const Eigen::MatrixXd& info_f,
                                                                      const Eigen::MatrixXd& E_f,
                                                                      const Eigen::VectorXd& lambda_f,
                                                                      const Eigen::VectorXd& e_nodes,
                                                                      double info_scale)
    {
        if (local_nodes.size() != 2)
        {
            throw std::runtime_error("Binary time factor gradient expects two local nodes.");
        }

        const Eigen::VectorXd v = applyFactorInformation(info_f, e_f);
        const Eigen::VectorXd u = applyFactorInformation(info_f, E_f * lambda_f);
        const Eigen::VectorXd w = applyFactorInformation(info_f, E_f * e_nodes);

        if (v.size() != 18 || u.size() != 18 || w.size() != 18)
        {
            throw std::runtime_error("Binary time factor gradient expects 18-dimensional residual blocks.");
        }

        const double dt = local_nodes[1].time - local_nodes[0].time;
        const double dt2 = dt * dt / 2.0;
        const double dt3 = dt * dt * dt / 3.0;

        const Eigen::VectorXd u1 = blockSegment(u, 0, 6);
        const Eigen::VectorXd u2 = blockSegment(u, 1, 6);
        const Eigen::VectorXd u3 = blockSegment(u, 2, 6);
        const Eigen::VectorXd v1 = blockSegment(v, 0, 6);
        const Eigen::VectorXd v2 = blockSegment(v, 1, 6);
        const Eigen::VectorXd v3 = blockSegment(v, 2, 6);
        const Eigen::VectorXd w1 = blockSegment(w, 0, 6);
        const Eigen::VectorXd w2 = blockSegment(w, 1, 6);
        const Eigen::VectorXd w3 = blockSegment(w, 2, 6);

        spacetime::FactorGradientContrib contrib;
        contrib.dL_dtheta_state = Eigen::VectorXd::Zero(12);
        contrib.dL_dtheta_info = Eigen::VectorXd::Zero(12);

        contrib.dL_dtheta_state.segment<6>(0) =
            (dt3 * (u1.array() * v1.array())).matrix() +
            (dt2 * ((u1.array() * v3.array()).matrix() + (u3.array() * v1.array()).matrix())) +
            (dt * (u3.array() * v3.array()).matrix());
        contrib.dL_dtheta_state.segment<6>(6) = dt * (u2.array() * v2.array()).matrix();

        contrib.dL_dtheta_info.segment<6>(0) = info_scale *
            ((dt3 * (w1.array().square()).matrix()) +
             (2.0 * dt2 * (w1.array() * w3.array()).matrix()) +
             (dt * (w3.array().square()).matrix()));
        contrib.dL_dtheta_info.segment<6>(6) = info_scale * (dt * (w2.array().square()).matrix());

        return contrib;
    }

    spacetime::FactorGradientContrib computeBinarySpaceFactorGradient(const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                                                       const Eigen::VectorXd& e_f,
                                                                       const Eigen::MatrixXd& info_f,
                                                                       const Eigen::MatrixXd& E_f,
                                                                       const Eigen::VectorXd& lambda_f,
                                                                       const Eigen::VectorXd& e_nodes,
                                                                       double info_scale)
    {
        if (local_nodes.size() != 2)
        {
            throw std::runtime_error("Binary space factor gradient expects two local nodes.");
        }

        const Eigen::VectorXd v = applyFactorInformation(info_f, e_f);
        const Eigen::VectorXd u = applyFactorInformation(info_f, E_f * lambda_f);
        const Eigen::VectorXd w = applyFactorInformation(info_f, E_f * e_nodes);

        if (v.size() != 18 || u.size() != 18 || w.size() != 18)
        {
            throw std::runtime_error("Binary space factor gradient expects 18-dimensional residual blocks.");
        }

        const double ds = local_nodes[1].arclength - local_nodes[0].arclength;
        const double ds2 = ds * ds / 2.0;
        const double ds3 = ds * ds * ds / 3.0;

        const Eigen::VectorXd u1 = blockSegment(u, 0, 6);
        const Eigen::VectorXd u2 = blockSegment(u, 1, 6);
        const Eigen::VectorXd u3 = blockSegment(u, 2, 6);
        const Eigen::VectorXd v1 = blockSegment(v, 0, 6);
        const Eigen::VectorXd v2 = blockSegment(v, 1, 6);
        const Eigen::VectorXd v3 = blockSegment(v, 2, 6);
        const Eigen::VectorXd w1 = blockSegment(w, 0, 6);
        const Eigen::VectorXd w2 = blockSegment(w, 1, 6);
        const Eigen::VectorXd w3 = blockSegment(w, 2, 6);

        const Eigen::Matrix<double, 6, 6> varpihat_t = se3::curlyhat(local_nodes[0].varpi).template cast<double>().transpose();
        const Eigen::VectorXd a = varpihat_t * u3;
        const Eigen::VectorXd b = varpihat_t * v3;
        const Eigen::VectorXd c = varpihat_t * w3;

        spacetime::FactorGradientContrib contrib;
        contrib.dL_dtheta_state = Eigen::VectorXd::Zero(12);
        contrib.dL_dtheta_info = Eigen::VectorXd::Zero(12);

        contrib.dL_dtheta_state.segment<6>(0) =
            (ds3 * (u1.array() * v1.array())).matrix() +
            (ds2 * ((u1.array() * v2.array()).matrix() + (u2.array() * v1.array()).matrix())) +
            (ds * (u2.array() * v2.array()).matrix()) -
            (ds3 * ((v1.array() * a.array()).matrix() + (u1.array() * b.array()).matrix())) -
            (ds2 * ((v2.array() * a.array()).matrix() + (u2.array() * b.array()).matrix())) +
            (ds3 * (a.array() * b.array()).matrix());
        contrib.dL_dtheta_state.segment<6>(6) = ds * (u3.array() * v3.array()).matrix();

        contrib.dL_dtheta_info.segment<6>(0) = info_scale *
            ((ds3 * (w1.array().square()).matrix()) +
             (2.0 * ds2 * (w1.array() * w2.array()).matrix()) +
             (ds * (w2.array().square()).matrix()) -
             (2.0 * ds3 * (w1.array() * c.array()).matrix()) -
             (2.0 * ds2 * (w2.array() * c.array()).matrix()) +
             (ds3 * (c.array().square()).matrix()));
        contrib.dL_dtheta_info.segment<6>(6) = info_scale * (ds * (w3.array().square()).matrix());

        return contrib;
    }

    spacetime::FactorGradientContrib computeProposalFactorGradient(const Spacetime::Factors::Factor& factor,
                                                                    const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                                                    const Eigen::VectorXd& e_f,
                                                                    const Eigen::MatrixXd& Q_f,
                                                                    const Eigen::MatrixXd& E_f,
                                                                    const Eigen::VectorXd& lambda_f,
                                                                    const Eigen::VectorXd& e_nodes,
                                                                    double info_scale)
    {
        // If this is Time factor
        if (dynamic_cast<const Spacetime::Factors::BinaryTimeFactor*>(&factor) != nullptr)
        {
            return computeBinaryTimeFactorGradient(local_nodes, e_f, Q_f, E_f, lambda_f, e_nodes, info_scale);
        }

        // If this is Space factor
        if (dynamic_cast<const Spacetime::Factors::BinarySpaceFactor*>(&factor) != nullptr)
        {
            return computeBinarySpaceFactorGradient(local_nodes, e_f, Q_f, E_f, lambda_f, e_nodes, info_scale);
        }

        // If this is unary factor
        if (dynamic_cast<const Spacetime::Factors::UnaryFactor*>(&factor) != nullptr)
        {
            return computeDiagonalFactorGradient(e_f, Q_f, E_f, lambda_f, e_nodes, info_scale);
        }

        throw std::runtime_error("Unknown factor type.");
    }

    Eigen::Matrix<double, 18, 18> computeNodeMarginalInformationBlock(const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                                                                       int total_dim,
                                                                       int node_index)
    {
        constexpr int node_dim = 18;
        constexpr double kMarginalCovarianceFloor = 1e-4;
        const int start = node_index * node_dim;

        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(total_dim, node_dim);
        rhs.block(start, 0, node_dim, node_dim).setIdentity();

        Eigen::MatrixXd solved_columns = solver.solve(rhs);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to solve for node marginal covariance columns.");
        }

        Eigen::Matrix<double, 18, 18> marginal_cov = solved_columns.block(start, 0, node_dim, node_dim);
        marginal_cov = 0.5 * (marginal_cov + marginal_cov.transpose());

        // Stabilize inversion by flooring tiny/negative modes in the marginal covariance block.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 18, 18>> eigensolver(marginal_cov);
        if (eigensolver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to eigendecompose node marginal covariance block.");
        }
        Eigen::Matrix<double, 18, 1> eigenvalues = eigensolver.eigenvalues();
        eigenvalues = eigenvalues.array().max(kMarginalCovarianceFloor);
        marginal_cov = eigensolver.eigenvectors() * eigenvalues.asDiagonal() * eigensolver.eigenvectors().transpose();

        Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(18, 18);
        return solvePositiveDefiniteSystem(marginal_cov, identity);
    }

    MarginalCacheData buildMarginalCache(const Eigen::SparseMatrix<double>& H,
                                         const std::vector<spacetime::ValidationTarget>& validation_targets)
    {
        MarginalCacheData cache;
        std::unordered_set<int> unique_nodes;
        unique_nodes.reserve(validation_targets.size());
        for (const auto& target : validation_targets)
        {
            unique_nodes.insert(target.node_index);
        }

        auto t_factor_start = std::chrono::steady_clock::now();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to factorize H while building marginal cache.");
        }
        auto t_factor_end = std::chrono::steady_clock::now();
        cache.factorization_ms = std::chrono::duration<double, std::milli>(t_factor_end - t_factor_start).count();

        auto t_solve_start = std::chrono::steady_clock::now();
        cache.node_information_blocks.reserve(unique_nodes.size());
        for (const int node_index : unique_nodes)
        {
            cache.node_information_blocks.emplace(
                node_index,
                computeNodeMarginalInformationBlock(solver, H.rows(), node_index));
        }
        auto t_solve_end = std::chrono::steady_clock::now();
        cache.marginal_solve_ms = std::chrono::duration<double, std::milli>(t_solve_end - t_solve_start).count();

        return cache;
    }

    const Matrix18d& getCachedNodeInformationBlock(const MarginalCacheData& cache, int node_index)
    {
        const auto it = cache.node_information_blocks.find(node_index);
        if (it == cache.node_information_blocks.end())
        {
            throw std::runtime_error("Node information block was not found in marginal cache.");
        }
        return it->second;
    }

    spacetime::ValidationLossData computeValidationLossWithMarginals(
        const std::vector<spacetime::ValidationTarget>& validation_targets,
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& solved_nodes,
        const MarginalCacheData& cache)
    {
        spacetime::ValidationLossData out;
        if (validation_targets.empty() || solved_nodes.empty())
        {
            return out;
        }

        constexpr int node_dim = 18;
        const int validation_count = static_cast<int>(validation_targets.size());
        out.nees_per_node = Eigen::VectorXd::Zero(validation_count);
        out.node_errors_stacked = Eigen::VectorXd::Zero(node_dim * validation_count);

        for (int i = 0; i < validation_count; ++i)
        {
            const auto& target = validation_targets[static_cast<size_t>(i)];
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes.size()))
            {
                throw std::runtime_error("computeValidationLossWithMarginals: validation node index out of range.");
            }

            const auto& estimate = solved_nodes[static_cast<size_t>(target.node_index)];
            const Vector18d e = computeNodeValidationError(estimate, target.ground_truth);
            out.node_errors_stacked.segment<18>(i * node_dim) = e;

            const Matrix18d& p_inv_block = getCachedNodeInformationBlock(cache, target.node_index);
            out.nees_per_node(i) = e.transpose() * p_inv_block * e;
        }

        out.mean_nees = out.nees_per_node.mean();
        out.delta = out.mean_nees - 18.0;
        out.loss = 0.5 * out.delta * out.delta;
        return out;
    }

    Eigen::VectorXd computeAdjointRhsWithMarginals(const Eigen::SparseMatrix<double>& H,
                                                   const std::vector<spacetime::ValidationTarget>& validation_targets,
                                                   const spacetime::ValidationLossData& validation,
                                                   const MarginalCacheData& cache)
    {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(H.rows());
        if (validation_targets.empty())
        {
            return rhs;
        }

        constexpr int node_dim = 18;
        const int validation_count = static_cast<int>(validation_targets.size());
        if (validation.node_errors_stacked.size() != validation_count * node_dim)
        {
            throw std::runtime_error("computeAdjointRhsWithMarginals: validation error vector has unexpected size.");
        }

        const double scale = 2.0 * validation.delta / static_cast<double>(validation_count);
        for (int i = 0; i < validation_count; ++i)
        {
            const auto& target = validation_targets[static_cast<size_t>(i)];
            if (target.node_index < 0 || (target.node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("computeAdjointRhsWithMarginals: validation node index out of range.");
            }

            const Vector18d e = validation.node_errors_stacked.segment<18>(i * node_dim);
            const Matrix18d& p_inv_block = getCachedNodeInformationBlock(cache, target.node_index);

            Matrix18d J = Matrix18d::Identity();
            J.block<6, 6>(0, 0) = se3::vec2jacinv(e.segment<6>(0));
            rhs.segment<18>(target.node_index * node_dim) += J.transpose() * p_inv_block * e;
        }

        rhs *= scale;
        return rhs;
    }

    Eigen::Matrix<double, 6, 6> diagonal6(const Eigen::VectorXd& values)
    {
        if (values.size() != 6)
        {
            throw std::runtime_error("Expected 6D diagonal parameter vector.");
        }

        Eigen::Matrix<double, 6, 6> out = Eigen::Matrix<double, 6, 6>::Zero();
        for (int i = 0; i < 6; ++i)
        {
            out(i, i) = values(i);
        }
        return out;
    }

    Matrix18d buildTimeCovariance(const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                  const Eigen::VectorXd& theta_local)
    {
        if (local_nodes.size() != 2 || theta_local.size() != 12)
        {
            throw std::runtime_error("buildTimeCovariance expects two local nodes and a 12D parameter vector.");
        }

        const double dt = local_nodes[1].time - local_nodes[0].time;
        const double dt2 = dt * dt / 2.0;
        const double dt3 = dt * dt * dt / 3.0;

        const Eigen::Matrix<double, 6, 6> Q1 = diagonal6(theta_local.segment<6>(0));
        const Eigen::Matrix<double, 6, 6> Q3 = diagonal6(theta_local.segment<6>(6));

        Matrix18d Q = Matrix18d::Zero();
        Q.block<6, 6>(0, 0) = dt3 * Q1;
        Q.block<6, 6>(0, 12) = dt2 * Q1;
        Q.block<6, 6>(6, 6) = dt * Q3;
        Q.block<6, 6>(12, 0) = dt2 * Q1;
        Q.block<6, 6>(12, 12) = dt * Q1;
        return Q;
    }

    Matrix18d buildSpaceCovariance(const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                   const Eigen::VectorXd& theta_local)
    {
        if (local_nodes.size() != 2 || theta_local.size() != 12)
        {
            throw std::runtime_error("buildSpaceCovariance expects two local nodes and a 12D parameter vector.");
        }

        const double ds = local_nodes[1].arclength - local_nodes[0].arclength;
        const double ds2 = ds * ds / 2.0;
        const double ds3 = ds * ds * ds / 3.0;

        const Eigen::Matrix<double, 6, 6> Q2 = diagonal6(theta_local.segment<6>(0));
        const Eigen::Matrix<double, 6, 6> Q3 = diagonal6(theta_local.segment<6>(6));
        const Eigen::Matrix<double, 6, 6> varpihat = se3::curlyhat(local_nodes[0].varpi).template cast<double>();

        Matrix18d Q = Matrix18d::Zero();
        Q.block<6, 6>(0, 0) = ds3 * Q2;
        Q.block<6, 6>(0, 6) = ds2 * Q2;
        Q.block<6, 6>(0, 12) = -ds3 * Q2 * varpihat.transpose();
        Q.block<6, 6>(6, 0) = ds2 * Q2;
        Q.block<6, 6>(6, 6) = ds * Q2;
        Q.block<6, 6>(6, 12) = -ds2 * Q2 * varpihat.transpose();
        Q.block<6, 6>(12, 0) = -ds3 * varpihat * Q2;
        Q.block<6, 6>(12, 6) = -ds2 * varpihat * Q2;
        Q.block<6, 6>(12, 12) = ds * Q3 + ds3 * varpihat * Q2 * varpihat.transpose();
        return Q;
    }

    Eigen::VectorXd finiteDifferenceGradient(const Eigen::VectorXd& theta0,
                                             double epsilon,
                                             const std::function<double(const Eigen::VectorXd&)>& objective)
    {
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(theta0.size());
        for (int i = 0; i < theta0.size(); ++i)
        {
            const double step = epsilon * std::max(1.0, std::abs(theta0(i)));
            Eigen::VectorXd theta_plus = theta0;
            Eigen::VectorXd theta_minus = theta0;
            theta_plus(i) += step;
            theta_minus(i) = std::max(1e-12, theta_minus(i) - step);
            gradient(i) = (objective(theta_plus) - objective(theta_minus)) / (theta_plus(i) - theta_minus(i));
        }
        return gradient;
    }

    void printGradientCheckSummary(const std::string& name,
                                   const Eigen::VectorXd& analytic,
                                   const Eigen::VectorXd& numeric)
    {
        if (analytic.size() != numeric.size())
        {
            throw std::runtime_error("Gradient check vectors have mismatched sizes.");
        }

        double max_abs_error = 0.0;
        double max_rel_error = 0.0;
        int worst_index = -1;
        for (int i = 0; i < analytic.size(); ++i)
        {
            const double abs_error = std::abs(analytic(i) - numeric(i));
            const double rel_error = abs_error / std::max(1e-10, std::abs(numeric(i)));
            if (abs_error > max_abs_error)
            {
                max_abs_error = abs_error;
                max_rel_error = rel_error;
                worst_index = i;
            }
        }

        std::cout << "Gradient FD check [" << name << "]: max_abs_error=" << max_abs_error
                  << ", max_rel_error=" << max_rel_error
                  << ", worst_index=" << worst_index << std::endl;
    }

    void runKernelFiniteDifferenceChecks(const Eigen::VectorXd& theta,
                                         const FactorKernelSample& unary_sample,
                                         const FactorKernelSample& time_sample,
                                         const FactorKernelSample& space_sample,
                                         double epsilon)
    {
        if (unary_sample.available)
        {
            const spacetime::FactorGradientContrib contrib = computeDiagonalFactorGradient(
                unary_sample.e_f,
                unary_sample.info_f,
                unary_sample.E_f,
                unary_sample.lambda_f,
                unary_sample.e_nodes,
                unary_sample.info_scale);
            const Eigen::VectorXd analytic = contrib.dL_dtheta_state + contrib.dL_dtheta_info;

            const Eigen::VectorXd theta_unary = theta.segment<18>(kP0Offset);
            const Eigen::VectorXd a = unary_sample.E_f * unary_sample.lambda_f;
            const Eigen::VectorXd b = unary_sample.e_f;
            const Eigen::VectorXd c = unary_sample.E_f * unary_sample.e_nodes;

            const auto objective = [&](const Eigen::VectorXd& theta_local) {
                Eigen::MatrixXd info = theta_local.cwiseInverse().asDiagonal();
                return (a.transpose() * info * b)(0, 0) - unary_sample.info_scale * (c.transpose() * info * c)(0, 0);
            };

            const Eigen::VectorXd numeric = finiteDifferenceGradient(theta_unary, epsilon, objective);
            printGradientCheckSummary("unary", analytic, numeric);
        }

        if (time_sample.available)
        {
            const spacetime::FactorGradientContrib contrib = computeBinaryTimeFactorGradient(
                time_sample.local_nodes,
                time_sample.e_f,
                time_sample.info_f,
                time_sample.E_f,
                time_sample.lambda_f,
                time_sample.e_nodes,
                time_sample.info_scale);
            const Eigen::VectorXd analytic = contrib.dL_dtheta_state + contrib.dL_dtheta_info;

            Eigen::VectorXd theta_time(12);
            theta_time.segment<6>(0) = theta.segment<6>(kQ1Offset);
            theta_time.segment<6>(6) = theta.segment<6>(kQ3Offset);
            const Eigen::VectorXd a = time_sample.E_f * time_sample.lambda_f;
            const Eigen::VectorXd b = time_sample.e_f;
            const Eigen::VectorXd c = time_sample.E_f * time_sample.e_nodes;

            const auto objective = [&](const Eigen::VectorXd& theta_local) {
                const Matrix18d Q = buildTimeCovariance(time_sample.local_nodes, theta_local);
                const Matrix18d info = Q.inverse();
                return -(a.transpose() * info * b)(0, 0) - time_sample.info_scale * (c.transpose() * info * c)(0, 0);
            };

            const Eigen::VectorXd numeric = finiteDifferenceGradient(theta_time, epsilon, objective);
            printGradientCheckSummary("time", analytic, numeric);
        }

        if (space_sample.available)
        {
            const spacetime::FactorGradientContrib contrib = computeBinarySpaceFactorGradient(
                space_sample.local_nodes,
                space_sample.e_f,
                space_sample.info_f,
                space_sample.E_f,
                space_sample.lambda_f,
                space_sample.e_nodes,
                space_sample.info_scale);
            const Eigen::VectorXd analytic = contrib.dL_dtheta_state + contrib.dL_dtheta_info;

            Eigen::VectorXd theta_space(12);
            theta_space.segment<6>(0) = theta.segment<6>(kQ2Offset);
            theta_space.segment<6>(6) = theta.segment<6>(kQ3Offset);
            const Eigen::VectorXd a = space_sample.E_f * space_sample.lambda_f;
            const Eigen::VectorXd b = space_sample.e_f;
            const Eigen::VectorXd c = space_sample.E_f * space_sample.e_nodes;

            const auto objective = [&](const Eigen::VectorXd& theta_local) {
                const Matrix18d Q = buildSpaceCovariance(space_sample.local_nodes, theta_local);
                const Matrix18d info = Q.inverse();
                return -(a.transpose() * info * b)(0, 0) - space_sample.info_scale * (c.transpose() * info * c)(0, 0);
            };

            const Eigen::VectorXd numeric = finiteDifferenceGradient(theta_space, epsilon, objective);
            printGradientCheckSummary("space", analytic, numeric);
        }
    }

    Eigen::Matrix<double, 18, 1> computeNodeValidationError(const Spacetime::SystemState<DTYPE>::Node &estimate,
                                                            const Spacetime::SystemState<DTYPE>::Node &ground_truth)
    {
        Eigen::Matrix<double, 18, 1> e;
        e.template segment<6>(0) = se3::tran2vec(ground_truth.pose * invertTransformation(estimate.pose)).template cast<double>();
        e.template segment<6>(6) = (ground_truth.epsilon - estimate.epsilon).template cast<double>();
        e.template segment<6>(12) = (ground_truth.varpi - estimate.varpi).template cast<double>();
        return e;
    }

    struct TestProblemData
    {
        Spacetime::RobotTopology robot_topology;
        Spacetime::Options estimator_options;
        Eigen::VectorXd initial_theta;
        std::vector<spacetime::OptimizationProblem::Node> nodes;
        std::vector<std::shared_ptr<Spacetime::Factors::Factor>> factors;
        std::vector<std::vector<int>> factor_node_indices;
        std::vector<spacetime::ValidationTarget> validation_targets;
        std::filesystem::path csv_path;
    };

    TestProblemData buildMinimalTestProblemData()
    {
        TestProblemData data;

        const std::filesystem::path trial_config_path = resolveExistingPath({
            "assets/config/trial/minimal.json",
            "../assets/config/trial/minimal.json",
            "../../assets/config/trial/minimal.json"
        });
        const std::filesystem::path robot_config_path = resolveExistingPath({
            "assets/config/robot/minimal.json",
            "../assets/config/robot/minimal.json",
            "../../assets/config/robot/minimal.json"
        });
        const std::filesystem::path estimator_config_path = resolveExistingPath({
            "assets/config/estimator/batch.json",
            "../assets/config/estimator/batch.json",
            "../../assets/config/estimator/batch.json"
        });

        const Json::Value trial_config = loadJsonFile(trial_config_path);
        const Json::Value robot_config = loadJsonFile(robot_config_path);
        const Json::Value estimator_config = loadJsonFile(estimator_config_path);

        data.initial_theta = loadInitialThetaFromRobotConfig(robot_config);

        loadRobotTopologyFromJson(robot_config, data.robot_topology);
        loadTrialInitialConditionFromJson(trial_config, data.robot_topology);
        const auto [trial_start_time, trial_end_time] = loadTrialTimeWindowFromJson(trial_config);
        const auto validation_interval = trial_config["options"]["validation_interval"].asInt();

        const std::filesystem::path repo_root = std::filesystem::absolute(trial_config_path)
                                                    .parent_path()
                                                    .parent_path()
                                                    .parent_path()
                                                    .parent_path();
        const std::filesystem::path data_dir = repo_root / trial_config["data"]["folder_path"].asString();

        for (const auto& entry : std::filesystem::directory_iterator(data_dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".csv")
            {
                data.csv_path = entry.path();
                break;
            }
        }

        if (data.csv_path.empty())
        {
            throw std::runtime_error("No CSV file found in minimal data directory: " + data_dir.string());
        }

        const std::vector<std::vector<std::string>> csv_rows = readCsvRows(data.csv_path);
        if (csv_rows.size() < 3)
        {
            throw std::runtime_error("Minimal CSV must contain at least one pose measurement row.");
        }

        Eigen::Matrix<double, 6, 6> pose_weight = Eigen::Matrix<double, 6, 6>::Zero();
        for (int i = 0; i < 6; ++i)
        {
            pose_weight(i, i) = robot_config["weights"]["R_pose"][i].asDouble();
        }

        data.nodes.clear();
        data.factors.clear();
        data.factor_node_indices.clear();
        data.validation_targets.clear();

        for (size_t row_index = 2; row_index < csv_rows.size(); ++row_index)
        {
            const auto& row = csv_rows[row_index];
            if (row.size() < 8)
            {
                continue;
            }

            const double row_time = std::stod(row[0]);
            if (row_time < trial_start_time || row_time > trial_end_time)
            {
                continue;
            }

            spacetime::OptimizationProblem::Node node;
            node.pose = data.robot_topology.T0;
            node.epsilon = data.robot_topology.epsilon0;
            node.varpi = data.robot_topology.varpi0;
            node.arclength = std::stod(row[1]);
            node.time = row_time;
            data.nodes.push_back(node);

            Spacetime::SensorMeasurement measurement;
            measurement.type = Spacetime::SensorMeasurement::Pose;
            measurement.mask = Eigen::Matrix<int, 6, 1>::Ones();
            measurement.value = se3::vec2tran(parsePoseRow(row));
            measurement.s = std::stod(row[1]);
            measurement.t = node.time;

            data.factors.push_back(std::make_shared<Spacetime::Factors::PoseMeasurementFactor>(pose_weight, measurement));
            data.factor_node_indices.push_back({static_cast<int>(data.nodes.size() - 1)});

            // Use a sparse subset of nodes as validation targets to exercise the upper-level loop.
            if ((data.nodes.size() - 1) % validation_interval == 0)
            {
                spacetime::ValidationTarget target;
                target.node_index = static_cast<int>(data.nodes.size() - 1);
                target.ground_truth = node;
                target.ground_truth.pose = measurement.value;
                data.validation_targets.push_back(target);
            }
        }

        if (data.nodes.empty())
        {
            throw std::runtime_error("No valid nodes were created from the minimal dataset.");
        }
        if (data.validation_targets.empty())
        {
            spacetime::ValidationTarget target;
            target.node_index = 0;
            target.ground_truth = data.nodes.front();
            if (!data.factors.empty())
            {
                const auto pose_factor = std::dynamic_pointer_cast<Spacetime::Factors::PoseMeasurementFactor>(data.factors.front());
                if (pose_factor)
                {
                    target.ground_truth.pose = pose_factor->getMeas().value;
                }
            }
            data.validation_targets.push_back(target);
        }

        loadEstimatorOptionsFromJson(estimator_config, data.estimator_options);
        return data;
    }
}

namespace spacetime {
    void GradientAccumulator::accumulate(const std::vector<FactorGradientContrib> &per_factor_contribs,
                                         Eigen::VectorXd &dL_dtheta)
    {
        if (per_factor_contribs.empty())
        {
            dL_dtheta.resize(0);
            return;
        }

        const int gradient_size = static_cast<int>(per_factor_contribs.front().dL_dtheta_state.size());
        if (dL_dtheta.size() == 0)
        {
            dL_dtheta = Eigen::VectorXd::Zero(gradient_size);
        }
        else if (dL_dtheta.size() != gradient_size)
        {
            throw std::runtime_error("GradientAccumulator::accumulate: gradient size mismatch.");
        }

        for (const auto &contrib : per_factor_contribs)
        {
            if (contrib.dL_dtheta_state.size() != gradient_size || contrib.dL_dtheta_info.size() != gradient_size)
            {
                throw std::runtime_error("GradientAccumulator::accumulate: factor contribution size mismatch.");
            }

            dL_dtheta += contrib.dL_dtheta_state;
            dL_dtheta += contrib.dL_dtheta_info;
        }
    }
}

namespace spacetime {
    OptimizationProblem::OptimizationProblem()
    {
        theta_ = Eigen::VectorXd::Ones(kThetaSize);
        estimator_options_.init_guess_type = Spacetime::Options::InitialGuessType::Custom;
        estimator_options_.solver = Spacetime::Options::Solver::Newton;
        estimator_options_.estimator_type = Spacetime::Options::EstimatorType::Batch;
        estimator_options_.max_optimization_iterations = 10;
        estimator_options_.convergence_threshold = 1e-8;
        estimator_options_.initial_perturbation = 0.0;
        estimator_options_.LM_damping = 0.0;
        estimator_options_.use_autodiff = false;
        estimator_options_.kirchhoff_rods = false;
        estimator_options_.interpolate_measurements = false;
        estimator_options_.compute_covariances = false;
        estimator_options_.marginalize = false;
        estimator_options_.reverse_order = false;
        estimator_options_.extract_from_front = false;

        assembleFromTheta();
    }

    void OptimizationProblem::setTheta(const Eigen::VectorXd& theta)
    {
        if (theta.size() != kThetaSize)
        {
            throw std::runtime_error("OptimizationProblem::setTheta expects exactly 36 parameters arranged as P0(18), Q1(6), Q2(6), and Q3(6).");
        }

        theta_ = theta;
        assembleFromTheta();
    }

    void OptimizationProblem::setProblemData(const std::vector<Node>& nodes,
                       const std::vector<std::shared_ptr<Spacetime::Factors::Factor>>& factors,
                       const std::vector<std::vector<int>>& factor_node_indices)
    {
        nodes_ = nodes;
        factors_ = factors;
        factor_node_indices_ = factor_node_indices;
        solved_nodes_ = nodes_;
    }

    void OptimizationProblem::setEstimatorOptions(const Spacetime::Options &options)
    {
        estimator_options_ = options;
        estimator_options_set_ = true;
    }

    void OptimizationProblem::setRobotTopology(const Spacetime::RobotTopology &topology)
    {
        robot_topology_ = topology;
        robot_topology_set_ = true;
    }

    void OptimizationProblem::assembleFromTheta()
    {
        hyperparameters_ = Spacetime::Hyperparameters();
        fillDiagonalFromTheta(hyperparameters_.P0, theta_, kP0Offset, "P0");
        fillDiagonalFromTheta(hyperparameters_.Q1, theta_, kQ1Offset, "Q1");
        fillDiagonalFromTheta(hyperparameters_.Q2, theta_, kQ2Offset, "Q2");
        fillDiagonalFromTheta(hyperparameters_.Q3, theta_, kQ3Offset, "Q3");
    }

    LowerLevelSolution OptimizationProblem::solveLowerLevel(bool verbose) // Returns x*, H and linearisation data.
    {
        LowerLevelSolution solution;

        if (nodes_.empty() || factors_.empty())
        {
            return solution;
        }

        if (!factor_node_indices_.empty() && factor_node_indices_.size() != factors_.size())
        {
            throw std::runtime_error(
                "OptimizationProblem: factor graph metadata is inconsistent. Factor node count does not match node index list count.");
        }

        constexpr int node_dim = 18;
        constexpr double damping = 1e-6;

        std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> measurement_factors;
        measurement_factors.reserve(factors_.size());
        for (std::size_t factor_idx = 0; factor_idx < factors_.size(); ++factor_idx)
        {
            const auto &factor = factors_[factor_idx];
            auto measurement_factor = std::dynamic_pointer_cast<Spacetime::Factors::MeasurementFactor>(factor);
            if (!measurement_factor)
            {
                throw std::runtime_error("OptimizationProblem: solveLowerLevel expects measurement factors when delegating to Estimator::computeStateEstimate().");
            }

            // Route measurement factors to the user-provided node assignment when available.
            measurement_factor->setRoutedNodeIndex(-1);
            if (!factor_node_indices_.empty() && factor_idx < factor_node_indices_.size())
            {
                const auto &assigned_nodes = factor_node_indices_[factor_idx];
                if (!assigned_nodes.empty())
                {
                    const int node_index = assigned_nodes.front();
                    if (node_index < 0 || node_index >= static_cast<int>(nodes_.size()))
                    {
                        throw std::runtime_error("OptimizationProblem: factor_node_indices contains an out-of-range node index.");
                    }
                    measurement_factor->setRoutedNodeIndex(node_index);
                }
            }
            /*
            if (verbose)
            {
                //Print factor information for debugging
                for (const auto &node_index : factor_node_indices_[&factor - &factors_[0]])
                {
                    auto m_meas = measurement_factor->getMeas();
                    auto value = m_meas.value;
                    std::cout << "Factor connects to node index: " << node_index << ", measurement value: " << value.transpose() << std::endl;
                }
            }*/
            measurement_factors.push_back(measurement_factor);
        }

        Spacetime::RobotTopology topology = robot_topology_;
        if (!robot_topology_set_)
        {
            topology.N = static_cast<unsigned int>(nodes_.size());
            topology.K = 1;
            topology.Ms = 1;
            topology.Mt = 1;
            topology.L = 0.0;
            topology.radius = 0.01;
            topology.lock_first_position = false;
            topology.lock_first_pose = false;
            topology.lock_last_strain = false;
            topology.time_nodes_on_measurements = false;
            topology.use_1D_estimator = true;
            topology.fbg_theta_offset = 0.0;
            topology.t0 = 0.0;
            topology.T0 = Eigen::Matrix<double, 4, 4>::Identity();
            topology.epsilon0 = Eigen::Matrix<double, 6, 1>::Zero();
            topology.varpi0 = Eigen::Matrix<double, 6, 1>::Zero();
        }
        else{
            std::cout << "Using provided robot topology with N=" << topology.N << ", K=" << topology.K << ", Ms=" << topology.Ms << ", Mt=" << topology.Mt << std::endl;
        }
        topology.N = std::max(1u, topology.N);
        topology.K = static_cast<unsigned int>(std::max<size_t>(1, nodes_.size() / topology.N));
        topology.Ms = std::max(1u, topology.Ms);
        topology.Mt = std::max(1u, topology.Mt);

        if (nodes_.size() % topology.N != 0)
        {
            throw std::runtime_error("OptimizationProblem: node count is not divisible by configured topology.N.");
        }

        if (!nodes_.empty())
        {
            topology.t0 = nodes_.front().time;
            topology.T0 = nodes_.front().pose;
            topology.epsilon0 = nodes_.front().epsilon;
            topology.varpi0 = nodes_.front().varpi;
        }

        double min_s = nodes_.front().arclength;
        double max_s = nodes_.front().arclength;
        double min_t = nodes_.front().time;
        double max_t = nodes_.front().time;
        for (const auto &node : nodes_)
        {
            min_s = std::min(min_s, node.arclength);
            max_s = std::max(max_s, node.arclength);
            min_t = std::min(min_t, node.time);
            max_t = std::max(max_t, node.time);
        }
        topology.L = max_s - min_s;
        topology.T = max_t - min_t;
        topology.spatial_node_positions.resize(static_cast<int>(topology.N));
        for (unsigned int i = 0; i < topology.N; ++i)
        {
            topology.spatial_node_positions(static_cast<int>(i)) = nodes_[i].arclength;
        }

        if (theta_.size() != kThetaSize)
        {
            throw std::runtime_error("OptimizationProblem: theta has not been initialized with 36 parameters.");
        }

        const Spacetime::Hyperparameters hyper = hyperparameters_;

        Spacetime::Options options = estimator_options_;
        if (!estimator_options_set_)
        {
            warning() << "OptimizationProblem: estimator options not explicitly set. Using built-in defaults.";
        }

        options.init_guess_type = Spacetime::Options::InitialGuessType::Custom;
        options.custom_guess.estimation_nodes = nodes_;
        options.estimator_type = Spacetime::Options::EstimatorType::Batch;

        Spacetime::Estimator estimator(topology, hyper, options);
        const auto estimate = estimator.computeStateEstimate(measurement_factors, verbose);
        if (!estimate.success)
        {
            throw std::runtime_error("OptimizationProblem: Estimator::computeStateEstimate failed in solveLowerLevel().");
        }

        Eigen::SparseMatrix<double> H_est;
        std::vector<Eigen::VectorXd> factor_errors;
        std::vector<Eigen::MatrixXd> factor_jacobians;
        std::vector<Eigen::MatrixXd> factor_Qs;
        std::vector<std::vector<int>> factor_node_offsets;
        estimator.extractJacobianHessian(estimate.state, H_est, factor_errors, factor_jacobians, factor_Qs, factor_node_offsets);
        /*
        if(verbose)
        {
            std::cout << "Lower-level solution H matrix (sparse):" << std::endl;
            for (int k = 0; k < H_est.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(H_est, k); it; ++it)
                {
                    std::cout << "H(" << it.row() << ", " << it.col() << ") = " << it.value() << std::endl;
                }
            }
        }
        */
        solution.H = H_est;
        Eigen::SparseMatrix<double> identity(solution.H.rows(), solution.H.cols());
        identity.setIdentity();
        solution.H += damping * identity;

        solution.linearizations.clear();
        solution.linearizations.reserve(factor_errors.size());
        for (size_t i = 0; i < factor_errors.size(); ++i)
        {
            std::vector<int> node_indices;
            node_indices.reserve(factor_node_offsets[i].size());
            for (int offset : factor_node_offsets[i])
            {
                if (offset % node_dim != 0)
                {
                    throw std::runtime_error("OptimizationProblem: estimator linearization index is not aligned to node dimension.");
                }
                node_indices.push_back(offset / node_dim);
            }

            solution.linearizations.push_back(LinearizationData{
                factor_errors[i],
                factor_jacobians[i],
                factor_Qs[i],
                node_indices
            });
        }

        solution.x = Eigen::VectorXd::Zero(static_cast<int>(estimate.state.estimation_nodes.size()) * node_dim);
        const size_t perturbation_nodes = std::min(nodes_.size(), estimate.state.estimation_nodes.size());
        for (size_t i = 0; i < perturbation_nodes; ++i)
        {
            Eigen::Vector<double, node_dim> perturb;
            Node reference = nodes_[i];
            getPerturbation(estimate.state.estimation_nodes[i], reference, perturb);
            solution.x.segment<node_dim>(static_cast<int>(i) * node_dim) = perturb;
        }

        robot_topology_ = topology;
        robot_topology_set_ = true;
        solved_nodes_ = estimate.state.estimation_nodes;
        return solution;
    }

    void OptimizationProblem::setValidationTargets(const std::vector<ValidationTarget> &targets)
    {
        validation_targets_ = targets;
    }

    ValidationLossData OptimizationProblem::computeValidationLoss(const LowerLevelSolution &solution) const
    {
        //Compute Mean NEES and the loss function (= 0.5 * (mean_nees - 18)^2) based on the current solution and the validation targets.
        ValidationLossData out;

        if (validation_targets_.empty() || solved_nodes_.empty())
        {
            return out;
        }

        constexpr int node_dim = 18;
        const int validation_count = static_cast<int>(validation_targets_.size());
        out.nees_per_node = Eigen::VectorXd::Zero(validation_count);
        out.node_errors_stacked = Eigen::VectorXd::Zero(node_dim * validation_count);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> h_solver;
        h_solver.compute(solution.H);
        if (h_solver.info() != Eigen::Success)
        {
            throw std::runtime_error("OptimizationProblem::computeValidationLoss: failed to factorize H for node marginals.");
        }

        for (int i = 0; i < validation_count; ++i)
        {
            const auto &target = validation_targets_[static_cast<size_t>(i)];
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes_.size()))
            {
                throw std::runtime_error("OptimizationProblem::computeValidationLoss: validation node index out of range.");
            }

            const auto &estimate = solved_nodes_[static_cast<size_t>(target.node_index)];
            const Eigen::Matrix<double, 18, 1> e = computeNodeValidationError(estimate, target.ground_truth);
            out.node_errors_stacked.segment<18>(i * node_dim) = e;

            const Eigen::Matrix<double, 18, 18> P_inv_block =
                computeNodeMarginalInformationBlock(h_solver, solution.H.rows(), target.node_index);
            out.nees_per_node(i) = e.transpose() * P_inv_block * e;
        }

        out.mean_nees = out.nees_per_node.mean();
        out.delta = out.mean_nees - 18; 
        out.loss = 0.5 * out.delta * out.delta;
        return out;
    }

    Eigen::VectorXd OptimizationProblem::computeAdjointRhs(const LowerLevelSolution &solution,
                                                           const ValidationLossData &validation) const
    {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(solution.H.rows());
        if (validation_targets_.empty())
        {
            return rhs; // No validation targets, return zero vector
        }

        constexpr int node_dim = 18;
        const int validation_count = static_cast<int>(validation_targets_.size());
        if (validation.node_errors_stacked.size() != validation_count * node_dim)
        {
            throw std::runtime_error("OptimizationProblem::computeAdjointRhs: validation error vector has unexpected size.");
        }

        const double scale = 2.0 * validation.delta / static_cast<double>(validation_count);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> h_solver;
        h_solver.compute(solution.H);
        if (h_solver.info() != Eigen::Success)
        {
            throw std::runtime_error("OptimizationProblem::computeAdjointRhs: failed to factorize H for node marginals.");
        }

        for (int i = 0; i < validation_count; ++i)
        {
            const auto &target = validation_targets_[static_cast<size_t>(i)];
            if (target.node_index < 0 || (target.node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("OptimizationProblem::computeAdjointRhs: validation node index out of range.");
            }

            const Eigen::Matrix<double, 18, 1> e = validation.node_errors_stacked.segment<18>(i * node_dim);
            const Eigen::Matrix<double, 18, 18> P_inv_block =
                computeNodeMarginalInformationBlock(h_solver, solution.H.rows(), target.node_index);

            Eigen::Matrix<double, 18, 18> J = Eigen::Matrix<double, 18, 18>::Identity();
            J.block<6, 6>(0, 0) = se3::vec2jacinv(e.segment<6>(0));

            rhs.segment<18>(target.node_index * node_dim) += J.transpose() * P_inv_block * e;
        }

        rhs *= scale;
        return rhs;
    }

    Eigen::VectorXd OptimizationProblem::computeValidationErrors() const
    {
        if (validation_targets_.empty() || solved_nodes_.empty())
        {
            return Eigen::VectorXd();
        }

        constexpr int node_dim = 18;
        Eigen::VectorXd errors = Eigen::VectorXd::Zero(static_cast<int>(validation_targets_.size()) * node_dim);
        for (size_t i = 0; i < validation_targets_.size(); ++i)
        {
            const auto &target = validation_targets_[i];
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes_.size()))
            {
                throw std::runtime_error("OptimizationProblem::computeValidationErrors: validation node index out of range.");
            }
            errors.segment<18>(static_cast<int>(i) * node_dim) =
                computeNodeValidationError(solved_nodes_[static_cast<size_t>(target.node_index)], target.ground_truth);
        }

        return errors;
    }

    const std::vector<std::shared_ptr<Spacetime::Factors::Factor>>& OptimizationProblem::factors() const
    {
        return factors_;
    }

    const Eigen::VectorXd& OptimizationProblem::theta() const
    {
        return theta_;
    }

    const std::vector<OptimizationProblem::Node>& OptimizationProblem::solvedNodes() const
    {
        return solved_nodes_;
    }

    const std::vector<ValidationTarget>& OptimizationProblem::validationTargets() const
    {
        return validation_targets_;
    }

    const Spacetime::RobotTopology& OptimizationProblem::robotTopology() const
    {
        return robot_topology_;
    }

    Optimizer::Optimizer(const OptimizationProblem& problem)
        : problem_(std::make_shared<OptimizationProblem>(problem))
    {
    }

    void Optimizer::setConfig(const OptimizerConfig& cfg)
    {
        optimizer_config = cfg;
    }

    Optimizer::Result Optimizer::optimize()
    {
        return step(Eigen::VectorXd());
    }

    Optimizer::Result Optimizer::step(const Eigen::VectorXd& phi) //Main loop
    {
        Result result;
        if (!problem_)
        {
            throw std::runtime_error("Optimizer::step requires a valid OptimizationProblem.");
        }

        const auto &validation_targets = problem_->validationTargets();
        if (validation_targets.empty())
        {
            throw std::runtime_error("Optimizer::step requires validation targets for NEES-based outer optimization.");
        }

        Eigen::VectorXd current_phi;
        if (phi.size() == 0)
        {
            const Eigen::VectorXd &initial_theta = problem_->theta();
            if (initial_theta.size() != kThetaSize)
            {
                throw std::runtime_error("Optimizer::step could not infer an initial 36-vector theta from the problem.");
            }
            current_phi = optimizer_config.use_exponential_param ? initial_theta.array().log().matrix() : initial_theta;
        }
        else
        {
            current_phi = phi;
        }

        //theta = exp(phi) if use_exponential_param is true, otherwise theta = phi
        auto mapTheta = [&](const Eigen::VectorXd &unconstrained) -> Eigen::VectorXd {
            if (optimizer_config.use_exponential_param)
            {
                return unconstrained.array().exp().matrix();
            }
            return unconstrained;
        };

        double previous_loss = std::numeric_limits<double>::infinity();
        const int validation_count = static_cast<int>(validation_targets.size());
        const double validation_scale = -1.0 / static_cast<double>(validation_count); //Scale factor for the gradient of the loss function with respect to the validation errors
        Eigen::VectorXd adam_m = Eigen::VectorXd::Zero(current_phi.size());
        Eigen::VectorXd adam_v = Eigen::VectorXd::Zero(current_phi.size());
        double beta1_power = 1.0;
        double beta2_power = 1.0;

        //Main optimization loop
        for (int iter = 0; iter < optimizer_config.max_iterations; ++iter)
        {
            const Eigen::VectorXd theta = mapTheta(current_phi);
            problem_->setTheta(theta);

            const LowerLevelSolution solution = problem_->solveLowerLevel(optimizer_config.verbose);
            if (solution.H.rows() == 0)
            {
                throw std::runtime_error("Optimizer::step received an empty lower-level solution.");
            }

            const auto &solved_nodes = problem_->solvedNodes();
            const auto &linearizations = solution.linearizations;
            const auto &topology = problem_->robotTopology();

            if (solved_nodes.empty() || topology.N == 0 || topology.K == 0)
            {
                throw std::runtime_error("Optimizer::step requires a solved topology with valid N and K.");
            }

            const MarginalCacheData marginal_cache = buildMarginalCache(solution.H, validation_targets);
            const ValidationLossData validation = computeValidationLossWithMarginals(validation_targets, solved_nodes, marginal_cache);
            const Eigen::VectorXd adjoint_rhs = computeAdjointRhsWithMarginals(solution.H, validation_targets, validation, marginal_cache);
            for (size_t i=0; i<theta.size(); ++i){
                std::cout << "Theta[" << i << "] = " << theta[i] << std::endl;
            }
            std::cout << "Validation mean NEES = " << validation.mean_nees << std::endl;
            std::cout << "Validation loss = " << validation.loss << std::endl;
            if(optimizer_config.verbose){
                for (int i = 0; i < validation_count; ++i)
                {
                    const int node_index = validation_targets[static_cast<std::size_t>(i)].node_index;
                    const Eigen::VectorXd e = validation.node_errors_stacked.segment<18>(i * 18);
                    const Matrix18d &p_inv_block = getCachedNodeInformationBlock(marginal_cache, node_index);

                    const double pose_norm = e.segment<6>(0).norm();
                    const double strain_norm = e.segment<6>(6).norm();
                    const double varpi_norm = e.segment<6>(12).norm();
                    const double residual_norm = e.norm();
                    const double info_trace = p_inv_block.trace();

                    std::cout << "Validation target " << i
                              << " (node=" << node_index << ")"
                              << ": NEES=" << validation.nees_per_node(i)
                              << ", ||e||=" << residual_norm
                              << ", ||e_pose||=" << pose_norm
                              << ", ||e_strain||=" << strain_norm
                              << ", ||e_varpi||=" << varpi_norm
                              << ", trace(P_inv)=" << info_trace
                              << std::endl;
                }

                std::cout << "Marginal cache profile: nodes=" << marginal_cache.node_information_blocks.size()
                          << ", factorization_ms=" << marginal_cache.factorization_ms
                          << ", solve_ms=" << marginal_cache.marginal_solve_ms << std::endl;
            }

            result.loss = validation.loss;
            result.iterations = iter + 1;
            result.phi = current_phi;
            result.theta = theta;

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> adjoint_solver;
            adjoint_solver.compute(solution.H);
            if (adjoint_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Optimizer::step failed to factorize the lower-level information matrix.");
            }

            const Eigen::VectorXd lambda = adjoint_solver.solve(adjoint_rhs);
            if (adjoint_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Optimizer::step failed to solve the adjoint system.");
            }

            const std::size_t unary_count = static_cast<std::size_t>(topology.K);
            const std::size_t binary_space_count = static_cast<std::size_t>(topology.K) * static_cast<std::size_t>(std::max<unsigned int>(1u, topology.N) - 1u);
            const std::size_t binary_time_count = topology.use_1D_estimator ? 0u : static_cast<std::size_t>(std::max<unsigned int>(1u, topology.K) - 1u) * static_cast<std::size_t>(topology.N);
            const std::size_t built_in_count = unary_count + binary_space_count + binary_time_count;

            if (linearizations.size() < built_in_count)
            {
                throw std::runtime_error("Optimizer::step: lower-level linearizations do not include all built-in factors.");
            }

            Eigen::VectorXd global_validation_errors = Eigen::VectorXd::Zero(static_cast<int>(solved_nodes.size()) * 18);
            for (std::size_t i = 0; i < validation_targets.size(); ++i)
            {
                const auto &target = validation_targets[i];
                const int node_index = target.node_index;
                if (node_index < 0 || node_index >= static_cast<int>(solved_nodes.size()))
                {
                    throw std::runtime_error("Optimizer::step: validation target node index out of range.");
                }
                global_validation_errors.segment<18>(node_index * 18) = validation.node_errors_stacked.segment<18>(static_cast<int>(i) * 18);
            }

            Eigen::VectorXd dL_dtheta = Eigen::VectorXd::Zero(kThetaSize);
            FactorKernelSample unary_sample;
            FactorKernelSample time_sample;
            FactorKernelSample space_sample;

            // Compute the gradient contributions from each built-in factor type
            for (std::size_t i = 0; i < built_in_count; ++i)
            {
                const auto &lin = linearizations[i];
                if (lin.node_indices.empty())
                {
                    continue;
                }

                std::vector<Spacetime::SystemState<DTYPE>::Node> local_nodes;
                local_nodes.reserve(lin.node_indices.size());
                Eigen::VectorXd lambda_f = Eigen::VectorXd::Zero(static_cast<int>(lin.node_indices.size()) * 18);
                Eigen::VectorXd e_nodes = Eigen::VectorXd::Zero(static_cast<int>(lin.node_indices.size()) * 18);

                for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                {
                    const int node_index = lin.node_indices[j];
                    if (node_index < 0 || node_index >= static_cast<int>(solved_nodes.size()))
                    {
                        throw std::runtime_error("Optimizer::step: built-in factor node index out of range.");
                    }

                    local_nodes.push_back(solved_nodes[static_cast<std::size_t>(node_index)]);
                    lambda_f.segment<18>(static_cast<int>(j) * 18) = lambda.segment<18>(node_index * 18);
                    e_nodes.segment<18>(static_cast<int>(j) * 18) = global_validation_errors.segment<18>(node_index * 18);
                }

                if (lin.node_indices.size() == 1)
                {
                    // If the unary sample is not yet available, store the first one for gradient checking
                    if (!unary_sample.available)
                    {
                        unary_sample.available = true;
                        unary_sample.local_nodes = local_nodes;
                        unary_sample.e_f = lin.e;
                        unary_sample.info_f = lin.Q;
                        unary_sample.E_f = lin.E;
                        unary_sample.lambda_f = lambda_f;
                        unary_sample.e_nodes = e_nodes;
                        unary_sample.info_scale = validation_scale * validation.delta;
                    }
                    const FactorGradientContrib contrib = computeDiagonalFactorGradient(lin.e, lin.Q, lin.E, lambda_f, e_nodes, validation_scale * validation.delta);
                    dL_dtheta.segment<18>(0) += contrib.dL_dtheta_state + contrib.dL_dtheta_info;
                    continue;
                }

                if (lin.node_indices.size() != 2)
                {
                    throw std::runtime_error("Optimizer::step: built-in factor must involve either one node (unary) or two nodes (binary).");
                }

                const auto &node0 = local_nodes[0];
                const auto &node1 = local_nodes[1];
                const bool same_time = std::abs(node1.time - node0.time) < TOLERANCE;
                const bool same_arclength = std::abs(node1.arclength - node0.arclength) < TOLERANCE;

                if (same_time && !same_arclength)
                {
                    // If the space sample is not yet available, store the first one for gradient checking
                    if (!space_sample.available)
                    {
                        space_sample.available = true;
                        space_sample.local_nodes = local_nodes;
                        space_sample.e_f = lin.e;
                        space_sample.info_f = lin.Q;
                        space_sample.E_f = lin.E;
                        space_sample.lambda_f = lambda_f;
                        space_sample.e_nodes = e_nodes;
                        space_sample.info_scale = validation_scale * validation.delta;
                    }
                    const FactorGradientContrib contrib = computeBinarySpaceFactorGradient(local_nodes, lin.e, lin.Q, lin.E, lambda_f, e_nodes, validation_scale * validation.delta);
                    dL_dtheta.segment<6>(24) += contrib.dL_dtheta_state.segment<6>(0) + contrib.dL_dtheta_info.segment<6>(0); // Q2
                    dL_dtheta.segment<6>(30) += contrib.dL_dtheta_state.segment<6>(6) + contrib.dL_dtheta_info.segment<6>(6); // Q3
                    continue;
                }

                if (!same_time && same_arclength)
                {
                    if (!time_sample.available)
                    {
                        time_sample.available = true;
                        time_sample.local_nodes = local_nodes;
                        time_sample.e_f = lin.e;
                        time_sample.info_f = lin.Q;
                        time_sample.E_f = lin.E;
                        time_sample.lambda_f = lambda_f;
                        time_sample.e_nodes = e_nodes;
                        time_sample.info_scale = validation_scale * validation.delta;
                    }

                    const FactorGradientContrib contrib = computeBinaryTimeFactorGradient(local_nodes, lin.e, lin.Q, lin.E, lambda_f, e_nodes, validation_scale * validation.delta);
                    dL_dtheta.segment<6>(18) += contrib.dL_dtheta_state.segment<6>(0) + contrib.dL_dtheta_info.segment<6>(0); // Q1
                    dL_dtheta.segment<6>(30) += contrib.dL_dtheta_state.segment<6>(6) + contrib.dL_dtheta_info.segment<6>(6); // Q3
                    continue;
                }

                throw std::runtime_error("Optimizer::step: could not classify built-in binary factor as space or time from node coordinates.");
            }

            if (optimizer_config.enable_gradient_fd_check && iter == 0)
            {
                runKernelFiniteDifferenceChecks(
                    theta,
                    unary_sample,
                    time_sample,
                    space_sample,
                    optimizer_config.gradient_fd_epsilon);
            }

            Eigen::VectorXd dL_dphi = dL_dtheta;
            if (optimizer_config.use_exponential_param)
            {
                dL_dphi = dL_dtheta.array() * theta.array();
            }

            const double grad_norm = dL_dphi.norm();
            if (optimizer_config.verbose)
            {
                std::cout << "Outer iteration " << iter
                          << ": loss=" << validation.loss
                          << ", delta=" << validation.delta
                          << ", grad_norm=" << grad_norm
                          << ", theta_min=" << theta.minCoeff()
                          << ", theta_max=" << theta.maxCoeff()
                          << std::endl;
                for (size_t i = 0; i < dL_dphi.size(); ++i)
                {
                    std::cout << "dL/dphi[" << i << "] = " << dL_dphi[i] << ", dL/dtheta[" << i << "] = " << dL_dtheta[i] << std::endl; 
                }
            }

            if (grad_norm < optimizer_config.tol_grad || std::abs(previous_loss - validation.loss) < optimizer_config.tol_loss)
            {
                result.converged = true;
                break;
            }

            if (optimizer_config.use_adam)
            {
                adam_m = optimizer_config.adam_beta1 * adam_m + (1.0 - optimizer_config.adam_beta1) * dL_dphi;
                adam_v = optimizer_config.adam_beta2 * adam_v + (1.0 - optimizer_config.adam_beta2) * dL_dphi.array().square().matrix();
                beta1_power *= optimizer_config.adam_beta1;
                beta2_power *= optimizer_config.adam_beta2;

                const double inv_one_minus_beta1_power = 1.0 / (1.0 - beta1_power);
                const double inv_one_minus_beta2_power = 1.0 / (1.0 - beta2_power);
                const Eigen::VectorXd m_hat = adam_m * inv_one_minus_beta1_power;
                const Eigen::VectorXd v_hat = adam_v * inv_one_minus_beta2_power;
                const Eigen::VectorXd adam_step = m_hat.array() / (v_hat.array().sqrt() + optimizer_config.adam_epsilon);

                current_phi = current_phi - optimizer_config.learning_rate * adam_step;
            }
            else
            {
                current_phi = current_phi - optimizer_config.learning_rate * dL_dphi;
            }
            previous_loss = validation.loss;
        }

        result.phi = current_phi;
        result.theta = mapTheta(current_phi);
        problem_->setTheta(result.theta);
        return result;
    }

    //Solve adjoint H * lambda = b using Eigen's SimplicialLDLT solver for sparse matrices.
    Eigen::VectorXd AdjointSolver::solve(const Eigen::SparseMatrix<double>& H,
                                  const Eigen::VectorXd& b) {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to factorize H in AdjointSolver.");
        }
        return solver.solve(b);
    }

}

int TestLower()
{
    std::cout << "Lower-level optimizer test" << std::endl;
    const TestProblemData data = buildMinimalTestProblemData();

    spacetime::OptimizationProblem problem;
    problem.setEstimatorOptions(data.estimator_options);
    problem.setRobotTopology(data.robot_topology);
    problem.setTheta(data.initial_theta);
    problem.setProblemData(data.nodes, data.factors, data.factor_node_indices);

    const auto result = problem.solveLowerLevel(true);
    std::cout << "Lower-level problem built from " << data.nodes.size() << " pose examples loaded from " << data.csv_path.string() << std::endl;
    std::cout << "Factors: " << data.factors.size() << std::endl;
    std::cout << "H size: " << result.H.rows() << " x " << result.H.cols() << std::endl;
    std::cout << "dx norm: " << result.x.norm() << std::endl;
    std::cout << "Linearizations: " << result.linearizations.size() << std::endl;
    for (size_t i = 0; i < result.linearizations.size(); ++i)
    {
        const auto& lin = result.linearizations[i];
        std::cout << "Factor " << i << ": e norm = " << lin.e.norm() << ", E size = " << lin.E.rows() << " x " << lin.E.cols() << ", Q size = " << lin.Q.rows() << " x " << lin.Q.cols() << std::endl;
    }
    for (size_t i=0; i < result.x.size(); ++i)
    {
        std::cout << "dx[" << i << "] = " << result.x[i] << std::endl;
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        const std::string mode = argv[1];
        if (mode == "--lower")
        {
            return TestLower();
        }
    }

    std::cout << "Upper-level optimizer test" << std::endl;

    const TestProblemData data = buildMinimalTestProblemData();

    const std::filesystem::path bilevel_config_path = resolveExistingPath({
        "assets/config/bilevel_optimizer/bilevel_config.json",
        "../assets/config/bilevel_optimizer/bilevel_config.json",
        "../../assets/config/bilevel_optimizer/bilevel_config.json"
    });
    const Json::Value bilevel_config = loadJsonFile(bilevel_config_path);

    spacetime::OptimizationProblem problem;
    problem.setEstimatorOptions(data.estimator_options);
    problem.setRobotTopology(data.robot_topology);
    problem.setTheta(data.initial_theta);
    problem.setProblemData(data.nodes, data.factors, data.factor_node_indices);
    problem.setValidationTargets(data.validation_targets);

    spacetime::Optimizer optimizer(problem);
    spacetime::Optimizer::OptimizerConfig config;
    loadBilevelOptimizerConfigFromJson(bilevel_config, config);
    optimizer.setConfig(config);

    const spacetime::Optimizer::Result result = optimizer.optimize();
    std::cout << "Upper-level optimization finished" << std::endl;
    std::cout << "Converged: " << (result.converged ? "true" : "false") << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final loss: " << result.loss << std::endl;
    std::cout << "Theta min/max: " << result.theta.minCoeff() << " / " << result.theta.maxCoeff() << std::endl;

    return 0;
}