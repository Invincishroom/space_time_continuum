#include "spacetime/bilevel_optimizer.hpp"
#include "spacetime/bilevel_optimizer/OscillationDetector.hpp"
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
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include<iostream>
#include <iomanip>
#include<random>
#include<algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include<json/json.h>
#include <unistd.h>

namespace // anonymous
{
    using Matrix18d = Eigen::Matrix<double, 18, 18>;
    using Vector18d = Eigen::Matrix<double, 18, 1>;

    struct TemporalValidationData
    {
        Spacetime::SystemState<DTYPE>::Node estimate;
        std::vector<int> support;
        Eigen::MatrixXd pose_jacobian;
        Eigen::Matrix<double, 6, 6> bridge_map = Eigen::Matrix<double, 6, 6>::Zero();
        double duration = 0.0;
        double offset = 0.0;
    };

    TemporalValidationData interpolateValidationTime(
        const spacetime::ValidationTarget& target,
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& nodes)
    {
        TemporalValidationData out;
        const auto& snapped = nodes.at(static_cast<std::size_t>(target.node_index));
        out.estimate = snapped;
        out.support = {target.node_index};
        out.pose_jacobian = Eigen::MatrixXd::Zero(6, 18);
        out.pose_jacobian.leftCols<6>().setIdentity();
        if (!target.interpolate_time)
        {
            return out;
        }
        if (target.observed_mask.tail<12>().any())
        {
            throw std::runtime_error("Temporal GT validation supports pose observations only.");
        }

        int left = -1;
        int right = -1;
        const double time = target.ground_truth.time;
        for (int index = 0; index < static_cast<int>(nodes.size()); ++index)
        {
            const auto& node = nodes[static_cast<std::size_t>(index)];
            if (std::abs(node.arclength - snapped.arclength) > TOLERANCE)
            {
                continue;
            }
            if (std::abs(node.time - time) < 1e-10)
            {
                out.estimate = node;
                out.support = {index};
                return out;
            }
            if (node.time < time && (left < 0 || node.time > nodes[left].time))
            {
                left = index;
            }
            if (node.time > time && (right < 0 || node.time < nodes[right].time))
            {
                right = index;
            }
        }
        if (left < 0 || right < 0)
        {
            throw std::runtime_error("Vicon validation timestamp is outside the state grid.");
        }

        const auto& node0 = nodes[left];
        const auto& node1 = nodes[right];
        out.duration = node1.time - node0.time;
        out.offset = time - node0.time;
        const double alpha = out.offset / out.duration;
        const double pose_weight = alpha * alpha * (3.0 - 2.0 * alpha);
        const double velocity0_weight = out.duration * alpha * (1.0 - alpha) * (1.0 - alpha);
        const double velocity1_weight = out.duration * alpha * alpha * (alpha - 1.0);
        const Eigen::Matrix4d relative = (node1.pose * invertTransformation(node0.pose)).template cast<double>();
        const Eigen::Matrix<double, 6, 1> xi = se3::tran2vec(relative);
        const Eigen::Matrix<double, 6, 6> inverse_jacobian = se3::vec2jacinv(xi);
        const Eigen::Matrix<double, 6, 1> velocity0 = node0.varpi.template cast<double>();
        const Eigen::Matrix<double, 6, 1> velocity1 = node1.varpi.template cast<double>();
        const Eigen::Matrix<double, 6, 1> query_xi =
            pose_weight * xi + velocity0_weight * velocity0 + velocity1_weight * inverse_jacobian * velocity1;
        out.bridge_map = se3::vec2jac(query_xi);
        const Eigen::Matrix4d increment = se3::vec2tran(query_xi);
        out.estimate.pose = (increment * node0.pose.template cast<double>()).template cast<DTYPE>();
        out.estimate.epsilon = (out.bridge_map *
            ((1.0 - alpha) * node0.epsilon.template cast<double>() +
             alpha * inverse_jacobian * node1.epsilon.template cast<double>())).template cast<DTYPE>();
        out.estimate.varpi = (out.bridge_map *
            (6.0 * alpha * (1.0 - alpha) / out.duration * xi +
             (3.0 * alpha * alpha - 4.0 * alpha + 1.0) * velocity0 +
             (3.0 * alpha * alpha - 2.0 * alpha) * inverse_jacobian * velocity1)).template cast<DTYPE>();
        out.estimate.time = time;
        out.support = {left, right};

        Eigen::Matrix<double, 6, 6> velocity_transport_derivative;
        const double step = 1e-5;
        for (int dimension = 0; dimension < 6; ++dimension)
        {
            Eigen::Matrix<double, 6, 1> plus = xi;
            Eigen::Matrix<double, 6, 1> minus = xi;
            plus(dimension) += step;
            minus(dimension) -= step;
            velocity_transport_derivative.col(dimension) =
                (se3::vec2jacinv(plus) * velocity1 - se3::vec2jacinv(minus) * velocity1) / (2.0 * step);
        }
        const Eigen::Matrix<double, 6, 6> pose_derivative =
            out.bridge_map * (pose_weight * Eigen::Matrix<double, 6, 6>::Identity() +
                              velocity1_weight * velocity_transport_derivative) * inverse_jacobian;
        out.pose_jacobian = Eigen::MatrixXd::Zero(6, 36);
        out.pose_jacobian.block<6, 6>(0, 0) = se3::tranAd(increment) - pose_derivative * se3::tranAd(relative);
        out.pose_jacobian.block<6, 6>(0, 12) = velocity0_weight * out.bridge_map;
        out.pose_jacobian.block<6, 6>(0, 18) = pose_derivative;
        out.pose_jacobian.block<6, 6>(0, 30) = velocity1_weight * out.bridge_map * inverse_jacobian;
        return out;
    }

    struct MarginalCacheData
    {
        std::unordered_map<int, Matrix18d> node_covariance_blocks;
        std::unordered_map<int, Matrix18d> node_information_blocks;
        double factorization_ms = 0.0;
        double marginal_solve_ms = 0.0;
    };

    struct JointValidationCacheData
    {
        bool full_state_observation = false;
        bool joint_observations = true;
        std::vector<int> node_indices;
        std::vector<TemporalValidationData> temporal_targets;
        std::vector<std::array<int, 18>> observed_columns;
        Eigen::SparseMatrix<double> observation_projection;
        Eigen::Matrix<double, 6, 6> temporal_q1 = Eigen::Matrix<double, 6, 6>::Zero();
        std::vector<Eigen::Matrix<int, 18, 1>> observed_masks;
        std::vector<int> observed_stacked_indices;
        Eigen::VectorXd node_errors_stacked;
        Eigen::MatrixXd h_inv_selector;
        Eigen::MatrixXd weighted_projection;
        Eigen::MatrixXd joint_covariance;
        Eigen::MatrixXd joint_information;
        std::vector<Eigen::MatrixXd> marginal_information;
        std::vector<int> marginal_offsets;
        Eigen::VectorXd weighted_errors;
        int effective_validation_count = 0;
        int observed_dim_total = 0;
        double logdet_sum = 0.0;
        double mahalanobis_sum = 0.0;
        double min_observed_cov_eig = 0.0;
        double max_observed_cov_eig = 0.0;
        double observed_cov_condition = 0.0;
        double setup_ms = 0.0;
        double selector_ms = 0.0;
        double projection_solve_ms = 0.0;
        double covariance_ms = 0.0;
        double eigendecomposition_ms = 0.0;
        double information_ms = 0.0;
    };

    struct JointGroupLogDetData
    {
        Eigen::VectorXd quadratic_state;
    };

    struct JointValidationGroupData
    {
        Eigen::VectorXd node_errors_stacked;
        Eigen::VectorXd weighted_errors;
        Eigen::VectorXd covariance_state_gradient;
        Eigen::VectorXd covariance_theta_gradient = Eigen::VectorXd::Zero(spacetime::OptimizationProblem::ParameterCount);
        double mahalanobis_sum = 0.0;
    };

    struct FactorKernelSample
    {
        bool available = false;
        std::vector<Spacetime::SystemState<DTYPE>::Node> local_nodes;
        Eigen::VectorXd e_f;
        Eigen::MatrixXd info_f;
        Eigen::MatrixXd E_f;
        Eigen::VectorXd adjoint_lambda_f;
        Eigen::VectorXd e_nodes;
        double info_scale = 0.0;
    };

    constexpr int kThetaSize = spacetime::OptimizationProblem::ParameterCount;
    constexpr int kNodeDim = 18;
    constexpr int kP0Offset = 0;
    constexpr int kQ1Offset = 18;
    constexpr int kQ2Offset = 24;
    constexpr int kQ3Offset = 30;
    constexpr int kQ4Offset = 36;

    using ObservedMask18 = Eigen::Matrix<int, 18, 1>;

    std::vector<int> collectObservedIndices(const ObservedMask18 &mask)
    {
        std::vector<int> indices;
        indices.reserve(kNodeDim);
        for (int i = 0; i < kNodeDim; ++i)
        {
            if (mask(i) != 0)
            {
                indices.push_back(i);
            }
        }
        return indices;
    }

    int countObservedDimensions(const ObservedMask18 &mask)
    {
        int count = 0;
        for (int i = 0; i < kNodeDim; ++i)
        {
            if (mask(i) != 0)
            {
                ++count;
            }
        }
        return count;
    }

    ObservedMask18 buildObservedMaskFromTrialDataFlags(const Json::Value &trial_data_root)
    {
        ObservedMask18 mask = ObservedMask18::Zero();
        const bool observe_pose = trial_data_root.get("aurora", false).asBool() ||
                                  trial_data_root.get("lidar", false).asBool();
        if (observe_pose)
        {
            mask.segment<6>(0).setOnes();
        }
        if (trial_data_root.get("fbgs", false).asBool())
        {
            mask.segment<6>(6).setOnes();
        }
        if (trial_data_root.get("velocity", false).asBool())
        {
            mask.segment<3>(12).setOnes();
        }
        if (trial_data_root.get("gyro", false).asBool())
        {
            mask.segment<3>(15).setOnes();
        }

        if (mask.isZero())
        {
            throw std::runtime_error("At least one observed validation dimension must be enabled in the trial data flags.");
        }
        return mask;
    }

    Vector18d applyObservedMask(const Vector18d &value, const ObservedMask18 &mask)
    {
        Vector18d masked = Vector18d::Zero();
        for (int i = 0; i < kNodeDim; ++i)
        {
            if (mask(i) != 0)
            {
                masked(i) = value(i);
            }
        }
        return masked;
    }

    std::vector<std::vector<spacetime::ValidationTarget>> groupValidationTargetsBySample(
        const std::vector<spacetime::ValidationTarget>& validation_targets)
    {
        std::vector<std::vector<spacetime::ValidationTarget>> groups;
        if (validation_targets.empty())
        {
            return groups;
        }

        std::unordered_map<int, std::size_t> group_index_by_id;
        group_index_by_id.reserve(validation_targets.size());
        for (const auto& target : validation_targets)
        {
            const int group_id = target.sample_group_id;
            auto it = group_index_by_id.find(group_id);
            if (it == group_index_by_id.end())
            {
                const std::size_t new_index = groups.size();
                group_index_by_id.emplace(group_id, new_index);
                groups.push_back({target});
            }
            else
            {
                groups[it->second].push_back(target);
            }
        }

        return groups;
    }

    Eigen::MatrixXd extractPrincipalSubmatrix(const Matrix18d &matrix, const std::vector<int> &indices)
    {
        Eigen::MatrixXd out(indices.size(), indices.size());
        for (int r = 0; r < static_cast<int>(indices.size()); ++r)
        {
            for (int c = 0; c < static_cast<int>(indices.size()); ++c)
            {
                out(r, c) = matrix(indices[static_cast<std::size_t>(r)], indices[static_cast<std::size_t>(c)]);
            }
        }
        return out;
    }

    Eigen::MatrixXd extractPrincipalSubmatrixDynamic(const Eigen::MatrixXd &matrix, const std::vector<int> &indices)
    {
        Eigen::MatrixXd out(indices.size(), indices.size());
        for (int r = 0; r < static_cast<int>(indices.size()); ++r)
        {
            for (int c = 0; c < static_cast<int>(indices.size()); ++c)
            {
                out(r, c) = matrix(indices[static_cast<std::size_t>(r)], indices[static_cast<std::size_t>(c)]);
            }
        }
        return out;
    }

    Matrix18d buildObservedInformationFromCovariance(const Matrix18d &covariance,
                                                     const ObservedMask18 &mask)
    {
        constexpr double kCovFloor = 1e-10;
        Matrix18d observed_info = Matrix18d::Zero();
        const std::vector<int> observed_indices = collectObservedIndices(mask);
        if (observed_indices.empty())
        {
            return observed_info;
        }

        Eigen::MatrixXd cov_sub = extractPrincipalSubmatrix(covariance, observed_indices);
        cov_sub = 0.5 * (cov_sub + cov_sub.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov_sub);
        if (eigensolver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to eigendecompose observed covariance sub-block.");
        }
        Eigen::VectorXd evals = eigensolver.eigenvalues().array().max(kCovFloor);
        cov_sub = eigensolver.eigenvectors() * evals.asDiagonal() * eigensolver.eigenvectors().transpose();

        const int observed_dim = static_cast<int>(observed_indices.size());
        const Eigen::MatrixXd rhs_identity = Eigen::MatrixXd::Identity(observed_dim, observed_dim);
        Eigen::LLT<Eigen::MatrixXd> llt(cov_sub);
        if (llt.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed LLT factorization for observed covariance sub-block.");
        }
        const Eigen::MatrixXd info_sub = llt.solve(rhs_identity);

        for (int r = 0; r < static_cast<int>(observed_indices.size()); ++r)
        {
            for (int c = 0; c < static_cast<int>(observed_indices.size()); ++c)
            {
                observed_info(observed_indices[static_cast<std::size_t>(r)], observed_indices[static_cast<std::size_t>(c)]) =
                    info_sub(r, c);
            }
        }
        return observed_info;
    }

    double computeObservedLogDetFromCovariance(const Matrix18d &covariance,
                                               const ObservedMask18 &mask)
    {
        constexpr double kCovFloor = 1e-12;
        const std::vector<int> observed_indices = collectObservedIndices(mask);
        if (observed_indices.empty())
        {
            return 0.0;
        }

        Eigen::MatrixXd cov_sub = extractPrincipalSubmatrix(covariance, observed_indices);
        cov_sub = 0.5 * (cov_sub + cov_sub.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov_sub);
        if (eigensolver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to eigendecompose observed covariance sub-block for log-det.");
        }

        const Eigen::VectorXd evals = eigensolver.eigenvalues().array().max(kCovFloor);
        return evals.array().log().sum();
    }

    Eigen::Matrix<double, 18, 1> computeNodeValidationError(const Spacetime::SystemState<DTYPE>::Node &estimate,
                                                            const Spacetime::SystemState<DTYPE>::Node &ground_truth);
    Matrix18d buildTimeCovariance(const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                  const Eigen::VectorXd& theta_local);
    Matrix18d buildSpaceCovariance(const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                   const Eigen::VectorXd& theta_local);

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

    std::filesystem::path resolveAssetsLogPath(const std::string& filename)
    {
        std::filesystem::path current = std::filesystem::current_path();
        while (true)
        {
            const std::filesystem::path assets_dir = current / "assets";
            if (std::filesystem::exists(assets_dir) && std::filesystem::is_directory(assets_dir))
            {
                const std::filesystem::path logs_dir = assets_dir / "logs";
                std::filesystem::create_directories(logs_dir);
                return logs_dir / filename;
            }

            if (!current.has_parent_path() || current == current.parent_path())
            {
                break;
            }

            current = current.parent_path();
        }

        throw std::runtime_error("Unable to locate assets/logs for optimizer output.");
    }

    std::string formatTimestamp()
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
#if defined(_WIN32)
        localtime_s(&local_time, &now_time);
#else
        localtime_r(&now_time, &local_time);
#endif
        std::ostringstream stream;
        stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
        return stream.str();
    }

    template <typename Derived>
    std::string formatVector(const Eigen::MatrixBase<Derived>& vector)
    {
        std::ostringstream stream;
        stream << '[';
        for (int i = 0; i < vector.size(); ++i)
        {
            if (i > 0)
            {
                stream << ',';
            }
            stream << vector(i);
        }
        stream << ']';
        return stream.str();
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
        if (root.isMember("enable_oscillation_detection"))
        {
            if (!root["enable_oscillation_detection"].isBool())
            {
                throw std::runtime_error("enable_oscillation_detection must be a boolean.");
            }
            config.enable_oscillation_detection = root["enable_oscillation_detection"].asBool();
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
        if (root.isMember("enable_unary_factor_fd_check"))
        {
            config.enable_unary_factor_fd_check = root["enable_unary_factor_fd_check"].asBool();
        }
        if (root.isMember("enable_time_factor_fd_check"))
        {
            config.enable_time_factor_fd_check = root["enable_time_factor_fd_check"].asBool();
        }
        if (root.isMember("enable_space_factor_fd_check"))
        {
            config.enable_space_factor_fd_check = root["enable_space_factor_fd_check"].asBool();
        }
        if (root.isMember("gradient_fd_epsilon"))
        {
            config.gradient_fd_epsilon = root["gradient_fd_epsilon"].asDouble();
        }
        if (root.isMember("gradient_fd_use_lambda_perturbation"))
        {
            config.gradient_fd_use_lambda_perturbation = root["gradient_fd_use_lambda_perturbation"].asBool();
        }
        if (root.isMember("freeze_p0_non_pose"))
        {
            config.freeze_p0_non_pose = root["freeze_p0_non_pose"].asBool();
        }
        if (root.isMember("lambda_update_mask"))
        {
            const Json::Value& mask = root["lambda_update_mask"];
            if (!mask.isArray() || (mask.size() != kThetaSize && mask.size() != kQ4Offset))
            {
                throw std::runtime_error("lambda_update_mask must contain 42 entries, or 36 legacy entries (Q4 frozen).");
            }
            std::vector<int> update_mask(kThetaSize);
            for (unsigned int index = 0; index < mask.size(); ++index)
            {
                const Json::Value& entry = mask[index];
                if (entry.isBool())
                {
                    update_mask[index] = entry.asBool() ? 1 : 0;
                }
                else if (entry.isInt() && (entry.asInt() == 0 || entry.asInt() == 1))
                {
                    update_mask[index] = entry.asInt();
                }
                else
                {
                    throw std::runtime_error("lambda_update_mask entries must be 0/1 or false/true.");
                }
            }
            config.lambda_update_mask = std::move(update_mask);
        }
        if (root.isMember("use_joint_validation_objective"))
        {
            config.use_joint_validation_objective = root["use_joint_validation_objective"].asBool();
        }
        if (root.isMember("log_joint_diagnostics"))
        {
            config.log_joint_diagnostics = root["log_joint_diagnostics"].asBool();
        }
        if (root.isMember("max_gradient_update_norm"))
        {
            config.max_gradient_update_norm = root["max_gradient_update_norm"].asDouble();
        }
        if (root.isMember("max_psi_step_norm"))
        {
            config.max_psi_step_norm = root["max_psi_step_norm"].asDouble();
        }
        else if (root.isMember("max_phi_step_norm"))
        {
            config.max_psi_step_norm = root["max_phi_step_norm"].asDouble();
        }
        if (root.isMember("min_theta"))
        {
            config.min_theta = root["min_theta"].asDouble();
        }
        if (root.isMember("max_theta"))
        {
            config.max_theta = root["max_theta"].asDouble();
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
                if (!values[i].isNumeric())
                    throw std::runtime_error(std::string("Initial theta weight '") + name + "' must be numeric.");
                const double value = values[i].asDouble();
                if (!std::isfinite(value) || !(value > 0.0))
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
        theta.segment<6>(kQ4Offset) = weights.isMember("Q4")
            ? readWeights("Q4", 6) : readWeights("Q3", 6);
        const Json::Value factor = weights.get("prior_factor", 1.0);
        if (!factor.isNumeric() || !std::isfinite(factor.asDouble()) || factor.asDouble() <= 0.0)
            throw std::runtime_error("prior_factor must be finite and strictly positive.");
        theta *= factor.asDouble();
        if (!theta.allFinite() || theta.minCoeff() <= 0.0)
            throw std::runtime_error("Scaled initial variances must be finite and strictly positive.");
        return theta;
    }

    Eigen::VectorXd loadSavedTheta(const std::filesystem::path &path)
    {
        const Json::Value saved = loadJsonFile(path);
        const Json::Value &values = saved["theta"];
        if (!values.isArray() || (values.size() != kThetaSize && values.size() != kQ4Offset))
            throw std::runtime_error("Saved theta must contain 42 entries, or 36 legacy entries with recorded Q4.");
        const int expected_version = values.size() == kThetaSize ? 2 : 1;
        if (saved.isMember("parameter_layout_version") &&
            (!saved["parameter_layout_version"].isInt() || saved["parameter_layout_version"].asInt() != expected_version))
            throw std::runtime_error("Saved parameter layout version does not match its theta array.");
        if (expected_version == 2 && saved.isMember("parameter_layout") &&
            saved["parameter_layout"].asString() != "P0(18),Q1(6),Q2(6),Q3(6),Q4(6)")
            throw std::runtime_error("Unsupported saved parameter layout.");
        Eigen::VectorXd theta(kThetaSize);
        for (unsigned int index = 0; index < values.size(); ++index)
        {
            if (!values[index].isNumeric())
                throw std::runtime_error("Saved theta entries must be numeric.");
            theta(index) = values[index].asDouble();
        }
        if (values.size() == kQ4Offset)
        {
            if (saved.isMember("fixed_Q4"))
            {
                const Json::Value &Q4 = saved["fixed_Q4"];
                if (!Q4.isArray() || Q4.size() != 6)
                    throw std::runtime_error("Recorded fixed_Q4 must contain six effective variances.");
                for (unsigned int index = 0; index < 6; ++index)
                {
                    if (!Q4[index].isNumeric())
                        throw std::runtime_error("Recorded fixed_Q4 entries must be numeric.");
                    theta(kQ4Offset + index) = Q4[index].asDouble();
                }
            }
            else
            {
                Json::Value robot;
                const auto robot_path = path.parent_path() / "robot_converged.json";
                const auto experiment_path = path.parent_path() / "experiment.json";
                if (std::filesystem::exists(experiment_path))
                    robot = loadJsonFile(experiment_path)["robot_config"];
                if (!robot["weights"].isMember("Q4") && std::filesystem::exists(robot_path))
                    robot = loadJsonFile(robot_path);
                if (!robot["weights"].isMember("Q4"))
                    throw std::runtime_error("Legacy theta requires recorded fixed_Q4 or a companion robot/experiment config containing Q4; current config is not used.");
                theta.tail<6>() = loadInitialThetaFromRobotConfig(robot).tail<6>();
            }
        }
        if (!theta.allFinite() || theta.minCoeff() <= 0.0)
            throw std::runtime_error("Saved theta entries must be finite and strictly positive.");
        return theta;
    }

    void loadRobotTopologyFromJson(const Json::Value &root, Spacetime::RobotTopology &topology)
    {
        topology.N = root["topology"].isMember("N") ? root["topology"]["N"].asUInt() : 1;
        if (root["topology"].isMember("K_per_second"))
        {
            topology.K = root["topology"]["K_per_second"].asUInt();
        }
        else if (root["topology"].isMember("K"))
        {
            topology.K = root["topology"]["K"].asUInt();
        }
        else
        {
            throw std::runtime_error("Robot topology must define either K or K_per_second.");
        }
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
    void fillDiagonalFromVariance(MatrixType &matrix, const Eigen::VectorXd &variance, int offset, const char *name)
    {
        const int diagonal_size = static_cast<int>(matrix.rows());
        if (matrix.rows() != matrix.cols())
        {
            throw std::runtime_error(std::string("Expected diagonal square matrix for ") + name + ".");
        }
        if (variance.size() != kThetaSize)
        {
            throw std::runtime_error("Variance must contain exactly 42 diagonal parameters for P0, Q1, Q2, Q3, and Q4.");
        }
        if (offset + diagonal_size > variance.size())
        {
            throw std::runtime_error(std::string("Variance layout is inconsistent while assembling ") + name + ".");
        }

        matrix.setIdentity();
        for (int i = 0; i < diagonal_size; ++i)
        {
            const double value = variance(offset + i);
            if (!std::isfinite(value) || !(value > 0.0))
            {
                throw std::runtime_error(std::string(name) + " must be strictly positive on the diagonal.");
            }
            matrix(i, i) = value;
        }
    }

    Spacetime::Hyperparameters buildHyperparametersFromVariance(const Eigen::VectorXd &variance)
    {
        Spacetime::Hyperparameters hyperparameters;
        fillDiagonalFromVariance(hyperparameters.P0, variance, kP0Offset, "P0");
        fillDiagonalFromVariance(hyperparameters.Q1, variance, kQ1Offset, "Q1");
        fillDiagonalFromVariance(hyperparameters.Q2, variance, kQ2Offset, "Q2");
        fillDiagonalFromVariance(hyperparameters.Q3, variance, kQ3Offset, "Q3");
        fillDiagonalFromVariance(hyperparameters.Q4, variance, kQ4Offset, "Q4");
        return hyperparameters;
    }

    Eigen::Matrix<double, 18, 1> empiricalValidationVarianceFromSamples(
        const std::vector<Spacetime::SystemState<DTYPE>::Node> &mean_nodes,
        const std::vector<Spacetime::SystemState<DTYPE>> &samples)
    {
        Eigen::Matrix<double, 18, 1> sum = Eigen::Matrix<double, 18, 1>::Zero();
        Eigen::Matrix<double, 18, 1> sum_sq = Eigen::Matrix<double, 18, 1>::Zero();
        std::size_t count = 0;

        if (samples.empty() || mean_nodes.empty())
        {
            return Eigen::Matrix<double, 18, 1>::Zero();
        }

        for (const auto &sample : samples)
        {
            if (sample.estimation_nodes.size() != mean_nodes.size())
            {
                throw std::runtime_error("Sample node count mismatch while computing empirical validation variance.");
            }

            for (std::size_t node_index = 0; node_index < mean_nodes.size(); ++node_index)
            {
                const Eigen::Matrix<double, 18, 1> e = computeNodeValidationError(
                    mean_nodes[node_index],
                    sample.estimation_nodes[node_index]);
                sum += e;
                sum_sq += e.array().square().matrix();
                ++count;
            }
        }

        if (count == 0)
        {
            return Eigen::Matrix<double, 18, 1>::Zero();
        }

        const double denom = static_cast<double>(count);
        const Eigen::Matrix<double, 18, 1> mean = sum / denom;
        return (sum_sq / denom - mean.array().square().matrix()).cwiseMax(0.0);
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

    //u_f = Q_f^{-1} e_f adjoint_lambda_f;
    //v_f = Q_f^{-1} e_f;
    //w_f = Q_f^{-1} E_f r_{quad,f};
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
                                                                    const Eigen::VectorXd& adjoint_lambda_f,
                                                                    const Eigen::VectorXd& e_nodes,
                                                                    double info_scale)
    {
        const Eigen::VectorXd v = applyFactorInformation(info_f, e_f);
        const Eigen::VectorXd u = applyFactorInformation(info_f, E_f * adjoint_lambda_f);
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
                                                                      const Eigen::VectorXd& adjoint_lambda_f,
                                                                      const Eigen::VectorXd& e_nodes,
                                                                      double info_scale)
    {
        if (local_nodes.size() != 2)
        {
            throw std::runtime_error("Binary time factor gradient expects two local nodes.");
        }

        const Eigen::VectorXd v = applyFactorInformation(info_f, e_f);
        const Eigen::VectorXd u = applyFactorInformation(info_f, E_f * adjoint_lambda_f);
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

        contrib.dL_dtheta_state.segment<6>(0) = -(
            (dt3 * (u1.array() * v1.array())).matrix() +
            (dt2 * ((u1.array() * v3.array()).matrix() + (u3.array() * v1.array()).matrix())) +
            (dt * (u3.array() * v3.array()).matrix()));
        contrib.dL_dtheta_state.segment<6>(6) = -dt * (u2.array() * v2.array()).matrix();

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
                                                                       const Eigen::VectorXd& adjoint_lambda_f,
                                                                       const Eigen::VectorXd& e_nodes,
                                                                       double info_scale)
    {
        if (local_nodes.size() != 2)
        {
            throw std::runtime_error("Binary space factor gradient expects two local nodes.");
        }

        const Eigen::VectorXd v = applyFactorInformation(info_f, e_f);
        const Eigen::VectorXd u = applyFactorInformation(info_f, E_f * adjoint_lambda_f);
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

        spacetime::FactorGradientContrib contrib;
        contrib.dL_dtheta_state = Eigen::VectorXd::Zero(12);
        contrib.dL_dtheta_info = Eigen::VectorXd::Zero(12);

        contrib.dL_dtheta_state.segment<6>(0) = -(
            (ds3 * (u1.array() * v1.array())).matrix() +
            (ds2 * ((u1.array() * v2.array()).matrix() + (u2.array() * v1.array()).matrix())) +
            (ds * (u2.array() * v2.array())).matrix());
        contrib.dL_dtheta_state.segment<6>(6) = -ds * (u3.array() * v3.array()).matrix();

        contrib.dL_dtheta_info.segment<6>(0) = info_scale *
            ((ds3 * (w1.array().square()).matrix()) +
             (2.0 * ds2 * (w1.array() * w2.array()).matrix()) +
             (ds * (w2.array().square()).matrix()));
        contrib.dL_dtheta_info.segment<6>(6) = info_scale * (ds * (w3.array().square()).matrix());

        return contrib;
    }

    Eigen::VectorXd computeUnaryLogDetGradientFromTrace(const Matrix18d& A)
    {
        Eigen::VectorXd contrib = Eigen::VectorXd::Zero(18);
        for (int i = 0; i < 18; ++i)
        {
            contrib(i) = 0.5 * A(i, i);
        }
        return contrib;
    }

    Eigen::VectorXd computeBinaryTimeLogDetGradientFromTrace(
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
        const Matrix18d& A)
    {
        if (local_nodes.size() != 2)
        {
            throw std::runtime_error("Binary time log-det gradient expects two local nodes.");
        }

        Eigen::VectorXd contrib = Eigen::VectorXd::Zero(12);
        for (int p = 0; p < 12; ++p)
        {
            Eigen::VectorXd basis = Eigen::VectorXd::Zero(12);
            basis(p) = 1.0;
            const Matrix18d dQ = buildTimeCovariance(local_nodes, basis);
            contrib(p) = 0.5 * (A * dQ).trace();
        }
        return contrib;
    }

    Eigen::VectorXd computeBinarySpaceLogDetGradientFromTrace(
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
        const Matrix18d& A)
    {
        if (local_nodes.size() != 2)
        {
            throw std::runtime_error("Binary space log-det gradient expects two local nodes.");
        }

        Eigen::VectorXd contrib = Eigen::VectorXd::Zero(12);
        for (int p = 0; p < 12; ++p)
        {
            Eigen::VectorXd basis = Eigen::VectorXd::Zero(12);
            basis(p) = 1.0;
            const Matrix18d dQ = buildSpaceCovariance(local_nodes, basis);
            contrib(p) = 0.5 * (A * dQ).trace();
        }
        return contrib;
    }

    spacetime::FactorGradientContrib computeProposalFactorGradient(const Spacetime::Factors::Factor& factor,
                                                                    const std::vector<Spacetime::SystemState<DTYPE>::Node>& local_nodes,
                                                                    const Eigen::VectorXd& e_f,
                                                                    const Eigen::MatrixXd& Q_f,
                                                                    const Eigen::MatrixXd& E_f,
                                                                    const Eigen::VectorXd& adjoint_lambda_f,
                                                                    const Eigen::VectorXd& e_nodes,
                                                                    double info_scale)
    {
        // If this is Time factor
        if (dynamic_cast<const Spacetime::Factors::BinaryTimeFactor*>(&factor) != nullptr)
        {
            return computeBinaryTimeFactorGradient(local_nodes, e_f, Q_f, E_f, adjoint_lambda_f, e_nodes, info_scale);
        }

        // If this is Space factor
        if (dynamic_cast<const Spacetime::Factors::BinarySpaceFactor*>(&factor) != nullptr)
        {
            return computeBinarySpaceFactorGradient(local_nodes, e_f, Q_f, E_f, adjoint_lambda_f, e_nodes, info_scale);
        }

        // If this is unary factor
        if (dynamic_cast<const Spacetime::Factors::UnaryFactor*>(&factor) != nullptr)
        {
            return computeDiagonalFactorGradient(e_f, Q_f, E_f, adjoint_lambda_f, e_nodes, info_scale);
        }

        throw std::runtime_error("Unknown factor type.");
    }

    Eigen::Matrix<double, 18, 18> computeNodeMarginalCovarianceBlock(const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                                                                     int total_dim,
                                                                     int node_index)
    {
        constexpr int node_dim = 18;
        constexpr double kMarginalCovarianceFloor = 1e-8;
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

        return marginal_cov;
    }

    Eigen::Matrix<double, 18, 18> computeNodeMarginalInformationBlock(const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                                                                       int total_dim,
                                                                       int node_index)
    {
        const Eigen::Matrix<double, 18, 18> marginal_cov = computeNodeMarginalCovarianceBlock(solver, total_dim, node_index);
        const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(18, 18);
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
        cache.node_covariance_blocks.reserve(unique_nodes.size());
        cache.node_information_blocks.reserve(unique_nodes.size());
        for (const int node_index : unique_nodes)
        {
            const Matrix18d node_cov = computeNodeMarginalCovarianceBlock(solver, H.rows(), node_index);
            cache.node_covariance_blocks.emplace(node_index, node_cov);
            const Eigen::MatrixXd rhs_identity = Eigen::MatrixXd::Identity(18, 18);
            cache.node_information_blocks.emplace(
                node_index,
                solvePositiveDefiniteSystem(node_cov, rhs_identity));
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

    const Matrix18d& getCachedNodeCovarianceBlock(const MarginalCacheData& cache, int node_index)
    {
        const auto it = cache.node_covariance_blocks.find(node_index);
        if (it == cache.node_covariance_blocks.end())
        {
            throw std::runtime_error("Node covariance block was not found in marginal cache.");
        }
        return it->second;
    }

    double computeLogDetFromInformationBlock(const Matrix18d& information_block)
    {
        constexpr double kEigenvalueFloor = 1e-12;
        Eigen::SelfAdjointEigenSolver<Matrix18d> eigensolver(information_block);
        if (eigensolver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to eigendecompose node information block while computing NLL.");
        }

        const Eigen::Array<double, 18, 1> eigenvalues = eigensolver.eigenvalues().array().max(kEigenvalueFloor);
        return -eigenvalues.log().sum();
    }

    double temporalBridgeKernel(const TemporalValidationData& first, const TemporalValidationData& second)
    {
        if (first.support.size() != 2 || first.support != second.support)
        {
            return 0.0;
        }
        const double alpha = std::min(first.offset, second.offset) / first.duration;
        const double beta = std::max(first.offset, second.offset) / first.duration;
        return std::pow(first.duration, 3) * alpha * alpha * (1.0 - beta) * (1.0 - beta) *
               (3.0 * beta - alpha * (1.0 + 2.0 * beta)) / 6.0;
    }

    JointValidationCacheData buildJointValidationCache(
        const Eigen::SparseMatrix<double>& H,
        const std::vector<spacetime::ValidationTarget>& validation_targets,
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& solved_nodes,
        const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& h_solver,
        const Eigen::VectorXd* theta = nullptr,
        bool joint_observations = true)
    {
        JointValidationCacheData cache;
        cache.joint_observations = joint_observations;
        if (theta)
        {
            cache.temporal_q1 = theta->segment<6>(kQ1Offset).asDiagonal();
        }
        const auto setup_start = std::chrono::steady_clock::now();
        if (validation_targets.empty() || solved_nodes.empty())
        {
            return cache;
        }

        constexpr int node_dim = 18;
        const int validation_count = static_cast<int>(validation_targets.size());
        const int joint_dim = validation_count * node_dim;
        cache.full_state_observation = joint_observations && validation_count == static_cast<int>(solved_nodes.size()) &&
                           joint_dim == H.rows();
        cache.node_indices.reserve(validation_targets.size());
        cache.observed_masks.reserve(validation_targets.size());
        cache.node_errors_stacked = Eigen::VectorXd::Zero(joint_dim);

        for (int i = 0; i < validation_count; ++i)
        {
            const auto& target = validation_targets[static_cast<size_t>(i)];
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes.size()))
            {
                throw std::runtime_error("buildJointValidationCache: validation node index out of range.");
            }

            cache.node_indices.push_back(target.node_index);
            cache.observed_masks.push_back(target.observed_mask);

            cache.temporal_targets.push_back(interpolateValidationTime(target, solved_nodes));
            if (cache.temporal_targets.back().support.size() == 2 && !theta)
            {
                throw std::runtime_error("Temporal validation covariance requires process variances.");
            }
            const Vector18d e_full = computeNodeValidationError(cache.temporal_targets.back().estimate, target.ground_truth);
            const Vector18d e_masked = applyObservedMask(e_full, target.observed_mask);
            cache.node_errors_stacked.segment<18>(i * node_dim) = e_masked;

            const int observed_dims = countObservedDimensions(target.observed_mask);
            cache.full_state_observation = cache.full_state_observation &&
                                           target.node_index == i &&
                                           observed_dims == node_dim;
            if (observed_dims > 0)
            {
                ++cache.effective_validation_count;
                cache.observed_dim_total += observed_dims;
            }
        }

        if (cache.full_state_observation)
        {
            const Eigen::VectorXd diagonal = h_solver.vectorD();
            if (diagonal.size() != H.rows() || (diagonal.array() <= 0.0).any())
            {
                throw std::runtime_error("buildJointValidationCache: H must be positive definite for full-state log-det evaluation.");
            }
            cache.logdet_sum = -diagonal.array().log().sum();
            cache.setup_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - setup_start).count();
            return cache;
        }

        for (int i = 0; i < validation_count; ++i)
        {
            const auto& mask = cache.observed_masks[static_cast<std::size_t>(i)];
            for (int d = 0; d < node_dim; ++d)
            {
                if (mask(d) != 0)
                {
                    cache.observed_stacked_indices.push_back(i * node_dim + d);
                }
            }
        }

        const int observed_dim = static_cast<int>(cache.observed_stacked_indices.size());
        const auto selector_start = std::chrono::steady_clock::now();
        cache.setup_ms = std::chrono::duration<double, std::milli>(selector_start - setup_start).count();
        std::vector<Eigen::Triplet<double>> projection_entries;
        cache.observed_columns.resize(validation_targets.size());
        for (auto& columns : cache.observed_columns)
        {
            columns.fill(-1);
        }
        for (int column = 0; column < observed_dim; ++column)
        {
            const int stacked_index = cache.observed_stacked_indices[static_cast<std::size_t>(column)];
            const int target_index = stacked_index / node_dim;
            const int dimension = stacked_index % node_dim;
            cache.observed_columns[target_index][dimension] = column;
            const auto& temporal = cache.temporal_targets[target_index];
            if (validation_targets[target_index].interpolate_time && dimension < 6)
            {
                for (int support = 0; support < static_cast<int>(temporal.support.size()); ++support)
                {
                    for (int local = 0; local < node_dim; ++local)
                    {
                        const double value = temporal.pose_jacobian(dimension, support * node_dim + local);
                        if (value != 0.0)
                        {
                            projection_entries.emplace_back(column, temporal.support[support] * node_dim + local, value);
                        }
                    }
                }
            }
            else
            {
                projection_entries.emplace_back(column, cache.node_indices[target_index] * node_dim + dimension, 1.0);
            }
        }
        cache.observation_projection.resize(observed_dim, H.rows());
        cache.observation_projection.setFromTriplets(projection_entries.begin(), projection_entries.end());
        const Eigen::MatrixXd rhs_selector = Eigen::MatrixXd(cache.observation_projection.transpose());
        const auto projection_solve_start = std::chrono::steady_clock::now();
        cache.selector_ms = std::chrono::duration<double, std::milli>(
            projection_solve_start - selector_start).count();
        cache.h_inv_selector = h_solver.solve(rhs_selector);
        if (h_solver.info() != Eigen::Success)
        {
            throw std::runtime_error("buildJointValidationCache: failed to solve for H^{-1}E_V projection.");
        }

        const auto covariance_start = std::chrono::steady_clock::now();
        cache.projection_solve_ms = std::chrono::duration<double, std::milli>(
            covariance_start - projection_solve_start).count();
        if (!joint_observations)
        {
            cache.weighted_errors = Eigen::VectorXd::Zero(joint_dim);
            cache.weighted_projection = Eigen::MatrixXd::Zero(H.rows(), observed_dim);
            int offset = 0;
            bool have_spectrum = false;
            for (int target_index = 0; target_index < validation_count; ++target_index)
            {
                const int dimensions = countObservedDimensions(cache.observed_masks[target_index]);
                cache.marginal_offsets.push_back(offset);
                cache.marginal_information.emplace_back(dimensions, dimensions);
                if (dimensions == 0) continue;
                const auto block_start = std::chrono::steady_clock::now();
                Eigen::MatrixXd covariance = cache.observation_projection.middleRows(offset, dimensions) *
                    cache.h_inv_selector.middleCols(offset, dimensions);
                const auto& temporal = cache.temporal_targets[target_index];
                const Eigen::Matrix<double, 6, 6> bridge = temporalBridgeKernel(temporal, temporal) *
                    temporal.bridge_map * cache.temporal_q1 * temporal.bridge_map.transpose();
                for (int row = 0; row < 6; ++row)
                {
                    const int observed_row = cache.observed_columns[target_index][row];
                    if (observed_row < 0) continue;
                    for (int column = 0; column < 6; ++column)
                    {
                        const int observed_column = cache.observed_columns[target_index][column];
                        if (observed_column >= 0)
                            covariance(observed_row - offset, observed_column - offset) += bridge(row, column);
                    }
                }
                covariance = (0.5 * (covariance + covariance.transpose())).eval();
                const auto eigen_start = std::chrono::steady_clock::now();
                cache.covariance_ms += std::chrono::duration<double, std::milli>(eigen_start - block_start).count();
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);
                if (eigensolver.info() != Eigen::Success)
                    throw std::runtime_error("buildJointValidationCache: failed to eigendecompose marginal covariance.");
                const auto information_start = std::chrono::steady_clock::now();
                cache.eigendecomposition_ms += std::chrono::duration<double, std::milli>(information_start - eigen_start).count();
                const double minimum = eigensolver.eigenvalues().minCoeff();
                const double maximum = eigensolver.eigenvalues().maxCoeff();
                cache.min_observed_cov_eig = have_spectrum ? std::min(cache.min_observed_cov_eig, minimum) : minimum;
                cache.max_observed_cov_eig = have_spectrum ? std::max(cache.max_observed_cov_eig, maximum) : maximum;
                have_spectrum = true;
                cache.logdet_sum += eigensolver.eigenvalues().array().max(1e-22).log().sum();
                auto& information = cache.marginal_information.back();
                information = eigensolver.eigenvectors() *
                    eigensolver.eigenvalues().array().max(1e-20).inverse().matrix().asDiagonal() *
                    eigensolver.eigenvectors().transpose();
                if (!information.allFinite())
                    throw std::runtime_error("buildJointValidationCache: non-finite marginal information.");
                cache.weighted_projection.middleCols(offset, dimensions).noalias() =
                    cache.h_inv_selector.middleCols(offset, dimensions) * information;
                cache.information_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - information_start).count();
                offset += dimensions;
            }
            if (have_spectrum)
                cache.observed_cov_condition = cache.max_observed_cov_eig / std::max(cache.min_observed_cov_eig, 1e-22);
            return cache;
        }
        cache.joint_covariance = cache.observation_projection * cache.h_inv_selector;
        for (int first = 0; first < validation_count; ++first)
        {
            for (int second = 0; second < validation_count; ++second)
            {
                const double kernel = temporalBridgeKernel(cache.temporal_targets[first], cache.temporal_targets[second]);
                const Eigen::Matrix<double, 6, 6> bridge = kernel * cache.temporal_targets[first].bridge_map *
                    cache.temporal_q1 * cache.temporal_targets[second].bridge_map.transpose();
                for (int row = 0; row < node_dim; ++row)
                {
                    const int observed_row = cache.observed_columns[first][row];
                    if (observed_row < 0) continue;
                    for (int column = 0; column < node_dim; ++column)
                    {
                        const int observed_column = cache.observed_columns[second][column];
                        if (observed_column < 0) continue;
                        if (!joint_observations && first != second)
                        {
                            cache.joint_covariance(observed_row, observed_column) = 0.0;
                        }
                        else if (row < 6 && column < 6)
                        {
                            cache.joint_covariance(observed_row, observed_column) += bridge(row, column);
                        }
                    }
                }
            }
        }
        cache.joint_covariance = 0.5 * (cache.joint_covariance + cache.joint_covariance.transpose());

        cache.joint_information = Eigen::MatrixXd::Zero(observed_dim, observed_dim);
        cache.weighted_errors = Eigen::VectorXd::Zero(joint_dim);
        if (observed_dim == 0)
        {
            cache.covariance_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - covariance_start).count();
            return cache;
        }

        constexpr double kCovFloor = 1e-20;
        constexpr double kLogDetFloor = 1e-22;
        Eigen::MatrixXd joint_cov_observed = cache.joint_covariance;
        joint_cov_observed = 0.5 * (joint_cov_observed + joint_cov_observed.transpose());

        const auto eigendecomposition_start = std::chrono::steady_clock::now();
        cache.covariance_ms = std::chrono::duration<double, std::milli>(
            eigendecomposition_start - covariance_start).count();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(joint_cov_observed);
        if (eigensolver.info() != Eigen::Success)
        {
            throw std::runtime_error("buildJointValidationCache: failed to eigendecompose joint observed covariance.");
        }
        const auto information_start = std::chrono::steady_clock::now();
        cache.eigendecomposition_ms = std::chrono::duration<double, std::milli>(
            information_start - eigendecomposition_start).count();
        if (eigensolver.eigenvalues().size() > 0)
        {
            cache.min_observed_cov_eig = eigensolver.eigenvalues().minCoeff();
            cache.max_observed_cov_eig = eigensolver.eigenvalues().maxCoeff();
            const double denom_cond = std::max(cache.min_observed_cov_eig, kLogDetFloor);
            cache.observed_cov_condition = cache.max_observed_cov_eig / denom_cond;
        }
        const Eigen::VectorXd evals_for_logdet = eigensolver.eigenvalues().array().max(kLogDetFloor);
        cache.logdet_sum = evals_for_logdet.array().log().sum();

        const Eigen::VectorXd inverse_eigenvalues =
            eigensolver.eigenvalues().array().max(kCovFloor).inverse();
        cache.joint_information = eigensolver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
                                  eigensolver.eigenvectors().transpose();
        if (!cache.joint_information.allFinite())
        {
            throw std::runtime_error("buildJointValidationCache: non-finite joint observed information.");
        }
        cache.weighted_projection = cache.h_inv_selector * cache.joint_information;
        cache.information_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - information_start).count();
        return cache;
    }

    void accumulateTemporalCovarianceGradients(
        const std::vector<spacetime::ValidationTarget>& targets,
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& nodes,
        const JointValidationCacheData& cache,
        JointValidationGroupData& group)
    {
        if (cache.full_state_observation ||
            std::none_of(cache.temporal_targets.begin(), cache.temporal_targets.end(), [](const TemporalValidationData& temporal) {
                return temporal.support.size() == 2;
            }))
        {
            return;
        }

        Eigen::VectorXd weighted(cache.observed_stacked_indices.size());
        for (int index = 0; index < weighted.size(); ++index)
        {
            weighted(index) = group.weighted_errors(cache.observed_stacked_indices[index]);
        }
        Eigen::MatrixXd covariance_sensitivity;
        Eigen::MatrixXd projection_sensitivity;
        std::vector<Eigen::MatrixXd> marginal_sensitivities;
        if (cache.joint_observations)
        {
            covariance_sensitivity = 0.5 * (cache.joint_information - weighted * weighted.transpose());
            projection_sensitivity = 2.0 * covariance_sensitivity * cache.h_inv_selector.transpose();
        }
        else
        {
            for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index)
            {
                const int offset = cache.marginal_offsets[target_index];
                const auto& information = cache.marginal_information[target_index];
                const int dimensions = information.rows();
                const Eigen::VectorXd local_weight = weighted.segment(offset, dimensions);
                marginal_sensitivities.push_back(0.5 * (information - local_weight * local_weight.transpose()));
            }
        }
        group.covariance_state_gradient = Eigen::VectorXd::Zero(cache.observation_projection.cols());

        for (int first = 0; first < static_cast<int>(targets.size()); ++first)
        {
            const auto& temporal = cache.temporal_targets[first];
            if (temporal.support.size() != 2) continue;
            Eigen::Matrix<double, 6, 36> local_projection_sensitivity = Eigen::Matrix<double, 6, 36>::Zero();
            for (int support = 0; support < 2; ++support)
            {
                Eigen::MatrixXd marginal_projection;
                if (!cache.joint_observations)
                {
                    marginal_projection = 2.0 * marginal_sensitivities[first] * cache.h_inv_selector.block(
                        temporal.support[support] * 18, cache.marginal_offsets[first],
                        18, cache.marginal_information[first].rows()).transpose();
                }
                for (int dimension = 0; dimension < 6; ++dimension)
                {
                    const int observed = cache.observed_columns[first][dimension];
                    if (observed < 0) continue;
                    if (cache.joint_observations)
                        local_projection_sensitivity.block<1, 18>(dimension, support * 18) =
                            projection_sensitivity.block<1, 18>(observed, temporal.support[support] * 18);
                    else
                        local_projection_sensitivity.block<1, 18>(dimension, support * 18) =
                            marginal_projection.row(observed - cache.marginal_offsets[first]);
                }
            }

            Eigen::Matrix<double, 6, 6> bridge_map_sensitivity = Eigen::Matrix<double, 6, 6>::Zero();
            const int second_begin = cache.joint_observations ? 0 : first;
            const int second_end = cache.joint_observations ? static_cast<int>(targets.size()) : first + 1;
            for (int second = second_begin; second < second_end; ++second)
            {
                const double kernel = temporalBridgeKernel(temporal, cache.temporal_targets[second]);
                if (kernel == 0.0) continue;
                Eigen::Matrix<double, 6, 6> sensitivity = Eigen::Matrix<double, 6, 6>::Zero();
                for (int row = 0; row < 6; ++row)
                {
                    const int observed_row = cache.observed_columns[first][row];
                    if (observed_row < 0) continue;
                    for (int column = 0; column < 6; ++column)
                    {
                        const int observed_column = cache.observed_columns[second][column];
                        if (observed_column >= 0)
                        {
                            sensitivity(row, column) = cache.joint_observations
                                ? covariance_sensitivity(observed_row, observed_column)
                                : marginal_sensitivities[first](observed_row - cache.marginal_offsets[first],
                                                               observed_column - cache.marginal_offsets[first]);
                        }
                    }
                }
                const auto& second_map = cache.temporal_targets[second].bridge_map;
                bridge_map_sensitivity += 2.0 * kernel * sensitivity * second_map * cache.temporal_q1;
                for (int dimension = 0; dimension < 6; ++dimension)
                {
                    group.covariance_theta_gradient(kQ1Offset + dimension) +=
                        kernel * temporal.bridge_map.col(dimension).dot(sensitivity * second_map.col(dimension));
                }
            }

            std::vector<Spacetime::SystemState<DTYPE>::Node> endpoints = {
                nodes[temporal.support[0]], nodes[temporal.support[1]]};
            auto local_target = targets[first];
            local_target.node_index = 0;
            const double step = 2e-4;
            for (int support = 0; support < 2; ++support)
            {
                for (int dimension = 0; dimension < 18; ++dimension)
                {
                    if (dimension >= 6 && dimension < 12) continue;
                    auto plus = endpoints;
                    auto minus = endpoints;
                    Eigen::Vector<DTYPE, 18> perturbation = Eigen::Vector<DTYPE, 18>::Zero();
                    perturbation(dimension) = step;
                    Spacetime::perturbNode(endpoints[support], perturbation, plus[support]);
                    perturbation *= -1.0;
                    Spacetime::perturbNode(endpoints[support], perturbation, minus[support]);
                    const auto plus_projection = interpolateValidationTime(local_target, plus);
                    const auto minus_projection = interpolateValidationTime(local_target, minus);
                    const double derivative =
                        (local_projection_sensitivity.cwiseProduct(plus_projection.pose_jacobian - minus_projection.pose_jacobian).sum() +
                         bridge_map_sensitivity.cwiseProduct(plus_projection.bridge_map - minus_projection.bridge_map).sum()) / (2.0 * step);
                    group.covariance_state_gradient(temporal.support[support] * 18 + dimension) += derivative;
                }
            }
        }
    }

    JointValidationGroupData buildJointValidationGroupData(
        const Eigen::SparseMatrix<double>& H,
        const std::vector<spacetime::ValidationTarget>& validation_targets,
        const std::vector<Spacetime::SystemState<DTYPE>::Node>& solved_nodes,
        const JointValidationCacheData& shared_cache,
        bool compute_sensitivities = true)
    {
        constexpr int node_dim = 18;
        if (validation_targets.size() != shared_cache.node_indices.size() ||
            validation_targets.size() != shared_cache.observed_masks.size())
        {
            throw std::runtime_error("Joint validation groups must have identical sizes.");
        }

        JointValidationGroupData group;
        group.node_errors_stacked = Eigen::VectorXd::Zero(static_cast<int>(validation_targets.size()) * node_dim);
        for (int i = 0; i < static_cast<int>(validation_targets.size()); ++i)
        {
            const auto& target = validation_targets[static_cast<std::size_t>(i)];
            if (target.node_index != shared_cache.node_indices[static_cast<std::size_t>(i)] ||
                target.observed_mask != shared_cache.observed_masks[static_cast<std::size_t>(i)])
            {
                throw std::runtime_error("Joint validation groups must use identical node ordering and observed masks.");
            }
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes.size()))
            {
                throw std::runtime_error("Joint validation target node index out of range.");
            }

            if (target.interpolate_time &&
                std::abs(target.ground_truth.time - shared_cache.temporal_targets[i].estimate.time) > 1e-10)
            {
                throw std::runtime_error("Joint validation groups must use identical timestamps.");
            }
            const Vector18d error = computeNodeValidationError(
                shared_cache.temporal_targets[i].estimate, target.ground_truth);
            group.node_errors_stacked.segment<18>(i * node_dim) = applyObservedMask(error, target.observed_mask);
        }

        if (shared_cache.full_state_observation)
        {
            group.weighted_errors = H * group.node_errors_stacked;
        }
        else
        {
            Eigen::VectorXd observed_errors(shared_cache.observed_stacked_indices.size());
            for (int i = 0; i < observed_errors.size(); ++i)
            {
                observed_errors(i) = group.node_errors_stacked(
                    shared_cache.observed_stacked_indices[static_cast<std::size_t>(i)]);
            }
            Eigen::VectorXd observed_weighted_errors;
            if (shared_cache.joint_observations)
            {
                observed_weighted_errors = shared_cache.joint_information * observed_errors;
            }
            else
            {
                observed_weighted_errors = Eigen::VectorXd::Zero(observed_errors.size());
                for (int target_index = 0; target_index < static_cast<int>(validation_targets.size()); ++target_index)
                {
                    const int offset = shared_cache.marginal_offsets[target_index];
                    const auto& information = shared_cache.marginal_information[target_index];
                    const int dimensions = information.rows();
                    observed_weighted_errors.segment(offset, dimensions).noalias() =
                        information * observed_errors.segment(offset, dimensions);
                }
            }
            group.weighted_errors = Eigen::VectorXd::Zero(group.node_errors_stacked.size());
            for (int i = 0; i < observed_weighted_errors.size(); ++i)
            {
                group.weighted_errors(
                    shared_cache.observed_stacked_indices[static_cast<std::size_t>(i)]) = observed_weighted_errors(i);
            }
        }
        group.mahalanobis_sum = group.node_errors_stacked.dot(group.weighted_errors);
        if (compute_sensitivities)
        {
            accumulateTemporalCovarianceGradients(validation_targets, solved_nodes, shared_cache, group);
        }
        return group;
    }

    spacetime::ValidationLossData computeValidationLossWithJoint(
        const std::vector<spacetime::ValidationTarget>& validation_targets,
        const JointValidationCacheData& cache,
        const JointValidationGroupData& group)
    {
        spacetime::ValidationLossData out;
        if (validation_targets.empty())
        {
            return out;
        }

        out.nees_per_node = Eigen::VectorXd::Zero(static_cast<int>(validation_targets.size()));
        if (!cache.joint_observations)
        {
            for (int index = 0; index < out.nees_per_node.size(); ++index)
            {
                out.nees_per_node(index) = group.node_errors_stacked.segment<18>(index * 18).dot(
                    group.weighted_errors.segment<18>(index * 18));
            }
        }
        out.node_errors_stacked = group.node_errors_stacked;
        out.mahalanobis_sum = group.mahalanobis_sum;
        out.logdet_sum = cache.logdet_sum;
        const double denom = static_cast<double>(std::max(1, cache.effective_validation_count));
        out.mean_nees = out.mahalanobis_sum / denom;
        out.mean_logdet = out.logdet_sum / denom;
        const double mean_observed_dims = static_cast<double>(cache.observed_dim_total) / denom;
        out.delta = out.mean_nees - mean_observed_dims;
        out.loss = 0.5 * (out.logdet_sum + out.mahalanobis_sum);
        return out;
    }

    Eigen::VectorXd computeAdjointRhsWithJoint(const Eigen::SparseMatrix<double>& H,
                                               const std::vector<spacetime::ValidationTarget>& validation_targets,
                                               const JointValidationCacheData& cache,
                                               const JointValidationGroupData& group)
    {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(H.rows());
        if (validation_targets.empty())
        {
            return rhs;
        }

        constexpr int node_dim = 18;
        for (int i = 0; i < static_cast<int>(validation_targets.size()); ++i)
        {
            const int node_index = cache.node_indices[static_cast<size_t>(i)];
            if (node_index < 0 || (node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("computeAdjointRhsWithJoint: validation node index out of range.");
            }

            const Vector18d e = group.node_errors_stacked.segment<18>(i * node_dim);
            if (countObservedDimensions(cache.observed_masks[static_cast<size_t>(i)]) == 0)
            {
                continue;
            }

            const Vector18d weighted_error = group.weighted_errors.segment<18>(i * node_dim);
            Matrix18d J = Matrix18d::Identity();
            const Eigen::Matrix<double, 6, 1> log_argument =
                validation_targets[i].interpolate_time ? Eigen::Matrix<double, 6, 1>(-e.head<6>()) : Eigen::Matrix<double, 6, 1>(e.head<6>());
            J.block<6, 6>(0, 0) = se3::vec2jacinv(log_argument);
            const Vector18d local_gradient = J.transpose() * weighted_error;
            if (validation_targets[i].interpolate_time)
            {
                const auto& temporal = cache.temporal_targets[i];
                const Eigen::VectorXd endpoint_gradient = temporal.pose_jacobian.transpose() * local_gradient.head<6>();
                for (int support = 0; support < static_cast<int>(temporal.support.size()); ++support)
                {
                    rhs.segment<18>(temporal.support[support] * node_dim) += endpoint_gradient.segment<18>(support * node_dim);
                }
            }
            else
            {
                rhs.segment<18>(node_index * node_dim) += local_gradient;
            }
        }
        if (group.covariance_state_gradient.size() == rhs.size())
        {
            rhs -= group.covariance_state_gradient;
        }
        return rhs;
    }

    Eigen::VectorXd computeQuadraticRhsWithJoint(const Eigen::SparseMatrix<double>& H,
                                                 const std::vector<spacetime::ValidationTarget>& validation_targets,
                                                 const JointValidationCacheData& cache,
                                                 const JointValidationGroupData& group)
    {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(H.rows());
        if (validation_targets.empty())
        {
            return rhs;
        }

        if (!cache.full_state_observation)
        {
            Eigen::VectorXd observed_weight(cache.observed_stacked_indices.size());
            for (int index = 0; index < observed_weight.size(); ++index)
            {
                observed_weight(index) = group.weighted_errors(cache.observed_stacked_indices[index]);
            }
            return cache.observation_projection.transpose() * observed_weight;
        }

        constexpr int node_dim = 18;
        for (int i = 0; i < static_cast<int>(validation_targets.size()); ++i)
        {
            const int node_index = cache.node_indices[static_cast<size_t>(i)];
            if (node_index < 0 || (node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("computeQuadraticRhsWithJoint: validation node index out of range.");
            }
            rhs.segment<18>(node_index * node_dim) += group.weighted_errors.segment<18>(i * node_dim);
        }
        return rhs;
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
        double logdet_sum = 0.0;
        int effective_validation_count = 0;
        int observed_dim_total = 0;

        for (int i = 0; i < validation_count; ++i)
        {
            const auto& target = validation_targets[static_cast<size_t>(i)];
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes.size()))
            {
                throw std::runtime_error("computeValidationLossWithMarginals: validation node index out of range.");
            }

            const auto& estimate = solved_nodes[static_cast<size_t>(target.node_index)];
            const Vector18d e_full = computeNodeValidationError(estimate, target.ground_truth);
            const Vector18d e = applyObservedMask(e_full, target.observed_mask);
            out.node_errors_stacked.segment<18>(i * node_dim) = e;

            const int observed_dims = countObservedDimensions(target.observed_mask);
            if (observed_dims == 0)
            {
                out.nees_per_node(i) = 0.0;
                continue;
            }

            const Matrix18d& p_cov_block = getCachedNodeCovarianceBlock(cache, target.node_index);
            const Matrix18d p_inv_observed = buildObservedInformationFromCovariance(p_cov_block, target.observed_mask);
            out.nees_per_node(i) = e.transpose() * p_inv_observed * e;
            logdet_sum += computeObservedLogDetFromCovariance(p_cov_block, target.observed_mask);
            ++effective_validation_count;
            observed_dim_total += observed_dims;
        }

        out.mahalanobis_sum = out.nees_per_node.sum();
        out.logdet_sum = logdet_sum;
        const double denom = static_cast<double>(std::max(1, effective_validation_count));
        out.mean_logdet = logdet_sum / denom;
        out.mean_nees = out.mahalanobis_sum / denom;
        const double mean_observed_dims = static_cast<double>(observed_dim_total) / denom;
        out.delta = out.mean_nees - mean_observed_dims;
        out.loss = 0.5 * (out.logdet_sum + out.mahalanobis_sum);
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

        for (int i = 0; i < validation_count; ++i)
        {
            const auto& target = validation_targets[static_cast<size_t>(i)];
            if (target.node_index < 0 || (target.node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("computeAdjointRhsWithMarginals: validation node index out of range.");
            }

            const Vector18d e = validation.node_errors_stacked.segment<18>(i * node_dim);
            if (countObservedDimensions(target.observed_mask) == 0)
            {
                continue;
            }

            const Matrix18d& p_cov_block = getCachedNodeCovarianceBlock(cache, target.node_index);
            const Matrix18d p_inv_observed = buildObservedInformationFromCovariance(p_cov_block, target.observed_mask);

            Matrix18d J = Matrix18d::Identity();
            J.block<6, 6>(0, 0) = se3::vec2jacinv(e.segment<6>(0));
            rhs.segment<18>(target.node_index * node_dim) += J.transpose() * p_inv_observed * e;
        }
        return rhs;
    }

    Eigen::VectorXd computeQuadraticRhsWithMarginals(const Eigen::SparseMatrix<double>& H,
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
            throw std::runtime_error("computeQuadraticRhsWithMarginals: validation error vector has unexpected size.");
        }

        for (int i = 0; i < validation_count; ++i)
        {
            const auto& target = validation_targets[static_cast<size_t>(i)];
            if (target.node_index < 0 || (target.node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("computeQuadraticRhsWithMarginals: validation node index out of range.");
            }

            const Vector18d e = validation.node_errors_stacked.segment<18>(i * node_dim);
            if (countObservedDimensions(target.observed_mask) == 0)
            {
                continue;
            }

            const Matrix18d& p_cov_block = getCachedNodeCovarianceBlock(cache, target.node_index);
            const Matrix18d p_inv_observed = buildObservedInformationFromCovariance(p_cov_block, target.observed_mask);
            rhs.segment<18>(target.node_index * node_dim) += p_inv_observed * e;
        }
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
        const Eigen::Matrix<double, 6, 6> Q4 = diagonal6(theta_local.segment<6>(6));

        Matrix18d Q = Matrix18d::Zero();
        Q.block<6, 6>(0, 0) = ds3 * Q2;
        Q.block<6, 6>(0, 6) = ds2 * Q2;
        Q.block<6, 6>(6, 0) = ds2 * Q2;
        Q.block<6, 6>(6, 6) = ds * Q2;
        Q.block<6, 6>(12, 12) = ds * Q4;
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

    Eigen::VectorXd finiteDifferenceGradientViaLambda(const Eigen::VectorXd& theta0,
                                                      double epsilon,
                                                      const std::function<double(const Eigen::VectorXd&)>& objective)
    {
        if ((theta0.array() <= 0.0).any())
        {
            throw std::runtime_error("finiteDifferenceGradientViaLambda expects strictly positive theta values.");
        }

        const Eigen::VectorXd lambda0 = theta0.cwiseInverse();
        Eigen::VectorXd dL_dlambda = Eigen::VectorXd::Zero(theta0.size());

        for (int i = 0; i < lambda0.size(); ++i)
        {
            const double step = epsilon * std::max(1.0, std::abs(lambda0(i)));
            Eigen::VectorXd lambda_plus = lambda0;
            Eigen::VectorXd lambda_minus = lambda0;
            lambda_plus(i) += step;
            lambda_minus(i) = std::max(1e-12, lambda_minus(i) - step);

            const Eigen::VectorXd theta_plus = lambda_plus.cwiseInverse();
            const Eigen::VectorXd theta_minus = lambda_minus.cwiseInverse();
            dL_dlambda(i) = (objective(theta_plus) - objective(theta_minus)) / (lambda_plus(i) - lambda_minus(i));
        }

        // Convert dL/dlambda to dL/dtheta using lambda = 1/theta:
        // dL/dtheta = dL/dlambda * d(1/theta)/dtheta = -dL/dlambda * lambda^2.
        return (-dL_dlambda.array() * lambda0.array().square()).matrix();
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

        std::cout << "NLL kernel gradient FD check [" << name << "]: max_abs_error=" << max_abs_error
                  << ", max_rel_error=" << max_rel_error
                  << ", worst_index=" << worst_index << std::endl;
    }

    void runUnaryFactorFiniteDifferenceCheck(const Eigen::VectorXd& variance,
                                             const FactorKernelSample& unary_sample,
                                             double epsilon,
                                             bool perturb_lambda)
    {
        if (unary_sample.available)
        {
            const spacetime::FactorGradientContrib contrib = computeDiagonalFactorGradient(
                unary_sample.e_f,
                unary_sample.info_f,
                unary_sample.E_f,
                unary_sample.adjoint_lambda_f,
                unary_sample.e_nodes,
                unary_sample.info_scale);
            const Eigen::VectorXd analytic = contrib.dL_dtheta_state + contrib.dL_dtheta_info;

            const Eigen::VectorXd variance_unary = variance.segment<18>(kP0Offset);
            const Eigen::VectorXd a = unary_sample.E_f * unary_sample.adjoint_lambda_f;
            const Eigen::VectorXd b = unary_sample.e_f;
            const Eigen::VectorXd c = unary_sample.E_f * unary_sample.e_nodes;

            const auto objective = [&](const Eigen::VectorXd& variance_local) {
                Eigen::MatrixXd info = variance_local.cwiseInverse().asDiagonal();
                return (a.transpose() * info * b)(0, 0) - unary_sample.info_scale * (c.transpose() * info * c)(0, 0);
            };

            const Eigen::VectorXd numeric = perturb_lambda
                ? finiteDifferenceGradientViaLambda(variance_unary, epsilon, objective)
                : finiteDifferenceGradient(variance_unary, epsilon, objective);
            printGradientCheckSummary("unary", analytic, numeric);
        }
    }

    void runVarianceFiniteDifferenceChecks(const Eigen::VectorXd& variance,
                                         const FactorKernelSample& unary_sample,
                                         const FactorKernelSample& time_sample,
                                         const FactorKernelSample& space_sample,
                                         double epsilon,
                                         bool perturb_lambda,
                                         bool run_unary,
                                         bool run_time,
                                         bool run_space)
    {
        if (run_unary)
        {
            runUnaryFactorFiniteDifferenceCheck(variance, unary_sample, epsilon, perturb_lambda);
        }

        if (run_time && time_sample.available)
        {
            const spacetime::FactorGradientContrib contrib = computeBinaryTimeFactorGradient(
                time_sample.local_nodes,
                time_sample.e_f,
                time_sample.info_f,
                time_sample.E_f,
                time_sample.adjoint_lambda_f,
                time_sample.e_nodes,
                time_sample.info_scale);
            const Eigen::VectorXd analytic = contrib.dL_dtheta_state + contrib.dL_dtheta_info;

            Eigen::VectorXd variance_time(12);
            variance_time.segment<6>(0) = variance.segment<6>(kQ1Offset);
            variance_time.segment<6>(6) = variance.segment<6>(kQ3Offset);
            const Eigen::VectorXd a = time_sample.E_f * time_sample.adjoint_lambda_f;
            const Eigen::VectorXd b = time_sample.e_f;
            const Eigen::VectorXd c = time_sample.E_f * time_sample.e_nodes;

            const auto objective = [&](const Eigen::VectorXd& variance_local) {
                const Matrix18d Q = buildTimeCovariance(time_sample.local_nodes, variance_local);
                const Matrix18d info = Q.inverse();
                return (a.transpose() * info * b)(0, 0) - time_sample.info_scale * (c.transpose() * info * c)(0, 0);
            };

            const Eigen::VectorXd numeric = perturb_lambda
                ? finiteDifferenceGradientViaLambda(variance_time, epsilon, objective)
                : finiteDifferenceGradient(variance_time, epsilon, objective);
            printGradientCheckSummary("time", analytic, numeric);
        }

        if (run_space && space_sample.available)
        {
            const spacetime::FactorGradientContrib contrib = computeBinarySpaceFactorGradient(
                space_sample.local_nodes,
                space_sample.e_f,
                space_sample.info_f,
                space_sample.E_f,
                space_sample.adjoint_lambda_f,
                space_sample.e_nodes,
                space_sample.info_scale);
            const Eigen::VectorXd analytic = contrib.dL_dtheta_state + contrib.dL_dtheta_info;

            Eigen::VectorXd variance_space(12);
            variance_space.head<6>() = variance.segment<6>(kQ2Offset);
            variance_space.tail<6>() = variance.segment<6>(kQ4Offset);
            const Eigen::VectorXd a = space_sample.E_f * space_sample.adjoint_lambda_f;
            const Eigen::VectorXd b = space_sample.e_f;
            const Eigen::VectorXd c = space_sample.E_f * space_sample.e_nodes;

            const auto objective = [&](const Eigen::VectorXd& variance_local) {
                const Matrix18d Q = buildSpaceCovariance(space_sample.local_nodes, variance_local);
                const Matrix18d info = Q.inverse();
                return (a.transpose() * info * b)(0, 0) - space_sample.info_scale * (c.transpose() * info * c)(0, 0);
            };

            const Eigen::VectorXd numeric = perturb_lambda
                ? finiteDifferenceGradientViaLambda(variance_space, epsilon, objective)
                : finiteDifferenceGradient(variance_space, epsilon, objective);
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

    spacetime::PoseValidationMetrics computePoseValidationMetrics(
        const Spacetime::Estimator& estimator,
        const Spacetime::SystemState<DTYPE>& state,
        const std::vector<spacetime::ValidationTarget>& targets,
        const Spacetime::RobotTopology& topology,
        bool tip_only)
    {
        spacetime::PoseValidationMetrics metrics;
        double translation_squared_sum = 0.0;
        double rotation_squared_sum = 0.0;
        for (const auto& target : targets)
        {
            const double arclength = target.ground_truth.arclength;
            const bool at_tip = std::abs(arclength - topology.L) <= TOLERANCE;
            const bool in_body = arclength > TOLERANCE && arclength < topology.L - TOLERANCE;
            if ((tip_only && !at_tip) || (!tip_only && !in_body))
            {
                continue;
            }
            const bool has_translation = (target.observed_mask.head<3>().array() != 0).all();
            const bool has_rotation = (target.observed_mask.segment<3>(3).array() != 0).all();
            if (!has_translation && !has_rotation)
            {
                continue;
            }
            auto query_node = target.ground_truth;
            estimator.interpolateMean2D(query_node, state);
            const Eigen::Matrix4d error_transform =
                (query_node.pose * invertTransformation(target.ground_truth.pose)).template cast<double>();
            if (has_translation)
            {
                const double error = error_transform.block<3, 1>(0, 3).norm();
                metrics.translation_mae += error;
                translation_squared_sum += error * error;
                ++metrics.translation_count;
            }
            if (has_rotation)
            {
                const double error = se3::tran2vec(error_transform).tail<3>().norm();
                metrics.rotation_mae += error;
                rotation_squared_sum += error * error;
                ++metrics.rotation_count;
            }
        }
        if (metrics.translation_count > 0)
        {
            metrics.translation_mae /= static_cast<double>(metrics.translation_count);
            metrics.translation_rmse = std::sqrt(translation_squared_sum / metrics.translation_count);
        }
        if (metrics.rotation_count > 0)
        {
            metrics.rotation_mae /= static_cast<double>(metrics.rotation_count);
            metrics.rotation_rmse = std::sqrt(rotation_squared_sum / metrics.rotation_count);
        }
        return metrics;
    }

    struct TestProblemData
    {
        struct MeasurementScenario
        {
            Spacetime::RobotTopology robot_topology;
            std::vector<spacetime::OptimizationProblem::Node> nodes;
            std::vector<std::shared_ptr<Spacetime::Factors::Factor>> factors;
            std::vector<std::vector<int>> factor_node_indices;
            std::vector<spacetime::ValidationTarget> validation_targets;
        };

        Spacetime::RobotTopology robot_topology;
        Spacetime::Options estimator_options;
        Eigen::VectorXd initial_theta;
        Eigen::VectorXd perturbation_multiplier = Eigen::VectorXd::Ones(kThetaSize);
        int validation_sample_count = 0;
        Eigen::Matrix<double, 6, 1> R_pose = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 3, 1> R_gyro = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 18, 1> validation_variance = Eigen::Matrix<double, 18, 1>::Zero();
        std::vector<spacetime::OptimizationProblem::Node> nodes;
        std::vector<std::shared_ptr<Spacetime::Factors::Factor>> factors;
        std::vector<std::vector<int>> factor_node_indices;
        std::vector<spacetime::ValidationTarget> validation_targets;
        std::vector<MeasurementScenario> measurement_scenarios;
        std::filesystem::path csv_path;
    };

    TestProblemData buildMinimalTestProblemData(bool physical_data = false,
                                              const std::string& trial_override = "",
                                              double window_duration_override = -1.0,
                                              int window_count_override = -1)
    {
        TestProblemData data;

        std::string trial_config_location = "assets/config/trial/minimal.json";
        if (physical_data)
        {
            const std::filesystem::path bilevel_config_path = resolveExistingPath({
                "assets/config/bilevel_optimizer/bilevel_config.json",
                "../assets/config/bilevel_optimizer/bilevel_config.json",
                "../../assets/config/bilevel_optimizer/bilevel_config.json"
            });
            const Json::Value bilevel_config = loadJsonFile(bilevel_config_path);
            trial_config_location = bilevel_config.get(
                "physical_trial_config", "assets/config/trial/fast_contact.json").asString();
            if (trial_config_location.empty())
            {
                throw std::runtime_error("physical_trial_config must specify a trial config path.");
            }
        }
        if (!trial_override.empty())
        {
            trial_config_location = trial_override;
        }
        const std::string robot_config_name = physical_data ? "robot1.json" : "minimal.json";
        const std::filesystem::path trial_config_path = resolveExistingPath({
            trial_config_location,
            std::filesystem::path("..") / trial_config_location,
            std::filesystem::path("../..") / trial_config_location
        });
        const std::filesystem::path robot_config_path = resolveExistingPath({
            "assets/config/robot/" + robot_config_name,
            "../assets/config/robot/" + robot_config_name,
            "../../assets/config/robot/" + robot_config_name
        });
        const std::filesystem::path estimator_config_path = resolveExistingPath({
            "assets/config/estimator/batch.json",
            "../assets/config/estimator/batch.json",
            "../../assets/config/estimator/batch.json"
        });

        const Json::Value trial_config = loadJsonFile(trial_config_path);
        const Json::Value robot_config = loadJsonFile(robot_config_path);
        const Json::Value estimator_config = loadJsonFile(estimator_config_path);

        // Initialize estimator options before any branch that may construct an
        // Estimator (e.g., prior-only ground-truth sampling).
        loadEstimatorOptionsFromJson(estimator_config, data.estimator_options);

        const Eigen::VectorXd ground_truth_theta = loadInitialThetaFromRobotConfig(robot_config);
        data.initial_theta = ground_truth_theta;

        loadRobotTopologyFromJson(robot_config, data.robot_topology);
        loadTrialInitialConditionFromJson(trial_config, data.robot_topology);
        const auto [trial_start_time, trial_end_time] = loadTrialTimeWindowFromJson(trial_config);
        const Json::Value options_root = trial_config["options"];
        const Json::Value trial_data_root = trial_config["data"];
        const Json::Value perturbation_multiplier_root = robot_config["weights"]["perturbation_multiplier"];
        if (!perturbation_multiplier_root.isNull())
        {
            if (!perturbation_multiplier_root.isArray() ||
                (perturbation_multiplier_root.size() != kThetaSize && perturbation_multiplier_root.size() != kQ4Offset))
            {
                throw std::runtime_error("perturbation_multiplier must contain 42 entries, or 36 legacy entries (Q4 unperturbed).");
            }

            for (unsigned int i = 0; i < perturbation_multiplier_root.size(); ++i)
            {
                data.perturbation_multiplier(i) = perturbation_multiplier_root[i].asDouble();
                if (!std::isfinite(data.perturbation_multiplier(i)) || !(data.perturbation_multiplier(i) > 0.0))
                {
                    throw std::runtime_error("Robot config weights 'perturbation_multiplier' must contain strictly positive values.");
                }
            }
        }
        const bool prior_only_ground_truth_mode = options_root.isMember("prior_only_ground_truth_mode") ? options_root["prior_only_ground_truth_mode"].asBool() : false;
        const int validation_sample_count = options_root.isMember("validation_sample_count") ? options_root["validation_sample_count"].asInt() : 100;
        const unsigned int validation_rng_seed = options_root.isMember("validation_rng_seed") ? options_root["validation_rng_seed"].asUInt() : 7u;
        const unsigned int measurement_rng_seed = options_root.isMember("measurement_rng_seed") ? options_root["measurement_rng_seed"].asUInt() : validation_rng_seed + 1u;
        const ObservedMask18 validation_observed_mask = buildObservedMaskFromTrialDataFlags(trial_data_root);

        data.initial_theta = data.initial_theta.cwiseProduct(data.perturbation_multiplier);
        data.validation_sample_count = validation_sample_count;

        for (int i = 0; i < 6; ++i)
        {
            data.R_pose(i) = robot_config["weights"]["R_pose"][i].asDouble();
        }
        for (int i = 0; i < 3; ++i)
        {
            data.R_gyro(i) = robot_config["weights"]["R_gyro"][i].asDouble();
        }

        const std::filesystem::path repo_root = std::filesystem::absolute(trial_config_path)
                                                    .parent_path()
                                                    .parent_path()
                                                    .parent_path()
                                                    .parent_path();
        const std::filesystem::path data_dir = repo_root / trial_config["data"]["folder_path"].asString();

        data.nodes.clear();
        data.factors.clear();
        data.factor_node_indices.clear();
        data.validation_targets.clear();
        data.measurement_scenarios.clear();

        if (prior_only_ground_truth_mode)
        {
            if (validation_sample_count <= 0)
            {
                throw std::runtime_error("validation_sample_count must be positive in prior_only_ground_truth_mode.");
            }
            if (data.robot_topology.N == 0 || data.robot_topology.K == 0)
            {
                throw std::runtime_error("Robot topology N and K must be positive in prior_only_ground_truth_mode.");
            }

            // Build the configured spatial-temporal grid. Multiple independent
            // GT examples are generated by drawing prior samples over this grid.
            data.nodes.clear();
            data.robot_topology.t0 = trial_start_time;
            data.robot_topology.T = (trial_end_time > trial_start_time)
                ? (trial_end_time - trial_start_time)
                : 1.0;
            if (data.robot_topology.N > 1 && !(data.robot_topology.L > 0.0))
            {
                throw std::runtime_error("Robot topology length must be positive when N is greater than one.");
            }

            for (int k = 0; k < static_cast<int>(data.robot_topology.K); ++k)
            {
                const double time_fraction = data.robot_topology.K > 1
                    ? static_cast<double>(k) / static_cast<double>(data.robot_topology.K - 1)
                    : 0.0;
                for (int n = 0; n < static_cast<int>(data.robot_topology.N); ++n)
                {
                    const double space_fraction = data.robot_topology.N > 1
                        ? static_cast<double>(n) / static_cast<double>(data.robot_topology.N - 1)
                        : 0.0;
                    spacetime::OptimizationProblem::Node node;
                    node.pose = Eigen::Matrix<double, 4, 4>::Identity();
                    node.epsilon = Eigen::Matrix<double, 6, 1>::Zero();
                    node.varpi = Eigen::Matrix<double, 6, 1>::Zero();
                    node.arclength = space_fraction * data.robot_topology.L;
                    node.time = trial_start_time + time_fraction * data.robot_topology.T;
                    data.nodes.push_back(node);
                }
            }

            Spacetime::Options sampling_options = data.estimator_options;
            sampling_options.compute_covariances = true;

            Spacetime::Estimator sampler(
                data.robot_topology,
                buildHyperparametersFromVariance(ground_truth_theta),
                sampling_options);
            sampler.setRandomSeed(validation_rng_seed);

            Spacetime::SystemState<DTYPE> mean_state;
            mean_state.estimation_nodes = data.nodes;
            mean_state.ib = false;

            const std::vector<Spacetime::SystemState<DTYPE>> samples =
                sampler.samplePrior(mean_state, data.robot_topology, validation_sample_count);
            if (samples.empty())
            {
                throw std::runtime_error("Prior sampling did not return any samples.");
            }

            const bool simulate_pose = trial_data_root.get("aurora", false).asBool() ||
                                       trial_data_root.get("lidar", false).asBool();
            const bool simulate_gyro = trial_data_root.get("gyro", false).asBool();
            Eigen::Matrix<double, 6, 6> pose_covariance = data.R_pose.asDiagonal();
            Eigen::Matrix<double, 3, 3> gyro_covariance = data.R_gyro.asDiagonal();
            std::default_random_engine measurement_rng(measurement_rng_seed);
            std::normal_distribution<double> standard_normal(0.0, 1.0);

            data.measurement_scenarios.reserve(samples.size());
            for (std::size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx)
            {
                const auto &sampled_state = samples[sample_idx];
                if (sampled_state.estimation_nodes.size() != data.nodes.size())
                {
                    throw std::runtime_error("Prior sampling returned an unexpected number of nodes.");
                }

                TestProblemData::MeasurementScenario scenario;
                scenario.robot_topology = data.robot_topology;
                scenario.nodes = data.nodes;
                for (std::size_t node_index = 0; node_index < sampled_state.estimation_nodes.size(); ++node_index)
                {
                    const auto& ground_truth_node = sampled_state.estimation_nodes[node_index];
                    const bool is_base_node = node_index % data.robot_topology.N == 0;
                    spacetime::ValidationTarget target;
                    target.node_index = static_cast<int>(node_index);
                    target.sample_group_id = static_cast<int>(sample_idx);
                    target.ground_truth = ground_truth_node;
                    target.observed_mask = validation_observed_mask;
                    data.validation_targets.push_back(target);
                    scenario.validation_targets.push_back(target);

                    if (simulate_pose && !is_base_node)
                    {
                        Eigen::Matrix<double, 6, 1> noise;
                        for (int d = 0; d < noise.size(); ++d)
                        {
                            noise(d) = std::sqrt(data.R_pose(d)) * standard_normal(measurement_rng);
                        }

                        Spacetime::SensorMeasurement measurement;
                        measurement.type = Spacetime::SensorMeasurement::Pose;
                        measurement.mask = Eigen::Matrix<int, 6, 1>::Ones();
                        measurement.value = invertTransformation(ground_truth_node.pose) * se3::vec2tran(-noise);
                        measurement.s = ground_truth_node.arclength;
                        measurement.t = ground_truth_node.time;

                        auto factor = std::make_shared<Spacetime::Factors::PoseMeasurementFactor>(
                            pose_covariance, measurement);
                        const Eigen::VectorXd generated_error = factor->getError({ground_truth_node}).template cast<double>();
                        if (!generated_error.isApprox(noise, 1e-9))
                        {
                            throw std::runtime_error("Generated pose measurement does not match the requested tangent-space noise.");
                        }
                        scenario.factors.push_back(factor);
                        scenario.factor_node_indices.push_back({static_cast<int>(node_index)});
                    }

                    if (simulate_gyro && !is_base_node)
                    {
                        Eigen::Matrix<double, 3, 1> noise;
                        for (int d = 0; d < noise.size(); ++d)
                        {
                            noise(d) = std::sqrt(data.R_gyro(d)) * standard_normal(measurement_rng);
                        }

                        Spacetime::SensorMeasurement measurement;
                        measurement.type = Spacetime::SensorMeasurement::Gyro;
                        measurement.mask = Eigen::Matrix<int, 3, 1>::Ones();
                        measurement.value = -ground_truth_node.varpi.template tail<3>().template cast<double>() + noise;
                        measurement.s = ground_truth_node.arclength;
                        measurement.t = ground_truth_node.time;

                        auto factor = std::make_shared<Spacetime::Factors::GyroMeasurementFactor>(
                            gyro_covariance, measurement);
                        const Eigen::VectorXd generated_error = factor->getError({ground_truth_node}).template cast<double>();
                        if (!generated_error.isApprox(noise, 1e-9))
                        {
                            throw std::runtime_error("Generated gyro measurement does not match the requested additive noise.");
                        }
                        scenario.factors.push_back(factor);
                        scenario.factor_node_indices.push_back({static_cast<int>(node_index)});
                    }
                }

                const std::size_t measured_node_count = sampled_state.estimation_nodes.size() - data.robot_topology.K;
                const std::size_t factors_per_measured_node = static_cast<std::size_t>(simulate_pose) +
                                                              static_cast<std::size_t>(simulate_gyro);
                if (scenario.factors.size() != measured_node_count * factors_per_measured_node)
                {
                    throw std::runtime_error("Synthetic measurement generation produced an unexpected factor count.");
                }
                data.measurement_scenarios.push_back(std::move(scenario));
            }

            data.validation_variance = empiricalValidationVarianceFromSamples(
                data.nodes,
                samples);
        }
        else
        {
            if (!physical_data)
            {
                throw std::runtime_error("Physical data mode must be selected with --physical when prior_only_ground_truth_mode is false.");
            }

            if (!(trial_end_time > trial_start_time) || !std::isfinite(trial_end_time))
            {
                throw std::runtime_error("Physical data requires a finite interval with end_time greater than start_time.");
            }

            const Json::Value topology_root = robot_config["topology"];
            if (!topology_root.isMember("K_per_second"))
            {
                throw std::runtime_error("Physical data robot topology must define K_per_second.");
            }
            const double K_per_second = topology_root["K_per_second"].asDouble();
            if (!(K_per_second > 0.0))
            {
                throw std::runtime_error("K_per_second must be positive for physical data.");
            }

            const double configured_window_duration = window_duration_override >= 0.0
                ? window_duration_override : options_root.get("physical_window_duration", 0.0).asDouble();
            if (!std::isfinite(configured_window_duration) || configured_window_duration < 0.0)
            {
                throw std::runtime_error("physical_window_duration must be finite and nonnegative.");
            }
            const bool use_physical_windows = configured_window_duration > 0.0;
            const double window_duration = use_physical_windows
                ? configured_window_duration
                : trial_end_time - trial_start_time;
            const int requested_window_count = use_physical_windows
                ? (window_count_override > 0 ? window_count_override : options_root.get("measurement_sample_count", 10).asInt())
                : 1;
            if (requested_window_count <= 0)
            {
                throw std::runtime_error("measurement_sample_count must be positive when physical windowing is enabled.");
            }

            const double window_stride = use_physical_windows ? window_duration / 2.0 : window_duration;
            const int available_window_count = std::max(1, 1 + static_cast<int>(std::ceil(
                (trial_end_time - trial_start_time - window_duration) / window_stride)));
            const int window_count = std::min(requested_window_count, available_window_count);
            const double measurement_factor = robot_config["weights"].get("measurement_factor", 1.0).asDouble();
            const Eigen::Matrix<double, 6, 6> pose_covariance = measurement_factor * data.R_pose.asDiagonal();
            const Eigen::Matrix<double, 3, 3> gyro_covariance = measurement_factor * data.R_gyro.asDiagonal();
            const bool include_aurora = trial_data_root.get("aurora", false).asBool();
            const bool include_gyro = trial_data_root.get("gyro", false).asBool();
            const bool exclude_base_measurements = options_root.get("exclude_base_measurements", false).asBool();

            struct PhysicalRecord
            {
                std::string sensor_type;
                Spacetime::SensorMeasurement measurement;
            };
            std::vector<PhysicalRecord> records;

            for (const auto& entry : std::filesystem::directory_iterator(data_dir))
            {
                if (!entry.is_regular_file() || entry.path().extension() != ".csv")
                {
                    continue;
                }

                const std::vector<std::vector<std::string>> csv_rows = readCsvRows(entry.path());
                if (csv_rows.size() < 3 || csv_rows.front().size() < 2)
                {
                    continue;
                }

                std::string sensor_type = csv_rows.front()[1];
                sensor_type.erase(std::remove(sensor_type.begin(), sensor_type.end(), '\r'), sensor_type.end());
                if (sensor_type != "vicon" && sensor_type != "aurora" && sensor_type != "gyro")
                {
                    continue;
                }

                for (std::size_t row_index = 2; row_index < csv_rows.size(); ++row_index)
                {
                    const auto& row = csv_rows[row_index];
                    const std::size_t expected_size = sensor_type == "gyro" ? 5 : 8;
                    if (row.size() < expected_size)
                    {
                        continue;
                    }

                    Spacetime::SensorMeasurement measurement;
                    measurement.t = std::stod(row[0]);
                    measurement.s = std::stod(row[1]);
                    if (measurement.t < trial_start_time || measurement.t > trial_end_time)
                    {
                        continue;
                    }

                    if (sensor_type == "gyro")
                    {
                        measurement.type = Spacetime::SensorMeasurement::Gyro;
                        measurement.mask = Eigen::Matrix<int, 3, 1>::Ones();
                        measurement.value = Eigen::MatrixXd::Zero(3, 1);
                        for (int i = 0; i < 3; ++i)
                        {
                            measurement.value(i, 0) = std::stod(row[2 + i]);
                        }
                    }
                    else
                    {
                        measurement.type = Spacetime::SensorMeasurement::Pose;
                        measurement.mask = Eigen::Matrix<int, 6, 1>::Ones();
                        measurement.value = se3::vec2tran(parsePoseRow(row));
                    }
                    records.push_back({sensor_type, measurement});
                }
            }

            std::sort(records.begin(), records.end(), [](const PhysicalRecord& lhs, const PhysicalRecord& rhs) {
                return lhs.measurement.t < rhs.measurement.t ||
                       (lhs.measurement.t == rhs.measurement.t && lhs.measurement.s < rhs.measurement.s);
            });

            for (int window_index = 0; window_index < window_count; ++window_index)
            {
                const double window_start = trial_start_time + static_cast<double>(window_index) * window_stride;
                const double window_end = std::min(window_start + window_duration, trial_end_time);
                const double actual_duration = window_end - window_start;
                if (!(actual_duration > 0.0))
                {
                    continue;
                }

                TestProblemData::MeasurementScenario scenario;
                scenario.robot_topology = data.robot_topology;
                scenario.robot_topology.t0 = window_start;
                scenario.robot_topology.T = actual_duration;
                scenario.robot_topology.K = static_cast<unsigned int>(std::ceil(K_per_second * actual_duration)) + 1;
                scenario.nodes.reserve(static_cast<std::size_t>(scenario.robot_topology.N) * scenario.robot_topology.K);

                for (unsigned int k = 0; k < scenario.robot_topology.K; ++k)
                {
                    const double time_fraction = scenario.robot_topology.K > 1
                        ? static_cast<double>(k) / static_cast<double>(scenario.robot_topology.K - 1)
                        : 0.0;
                    for (unsigned int n = 0; n < scenario.robot_topology.N; ++n)
                    {
                        const double space_fraction = scenario.robot_topology.N > 1
                            ? static_cast<double>(n) / static_cast<double>(scenario.robot_topology.N - 1)
                            : 0.0;
                        spacetime::OptimizationProblem::Node node;
                        node.arclength = space_fraction * scenario.robot_topology.L;
                        node.time = window_start + time_fraction * actual_duration;
                        node.pose = se3::vec2tran(node.arclength * scenario.robot_topology.epsilon0) * scenario.robot_topology.T0;
                        node.epsilon = scenario.robot_topology.epsilon0;
                        node.varpi = scenario.robot_topology.varpi0;
                        scenario.nodes.push_back(node);
                    }
                }

                for (const PhysicalRecord& record : records)
                {
                    const bool is_last_window = window_index + 1 == window_count;
                    if (record.measurement.t < window_start || record.measurement.t > window_end ||
                        (!is_last_window && record.measurement.t == window_end))
                    {
                        continue;
                    }

                    const bool excluded_base_measurement =
                        exclude_base_measurements &&
                        record.sensor_type != "vicon" &&
                        std::abs(record.measurement.s) < TOLERANCE;
                    if (excluded_base_measurement)
                    {
                        continue;
                    }
                    if (record.sensor_type == "aurora" && include_aurora)
                    {
                        scenario.factors.push_back(std::make_shared<Spacetime::Factors::PoseMeasurementFactor>(pose_covariance, record.measurement));
                    }
                    else if (record.sensor_type == "gyro" && include_gyro)
                    {
                        scenario.factors.push_back(std::make_shared<Spacetime::Factors::GyroMeasurementFactor>(gyro_covariance, record.measurement));
                    }
                    else if (record.sensor_type == "vicon")
                    {
                        const double time_fraction = (record.measurement.t - window_start) / actual_duration;
                        const double space_fraction = scenario.robot_topology.L > 0.0
                            ? record.measurement.s / scenario.robot_topology.L
                            : 0.0;
                        const int k = std::clamp(
                            static_cast<int>(std::llround(time_fraction * static_cast<double>(scenario.robot_topology.K - 1))),
                            0,
                            static_cast<int>(scenario.robot_topology.K - 1));
                        const int n = std::clamp(
                            static_cast<int>(std::llround(space_fraction * static_cast<double>(scenario.robot_topology.N - 1))),
                            0,
                            static_cast<int>(scenario.robot_topology.N - 1));
                        const int node_index = k * static_cast<int>(scenario.robot_topology.N) + n;

                        spacetime::ValidationTarget target;
                        target.node_index = node_index;
                        target.sample_group_id = window_index;
                        target.ground_truth = scenario.nodes[static_cast<std::size_t>(node_index)];
                        target.ground_truth.time = record.measurement.t;
                        target.ground_truth.arclength = scenario.nodes[static_cast<std::size_t>(node_index)].arclength;
                        target.interpolate_time = true;
                        target.ground_truth.pose = invertTransformation(record.measurement.value);
                        Spacetime::Factors::PoseMeasurementFactor convention_check(pose_covariance, record.measurement);
                        if (!convention_check.getError({target.ground_truth}).isZero(1e-9))
                        {
                            throw std::runtime_error("Vicon validation pose is inconsistent with pose measurement convention.");
                        }
                        target.observed_mask = ObservedMask18::Zero();
                        target.observed_mask.segment<6>(0).setOnes();
                        scenario.validation_targets.push_back(target);
                    }
                }

                if (!scenario.factors.empty() && !scenario.validation_targets.empty())
                {
                    data.measurement_scenarios.push_back(std::move(scenario));
                }
            }

            if (data.measurement_scenarios.empty())
            {
                throw std::runtime_error("Physical data did not produce any windows containing both measurements and Vicon validation targets.");
            }

            data.validation_sample_count = static_cast<int>(data.measurement_scenarios.size());
            data.robot_topology = data.measurement_scenarios.front().robot_topology;
            data.nodes = data.measurement_scenarios.front().nodes;
            data.factors = data.measurement_scenarios.front().factors;
            data.factor_node_indices = data.measurement_scenarios.front().factor_node_indices;
            data.validation_targets = data.measurement_scenarios.front().validation_targets;
            data.csv_path = data_dir;

            std::cout << "Loaded " << data.measurement_scenarios.size()
                      << " physical windows at K_per_second=" << K_per_second
                      << "; first window K=" << data.robot_topology.K
                      << ", N=" << data.robot_topology.N << std::endl;
        }

        if (data.nodes.empty())
        {
            throw std::runtime_error("No valid nodes were created from the minimal dataset.");
        }
        if (data.validation_targets.empty())
        {
            spacetime::ValidationTarget target;
            target.node_index = 0;
            target.sample_group_id = 0;
            target.ground_truth = data.nodes.front();
            target.observed_mask = validation_observed_mask;
            if (!data.factors.empty())
            {
                const auto pose_factor = std::dynamic_pointer_cast<Spacetime::Factors::PoseMeasurementFactor>(data.factors.front());
                if (pose_factor)
                {
                    target.ground_truth.pose = pose_factor->getMeas().value;
                    target.observed_mask.setZero();
                    for (int d = 0; d < 6; ++d)
                    {
                        target.observed_mask(d) = pose_factor->getMeas().mask(d, 0);
                    }
                }
            }
            data.validation_targets.push_back(target);
        }

        return data;
    }

    void writeReportJson(std::ostream& stream, const Json::Value& value)
    {
        Json::StreamWriterBuilder writer;
        writer["commentStyle"] = "None";
        stream << Json::writeString(writer, value);
        if (!stream) throw std::runtime_error("Failed to write JSON report.");
    }

    spacetime::OptimizationProblem makePhysicalProblem(
        const TestProblemData& data, const TestProblemData::MeasurementScenario& scenario,
        const Eigen::VectorXd& theta)
    {
        spacetime::OptimizationProblem problem;
        problem.setEstimatorOptions(data.estimator_options);
        problem.setRobotTopology(scenario.robot_topology);
        problem.setRunSummary({data.validation_sample_count, data.R_pose, data.R_gyro, data.validation_variance});
        problem.setTheta(theta);
        problem.setProblemData(scenario.nodes, scenario.factors, scenario.factor_node_indices);
        problem.setValidationTargets(scenario.validation_targets);
        return problem;
    }

    struct PhysicalEvaluationTotals
    {
        double nll = 0.0;
        double mahalanobis = 0.0;
        double translation_sum = 0.0;
        double translation_squared_sum = 0.0;
        double rotation_sum = 0.0;
        double rotation_squared_sum = 0.0;
        std::size_t windows = 0;
        std::size_t targets = 0;
        int observed_dimensions = 0;

        void add(const PhysicalEvaluationTotals& other)
        {
            nll += other.nll;
            mahalanobis += other.mahalanobis;
            translation_sum += other.translation_sum;
            translation_squared_sum += other.translation_squared_sum;
            rotation_sum += other.rotation_sum;
            rotation_squared_sum += other.rotation_squared_sum;
            windows += other.windows;
            targets += other.targets;
            observed_dimensions += other.observed_dimensions;
        }
    };

    void writePhysicalEvaluationRow(std::ostream& stream, const std::string& split,
                                    const std::string& trial, const std::string& window,
                                    double start, double end, const PhysicalEvaluationTotals& totals)
    {
        const double target_count = static_cast<double>(totals.targets);
        stream << split << ',' << trial << ',' << window << ',' << start << ',' << end << ','
               << totals.windows << ',' << totals.targets << ',' << totals.observed_dimensions << ','
               << totals.nll << ',' << totals.nll / totals.windows << ','
               << totals.nll / totals.observed_dimensions << ',' << totals.mahalanobis / target_count << ','
               << totals.translation_sum / target_count << ','
               << std::sqrt(totals.translation_squared_sum / target_count) << ','
               << totals.rotation_sum / target_count << ','
               << std::sqrt(totals.rotation_squared_sum / target_count) << '\n';
        stream.flush();
    }

    PhysicalEvaluationTotals evaluatePhysicalWindows(
        const TestProblemData& data, std::size_t begin, std::size_t count,
        const Eigen::VectorXd& theta, bool joint, const std::string& split,
        const std::string& trial, std::ostream& windows_log, std::ostream& summary_log)
    {
        PhysicalEvaluationTotals totals;
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto& scenario = data.measurement_scenarios.at(begin + index);
            auto problem = makePhysicalProblem(data, scenario, theta);
            const auto solution = problem.solveLowerLevel(false);
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(solution.H);
            if (solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Held-parameter evaluation Hessian factorization failed.");
            }
            const auto& targets = problem.validationTargets();
            const auto& nodes = problem.solvedNodes();
            const auto cache = buildJointValidationCache(solution.H, targets, nodes, solver, &theta, joint);
            const auto group = buildJointValidationGroupData(solution.H, targets, nodes, cache, false);
            const auto validation = computeValidationLossWithJoint(targets, cache, group);
            PhysicalEvaluationTotals window;
            window.windows = 1;
            window.targets = targets.size();
            window.observed_dimensions = cache.observed_dim_total;
            window.nll = validation.loss;
            window.mahalanobis = validation.mahalanobis_sum;
            for (const auto& target : targets)
            {
                const auto estimate = interpolateValidationTime(target, nodes).estimate;
                const Eigen::Matrix4d error = estimate.pose * invertTransformation(target.ground_truth.pose);
                const double translation = error.block<3, 1>(0, 3).norm();
                const double rotation = se3::tran2vec(error).tail<3>().norm();
                window.translation_sum += translation;
                window.translation_squared_sum += translation * translation;
                window.rotation_sum += rotation;
                window.rotation_squared_sum += rotation * rotation;
            }
            writePhysicalEvaluationRow(windows_log, split, trial, std::to_string(index),
                scenario.robot_topology.t0, scenario.robot_topology.t0 + scenario.robot_topology.T, window);
            totals.add(window);
            std::cout << "Frozen-parameter " << split << " evaluation: " << trial
                      << " window " << index + 1 << '/' << count << ", NLL=" << validation.loss << std::endl;
        }
        writePhysicalEvaluationRow(summary_log, split, trial, "all", 0.0, 0.0, totals);
        return totals;
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
            throw std::runtime_error("OptimizationProblem::setTheta expects 42 parameters: P0(18), Q1(6), Q2(6), Q3(6), Q4(6).");
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
        has_lower_solution_ = false;
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
        fillDiagonalFromVariance(hyperparameters_.P0, theta_, kP0Offset, "P0");
        fillDiagonalFromVariance(hyperparameters_.Q1, theta_, kQ1Offset, "Q1");
        fillDiagonalFromVariance(hyperparameters_.Q2, theta_, kQ2Offset, "Q2");
        fillDiagonalFromVariance(hyperparameters_.Q3, theta_, kQ3Offset, "Q3");
        fillDiagonalFromVariance(hyperparameters_.Q4, theta_, kQ4Offset, "Q4");
    }

    LowerLevelSolution OptimizationProblem::solveLowerLevel(bool verbose) // Returns x*, H and linearisation data.
    {
        LowerLevelSolution solution;

        if (nodes_.empty())
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
            throw std::runtime_error("OptimizationProblem: theta has not been initialized with 42 parameters.");
        }

        const Spacetime::Hyperparameters hyper = hyperparameters_;

        Spacetime::Options options = estimator_options_;
        if (!estimator_options_set_)
        {
            warning() << "OptimizationProblem: estimator options not explicitly set. Using built-in defaults.";
        }

        options.init_guess_type = Spacetime::Options::InitialGuessType::Custom;
        options.custom_guess.estimation_nodes = has_lower_solution_ ? solved_nodes_ : nodes_;
        options.estimator_type = Spacetime::Options::EstimatorType::Batch;

        Spacetime::Estimator estimator(topology, hyper, options);
        const auto t_lower_solve_start = std::chrono::steady_clock::now();
        const auto estimate = estimator.computeStateEstimate(measurement_factors, verbose);
        const auto t_lower_solve_end = std::chrono::steady_clock::now();
        solution.lower_solve_ms = std::chrono::duration<double, std::milli>(
            t_lower_solve_end - t_lower_solve_start).count();
        if (!estimate.success)
        {
            throw std::runtime_error(
                "OptimizationProblem: Estimator::computeStateEstimate failed in solveLowerLevel(): " +
                estimate.message.str());
        }

        solution.tip_metrics = computePoseValidationMetrics(estimator, estimate.state, validation_targets_, topology, true);
        solution.body_metrics = computePoseValidationMetrics(estimator, estimate.state, validation_targets_, topology, false);

        Eigen::SparseMatrix<double> H_est;
        std::vector<Eigen::VectorXd> factor_errors;
        std::vector<Eigen::MatrixXd> factor_jacobians;
        std::vector<Eigen::MatrixXd> factor_Qs;
        std::vector<std::vector<int>> factor_node_offsets;
        const auto t_extract_hessian_start = std::chrono::steady_clock::now();
        estimator.extractJacobianHessian(estimate.state, H_est, factor_errors, factor_jacobians, factor_Qs, factor_node_offsets);
        const auto t_extract_hessian_end = std::chrono::steady_clock::now();
        solution.extract_hessian_ms = std::chrono::duration<double, std::milli>(
            t_extract_hessian_end - t_extract_hessian_start).count();
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
        has_lower_solution_ = true;
        return solution;
    }

    void OptimizationProblem::setValidationTargets(const std::vector<ValidationTarget> &targets)
    {
        validation_targets_ = targets;
    }

    void OptimizationProblem::setRunSummary(const RunSummary &summary)
    {
        run_summary_ = summary;
        run_summary_set_ = true;
    }

    ValidationLossData OptimizationProblem::computeValidationLoss(const LowerLevelSolution &solution) const
    {
        // Compute validation NLL and keep NEES statistics for diagnostics/compatibility.
        ValidationLossData out;

        if (validation_targets_.empty() || solved_nodes_.empty())
        {
            return out;
        }

        constexpr int node_dim = 18;
        const int validation_count = static_cast<int>(validation_targets_.size());
        out.nees_per_node = Eigen::VectorXd::Zero(validation_count);
        out.node_errors_stacked = Eigen::VectorXd::Zero(node_dim * validation_count);
        double logdet_sum = 0.0;
        int effective_validation_count = 0;
        int observed_dim_total = 0;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> h_solver;
        h_solver.compute(solution.H);
        if (h_solver.info() != Eigen::Success)
        {
            throw std::runtime_error("OptimizationProblem::computeValidationLoss: failed to factorize H for node marginals.");
        }

        if (std::any_of(validation_targets_.begin(), validation_targets_.end(), [](const ValidationTarget& target) {
                return target.interpolate_time;
            }))
        {
            const auto cache = buildJointValidationCache(solution.H, validation_targets_, solved_nodes_, h_solver, &theta_, false);
            const auto group = buildJointValidationGroupData(solution.H, validation_targets_, solved_nodes_, cache, false);
            return computeValidationLossWithJoint(validation_targets_, cache, group);
        }

        for (int i = 0; i < validation_count; ++i)
        {
            const auto &target = validation_targets_[static_cast<size_t>(i)];
            if (target.node_index < 0 || target.node_index >= static_cast<int>(solved_nodes_.size()))
            {
                throw std::runtime_error("OptimizationProblem::computeValidationLoss: validation node index out of range.");
            }

            const auto &estimate = solved_nodes_[static_cast<size_t>(target.node_index)];
            const Eigen::Matrix<double, 18, 1> e_full = computeNodeValidationError(estimate, target.ground_truth);
            const Eigen::Matrix<double, 18, 1> e = applyObservedMask(e_full, target.observed_mask);
            out.node_errors_stacked.segment<18>(i * node_dim) = e;

            const int observed_dims = countObservedDimensions(target.observed_mask);
            if (observed_dims == 0)
            {
                out.nees_per_node(i) = 0.0;
                continue;
            }

            const Eigen::Matrix<double, 18, 18> P_cov_block =
                computeNodeMarginalCovarianceBlock(h_solver, solution.H.rows(), target.node_index);
            const Eigen::Matrix<double, 18, 18> P_inv_observed =
                buildObservedInformationFromCovariance(P_cov_block, target.observed_mask);
            out.nees_per_node(i) = e.transpose() * P_inv_observed * e;
            logdet_sum += computeObservedLogDetFromCovariance(P_cov_block, target.observed_mask);
            ++effective_validation_count;
            observed_dim_total += observed_dims;
        }

        out.mahalanobis_sum = out.nees_per_node.sum();
        out.logdet_sum = logdet_sum;
        const double denom = static_cast<double>(std::max(1, effective_validation_count));
        out.mean_logdet = logdet_sum / denom;
        out.mean_nees = out.mahalanobis_sum / denom;
        const double mean_observed_dims = static_cast<double>(observed_dim_total) / denom;
        out.delta = out.mean_nees - mean_observed_dims;
        out.loss = 0.5 * (out.logdet_sum + out.mahalanobis_sum);
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

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> h_solver;
        h_solver.compute(solution.H);
        if (h_solver.info() != Eigen::Success)
        {
            throw std::runtime_error("OptimizationProblem::computeAdjointRhs: failed to factorize H for node marginals.");
        }

        if (std::any_of(validation_targets_.begin(), validation_targets_.end(), [](const ValidationTarget& target) {
                return target.interpolate_time;
            }))
        {
            const auto cache = buildJointValidationCache(solution.H, validation_targets_, solved_nodes_, h_solver, &theta_, false);
            const auto group = buildJointValidationGroupData(solution.H, validation_targets_, solved_nodes_, cache);
            return computeAdjointRhsWithJoint(solution.H, validation_targets_, cache, group);
        }

        for (int i = 0; i < validation_count; ++i)
        {
            const auto &target = validation_targets_[static_cast<size_t>(i)];
            if (target.node_index < 0 || (target.node_index + 1) * node_dim > rhs.size())
            {
                throw std::runtime_error("OptimizationProblem::computeAdjointRhs: validation node index out of range.");
            }

            const Eigen::Matrix<double, 18, 1> e = validation.node_errors_stacked.segment<18>(i * node_dim);
            if (countObservedDimensions(target.observed_mask) == 0)
            {
                continue;
            }

            const Eigen::Matrix<double, 18, 18> P_cov_block =
                computeNodeMarginalCovarianceBlock(h_solver, solution.H.rows(), target.node_index);
            const Eigen::Matrix<double, 18, 18> P_inv_observed =
                buildObservedInformationFromCovariance(P_cov_block, target.observed_mask);

            Eigen::Matrix<double, 18, 18> J = Eigen::Matrix<double, 18, 18>::Identity();
            J.block<6, 6>(0, 0) = se3::vec2jacinv(e.segment<6>(0));

            rhs.segment<18>(target.node_index * node_dim) += J.transpose() * P_inv_observed * e;
        }
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
                computeNodeValidationError(interpolateValidationTime(target, solved_nodes_).estimate, target.ground_truth);
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

    const OptimizationProblem::RunSummary& OptimizationProblem::runSummary() const
    {
        return run_summary_;
    }

    Optimizer::Optimizer(const OptimizationProblem& problem)
        : problem_(std::make_shared<OptimizationProblem>(problem))
    {
        scenario_problems_.push_back(problem_);
    }

    void Optimizer::setScenarioProblems(const std::vector<OptimizationProblem>& problems)
    {
        if (problems.empty())
        {
            throw std::runtime_error("Optimizer::setScenarioProblems requires at least one problem.");
        }

        scenario_problems_.clear();
        scenario_problems_.reserve(problems.size());
        for (const auto& scenario_problem : problems)
        {
            scenario_problems_.push_back(std::make_shared<OptimizationProblem>(scenario_problem));
        }
        problem_ = scenario_problems_.front();
    }

    void Optimizer::setConfig(const OptimizerConfig& cfg)
    {
        optimizer_config = cfg;
        if (optimizer_config.lambda_update_mask.size() == kQ4Offset)
            optimizer_config.lambda_update_mask.resize(kThetaSize, 0);
        if (optimizer_config.lambda_update_mask.size() != kThetaSize ||
            std::any_of(optimizer_config.lambda_update_mask.begin(), optimizer_config.lambda_update_mask.end(),
                [](int value) { return value != 0 && value != 1; }))
            throw std::runtime_error("lambda_update_mask requires 42 binary entries, or 36 legacy binary entries.");
    }

    Optimizer::Result Optimizer::optimize()
    {
        return step(Eigen::VectorXd());
    }

    Optimizer::Result Optimizer::step(const Eigen::VectorXd& psi) //Main loop
    {
        Result result;
        if (!problem_)
        {
            throw std::runtime_error("Optimizer::step requires a valid OptimizationProblem.");
        }

        if (problem_->validationTargets().empty())
        {
            throw std::runtime_error("Optimizer::step requires validation targets for NLL-based outer optimization.");
        }

        Eigen::VectorXd current_psi;
        if (psi.size() == 0)
        {
            const Eigen::VectorXd &initial_theta = problem_->theta();
            if (initial_theta.size() != kThetaSize)
            {
                throw std::runtime_error("Optimizer::step could not infer an initial 42-vector theta from the problem.");
            }
            const Eigen::VectorXd initial_lambda = initial_theta.cwiseInverse();
            current_psi = optimizer_config.use_exponential_param ? initial_lambda.array().log().matrix() : initial_lambda;
        }
        else
        {
            current_psi = psi;
        }

        if (optimizer_config.lambda_update_mask.size() != kThetaSize)
        {
            throw std::runtime_error("lambda_update_mask must contain exactly 42 entries.");
        }
        std::vector<int> update_mask = optimizer_config.lambda_update_mask;
        for (int index = 0; index < kThetaSize; ++index)
        {
            if (update_mask[index] != 0 && update_mask[index] != 1)
            {
                throw std::runtime_error("lambda_update_mask entries must be 0 or 1.");
            }
            if (optimizer_config.freeze_p0_non_pose && index >= 6 && index < 18)
            {
                update_mask[index] = 0;
            }
        }
        const Eigen::VectorXd initial_psi = current_psi;

        //lambda = exp(psi) if use_exponential_param is true, otherwise lambda = psi
        auto mapLambda = [&](const Eigen::VectorXd &unconstrained) -> Eigen::VectorXd {
            if (optimizer_config.use_exponential_param)
            {
                return unconstrained.array().exp().matrix();
            }
            return unconstrained;
        };

        auto mapVariance = [&](const Eigen::VectorXd &lambda) -> Eigen::VectorXd {
            return lambda.cwiseInverse();
        };

        // L = 0.5 * (logdet(P) + e^T * P^{-1} * e)
        constexpr double kNllQuadraticScale = -0.5;
        Eigen::VectorXd adam_m = Eigen::VectorXd::Zero(current_psi.size());
        Eigen::VectorXd adam_v = Eigen::VectorXd::Zero(current_psi.size());
        double beta1_power = 1.0;
        double beta2_power = 1.0;
        double previous_loss = std::numeric_limits<double>::infinity();
        Eigen::VectorXd effective_learning_rates = Eigen::VectorXd::Constant(current_psi.size(), optimizer_config.learning_rate);
        OscillationDetector oscillation_detector;
        int oscillation_reductions = 0;
        std::vector<int> dimension_reductions(current_psi.size(), 0);
        const std::filesystem::path outer_loop_log_path = resolveAssetsLogPath("bilevel_optimizer_outer_loop_" + formatTimestamp() + ".log");
        std::ofstream outer_loop_log(outer_loop_log_path, std::ios::out | std::ios::trunc);
        if (!outer_loop_log.is_open())
        {
            throw std::runtime_error("Failed to open optimizer log file: " + outer_loop_log_path.string());
        }

        outer_loop_log << "iteration";
        for (int i = 0; i < kThetaSize; ++i)
        {
            outer_loop_log << ",lambda[" << i << "]";
        }
        for (int i = 0; i < kThetaSize; ++i)
        {
            outer_loop_log << ",dL_dlambda[" << i << "]";
        }
        outer_loop_log << ",mean_mahalanobis,mean_logdet,validation_nll"
                   << ",tip_mae_m,tip_rmse_m,tip_rotation_mae_rad,tip_rotation_rmse_rad"
                   << ",body_mae_m,body_rmse_m,body_rotation_mae_rad,body_rotation_rmse_rad"
                   << ",effective_learning_rate,oscillation_reductions";
        for (int index = 0; index < kThetaSize; ++index)
        {
            outer_loop_log << ",learning_rate[" << index << "],oscillation_reductions[" << index << "]";
        }
        outer_loop_log << '\n';

        //Main optimization loop
        for (int iter = 0; iter < optimizer_config.max_iterations; ++iter)
        {
            const Eigen::VectorXd lambda = mapLambda(current_psi);
            const Eigen::VectorXd theta = mapVariance(lambda);
            Eigen::VectorXd variance_gradient_sum = Eigen::VectorXd::Zero(kThetaSize);
            ValidationLossData averaged_validation;
            TipValidationMetrics tip_metric_sums;
            PoseValidationMetrics body_metric_sums;
            double g_joint_norm_sum = 0.0;
            double q_joint_norm_sum = 0.0;
            double adjoint_rhs_norm_sum = 0.0;
            double quadratic_rhs_norm_sum = 0.0;
            double adjoint_state_norm_sum = 0.0;
            double quadratic_state_norm_sum = 0.0;
            double pv_min_eig_sum = 0.0;
            double pv_max_eig_sum = 0.0;
            double pv_condition_sum = 0.0;
            double lower_solve_ms_sum = 0.0;
            double extract_hessian_ms_sum = 0.0;
            double joint_cache_ms_sum = 0.0;
            double joint_setup_ms_sum = 0.0;
            double joint_selector_ms_sum = 0.0;
            double joint_projection_solve_ms_sum = 0.0;
            double joint_covariance_ms_sum = 0.0;
            double joint_eigendecomposition_ms_sum = 0.0;
            double joint_information_ms_sum = 0.0;
            double adjoint_solves_ms_sum = 0.0;
            double factor_gradient_loop_ms_sum = 0.0;

            if (scenario_problems_.empty())
            {
                throw std::runtime_error("Optimizer::step requires at least one scenario problem.");
            }

            for (std::size_t scenario_index = 0; scenario_index < scenario_problems_.size(); ++scenario_index)
            {
                problem_ = scenario_problems_[scenario_index];
                problem_->setTheta(theta);
                const auto& validation_targets = problem_->validationTargets();
                if (validation_targets.empty())
                {
                    throw std::runtime_error("Optimizer::step requires validation targets for every scenario.");
                }

                LowerLevelSolution solution;
                try
                {
                    solution = problem_->solveLowerLevel(optimizer_config.verbose);
                }
                catch (const std::exception& error)
                {
                    throw std::runtime_error(
                        "Optimizer::step scenario " + std::to_string(scenario_index) +
                        " lower-level solve failed: " + error.what());
                }
                if (optimizer_config.verbose)
                {
                    std::cout << "Scenario " << scenario_index << ": lower-level solve complete" << std::endl;
                }
                lower_solve_ms_sum += solution.lower_solve_ms;
                extract_hessian_ms_sum += solution.extract_hessian_ms;
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

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> adjoint_solver;
            adjoint_solver.compute(solution.H);
            if (adjoint_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Optimizer::step failed to factorize the lower-level information matrix.");
            }
            if (optimizer_config.verbose)
            {
                std::cout << "Scenario " << scenario_index << ": Hessian factorization complete" << std::endl;
            }

            const bool use_joint_objective = optimizer_config.use_joint_validation_objective ||
                std::any_of(validation_targets.begin(), validation_targets.end(), [](const ValidationTarget& target) {
                    return target.interpolate_time;
                });
            ValidationLossData validation;
            Eigen::VectorXd adjoint_rhs;
            Eigen::VectorXd quadratic_rhs;
            Eigen::VectorXd direct_validation_gradient = Eigen::VectorXd::Zero(kThetaSize);
            Eigen::MatrixXd logdet_information;
            std::vector<int> logdet_node_indices;
            std::vector<JointGroupLogDetData> joint_group_logdet_data;
            JointValidationCacheData shared_joint_cache;
            Eigen::MatrixXd shared_joint_logdet_projection;
            Eigen::MatrixXd shared_joint_weighted_projection;
            double g_joint_norm = 0.0;
            double q_joint_norm = 0.0;
            double pv_min_eig = 0.0;
            double pv_max_eig = 0.0;
            double pv_condition = 0.0;
            double joint_cache_ms = 0.0;
            double adjoint_solves_ms = 0.0;
            double factor_gradient_loop_ms = 0.0;

            if (use_joint_objective)
            {
                const std::vector<std::vector<ValidationTarget>> grouped_validation_targets =
                    groupValidationTargetsBySample(validation_targets);
                if (grouped_validation_targets.empty())
                {
                    throw std::runtime_error("Optimizer::step: grouped validation target set is empty in joint mode.");
                }

                const double group_count = static_cast<double>(grouped_validation_targets.size());
                validation = ValidationLossData{};
                adjoint_rhs = Eigen::VectorXd::Zero(solution.H.rows());
                quadratic_rhs = Eigen::VectorXd::Zero(solution.H.rows());
                joint_group_logdet_data.clear();
                joint_group_logdet_data.reserve(grouped_validation_targets.size());
                std::vector<Eigen::VectorXd> group_quadratic_rhs_list;
                group_quadratic_rhs_list.reserve(grouped_validation_targets.size());

                double total_loss = 0.0;
                double total_mahalanobis_sum = 0.0;
                double total_logdet_sum = 0.0;
                double total_mean_nees = 0.0;
                double total_mean_logdet = 0.0;
                int total_effective_validation_count = 0;
                int total_observed_dim = 0;
                double g_joint_norm_sq_sum = 0.0;
                double q_joint_norm_sq_sum = 0.0;
                bool have_cov_diag = false;
                double pv_min_eig_acc = 0.0;
                double pv_max_eig_acc = 0.0;
                double pv_condition_acc = 0.0;

                const auto t_joint_cache_start = std::chrono::steady_clock::now();
                shared_joint_cache = buildJointValidationCache(
                    solution.H, grouped_validation_targets.front(), solved_nodes, adjoint_solver, &theta,
                    optimizer_config.use_joint_validation_objective);
                const auto t_joint_cache_end = std::chrono::steady_clock::now();
                joint_cache_ms = std::chrono::duration<double, std::milli>(
                    t_joint_cache_end - t_joint_cache_start).count();
                if (optimizer_config.verbose)
                {
                    std::cout << "Scenario " << scenario_index << ": "
                              << (shared_joint_cache.joint_observations ? "joint" : "marginal")
                              << " validation cache complete"
                              << "; state_dim=" << solution.H.rows()
                              << ", observed_dim=" << shared_joint_cache.observed_dim_total
                              << ", validation_targets=" << grouped_validation_targets.front().size()
                              << ", full_state=" << shared_joint_cache.full_state_observation << std::endl;
                }
                constexpr int node_dim = 18;
                if (!shared_joint_cache.full_state_observation)
                {
                    shared_joint_logdet_projection = shared_joint_cache.h_inv_selector;
                    shared_joint_weighted_projection = shared_joint_cache.weighted_projection;
                }

                for (const auto& target_group : grouped_validation_targets)
                {
                    const JointValidationGroupData group_data =
                        buildJointValidationGroupData(solution.H, target_group, solved_nodes, shared_joint_cache);
                    const ValidationLossData group_validation =
                        computeValidationLossWithJoint(target_group, shared_joint_cache, group_data);

                    total_loss += group_validation.loss;
                    direct_validation_gradient += group_data.covariance_theta_gradient / group_count;
                    total_mahalanobis_sum += group_validation.mahalanobis_sum;
                    total_logdet_sum += group_validation.logdet_sum;
                    total_mean_nees += group_validation.mean_nees;
                    total_mean_logdet += group_validation.mean_logdet;
                    total_effective_validation_count += shared_joint_cache.effective_validation_count;
                    total_observed_dim += shared_joint_cache.observed_dim_total;

                    adjoint_rhs += computeAdjointRhsWithJoint(solution.H, target_group, shared_joint_cache, group_data);
                    const Eigen::VectorXd group_quadratic_rhs =
                        computeQuadraticRhsWithJoint(solution.H, target_group, shared_joint_cache, group_data);
                    quadratic_rhs += group_quadratic_rhs;

                    Eigen::VectorXd g_joint = group_data.weighted_errors;
                    for (int i = 0; i < static_cast<int>(target_group.size()); ++i)
                    {
                        const Vector18d e = group_data.node_errors_stacked.segment<18>(i * node_dim);
                        Matrix18d J = Matrix18d::Identity();
                        J.block<6, 6>(0, 0) = se3::vec2jacinv(e.segment<6>(0));
                        g_joint.segment<18>(i * node_dim) = J.transpose() * g_joint.segment<18>(i * node_dim);
                    }
                    g_joint_norm_sq_sum += g_joint.squaredNorm();
                    q_joint_norm_sq_sum += group_data.weighted_errors.squaredNorm();

                    if (shared_joint_cache.effective_validation_count > 0)
                    {
                        pv_min_eig_acc += shared_joint_cache.min_observed_cov_eig;
                        pv_max_eig_acc += shared_joint_cache.max_observed_cov_eig;
                        pv_condition_acc += shared_joint_cache.observed_cov_condition;
                        have_cov_diag = true;
                    }

                    JointGroupLogDetData logdet_group;
                    if (shared_joint_cache.full_state_observation)
                    {
                        logdet_group.quadratic_state = group_data.node_errors_stacked;
                    }
                    else
                    {
                        if (shared_joint_cache.joint_observations)
                        {
                            group_quadratic_rhs_list.push_back(group_quadratic_rhs);
                        }
                        else
                        {
                            const double target_scale = std::sqrt(static_cast<double>(target_group.size()));
                            for (int target_index = 0; target_index < static_cast<int>(target_group.size()); ++target_index)
                            {
                                JointGroupLogDetData target_logdet;
                                target_logdet.quadratic_state = Eigen::VectorXd::Zero(solution.H.rows());
                                for (int dimension = 0; dimension < 18; ++dimension)
                                {
                                    const int column = shared_joint_cache.observed_columns[target_index][dimension];
                                    if (column >= 0)
                                    {
                                        target_logdet.quadratic_state += shared_joint_cache.h_inv_selector.col(column) *
                                            group_data.weighted_errors(target_index * 18 + dimension);
                                    }
                                }
                                target_logdet.quadratic_state *= target_scale;
                                joint_group_logdet_data.push_back(std::move(target_logdet));
                            }
                        }
                    }

                    if (shared_joint_cache.full_state_observation || shared_joint_cache.joint_observations)
                    {
                        joint_group_logdet_data.push_back(std::move(logdet_group));
                    }
                }

                if (!shared_joint_cache.full_state_observation && shared_joint_cache.joint_observations)
                {
                    const int group_count_int = static_cast<int>(group_quadratic_rhs_list.size());
                    Eigen::MatrixXd group_quadratic_rhs_matrix = Eigen::MatrixXd::Zero(solution.H.rows(), group_count_int);
                    for (int group_idx = 0; group_idx < group_count_int; ++group_idx)
                    {
                        group_quadratic_rhs_matrix.col(group_idx) = group_quadratic_rhs_list[static_cast<std::size_t>(group_idx)];
                    }

                    const auto t_group_quadratic_solve_start = std::chrono::steady_clock::now();
                    const Eigen::MatrixXd group_quadratic_states = adjoint_solver.solve(group_quadratic_rhs_matrix);
                    const auto t_group_quadratic_solve_end = std::chrono::steady_clock::now();
                    adjoint_solves_ms += std::chrono::duration<double, std::milli>(
                        t_group_quadratic_solve_end - t_group_quadratic_solve_start).count();
                    if (adjoint_solver.info() != Eigen::Success)
                    {
                        throw std::runtime_error("Optimizer::step failed to solve group quadratic sensitivity states.");
                    }

                    for (int group_idx = 0; group_idx < group_count_int; ++group_idx)
                    {
                        joint_group_logdet_data[static_cast<std::size_t>(group_idx)].quadratic_state =
                            group_quadratic_states.col(group_idx);
                    }
                }

                adjoint_rhs /= group_count;
                quadratic_rhs /= group_count;

                validation.mahalanobis_sum = total_mahalanobis_sum / group_count;
                validation.logdet_sum = total_logdet_sum / group_count;
                validation.loss = total_loss / group_count;

                if (total_effective_validation_count > 0)
                {
                    const double effective_count = static_cast<double>(total_effective_validation_count);
                    validation.mean_nees = total_mahalanobis_sum / effective_count;
                    validation.mean_logdet = total_logdet_sum / effective_count;
                    validation.delta = validation.mean_nees - (static_cast<double>(total_observed_dim) / effective_count);
                }
                else
                {
                    validation.mean_nees = 0.0;
                    validation.mean_logdet = 0.0;
                    validation.delta = 0.0;
                }

                g_joint_norm = std::sqrt(g_joint_norm_sq_sum / group_count);
                q_joint_norm = std::sqrt(q_joint_norm_sq_sum / group_count);
                if (have_cov_diag)
                {
                    pv_min_eig = pv_min_eig_acc / group_count;
                    pv_max_eig = pv_max_eig_acc / group_count;
                    pv_condition = pv_condition_acc / group_count;
                }
            }
            else
            {
                const MarginalCacheData marginal_cache = buildMarginalCache(solution.H, validation_targets);
                validation = computeValidationLossWithMarginals(validation_targets, solved_nodes, marginal_cache);
                adjoint_rhs = computeAdjointRhsWithMarginals(solution.H, validation_targets, validation, marginal_cache);
                quadratic_rhs = computeQuadraticRhsWithMarginals(solution.H, validation_targets, validation, marginal_cache);

                constexpr int node_dim = 18;
                std::unordered_map<int, int> validation_node_multiplicity;
                std::unordered_map<int, ObservedMask18> validation_node_observed_masks;
                logdet_node_indices.reserve(validation_targets.size());
                for (const auto &target : validation_targets)
                {
                    if (target.node_index < 0 || (target.node_index + 1) * node_dim > solution.H.rows())
                    {
                        throw std::runtime_error("Optimizer::step: validation target node index out of range for marginal log-det accumulation.");
                    }

                    auto [it, inserted] = validation_node_multiplicity.emplace(target.node_index, 1);
                    if (inserted)
                    {
                        logdet_node_indices.push_back(target.node_index);
                        validation_node_observed_masks.emplace(target.node_index, target.observed_mask);
                    }
                    else
                    {
                        ++(it->second);
                        validation_node_observed_masks[target.node_index] =
                            validation_node_observed_masks[target.node_index].cwiseMax(target.observed_mask);
                    }
                }

                const int unique_validation_count = static_cast<int>(logdet_node_indices.size());
                logdet_information = Eigen::MatrixXd::Zero(node_dim * unique_validation_count, node_dim * unique_validation_count);
                for (int i = 0; i < unique_validation_count; ++i)
                {
                    const int node_index = logdet_node_indices[static_cast<std::size_t>(i)];
                    const Matrix18d &p_cov_block = getCachedNodeCovarianceBlock(marginal_cache, node_index);
                    const Matrix18d p_inv_block = buildObservedInformationFromCovariance(
                        p_cov_block,
                        validation_node_observed_masks[node_index]);
                    const double multiplicity = static_cast<double>(validation_node_multiplicity[node_index]);
                    logdet_information.block(i * node_dim, i * node_dim, node_dim, node_dim) = multiplicity * p_inv_block;
                }
            }

            result.loss = validation.loss;
            result.iterations = iter + 1;
            result.psi = current_psi;
            result.lambda = lambda;
            result.theta = theta;
            if (!optimizer_config.parameter_checkpoint_path.empty())
            {
                Json::Value checkpoint;
                checkpoint["parameter_layout_version"] = 2;
                checkpoint["parameter_layout"] = "P0(18),Q1(6),Q2(6),Q3(6),Q4(6)";
                checkpoint["iterations"] = result.iterations;
                checkpoint["loss"] = result.loss;
                checkpoint["converged"] = false;
                for (int index = 0; index < kThetaSize; ++index)
                {
                    checkpoint["theta"].append(theta(index));
                    checkpoint["lambda"].append(lambda(index));
                }
                const auto temporary_path = optimizer_config.parameter_checkpoint_path + ".tmp";
                std::ofstream checkpoint_stream(temporary_path);
                if (!checkpoint_stream) throw std::runtime_error("Cannot write parameter checkpoint.");
                writeReportJson(checkpoint_stream, checkpoint);
                checkpoint_stream.close();
                if (!checkpoint_stream) throw std::runtime_error("Failed to write parameter checkpoint.");
                std::filesystem::rename(temporary_path, optimizer_config.parameter_checkpoint_path);
            }

            const auto t_adjoint_solve_start = std::chrono::steady_clock::now();
            const Eigen::VectorXd adjoint_lambda = adjoint_solver.solve(adjoint_rhs);
            const auto t_adjoint_solve_end = std::chrono::steady_clock::now();
            adjoint_solves_ms += std::chrono::duration<double, std::milli>(
                t_adjoint_solve_end - t_adjoint_solve_start).count();
            if (adjoint_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Optimizer::step failed to solve the adjoint system.");
            }
            const auto t_quadratic_solve_start = std::chrono::steady_clock::now();
            const Eigen::VectorXd quadratic_state = adjoint_solver.solve(quadratic_rhs);
            const auto t_quadratic_solve_end = std::chrono::steady_clock::now();
            adjoint_solves_ms += std::chrono::duration<double, std::milli>(
                t_quadratic_solve_end - t_quadratic_solve_start).count();
            if (adjoint_solver.info() != Eigen::Success)
            {
                throw std::runtime_error("Optimizer::step failed to solve the quadratic sensitivity system.");
            }

            constexpr int node_dim = 18;
            const int logdet_block_count = static_cast<int>(logdet_node_indices.size());
            Eigen::MatrixXd logdet_projection;
            if (!use_joint_objective)
            {
                Eigen::MatrixXd rhs_logdet = Eigen::MatrixXd::Zero(solution.H.rows(), node_dim * logdet_block_count);
                for (int u = 0; u < logdet_block_count; ++u)
                {
                    const int node_index = logdet_node_indices[static_cast<std::size_t>(u)];
                    rhs_logdet.block(node_index * node_dim, u * node_dim, node_dim, node_dim).setIdentity();
                }
                const auto t_logdet_projection_solve_start = std::chrono::steady_clock::now();
                logdet_projection = adjoint_solver.solve(rhs_logdet);
                const auto t_logdet_projection_solve_end = std::chrono::steady_clock::now();
                adjoint_solves_ms += std::chrono::duration<double, std::milli>(
                    t_logdet_projection_solve_end - t_logdet_projection_solve_start).count();
                if (adjoint_solver.info() != Eigen::Success)
                {
                    throw std::runtime_error("Optimizer::step failed to solve for the log-det sensitivity projection.");
                }
            }

            const std::size_t unary_count = static_cast<std::size_t>(topology.K);
            const std::size_t binary_space_count = static_cast<std::size_t>(topology.K) * static_cast<std::size_t>(std::max<unsigned int>(1u, topology.N) - 1u);
            const std::size_t binary_time_count = topology.use_1D_estimator ? 0u : static_cast<std::size_t>(std::max<unsigned int>(1u, topology.K) - 1u) * static_cast<std::size_t>(topology.N);
            const std::size_t built_in_count = unary_count + binary_space_count + binary_time_count;

            if (linearizations.size() < built_in_count)
            {
                throw std::runtime_error("Optimizer::step: lower-level linearizations do not include all built-in factors.");
            }

            for (std::size_t i = 0; i < validation_targets.size(); ++i)
            {
                const auto &target = validation_targets[i];
                const int node_index = target.node_index;
                if (node_index < 0 || node_index >= static_cast<int>(solved_nodes.size()))
                {
                    throw std::runtime_error("Optimizer::step: validation target node index out of range.");
                }
            }

            Eigen::VectorXd gradient_dL_dtheta = direct_validation_gradient;
            FactorKernelSample unary_sample;
            FactorKernelSample time_sample;
            FactorKernelSample space_sample;
            const auto t_factor_gradient_loop_start = std::chrono::steady_clock::now();

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
                Eigen::VectorXd adjoint_lambda_f = Eigen::VectorXd::Zero(static_cast<int>(lin.node_indices.size()) * 18);
                Eigen::VectorXd e_nodes = Eigen::VectorXd::Zero(static_cast<int>(lin.node_indices.size()) * 18);
                const int local_dim = static_cast<int>(lin.node_indices.size()) * 18;
                Matrix18d A_logdet = Matrix18d::Zero();

                for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                {
                    const int node_index = lin.node_indices[j];
                    if (node_index < 0 || node_index >= static_cast<int>(solved_nodes.size()))
                    {
                        throw std::runtime_error("Optimizer::step: built-in factor node index out of range.");
                    }

                    local_nodes.push_back(solved_nodes[static_cast<std::size_t>(node_index)]);
                    adjoint_lambda_f.segment<18>(static_cast<int>(j) * 18) = adjoint_lambda.segment<18>(node_index * 18);
                    e_nodes.segment<18>(static_cast<int>(j) * 18) = quadratic_state.segment<18>(node_index * 18);
                }

                std::vector<Eigen::VectorXd> joint_group_e_nodes;
                if (use_joint_objective)
                {
                    joint_group_e_nodes.reserve(joint_group_logdet_data.size());
                    for (const auto& group_data : joint_group_logdet_data)
                    {
                        Eigen::VectorXd group_e_nodes = Eigen::VectorXd::Zero(local_dim);
                        for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                        {
                            const int node_index = lin.node_indices[j];
                            group_e_nodes.segment<18>(static_cast<int>(j) * 18) =
                                group_data.quadratic_state.segment<18>(node_index * 18);
                        }
                        joint_group_e_nodes.push_back(std::move(group_e_nodes));
                    }
                }

                Eigen::MatrixXd Z_f = Eigen::MatrixXd::Zero(local_dim, local_dim);
                if (use_joint_objective)
                {
                    if (shared_joint_cache.full_state_observation)
                    {
                        Eigen::MatrixXd rhs_local = Eigen::MatrixXd::Zero(solution.H.rows(), local_dim);
                        for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                        {
                            const int node_index = lin.node_indices[j];
                            rhs_local.block(node_index * node_dim, static_cast<int>(j) * node_dim, node_dim, node_dim).setIdentity();
                        }
                        const Eigen::MatrixXd local_projection = adjoint_solver.solve(rhs_local);
                        if (adjoint_solver.info() != Eigen::Success)
                        {
                            throw std::runtime_error("Optimizer::step failed to solve full-state local covariance projection.");
                        }
                        for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                        {
                            const int node_index = lin.node_indices[j];
                            Z_f.block(static_cast<int>(j) * node_dim, 0, node_dim, local_dim) =
                                local_projection.block(node_index * node_dim, 0, node_dim, local_dim);
                        }
                    }
                    else
                    {
                        const int observed_dim = static_cast<int>(shared_joint_cache.observed_stacked_indices.size());
                        Eigen::MatrixXd G_local_group = Eigen::MatrixXd::Zero(local_dim, observed_dim);
                        Eigen::MatrixXd weighted_local_group(local_dim, observed_dim);
                        for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                        {
                            const int node_index = lin.node_indices[j];
                            G_local_group.block(static_cast<int>(j) * node_dim, 0, node_dim, observed_dim) =
                                shared_joint_logdet_projection.block(node_index * node_dim, 0, node_dim, observed_dim);
                            weighted_local_group.block(static_cast<int>(j) * node_dim, 0, node_dim, observed_dim) =
                                shared_joint_weighted_projection.block(node_index * node_dim, 0, node_dim, observed_dim);
                        }
                        Z_f = weighted_local_group * G_local_group.transpose();
                    }
                }
                else
                {
                    Eigen::MatrixXd G_local = Eigen::MatrixXd::Zero(local_dim, node_dim * logdet_block_count);
                    for (int u = 0; u < logdet_block_count; ++u)
                    {
                        for (std::size_t j = 0; j < lin.node_indices.size(); ++j)
                        {
                            const int node_index = lin.node_indices[j];
                            G_local.block(static_cast<int>(j) * node_dim, u * node_dim, node_dim, node_dim) =
                                logdet_projection.block(node_index * node_dim, u * node_dim, node_dim, node_dim);
                        }
                    }
                    Z_f = G_local * logdet_information * G_local.transpose();
                }

                if (lin.E.rows() != 18 || lin.Q.rows() != 18 || lin.Q.cols() != 18)
                {
                    throw std::runtime_error("Optimizer::step: built-in factor log-det accumulation expects 18x18 residual blocks.");
                }
                A_logdet = (lin.Q * (lin.E * Z_f * lin.E.transpose()) * lin.Q).template cast<double>();

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
                        unary_sample.adjoint_lambda_f = adjoint_lambda_f;
                        unary_sample.e_nodes = e_nodes;
                        unary_sample.info_scale = kNllQuadraticScale;
                    }
                    const Eigen::VectorXd zero_e_nodes = Eigen::VectorXd::Zero(18);
                    const FactorGradientContrib state_contrib = computeDiagonalFactorGradient(
                        lin.e,
                        lin.Q,
                        lin.E,
                        adjoint_lambda_f,
                        zero_e_nodes,
                        kNllQuadraticScale);
                    Eigen::VectorXd info_gradient = Eigen::VectorXd::Zero(18);
                    if (use_joint_objective)
                    {
                        for (const Eigen::VectorXd& group_e_nodes : joint_group_e_nodes)
                        {
                            info_gradient += computeDiagonalFactorGradient(
                                lin.e, lin.Q, lin.E, adjoint_lambda_f, group_e_nodes, kNllQuadraticScale).dL_dtheta_info;
                        }
                        info_gradient /= static_cast<double>(joint_group_e_nodes.size());
                    }
                    else
                    {
                        info_gradient = computeDiagonalFactorGradient(
                            lin.e, lin.Q, lin.E, adjoint_lambda_f, e_nodes, kNllQuadraticScale).dL_dtheta_info;
                    }

                    gradient_dL_dtheta.segment<18>(0) += state_contrib.dL_dtheta_state + info_gradient;
                    gradient_dL_dtheta.segment<18>(0) += computeUnaryLogDetGradientFromTrace(A_logdet);
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
                        space_sample.adjoint_lambda_f = adjoint_lambda_f;
                        space_sample.e_nodes = e_nodes;
                        space_sample.info_scale = kNllQuadraticScale;
                    }
                    const Eigen::VectorXd zero_e_nodes = Eigen::VectorXd::Zero(local_dim);
                    const FactorGradientContrib state_contrib = computeBinarySpaceFactorGradient(
                        local_nodes,
                        lin.e,
                        lin.Q,
                        lin.E,
                        adjoint_lambda_f,
                        zero_e_nodes,
                        kNllQuadraticScale);
                    Eigen::VectorXd info_gradient = Eigen::VectorXd::Zero(12);
                    if (use_joint_objective)
                    {
                        for (const Eigen::VectorXd& group_e_nodes : joint_group_e_nodes)
                        {
                            info_gradient += computeBinarySpaceFactorGradient(
                                local_nodes, lin.e, lin.Q, lin.E, adjoint_lambda_f,
                                group_e_nodes, kNllQuadraticScale).dL_dtheta_info;
                        }
                        info_gradient /= static_cast<double>(joint_group_e_nodes.size());
                    }
                    else
                    {
                        info_gradient = computeBinarySpaceFactorGradient(
                            local_nodes, lin.e, lin.Q, lin.E, adjoint_lambda_f,
                            e_nodes, kNllQuadraticScale).dL_dtheta_info;
                    }

                    gradient_dL_dtheta.segment<6>(24) += state_contrib.dL_dtheta_state.segment<6>(0) + info_gradient.segment<6>(0); // Q2
                    gradient_dL_dtheta.segment<6>(kQ4Offset) += state_contrib.dL_dtheta_state.segment<6>(6) + info_gradient.segment<6>(6);
                    const Eigen::VectorXd logdet_contrib = computeBinarySpaceLogDetGradientFromTrace(local_nodes, A_logdet);
                    gradient_dL_dtheta.segment<6>(24) += logdet_contrib.segment<6>(0);
                    gradient_dL_dtheta.segment<6>(kQ4Offset) += logdet_contrib.segment<6>(6);
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
                        time_sample.adjoint_lambda_f = adjoint_lambda_f;
                        time_sample.e_nodes = e_nodes;
                        time_sample.info_scale = kNllQuadraticScale;
                    }

                    const Eigen::VectorXd zero_e_nodes = Eigen::VectorXd::Zero(local_dim);
                    const FactorGradientContrib state_contrib = computeBinaryTimeFactorGradient(
                        local_nodes,
                        lin.e,
                        lin.Q,
                        lin.E,
                        adjoint_lambda_f,
                        zero_e_nodes,
                        kNllQuadraticScale);
                    Eigen::VectorXd info_gradient = Eigen::VectorXd::Zero(12);
                    if (use_joint_objective)
                    {
                        for (const Eigen::VectorXd& group_e_nodes : joint_group_e_nodes)
                        {
                            info_gradient += computeBinaryTimeFactorGradient(
                                local_nodes, lin.e, lin.Q, lin.E, adjoint_lambda_f,
                                group_e_nodes, kNllQuadraticScale).dL_dtheta_info;
                        }
                        info_gradient /= static_cast<double>(joint_group_e_nodes.size());
                    }
                    else
                    {
                        info_gradient = computeBinaryTimeFactorGradient(
                            local_nodes, lin.e, lin.Q, lin.E, adjoint_lambda_f,
                            e_nodes, kNllQuadraticScale).dL_dtheta_info;
                    }

                    gradient_dL_dtheta.segment<6>(18) += state_contrib.dL_dtheta_state.segment<6>(0) + info_gradient.segment<6>(0); // Q1
                    gradient_dL_dtheta.segment<6>(30) += state_contrib.dL_dtheta_state.segment<6>(6) + info_gradient.segment<6>(6); // Q3
                    const Eigen::VectorXd logdet_contrib = computeBinaryTimeLogDetGradientFromTrace(local_nodes, A_logdet);
                    gradient_dL_dtheta.segment<6>(18) += logdet_contrib.segment<6>(0);
                    gradient_dL_dtheta.segment<6>(30) += logdet_contrib.segment<6>(6);
                    continue;
                }

                throw std::runtime_error("Optimizer::step: could not classify built-in binary factor as space or time from node coordinates.");
            }
            const auto t_factor_gradient_loop_end = std::chrono::steady_clock::now();
            factor_gradient_loop_ms = std::chrono::duration<double, std::milli>(
                t_factor_gradient_loop_end - t_factor_gradient_loop_start).count();

            if (optimizer_config.enable_gradient_fd_check && iter == 0 && scenario_index == 0)
            {
                runVarianceFiniteDifferenceChecks(
                    theta,
                    unary_sample,
                    time_sample,
                    space_sample,
                    optimizer_config.gradient_fd_epsilon,
                    optimizer_config.gradient_fd_use_lambda_perturbation,
                    optimizer_config.enable_unary_factor_fd_check,
                    optimizer_config.enable_time_factor_fd_check,
                    optimizer_config.enable_space_factor_fd_check);
            }
                variance_gradient_sum += gradient_dL_dtheta;
                if (solution.tip_metrics.translation_count > 0)
                {
                    tip_metric_sums.translation_mae += solution.tip_metrics.translation_mae;
                    tip_metric_sums.translation_rmse += solution.tip_metrics.translation_rmse;
                    ++tip_metric_sums.translation_count;
                }
                if (solution.tip_metrics.rotation_count > 0)
                {
                    tip_metric_sums.rotation_mae += solution.tip_metrics.rotation_mae;
                    tip_metric_sums.rotation_rmse += solution.tip_metrics.rotation_rmse;
                    ++tip_metric_sums.rotation_count;
                }
                if (solution.body_metrics.translation_count > 0)
                {
                    body_metric_sums.translation_mae += solution.body_metrics.translation_mae;
                    body_metric_sums.translation_rmse += solution.body_metrics.translation_rmse;
                    ++body_metric_sums.translation_count;
                }
                if (solution.body_metrics.rotation_count > 0)
                {
                    body_metric_sums.rotation_mae += solution.body_metrics.rotation_mae;
                    body_metric_sums.rotation_rmse += solution.body_metrics.rotation_rmse;
                    ++body_metric_sums.rotation_count;
                }
                averaged_validation.loss += validation.loss;
                averaged_validation.mean_nees += validation.mean_nees;
                averaged_validation.mean_logdet += validation.mean_logdet;
                averaged_validation.mahalanobis_sum += validation.mahalanobis_sum;
                averaged_validation.logdet_sum += validation.logdet_sum;
                averaged_validation.delta += validation.delta;
                g_joint_norm_sum += g_joint_norm;
                q_joint_norm_sum += q_joint_norm;
                adjoint_rhs_norm_sum += adjoint_rhs.norm();
                quadratic_rhs_norm_sum += quadratic_rhs.norm();
                adjoint_state_norm_sum += adjoint_lambda.norm();
                quadratic_state_norm_sum += quadratic_state.norm();
                pv_min_eig_sum += pv_min_eig;
                pv_max_eig_sum += pv_max_eig;
                pv_condition_sum += pv_condition;
                joint_cache_ms_sum += joint_cache_ms;
                joint_setup_ms_sum += shared_joint_cache.setup_ms;
                joint_selector_ms_sum += shared_joint_cache.selector_ms;
                joint_projection_solve_ms_sum += shared_joint_cache.projection_solve_ms;
                joint_covariance_ms_sum += shared_joint_cache.covariance_ms;
                joint_eigendecomposition_ms_sum += shared_joint_cache.eigendecomposition_ms;
                joint_information_ms_sum += shared_joint_cache.information_ms;
                adjoint_solves_ms_sum += adjoint_solves_ms;
                factor_gradient_loop_ms_sum += factor_gradient_loop_ms;
            }

            const double scenario_count = static_cast<double>(scenario_problems_.size());
            const double tip_mae = tip_metric_sums.translation_count > 0
                ? tip_metric_sums.translation_mae / tip_metric_sums.translation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double tip_rmse = tip_metric_sums.translation_count > 0
                ? tip_metric_sums.translation_rmse / tip_metric_sums.translation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double tip_rotation_mae = tip_metric_sums.rotation_count > 0
                ? tip_metric_sums.rotation_mae / tip_metric_sums.rotation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double tip_rotation_rmse = tip_metric_sums.rotation_count > 0
                ? tip_metric_sums.rotation_rmse / tip_metric_sums.rotation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double body_mae = body_metric_sums.translation_count > 0
                ? body_metric_sums.translation_mae / body_metric_sums.translation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double body_rmse = body_metric_sums.translation_count > 0
                ? body_metric_sums.translation_rmse / body_metric_sums.translation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double body_rotation_mae = body_metric_sums.rotation_count > 0
                ? body_metric_sums.rotation_mae / body_metric_sums.rotation_count
                : std::numeric_limits<double>::quiet_NaN();
            const double body_rotation_rmse = body_metric_sums.rotation_count > 0
                ? body_metric_sums.rotation_rmse / body_metric_sums.rotation_count
                : std::numeric_limits<double>::quiet_NaN();
            const Eigen::VectorXd variance_gradient = variance_gradient_sum / scenario_count;
            averaged_validation.loss /= scenario_count;
            averaged_validation.mean_nees /= scenario_count;
            averaged_validation.mean_logdet /= scenario_count;
            averaged_validation.mahalanobis_sum /= scenario_count;
            averaged_validation.logdet_sum /= scenario_count;
            averaged_validation.delta /= scenario_count;
            const ValidationLossData& validation = averaged_validation;
            result.loss = validation.loss;
            result.iterations = iter + 1;
            result.psi = current_psi;
            result.lambda = lambda;
            result.theta = theta;

            Eigen::VectorXd dL_dlambda = -variance_gradient.array() / lambda.array().square();
            Eigen::VectorXd dL_dpsi = optimizer_config.use_exponential_param
                ? (dL_dlambda.array() * lambda.array()).matrix()
                : dL_dlambda;
            for (int index = 0; index < kThetaSize; ++index)
            {
                if (update_mask[index] == 0)
                {
                    dL_dlambda(index) = 0.0;
                    dL_dpsi(index) = 0.0;
                }
            }

            const double grad_norm = dL_dpsi.norm();
            const double loss_change = std::abs(validation.loss - previous_loss);
            const double g_joint_norm = g_joint_norm_sum / scenario_count;
            const double q_joint_norm = q_joint_norm_sum / scenario_count;
            const double adjoint_rhs_norm = adjoint_rhs_norm_sum / scenario_count;
            const double quadratic_rhs_norm = quadratic_rhs_norm_sum / scenario_count;
            const double adjoint_state_norm = adjoint_state_norm_sum / scenario_count;
            const double quadratic_state_norm = quadratic_state_norm_sum / scenario_count;
            const double pv_min_eig = pv_min_eig_sum / scenario_count;
            const double pv_max_eig = pv_max_eig_sum / scenario_count;
            const double pv_condition = pv_condition_sum / scenario_count;
            const double lower_solve_ms = lower_solve_ms_sum / scenario_count;
            const double extract_hessian_ms = extract_hessian_ms_sum / scenario_count;
            const double joint_cache_ms = joint_cache_ms_sum / scenario_count;
            const double adjoint_solves_ms = adjoint_solves_ms_sum / scenario_count;
            const double factor_gradient_loop_ms = factor_gradient_loop_ms_sum / scenario_count;
            const std::string objective_mode = optimizer_config.use_joint_validation_objective ? "joint" : "marginal";
            if (optimizer_config.verbose)
            {
                std::cout << "Outer iteration " << iter
                          << ": loss=" << validation.loss
                          << ", mean_mahalanobis=" << validation.mean_nees
                          << ", mean_logdet=" << validation.mean_logdet
                          << ", tip_mae_m=" << tip_mae
                          << ", tip_rmse_m=" << tip_rmse
                          << ", tip_rotation_mae_rad=" << tip_rotation_mae
                          << ", tip_rotation_rmse_rad=" << tip_rotation_rmse
                          << ", body_mae_m=" << body_mae
                          << ", body_rmse_m=" << body_rmse
                          << ", body_rotation_mae_rad=" << body_rotation_mae
                          << ", body_rotation_rmse_rad=" << body_rotation_rmse
                          << ", grad_norm=" << grad_norm
                          << ", minimum_learning_rate=" << effective_learning_rates.minCoeff()
                          << ", lambda_min=" << lambda.minCoeff()
                          << ", lambda_max=" << lambda.maxCoeff()
                          << ", theta_min=" << theta.minCoeff()
                          << ", theta_max=" << theta.maxCoeff()
                          << ", objective_mode=" << objective_mode
                          << std::endl;
                std::cout << "Timing breakdown (ms): lower_solve_ms=" << lower_solve_ms
                          << ", extract_hessian_ms=" << extract_hessian_ms
                          << ", joint_cache_ms=" << joint_cache_ms
                          << ", adjoint_solves_ms=" << adjoint_solves_ms
                          << ", factor_gradient_loop_ms=" << factor_gradient_loop_ms
                          << std::endl;
                if (optimizer_config.use_joint_validation_objective)
                {
                    const double joint_phase_ms_sum = joint_setup_ms_sum + joint_selector_ms_sum +
                        joint_projection_solve_ms_sum + joint_covariance_ms_sum +
                        joint_eigendecomposition_ms_sum + joint_information_ms_sum;
                    std::cout << "Joint cache timing breakdown (ms, per-scenario mean): setup_ms="
                              << joint_setup_ms_sum / scenario_count
                              << ", selector_ms=" << joint_selector_ms_sum / scenario_count
                              << ", projection_solve_ms=" << joint_projection_solve_ms_sum / scenario_count
                              << ", covariance_ms=" << joint_covariance_ms_sum / scenario_count
                              << ", eigendecomposition_ms=" << joint_eigendecomposition_ms_sum / scenario_count
                              << ", information_ms=" << joint_information_ms_sum / scenario_count
                              << ", other_ms=" << (joint_cache_ms_sum - joint_phase_ms_sum) / scenario_count
                              << std::endl;
                }
                if (optimizer_config.log_joint_diagnostics)
                {
                    std::cout << "Path diagnostics: g_joint_norm=" << g_joint_norm
                              << ", q_joint_norm=" << q_joint_norm
                              << ", adjoint_rhs_norm=" << adjoint_rhs_norm
                              << ", quadratic_rhs_norm=" << quadratic_rhs_norm
                              << ", adjoint_state_norm=" << adjoint_state_norm
                              << ", quadratic_state_norm=" << quadratic_state_norm
                              << ", pv_min_eig=" << pv_min_eig
                              << ", pv_max_eig=" << pv_max_eig
                              << ", pv_condition=" << pv_condition
                              << std::endl;
                }
            }

            outer_loop_log << iter;
            for (int i = 0; i < lambda.size(); ++i)
            {
                outer_loop_log << ',' << lambda[i];
            }
            for (int i = 0; i < dL_dlambda.size(); ++i)
            {
                outer_loop_log << ',' << dL_dlambda[i];
            }
            outer_loop_log << ',' << validation.mean_nees << ',' << validation.mean_logdet << ',' << validation.loss
                           << ',' << tip_mae << ',' << tip_rmse << ',' << tip_rotation_mae << ',' << tip_rotation_rmse
                           << ',' << body_mae << ',' << body_rmse << ',' << body_rotation_mae << ',' << body_rotation_rmse
                           << ',' << effective_learning_rates.minCoeff() << ',' << oscillation_reductions;
            for (int index = 0; index < kThetaSize; ++index)
            {
                outer_loop_log << ',' << effective_learning_rates(index) << ',' << dimension_reductions[index];
            }
            outer_loop_log << '\n';
            if (grad_norm < optimizer_config.tol_grad ||
                (iter > 0 && loss_change < optimizer_config.tol_loss))
            {
                result.converged = true;
                break;
            }
            previous_loss = validation.loss;

            Eigen::VectorXd update_direction = dL_dpsi;
            if (optimizer_config.use_adam)
            {
                adam_m = optimizer_config.adam_beta1 * adam_m + (1.0 - optimizer_config.adam_beta1) * dL_dpsi;
                adam_v = optimizer_config.adam_beta2 * adam_v + (1.0 - optimizer_config.adam_beta2) * dL_dpsi.array().square().matrix();
                beta1_power *= optimizer_config.adam_beta1;
                beta2_power *= optimizer_config.adam_beta2;

                const double inv_one_minus_beta1_power = 1.0 / (1.0 - beta1_power);
                const double inv_one_minus_beta2_power = 1.0 / (1.0 - beta2_power);
                const Eigen::VectorXd m_hat = adam_m * inv_one_minus_beta1_power;
                const Eigen::VectorXd v_hat = adam_v * inv_one_minus_beta2_power;
                update_direction = m_hat.array() / (v_hat.array().sqrt() + optimizer_config.adam_epsilon);
            }

            const double update_dir_norm = update_direction.norm();
            if (std::isfinite(update_dir_norm) && optimizer_config.max_gradient_update_norm > 0.0 &&
                update_dir_norm > optimizer_config.max_gradient_update_norm)
            {
                const double scale = optimizer_config.max_gradient_update_norm / update_dir_norm;
                update_direction *= scale;
                if (optimizer_config.verbose)
                {
                    std::cout << "Clipped update direction norm from " << update_dir_norm
                              << " to " << optimizer_config.max_gradient_update_norm << std::endl;
                }
            }

            Eigen::VectorXd delta_psi = -(effective_learning_rates.array() * update_direction.array()).matrix();
            const double delta_psi_norm = delta_psi.norm();
            if (std::isfinite(delta_psi_norm) && optimizer_config.max_psi_step_norm > 0.0 &&
                delta_psi_norm > optimizer_config.max_psi_step_norm)
            {
                const double scale = optimizer_config.max_psi_step_norm / delta_psi_norm;
                delta_psi *= scale;
                if (optimizer_config.verbose)
                {
                    std::cout << "Clipped psi step norm from " << delta_psi_norm
                              << " to " << optimizer_config.max_psi_step_norm << std::endl;
                }
            }

            const Eigen::VectorXd previous_psi = current_psi;
            current_psi += delta_psi;
            const Eigen::VectorXd requested_psi = current_psi;

            if (optimizer_config.use_exponential_param)
            {
                const double min_lambda = std::max(1.0 / std::max(optimizer_config.max_theta, optimizer_config.min_theta), 1e-12);
                const double max_lambda = std::max(1.0 / std::max(optimizer_config.min_theta, 1e-12), min_lambda);
                const double min_psi = std::log(min_lambda);
                const double max_psi = std::log(max_lambda);
                current_psi = current_psi.array().max(min_psi).min(max_psi).matrix();
            }
            for (int index = 0; index < kThetaSize; ++index)
            {
                if (update_mask[index] == 0)
                {
                    current_psi(index) = initial_psi(index);
                }
            }
            if (optimizer_config.enable_oscillation_detection)
            {
                std::vector<int> eligible = update_mask;
                for (int index = 0; index < kThetaSize; ++index)
                {
                    if (current_psi(index) != requested_psi(index))
                    {
                        eligible[index] = 0;
                    }
                }
                const auto detected_dimensions = oscillation_detector.observe(current_psi - previous_psi, eligible);
                for (const int dimension : detected_dimensions)
                {
                    const double previous_rate = effective_learning_rates(dimension);
                    effective_learning_rates(dimension) = oscillation_detector.reducedLearningRate(
                        dimension, previous_rate, std::abs(update_direction(dimension)));
                    ++oscillation_reductions;
                    ++dimension_reductions[dimension];
                    std::cout << "Persistent step oscillation at outer iteration " << iter
                              << ": dimension=" << dimension << ", learning_rate=" << previous_rate
                              << " -> " << effective_learning_rates(dimension)
                              << ", cooldown_updates=15" << std::endl;
                }
            }
        }

        result.psi = current_psi;
        result.lambda = mapLambda(current_psi);
        result.theta = mapVariance(result.lambda);
        problem_->setTheta(result.theta);

        const auto &summary = problem_->runSummary();
        Eigen::VectorXd validation_precision = summary.validation_variance;
        for (int i = 0; i < validation_precision.size(); ++i)
        {
            const double v = summary.validation_variance(i);
            validation_precision(i) = (v > 0.0) ? (1.0 / v) : 0.0;
        }
        outer_loop_log << "\nrun_summary\n";
        outer_loop_log << "N,K,tol_grad,tol_loss,learning_rate,R_pose,R_gyro,validation_sample_count,validation_precision,objective_mode,converged,enable_oscillation_detection,final_learning_rate,oscillation_reductions,final_learning_rates,dimension_reductions\n";
        outer_loop_log << problem_->robotTopology().N << ','
                       << problem_->robotTopology().K << ','
                       << optimizer_config.tol_grad << ','
                       << optimizer_config.tol_loss << ','
                       << optimizer_config.learning_rate << ','
                       << '"' << formatVector(summary.R_pose) << '"' << ','
                       << '"' << formatVector(summary.R_gyro) << '"' << ','
                       << summary.validation_sample_count << ','
                       << '"' << formatVector(validation_precision) << '"' << ','
                   << (optimizer_config.use_joint_validation_objective ? "joint" : "marginal") << ','
                       << (result.converged ? "true" : "false") << ','
                       << (optimizer_config.enable_oscillation_detection ? "true" : "false") << ','
                       << effective_learning_rates.minCoeff() << ',' << oscillation_reductions << ','
                       << '"' << formatVector(effective_learning_rates) << '"' << ',' << '"';
        for (int index = 0; index < kThetaSize; ++index)
        {
            if (index > 0) outer_loop_log << ',';
            outer_loop_log << dimension_reductions[index];
        }
        outer_loop_log << '"' << '\n';
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

int TestLower(bool physical_data)
{
    std::cout << "Lower-level optimizer test" << std::endl;
    const TestProblemData data = buildMinimalTestProblemData(physical_data);

    if (physical_data)
    {
        for (std::size_t scenario_index = 0; scenario_index < data.measurement_scenarios.size(); ++scenario_index)
        {
            const auto& scenario = data.measurement_scenarios[scenario_index];
            spacetime::OptimizationProblem scenario_problem;
            scenario_problem.setEstimatorOptions(data.estimator_options);
            scenario_problem.setRobotTopology(scenario.robot_topology);
            scenario_problem.setTheta(data.initial_theta);
            scenario_problem.setProblemData(scenario.nodes, scenario.factors, scenario.factor_node_indices);
            const auto scenario_result = scenario_problem.solveLowerLevel(false);
            std::cout << "Physical scenario " << scenario_index
                      << " solved: H=" << scenario_result.H.rows() << 'x' << scenario_result.H.cols()
                      << ", dx_norm=" << scenario_result.x.norm()
                      << ", linearizations=" << scenario_result.linearizations.size()
                      << std::endl;
        }
        return 0;
    }

    spacetime::OptimizationProblem problem;
    problem.setEstimatorOptions(data.estimator_options);
    problem.setRobotTopology(data.robot_topology);
    problem.setRunSummary({data.validation_sample_count, data.R_pose, data.R_gyro, data.validation_variance});
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
    bool run_lower = false;
    bool physical_data = false;
    bool inspect_data = false;
    int max_outer_iterations_override = -1;
    int max_scenarios_override = -1;
    std::string evaluation_parameters_path;
    for (int arg_index = 1; arg_index < argc; ++arg_index)
    {
        const std::string mode = argv[arg_index];
        if (mode == "--lower")
        {
            run_lower = true;
        }
        else if (mode == "--physical")
        {
            physical_data = true;
        }
        else if (mode == "--inspect")
        {
            inspect_data = true;
        }
        else if (mode == "--max-outer-iterations")
        {
            if (arg_index + 1 >= argc)
            {
                throw std::runtime_error("--max-outer-iterations requires a positive integer value.");
            }
            max_outer_iterations_override = std::stoi(argv[++arg_index]);
            if (max_outer_iterations_override <= 0)
            {
                throw std::runtime_error("--max-outer-iterations requires a positive integer value.");
            }
        }
        else if (mode == "--max-scenarios")
        {
            if (arg_index + 1 >= argc)
            {
                throw std::runtime_error("--max-scenarios requires a positive integer value.");
            }
            max_scenarios_override = std::stoi(argv[++arg_index]);
            if (max_scenarios_override <= 0)
            {
                throw std::runtime_error("--max-scenarios requires a positive integer value.");
            }
        }
        else if (mode == "--evaluate-parameters")
        {
            if (arg_index + 1 >= argc)
            {
                throw std::runtime_error("--evaluate-parameters requires a saved parameter JSON path.");
            }
            evaluation_parameters_path = argv[++arg_index];
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + mode + ". Supported arguments are --lower, --physical, --inspect, --max-outer-iterations N, --max-scenarios N, and --evaluate-parameters FILE.");
        }
    }
    if (run_lower)
    {
        return TestLower(physical_data);
    }

    std::cout << "Upper-level optimizer test" << std::endl;

    const std::filesystem::path bilevel_config_path = resolveExistingPath({
        "assets/config/bilevel_optimizer/bilevel_config.json",
        "../assets/config/bilevel_optimizer/bilevel_config.json",
        "../../assets/config/bilevel_optimizer/bilevel_config.json"
    });
    const Json::Value bilevel_config = loadJsonFile(bilevel_config_path);
    std::vector<std::string> training_trials;
    if (physical_data && bilevel_config.isMember("physical_training_trials"))
    {
        const auto& trials = bilevel_config["physical_training_trials"];
        if (!trials.isArray() || trials.empty())
        {
            throw std::runtime_error("physical_training_trials must be a nonempty array of trial config paths.");
        }
        for (const auto& trial : trials)
        {
            if (!trial.isString() || trial.asString().empty())
            {
                throw std::runtime_error("physical_training_trials entries must be nonempty strings.");
            }
            training_trials.push_back(trial.asString());
        }
    }
    const double common_window_duration = bilevel_config.get("physical_window_duration", -1.0).asDouble();
    const int common_window_count = bilevel_config.get("measurement_sample_count", -1).asInt();
    std::vector<std::string> testing_trials;
    if (physical_data && bilevel_config.isMember("physical_testing_trials"))
    {
        const auto& trials = bilevel_config["physical_testing_trials"];
        if (!trials.isArray()) throw std::runtime_error("physical_testing_trials must be an array.");
        for (const auto& trial : trials)
        {
            if (!trial.isString() || trial.asString().empty())
                throw std::runtime_error("physical_testing_trials entries must be nonempty strings.");
            const auto trial_path = std::filesystem::weakly_canonical(trial.asString());
            for (const auto& training_trial : training_trials)
            {
                if (trial_path == std::filesystem::weakly_canonical(training_trial))
                    throw std::runtime_error("A physical trial cannot belong to both training and testing splits.");
            }
            testing_trials.push_back(trial.asString());
        }
    }
    if (!evaluation_parameters_path.empty() && (!physical_data || training_trials.empty()))
        throw std::runtime_error("--evaluate-parameters requires --physical and physical_training_trials.");
    TestProblemData data = training_trials.empty()
        ? buildMinimalTestProblemData(physical_data)
        : buildMinimalTestProblemData(true, training_trials.front(), common_window_duration, common_window_count);
    std::vector<std::pair<std::string, std::size_t>> training_ranges;
    if (!training_trials.empty())
    {
        training_ranges.push_back({training_trials.front(), data.measurement_scenarios.size()});
        for (std::size_t index = 1; index < training_trials.size(); ++index)
        {
            auto trial_data = buildMinimalTestProblemData(true, training_trials[index], common_window_duration, common_window_count);
            training_ranges.push_back({training_trials[index], trial_data.measurement_scenarios.size()});
            data.measurement_scenarios.insert(data.measurement_scenarios.end(),
                std::make_move_iterator(trial_data.measurement_scenarios.begin()),
                std::make_move_iterator(trial_data.measurement_scenarios.end()));
        }
        data.validation_sample_count = static_cast<int>(data.measurement_scenarios.size());
    }
    if (max_scenarios_override > 0 &&
        data.measurement_scenarios.size() > static_cast<std::size_t>(max_scenarios_override))
    {
        data.measurement_scenarios.resize(static_cast<std::size_t>(max_scenarios_override));
        data.validation_sample_count = max_scenarios_override;
        std::size_t remaining = data.measurement_scenarios.size();
        for (auto& range : training_ranges)
        {
            range.second = std::min(range.second, remaining);
            remaining -= range.second;
        }
    }
    if (inspect_data)
    {
        for (std::size_t index = 0; index < data.measurement_scenarios.size(); ++index)
        {
            const auto& scenario = data.measurement_scenarios[index];
            std::cout << "Scenario " << index
                      << ": N=" << scenario.robot_topology.N
                      << ", K=" << scenario.robot_topology.K
                      << ", T=" << scenario.robot_topology.T
                      << ", nodes=" << scenario.nodes.size()
                      << ", measurements=" << scenario.factors.size()
                      << ", validation_targets=" << scenario.validation_targets.size()
                      << std::endl;
        }
        for (const auto& trial : testing_trials)
        {
            const auto testing_data = buildMinimalTestProblemData(true, trial, common_window_duration, common_window_count);
            std::cout << "Testing trial " << trial << ": windows=" << testing_data.measurement_scenarios.size() << std::endl;
        }
        return 0;
    }

    spacetime::OptimizationProblem problem;
    problem.setEstimatorOptions(data.estimator_options);
    problem.setRobotTopology(data.robot_topology);
    problem.setRunSummary({data.validation_sample_count, data.R_pose, data.R_gyro, data.validation_variance});
    problem.setTheta(data.initial_theta);
    problem.setProblemData(data.nodes, data.factors, data.factor_node_indices);
    problem.setValidationTargets(data.validation_targets);

    spacetime::Optimizer optimizer(problem);
    if (!data.measurement_scenarios.empty())
    {
        std::vector<spacetime::OptimizationProblem> scenario_problems;
        scenario_problems.reserve(data.measurement_scenarios.size());
        for (const auto& scenario : data.measurement_scenarios)
        {
            spacetime::OptimizationProblem scenario_problem;
            scenario_problem.setEstimatorOptions(data.estimator_options);
            scenario_problem.setRobotTopology(scenario.robot_topology);
            scenario_problem.setRunSummary({data.validation_sample_count, data.R_pose, data.R_gyro, data.validation_variance});
            scenario_problem.setTheta(data.initial_theta);
            scenario_problem.setProblemData(scenario.nodes, scenario.factors, scenario.factor_node_indices);
            scenario_problem.setValidationTargets(scenario.validation_targets);
            scenario_problems.push_back(std::move(scenario_problem));
        }
        optimizer.setScenarioProblems(scenario_problems);
    }
    spacetime::Optimizer::OptimizerConfig config;
    loadBilevelOptimizerConfigFromJson(bilevel_config, config);
    if (max_outer_iterations_override > 0)
    {
        config.max_iterations = max_outer_iterations_override;
    }
    std::filesystem::path output_dir;
    if (!training_trials.empty())
    {
        output_dir = resolveAssetsLogPath("shared_physical_" + formatTimestamp());
        std::filesystem::create_directories(output_dir);
        config.parameter_checkpoint_path = (output_dir / "checkpoint.json").string();
        Json::Value experiment;
        experiment["bilevel_config"] = bilevel_config;
        experiment["max_iterations"] = config.max_iterations;
        experiment["max_scenarios_override"] = max_scenarios_override;
        experiment["evaluation_parameters_path"] = evaluation_parameters_path;
        experiment["robot_config"] = loadJsonFile(resolveExistingPath({
            "assets/config/robot/robot1.json", "../assets/config/robot/robot1.json", "../../assets/config/robot/robot1.json"}));
        experiment["estimator_config"] = loadJsonFile(resolveExistingPath({
            "assets/config/estimator/batch.json", "../assets/config/estimator/batch.json", "../../assets/config/estimator/batch.json"}));
        for (const auto& trial : training_trials) experiment["trial_configs"][trial] = loadJsonFile(trial);
        for (const auto& trial : testing_trials) experiment["trial_configs"][trial] = loadJsonFile(trial);
        std::ofstream manifest(output_dir / "experiment.json");
        if (!manifest) throw std::runtime_error("Cannot write shared experiment manifest.");
        writeReportJson(manifest, experiment);
        std::cout << "Shared experiment output directory: " << output_dir << std::endl;
    }
    optimizer.setConfig(config);

    spacetime::Optimizer::Result result;
    if (evaluation_parameters_path.empty())
    {
        result = optimizer.optimize();
    }
    else
    {
        result.theta = loadSavedTheta(evaluation_parameters_path);
        result.lambda = result.theta.cwiseInverse();
    }
    std::cout << "Upper-level optimization finished" << std::endl;
    std::cout << "Converged: " << (result.converged ? "true" : "false") << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final loss: " << result.loss << std::endl;
    std::cout << "Lambda min/max: " << result.lambda.minCoeff() << " / " << result.lambda.maxCoeff() << std::endl;

    if (!training_trials.empty())
    {
        Json::Value saved;
        saved["parameter_layout_version"] = 2;
        saved["parameter_layout"] = "P0(18),Q1(6),Q2(6),Q3(6),Q4(6)";
        saved["config"] = bilevel_config;
        saved["iterations"] = result.iterations;
        saved["converged"] = result.converged;
        saved["evaluation_only"] = !evaluation_parameters_path.empty();
        saved["nll_convention"] = "0.5*(mahalanobis+logdet), Gaussian constant omitted";
        saved["metric_convention"] = "All Vicon pose targets including base; translation norm in m, rotation angle in rad; pooled across overlapping windows";
        for (int index = 0; index < kThetaSize; ++index)
        {
            saved["theta"].append(result.theta(index));
            saved["lambda"].append(result.lambda(index));
        }
        for (const auto& trial : training_trials) saved["trial_configs"][trial] = loadJsonFile(trial);
        for (const auto& trial : testing_trials) saved["trial_configs"][trial] = loadJsonFile(trial);
        std::ofstream parameters(output_dir / "parameters.json");
        if (!parameters) throw std::runtime_error("Cannot write shared-fit parameters.");
        writeReportJson(parameters, saved);
        parameters.close();
        std::ofstream windows_log(output_dir / "window_metrics.csv");
        std::ofstream summary_log(output_dir / "split_metrics.csv");
        if (!windows_log || !summary_log) throw std::runtime_error("Cannot write shared-fit metrics.");
        windows_log << std::setprecision(17);
        summary_log << std::setprecision(17);
        const std::string header = "split,trial,window,start_time,end_time,windows,targets,observed_dimensions,nll_sum,nll_mean_window,nll_per_dimension,mean_nees,translation_mae_m,translation_rmse_m,rotation_mae_rad,rotation_rmse_rad\n";
        windows_log << header;
        summary_log << header;
        std::size_t begin = 0;
        PhysicalEvaluationTotals training_totals;
        for (const auto& range : training_ranges)
        {
            if (range.second > 0)
                training_totals.add(evaluatePhysicalWindows(data, begin, range.second, result.theta,
                    config.use_joint_validation_objective, "training", range.first, windows_log, summary_log));
            begin += range.second;
        }
        writePhysicalEvaluationRow(summary_log, "training", "all", "all", 0.0, 0.0, training_totals);
        PhysicalEvaluationTotals testing_totals;
        for (const auto& trial : testing_trials)
        {
            const auto testing_data = buildMinimalTestProblemData(true, trial, common_window_duration, common_window_count);
            testing_totals.add(evaluatePhysicalWindows(testing_data, 0, testing_data.measurement_scenarios.size(),
                result.theta, config.use_joint_validation_objective, "testing", trial, windows_log, summary_log));
        }
        if (testing_totals.windows > 0)
            writePhysicalEvaluationRow(summary_log, "testing", "all", "all", 0.0, 0.0, testing_totals);
        std::cout << "Shared-fit parameters and frozen training/testing metrics: " << output_dir << std::endl;
    }

    return 0;
}