#include <fstream>
#include <filesystem>
#include <thread>

#include <json/json.h>
#include <json/value.h>
#include <QApplication>
#include <QShortcut>
#include <QElapsedTimer>
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include "spacetime.hpp"
#include "data_loader.hpp"
#include "export_tools.hpp"

// Single robot for testing
int main(int argc, char *argv[])
{
    bool headless = false;
    std::string output_path = "";
    std::vector<std::string> config_files;
    for (int i = 1; i < argc; i++)
    {
        std::cout << "Arg " << i << ": " << argv[i] << std::endl;
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h" || argc < 3)
        {
            std::cout << "Usage: " << argv[0] << " <config_file.json> [--output_path <output_path>] [--headless]" << std::endl;
            return 0;
        }
        else if (arg == "--output_path" && i + 1 < argc)
        {
            output_path = argv[++i];
            if (!std::filesystem::exists(output_path))
            {
                info() << "Creating output directory: " << output_path;
                std::filesystem::create_directories(output_path);
            }
            if (!std::filesystem::exists(output_path + "logs/"))
            {
                info() << "Creating logs directory: " << output_path + "logs/";
                std::filesystem::create_directories(output_path + "logs/");
            }
        }
        else if (arg == "--headless")
        {
            headless = true;
        }
        else
        {
            config_files.push_back(arg);
        }
    }
    // Define robot topology
    Spacetime::RobotTopology topology;
    Spacetime::Hyperparameters params;
    Spacetime::Options options;
    DataLoaderOptions data_options;
    Spacetime::VisualizerOptions vis_options;
    std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> measurements;
    std::vector<Spacetime::SensorMeasurement> ground_truth;
    std::shared_ptr<steam_icp::Map> map;

    for (auto config_file : config_files)
    {
        if (!loadConfig(config_file, topology, params, options, data_options, vis_options))
            return -1;
    }
    if (!loadData(data_options, options, params, measurements, ground_truth, map))
        return -1;
    vis_options.topology = topology;
    vis_options.estimator_options = options;

    // Run state estimator
    Spacetime::SystemState<DTYPE> state;
    Eigen::MatrixX<double> covariance;
    Spacetime::Estimator::Results result;

    Spacetime::Estimator *state_estimator;
    if (options.estimator_type == Spacetime::Options::EstimatorType::Batch)
    {
        info("Using batch estimator.");
        topology.T = measurements.back()->getMeas().t - topology.t0; // Set T based on measurement times and start time t0
        topology.K = (unsigned int)std::ceil(topology.K * topology.T) + 1; // Set K based on K_per_second and total time T
        state_estimator = new Spacetime::Estimator(topology, params, options);
    }
    else if (options.estimator_type == Spacetime::Options::EstimatorType::Window)
    {
        info("Using sliding window estimator.");
        topology.T = options.estimator_dt; 
        topology.K = (unsigned int)std::ceil(topology.K * topology.T) + 1; // Set K based on K_per_second and total window time T
        state_estimator = new Spacetime::SWEstimator(topology, params, options);
    }
    else if (options.estimator_type == Spacetime::Options::EstimatorType::Filter)
    {
        info("Using filter estimator.");
        topology.T = options.estimator_dt; // Set T to the estimator time step
        topology.K = 1; // For filter estimator, we set K=1 and process measurements sequentially
        state_estimator = new Spacetime::FilterEstimator(topology, params, options);
    }
    else
    {
        error() << "Unknown estimator type: " << options.estimator_type;
        return -1;
    }
    result = state_estimator->computeStateEstimate(measurements, true);
    info() << "Optimization message: " << result.message.str();

    state_estimator->interpolateStates(result.state);

    infoStart(); 
    Spacetime::printError(state_estimator, result.state, ground_truth);
    logReset(); 

    if (output_path != "")
    {
        info() << "Exporting results to " << output_path;
        Spacetime::exportTrajData(result.state, ground_truth, measurements, output_path);
        Spacetime::exportErrorData(state_estimator, result.state, ground_truth, output_path);
        Spacetime::exportRuntimeData(result, output_path);
        if (result.map)
            Spacetime::exportMapData(result.map, output_path);
        Spacetime::exportPointCloudData(state_estimator, result.state, measurements, output_path);
    }

    // Visualize results
    if (result.state.estimation_nodes.size() == 0)
    {
        error() << "No estimation nodes were computed. Unable to visualize results.";
        return -1;
    }
    else if (!headless)
    {
        info() << "Launching visualizer...";
        QApplication a(argc, argv);
        Spacetime::Visualizer vis = Spacetime::Visualizer(vis_options, output_path);
        vis.show();
        vis.animate(result, measurements, ground_truth);

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_N), &vis), &QShortcut::activated, [&]()
                         { 
                            vis.sendMessage("Generating new sample...");

                            std::thread([&]()
                            {
                                auto sampled_state = state_estimator->samplePrior(result.state, topology, 1)[0];
                                state_estimator->interpolateStates(sampled_state);

                                auto original_state = result.state;
                                result.state = sampled_state; 
                                
                                vis.sendMessage("Sample generated.", 2000); 
                                QMetaObject::invokeMethod(&vis, [&vis, &result, &measurements, ground_truth, original_state]() mutable
                                    {
                                        vis.animate(result, measurements, ground_truth);
                                        result.state = original_state; // Restore original state
                                    }, Qt::QueuedConnection);
                            }).detach(); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_M), &vis), &QShortcut::activated, [&]()
                         { vis.sendMessage("Displaying mean solution.", 2000); vis.animate(result, measurements, ground_truth); });

        return a.exec();
    }
    return 0;
}
