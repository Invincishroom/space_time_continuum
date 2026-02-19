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
#include "customMeasurementFactor.hpp"

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
    Spacetime::VisualizerOptions vis_options;
    std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> measurements;
    std::vector<Spacetime::SensorMeasurement> ground_truth;
    vis_options.topology = topology;
    vis_options.estimator_options = options;

    Spacetime::SensorMeasurement custom_meas;
    custom_meas.type = Spacetime::SensorMeasurement::Custom;
    custom_meas.value = Eigen::MatrixXd::Identity(4, 4);
    custom_meas.s = 0.5;
    custom_meas.t = 0.5;
    measurements.push_back(std::make_shared<CustomMeasurementFactor>(Eigen::Matrix<double, 6, 6>::Identity(), custom_meas));

    // Run state estimator
    Spacetime::SystemState<DTYPE> state;
    Eigen::MatrixX<double> covariance;
    Spacetime::Estimator::Results result;

    Spacetime::Estimator *state_estimator;

    info("Using batch estimator.");
    topology.T = measurements.back()->getMeas().t - topology.t0; // Set T based on measurement times and start time t0
    topology.K = (unsigned int)std::ceil(topology.K * topology.T) + 1; // Set K based on K_per_second and total time T
    debug() << "Setting K to " << topology.K << " based on K_per_second and total time T";
    state_estimator = new Spacetime::Estimator(topology, params, options);
    
    result = state_estimator->computeStateEstimate(measurements, true);
    info() << "Optimization message: " << result.message.str();

    return 0;
}
