//?2012 Vanderbilt University, All Rights Reserved
#include <filesystem>
#include <algorithm>

#include <QShortcut>
#include <QLabel>
#include <QApplication>
#include <QProgressBar>
#include <QPixmap>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkTetra.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolygon.h>
#include <vtkSmartPointer.h>
#include <vtkMath.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCleanPolyData.h>
#include <vtkDelaunay3D.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkMassProperties.h>
#include <vtkConvexPointSet.h>
#include <vtkUnstructuredGrid.h>
#include <vtkAppendFilter.h>
#include <vtkNamedColors.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkPropPicker.h>
#include <vtkCallbackCommand.h>
#include <vtkSTLReader.h>
#include <vtkLightCollection.h>
#include <vtkCubeSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkAssembly.h>
#include <Eigen/Eigenvalues>

#include <vtkPNGReader.h>
#include <vtkImageActor.h>

#include "spacetime/Visualizer.hpp"
#include "spacetime/visualizer/PlotWidget.hpp"
#include "spacetime/visualizer/TimelineWidget.hpp"
#include "spacetime/visualizer/InfoWidget.hpp"

#include "spacetime/utilities.hpp"
#include "spacetime/estimators/Estimator.hpp"

vtkNew<vtkNamedColors> colors;

namespace Spacetime
{
    // Callback function for mouse clicks
    void onLeftButtonDown(vtkObject *caller, [[maybe_unused]] long unsigned int eventId, void *clientData, [[maybe_unused]] void *callData)
    {
        auto interactor = static_cast<vtkRenderWindowInteractor *>(caller);
        auto picker = vtkSmartPointer<vtkPropPicker>::New();

        // Get the click position
        int *clickPos = interactor->GetEventPosition();

        // Perform picking
        auto visualizer = static_cast<Spacetime::Visualizer *>(clientData);
        picker->Pick(clickPos[0], clickPos[1], 0, visualizer->mp_ren);

        // Get the picked actor
        vtkProp *pickedActor = picker->GetViewProp();
        if (pickedActor)
        {
            // Check if the picked actor is one of the mp_axes_k or mp_axes_k1
            for (size_t i = 0; i < visualizer->mp_axes_k.size(); ++i)
            {
                if (pickedActor == visualizer->mp_axes_k[i])
                {
                    visualizer->m_selected_frame = visualizer->mp_axes_k[i];
                    int curr_n = i;
                    int curr_k = visualizer->m_time_idx / visualizer->m_options.topology.Mt;
                    info() << "Picked node: " << curr_n << " at time step: " << curr_k;
                    visualizer->m_state.estimation_nodes[curr_n + curr_k * visualizer->m_options.topology.N].print();
                    visualizer->m_selected_frame_arclength = visualizer->m_state.estimation_nodes[curr_n + curr_k * visualizer->m_options.topology.N].arclength;
                    return;
                }
                else if (pickedActor == visualizer->mp_axes_k1[i])
                {
                    visualizer->m_selected_frame = visualizer->mp_axes_k1[i];
                    int curr_n = i;
                    int curr_k = visualizer->m_time_idx / visualizer->m_options.topology.Mt + 1;
                    info() << "Picked node: " << curr_n << " at time step: " << curr_k;
                    visualizer->m_state.estimation_nodes[curr_n + curr_k * visualizer->m_options.topology.N].print();
                    visualizer->m_selected_frame_arclength = visualizer->m_state.estimation_nodes[curr_n + curr_k * visualizer->m_options.topology.N].arclength;
                    return;
                }
                else if (pickedActor == visualizer->mp_axes_interp[i])
                {
                    visualizer->m_selected_frame = visualizer->mp_axes_interp[i];
                    int curr_n = i;
                    int curr_k = visualizer->m_time_idx;
                    info() << "Picked node: " << curr_n << " at time step: " << curr_k;
                    visualizer->m_state.interpolation_nodes[curr_n + curr_k * visualizer->m_options.topology.N].print();
                    visualizer->m_selected_frame_arclength = visualizer->m_state.interpolation_nodes[curr_n + curr_k * visualizer->m_options.topology.N].arclength;
                    return;
                }
            }
        }
        else
        {
            visualizer->m_selected_frame = nullptr;
        }
    }

    /**
     * Constructor.
     * @brief Visualizer::Visualizer
     */
    Visualizer::Visualizer(VisualizerOptions options, std::string output_path) : QVTKOpenGLNativeWidget()
    {
        m_options = options;
        m_output_path = output_path;

        mp_renWin = RenderWindow;
        mp_ren = vtkRenderer::New();
        mp_renWin->AddRenderer(mp_ren);
        this->setGeometry(100, 100, 1280, 720);

        m_diagnosticsLabel = new InfoWidget(this);
        m_diagnosticsLabel->addHeader("Information");
        m_diagnosticsLabel->addLabel("Time Index [←, →]", &m_time_idx);
        m_diagnosticsLabel->addLabel("Time [←, →]", &m_time);
        m_diagnosticsLabel->addLabel("Speed [↑, ↓]", &m_speed);
        m_diagnosticsLabel->addLabel("Play [Space]", &m_play);
        m_diagnosticsLabel->addLabel("Camera position", [this]()
                                     {
            double pos[3];
            this->mp_ren->GetActiveCamera()->GetPosition(pos);
            std::ostringstream oss;
            oss << "<br>" << pos[0] << ",<br>" << pos[1] << ",<br>" << pos[2];
            return oss.str(); });
        m_diagnosticsLabel->addLabel("Camera focal", [this]()
                                     {
            double fp[3];
            this->mp_ren->GetActiveCamera()->GetFocalPoint(fp);
            std::ostringstream oss;
            oss << "<br>" << fp[0] << ",<br>" << fp[1] << ",<br>" << fp[2];
            return oss.str(); });
        m_diagnosticsLabel->addLabel("Camera up vector", [this]()
                                     {
            double up[3];
            this->mp_ren->GetActiveCamera()->OrthogonalizeViewUp();
            this->mp_ren->GetActiveCamera()->GetViewUp(up);
            std::ostringstream oss;
            oss << "<br>" << up[0] << ",<br>" << up[1] << ",<br>" << up[2];
            return oss.str(); });
        m_diagnosticsLabel->addLabel("Camera FOV", [this]()
                                     {
            double fov = this->mp_ren->GetActiveCamera()->GetViewAngle();
            std::ostringstream oss;
            oss << fov << "°";
            return oss.str(); });

        m_diagnosticsLabel->addHeader("Display Options");
        m_diagnosticsLabel->addLabel("Map [p]", &m_show_map);
        m_diagnosticsLabel->addLabel("Grid [w]", &m_show_grid);
        m_diagnosticsLabel->addLabel("Ground plane [s]", &m_options.show_ground_plane);
        m_diagnosticsLabel->addLabel("Lidar points [d]", &m_options.lidar.show);
        m_diagnosticsLabel->addLabel("Aurora frames [a]", &m_options.aurora.show);
        m_diagnosticsLabel->addLabel("Vicon [g]", &m_options.vicon.show);
        m_diagnosticsLabel->addLabel("Environment [f]", &m_options.environment.show);
        m_diagnosticsLabel->addLabel("Robot frames [e]", &m_options.robot.show_frames);
        m_diagnosticsLabel->addLabel("Interpolation nodes [i]", &m_options.robot.show_interpolation);
        m_diagnosticsLabel->addLabel("Covariance ellipsoids [c]", &m_options.robot.show_covariances);

        m_diagnosticsLabel->addHeader("Controls");
        m_diagnosticsLabel->addLabel("Quit [q]");
        m_diagnosticsLabel->addLabel("Record [x]");
        m_diagnosticsLabel->addLabel("Reset camera [r]");
        m_diagnosticsLabel->addLabel("Sample estimate [n]");
        m_diagnosticsLabel->addLabel("Display estimate mean [m]");
        m_diagnosticsLabel->addLabel("Toggle diagnostics [h]");

        m_diagnosticsLabel->setGeometry(10, 10, m_diagnosticsLabel->sizeHint().width(), m_diagnosticsLabel->sizeHint().height());

        // Set up the interactor
        vtkSmartPointer<vtkCallbackCommand> p_left_click_callback = vtkSmartPointer<vtkCallbackCommand>::New();
        p_left_click_callback->SetCallback(onLeftButtonDown);
        p_left_click_callback->SetClientData(this);

        mp_renWin->GetInteractor()->AddObserver(vtkCommand::LeftButtonPressEvent, p_left_click_callback);

        // Timers
        m_animation_timer.setSingleShot(true);
        connect(&m_animation_timer, SIGNAL(timeout()), this, SLOT(slotAnimationTimer()));

        m_infoTimer.setSingleShot(true);
        connect(&m_infoTimer, SIGNAL(timeout()), this, SLOT(hideMessage()));

        // Create a QShortcut for key bindings
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Up), this), &QShortcut::activated, [this]()
                         { this->m_speed /= 1.5; if (m_verbose) info() << "speed: " << this->m_speed; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Down), this), &QShortcut::activated, [this]()
                         { this->m_speed *= 1.5; if (m_verbose) info() << "speed: " << this->m_speed; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Left), this), &QShortcut::activated, [this]()
                         { this->m_time_idx -= std::min(this->m_time_idx, m_options.topology.Mt); if (m_verbose) info() << "New idx: " << this->m_time_idx; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Right), this), &QShortcut::activated, [this]()
                         { this->m_time_idx += m_options.topology.Mt; if (m_verbose) info() << "New idx: " << this->m_time_idx; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Space), this), &QShortcut::activated, [this]()
                         { this->m_play = !this->m_play; if (m_verbose) info() << "play: " << this->m_play; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_P), this), &QShortcut::activated, [this]()
                         { this->m_show_map = !this->m_show_map; if (m_verbose) info() << "show map: " << this->m_show_map; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_W), this), &QShortcut::activated, [this]()
                         { this->m_show_grid = !this->m_show_grid; if (m_verbose) info() << "show grid: " << this->m_show_grid; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Q), this), &QShortcut::activated, [this]()
                         { QApplication::quit(); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_S), this), &QShortcut::activated, [this]()
                         { this->m_options.show_ground_plane = !this->m_options.show_ground_plane; this->update(); if (m_verbose) info() << "show_ground_plane: " << this->m_options.show_ground_plane; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_D), this), &QShortcut::activated, [this]()
                         { this->m_options.lidar.show = !this->m_options.lidar.show; this->update(); if (m_verbose) info() << "show_lidar: " << this->m_options.lidar.show; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_A), this), &QShortcut::activated, [this]()
                         { this->m_options.aurora.show = !this->m_options.aurora.show; this->update(); if (m_verbose) info() << "show_aurora: " << this->m_options.aurora.show; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_F), this), &QShortcut::activated, [this]()
                         { this->m_options.environment.show = !this->m_options.environment.show; this->update(); if (m_verbose) info() << "show_environment: " << this->m_options.environment.show; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_E), this), &QShortcut::activated, [this]()
                         { this->m_options.robot.show_frames = !this->m_options.robot.show_frames; this->update(); if (m_verbose) info() << "show_estimation_nodes: " << this->m_options.robot.show_frames; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_I), this), &QShortcut::activated, [this]()
                         { this->m_options.robot.show_interpolation = !this->m_options.robot.show_interpolation; this->update(); if (m_verbose) info() << "show_interpolation_nodes: " << this->m_options.robot.show_interpolation; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_C), this), &QShortcut::activated, [this]()
                         { this->m_options.robot.show_covariances = !this->m_options.robot.show_covariances; this->update(); if (m_verbose) info() << "show_covariance: " << this->m_options.robot.show_covariances; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_G), this), &QShortcut::activated, [this]()
                         { this->m_options.vicon.show = !this->m_options.vicon.show; this->update(); if (m_verbose) info() << "show_gt: " << this->m_options.vicon.show; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_X), this), &QShortcut::activated, [this]()
                         { this->m_recording = !this->m_recording; this->update(); if (m_verbose) info() << "m_recording: " << this->m_recording; });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_R), this), &QShortcut::activated, [&]()
                         { this->resetCameraView(); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Equal), this), &QShortcut::activated, [&]()
                         { 
                            this->mp_ren->GetActiveCamera()->Zoom(1.005);
                            this->update();
                         });
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Minus), this), &QShortcut::activated, [&]()
                         { 
                            this->mp_ren->GetActiveCamera()->Zoom(0.995);
                            this->update();
                         });

        // Create timeline widget
        m_timelineWidget = new TimelineWidget(this);

        // Update function for the progress bar
        auto updateSlider = [this]()
        {
            m_timelineWidget->setValue(m_time_idx);
        };

        m_plotWidget = new PlotWidget(this);

        // Connect the update function to the animation timer
        QObject::connect(&m_animation_timer, &QTimer::timeout, this, updateSlider);

        // Connect a callback to handle window resize events
        connect(this, &Visualizer::resized, [this]()
                { 
                    m_timelineWidget->setGeometry(150, this->height() - 50, this->width() - 300, 50); 
                    m_plotWidget->setGeometry(this->width() - 520, 10, 510, this->height() - 20 - m_timelineWidget->height()); });

        // Create a QShortcut for toggling the controls menu
        QShortcut *toggleHelpShortcut = new QShortcut(QKeySequence(Qt::Key_H), this);

        // Connect the shortcut to a lambda function to toggle the visibility of the controls menu
        QObject::connect(toggleHelpShortcut, &QShortcut::activated, [&]()
                         { 
                            m_diagnosticsLabel->setVisible(m_diagnosticsLabel->isHidden());
                            m_plotWidget->setVisible(m_plotWidget->isHidden()); });

        initScene();
    }

    Visualizer::~Visualizer()
    {
    }

    int Visualizer::getDuration()
    {
        // Default duration in milliseconds (10 FPS)
        m_time_diff = 100.0;
        int duration = 100;

        // Ensure there are enough interpolation nodes and the current index is within bounds
        if (m_options.topology.N > 1 && m_time_idx + 1 < ((m_topology_K - 1) * m_options.topology.Mt + 1))
        {
            // Calculate the index for the current and next interpolation nodes
            size_t current_idx = m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1);
            size_t next_idx = (m_time_idx + 1) * ((m_options.topology.N - 1) * m_options.topology.Ms + 1);

            // Ensure the indices are within bounds
            if (next_idx < m_state.interpolation_nodes.size())
            {
                // Calculate the time difference between the current and next interpolation nodes
                m_time_diff = m_state.interpolation_nodes[next_idx].time - m_state.interpolation_nodes[current_idx].time;

                // Convert the time difference to milliseconds
                m_time_diff = 1000 * m_time_diff;
            }

            // Apply the speed reduction factor
            duration = static_cast<int>(m_speed * m_time_diff);
        }

        // Measure the elapsed time since the last call
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_last_time).count();

        // Adjust the duration to account for the elapsed time
        duration = std::max(0, static_cast<int>(duration - elapsed));

        return duration;
    }

    void Visualizer::sendMessage(QString message)
    {
        QMetaObject::invokeMethod(&m_infoTimer, "stop", Qt::QueuedConnection);
        m_infoLabel->setText(message);
        m_infoLabel->setVisible(true);
    }
    void Visualizer::sendMessage(QString message, int duration)
    {
        QMetaObject::invokeMethod(&m_infoTimer, "stop", Qt::QueuedConnection);
        m_infoLabel->setText(message);
        m_infoLabel->setVisible(true);
        QMetaObject::invokeMethod(&m_infoTimer, "start", Qt::QueuedConnection, Q_ARG(int, duration));
    }
    void Visualizer::hideMessage()
    {
        m_infoLabel->setVisible(false);
    }

    void Visualizer::animate(const Spacetime::Estimator::Results &result, const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements, const std::vector<Spacetime::SensorMeasurement> &ground_truth)
    {
        info() << "Visualizer::animate()";
        mp_map = result.map;
        m_state = result.state.cast<double>();

        m_topology_K = m_state.getK(m_options.topology);
        m_timelineWidget->setRange(0, int((m_topology_K - 1) * m_options.topology.Mt), m_state.interpolation_nodes[0].time, m_state.interpolation_nodes.back().time, m_options.topology.T);
        initMap();

        // Initialize m_last_time on the first call
        m_last_time = std::chrono::high_resolution_clock::now();

        if (!m_state.ib) // Visualizer assumes state is in inertial-body form
            m_state.convertStateMeanBodyInertial();
        m_selected_frame_arclength = m_state.interpolation_nodes.back().arclength; // Initialize selected frame to last node
        m_measurements = measurements;
        m_gt = ground_truth;
        m_time = m_state.interpolation_nodes[0].time;
        m_time_idx = 0;
        m_timestep = m_options.topology.T / (m_options.topology.K - 1) / m_options.topology.Mt;
        m_play = true;

        update();
        m_animation_timer.start(getDuration());
    }

    void Visualizer::getSpherePose(Spacetime::SystemState<double>::Node node, vtkSmartPointer<vtkMatrix4x4> &sphere_frame_vtk, Eigen::Vector3d &s)
    {
        if (node.getCovariance().hasNaN())
        {
            sphere_frame_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
            sphere_frame_vtk->Identity();
            s << 0, 0, 0;
            return;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> solver;

        Eigen::Vector3d pos = node.pose.block<3, 1>(0, 3);
        Eigen::Matrix3d rot = node.pose.block<3, 3>(0, 0);
        Eigen::Matrix3d cov = rot * node.getCovariance().block<3, 3>(0, 0) * rot.transpose();
        solver.compute(cov);
        Eigen::MatrixXd eigen_vectors = solver.eigenvectors().real();
        Eigen::VectorXd eigen_values = solver.eigenvalues().real();
        eigen_values = 3.73 * eigen_values.cwiseSqrt(); // 3.73 interval for 99.7% (3\sigma) confidence interval in R^3

        Eigen::Matrix3d R = eigen_vectors;
        s = eigen_values;

        // Make sure that R resembles a rotation matrix
        if ((R.col(0).cross(R.col(1))).dot(R.col(2)) < 0)
        {
            R << eigen_vectors.col(1), eigen_vectors.col(0), eigen_vectors.col(2);
            s << eigen_values(1),
                eigen_values(0),
                eigen_values(2);
        }

        // Create sphere pose
        Eigen::Matrix4d sphere_pose = Eigen::Matrix4d::Identity();

        sphere_pose.block<3, 3>(0, 0) = R;
        sphere_pose.block<3, 1>(0, 3) = pos;

        // Check if one of the singular values is below a treshold (important to not scale the sphere's dimensions to zero)
        for (Eigen::Index i = 0; i < s.size(); i++)
        {
            if (s(i) < 1e-8)
                s(i) = 1e-8;
        }

        sphere_frame_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                sphere_frame_vtk->SetElement(i, j, sphere_pose(i, j));
            }
        }
    }

    void Visualizer::update()
    {
        m_diagnosticsLabel->refresh();

        if (mp_floor)
            mp_floor->SetVisibility(m_options.show_ground_plane);
        if (mp_grid_actor)
            mp_grid_actor->SetVisibility(m_show_grid);
        if (mp_environment_actor)
            mp_environment_actor->SetVisibility(m_options.environment.show);
        if (mp_map_actor)
            mp_map_actor->SetVisibility(m_show_map);

        if (m_selected_frame_arclength != m_plotWidget->getArclength())
        {
            m_plotWidget->updateData(m_state, m_selected_frame_arclength);
        }

        // Backbone
        mp_backbone_actor->SetVisibility(m_options.robot.show_interpolation);
        if (m_options.robot.show_interpolation && m_state.interpolation_nodes.size() > 0)
        {
            for (unsigned int i = m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1); i < (m_time_idx + 1) * ((m_options.topology.N - 1) * m_options.topology.Ms + 1); i++)
            {
                Eigen::Vector3d pos = m_state.interpolation_nodes[i].pose.block<3, 1>(0, 3);

                mp_backbone_points->SetPoint(i - m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1), pos(0), pos(1), pos(2)); // Set the point of the line
            }
            mp_backbone_points->Modified();
        }

        // Render frame axes and [if applicable] frame covariances
        if (m_options.robot.show_frames && (m_state.estimation_nodes.size() > 0))
        {
            int estimation_time_idx = std::floor(m_time_idx / m_options.topology.Mt);
            // Show all axes
            for (unsigned int i = 0; i < mp_axes_k.size(); i++)
            {
                mp_axes_k[i]->SetVisibility(true);
                mp_axes_k1[i]->SetVisibility(true);

                mp_axes_k[i]->SetCylinderRadius(std::max(0.025, 0.25 * m_options.topology.radius));
                mp_axes_k[i]->SetTotalLength(std::max(0.015, 1.5 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius));
                mp_axes_k1[i]->SetCylinderRadius(std::max(0.025, 0.25 * m_options.topology.radius));
                mp_axes_k1[i]->SetTotalLength(std::max(0.015, 1.5 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius));
                mp_axes_interp[i]->SetCylinderRadius(std::max(0.025, 0.25 * m_options.topology.radius));
                mp_axes_interp[i]->SetTotalLength(std::max(0.015, 1.5 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius));

                if (mp_axes_k1[i] == m_selected_frame)
                {
                    mp_axes_k1[i]->SetCylinderRadius(0.05);
                    mp_axes_k1[i]->SetTotalLength(0.02, 0.02, 0.02);
                }
                else if (mp_axes_k[i] == m_selected_frame)
                {
                    mp_axes_k[i]->SetCylinderRadius(0.05);
                    mp_axes_k[i]->SetTotalLength(0.02, 0.02, 0.02);
                }
                else if (mp_axes_interp[i] == m_selected_frame)
                {
                    mp_axes_interp[i]->SetCylinderRadius(0.05);
                    mp_axes_interp[i]->SetTotalLength(0.02, 0.02, 0.02);
                }
            }
            for (unsigned int i = 0; i < mp_axes_interp.size(); i++)
            {
                if (m_options.robot.show_interpolation)
                    mp_axes_interp[i]->SetVisibility(true);
                else
                    mp_axes_interp[i]->SetVisibility(false);
            }

            // Update Robot frames
            for (unsigned int i = estimation_time_idx * m_options.topology.N; i < (estimation_time_idx + 1) * m_options.topology.N; i++)
            {
                Eigen::Matrix4d axes_pose_k = m_state.estimation_nodes[i].pose;

                vtkSmartPointer<vtkMatrix4x4> axes_frame_vtk_k = vtkSmartPointer<vtkMatrix4x4>::New();
                for (int k = 0; k < 4; k++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        axes_frame_vtk_k->SetElement(k, j, axes_pose_k(k, j));
                    }
                }
                mp_axes_k[i - estimation_time_idx * m_options.topology.N]->SetUserMatrix(axes_frame_vtk_k);

                if (m_options.robot.show_interpolation && ((size_t)i + m_options.topology.N < m_state.estimation_nodes.size()) && m_options.topology.Mt > 1)
                {
                    Eigen::Matrix4d axes_pose_k1 = m_state.estimation_nodes[i + m_options.topology.N].pose;
                    vtkSmartPointer<vtkMatrix4x4> axes_frame_vtk_k1 = vtkSmartPointer<vtkMatrix4x4>::New();
                    for (int k = 0; k < 4; k++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            axes_frame_vtk_k1->SetElement(k, j, axes_pose_k1(k, j));
                        }
                    }
                    mp_axes_k1[i - estimation_time_idx * m_options.topology.N]->SetUserMatrix(axes_frame_vtk_k1);
                }
                else
                {
                    mp_axes_k1[i - estimation_time_idx * m_options.topology.N]->SetVisibility(false);
                }
            }

            if (m_options.robot.show_interpolation && m_state.interpolation_nodes.size() > 0)
            {
                for (unsigned int i = m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1); i < (m_time_idx + 1) * ((m_options.topology.N - 1) * m_options.topology.Ms + 1); i++)
                {
                    Eigen::Matrix4d axes_pose_interp = m_state.interpolation_nodes[i].pose;
                    vtkSmartPointer<vtkMatrix4x4> axes_frame_vtk_interp = vtkSmartPointer<vtkMatrix4x4>::New();
                    for (int k = 0; k < 4; k++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            axes_frame_vtk_interp->SetElement(k, j, axes_pose_interp(k, j));
                        }
                    }
                    mp_axes_interp[i - m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1)]->SetUserMatrix(axes_frame_vtk_interp);
                }
            }
        }
        else
        {
            // Hide all axes
            for (unsigned int i = 0; i < mp_axes_k.size(); i++)
            {
                mp_axes_k[i]->SetVisibility(false);
                mp_axes_k1[i]->SetVisibility(false);
            }
            for (unsigned int i = 0; i < mp_axes_interp.size(); i++)
            {
                mp_axes_interp[i]->SetVisibility(false);
            }
        }

        // Measurement
        for (unsigned int m = 0; m < m_options.aurora.count; m++)
        {
            mp_aurora_meas[m]->SetVisibility(false);
        }
        mp_lidar_actor_active->SetVisibility(false);
        mp_lidar_actor_inactive->SetVisibility(false);
        for (int i = 0; i < mp_sensor_module_actors.size(); i++)
            mp_sensor_module_actors[i]->SetVisibility(false);

        unsigned int meas_idx = 0;
        int lidar_display_idx = 0;
        for (unsigned int m = 0; m < m_measurements.size(); m++)
        {
            const auto &measurement = m_measurements[m]->getMeas();
            if (measurement.t < m_time - m_timestep / 2.0 - TOLERANCE)
                continue;
            else if (measurement.t > m_time + m_timestep / 2.0 + TOLERANCE || measurement.t > m_state.estimation_nodes.back().time)
                break;

            if (m_options.aurora.show && measurement.type == Spacetime::SensorMeasurement::Type::Pose)
            {
                vtkSmartPointer<vtkMatrix4x4> meas_frame_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        meas_frame_vtk->SetElement(i, j, measurement.value(i, j));
                    }
                }
                if (meas_idx >= m_options.aurora.count)
                    continue;
                mp_aurora_meas[meas_idx]->SetUserMatrix(meas_frame_vtk);
                mp_aurora_meas[meas_idx]->SetVisibility(true);
                meas_idx++;
            }

            if (m_options.lidar.show && measurement.type == Spacetime::SensorMeasurement::Type::Lidar)
            {
                // Transform point cloud
                if (lidar_display_idx > mp_lidar_points_active->GetNumberOfPoints())
                    continue;

                Eigen::MatrixXd ps_b = measurement.value;
                Eigen::Matrix4d T_ib = invertTransformation(m_measurements[m]->getOperatingPoint().pose.cast<double>());
                Eigen::Matrix3d R_ib = T_ib.block<3, 3>(0, 0);
                Eigen::Vector3d t_ib = T_ib.block<3, 1>(0, 3);
                for (int i = 0; i < ps_b.rows(); i++)
                {
                    Eigen::Vector3d p_b = ps_b.row(i).transpose();
                    Eigen::Vector3d p_i = R_ib * p_b + t_ib;
                    if (measurement.mask(i))
                    {
                        mp_lidar_points_active->SetPoint(lidar_display_idx, p_i(0), p_i(1), p_i(2));
                        mp_lidar_points_inactive->SetPoint(lidar_display_idx, -1e2, -1e2, -1e2);
                    }
                    else
                    {
                        mp_lidar_points_inactive->SetPoint(lidar_display_idx, p_i(0), p_i(1), p_i(2));
                        mp_lidar_points_active->SetPoint(lidar_display_idx, -1e2, -1e2, -1e2);
                    }
                    lidar_display_idx++;
                    if (mp_lidar_points_active->GetNumberOfPoints() <= lidar_display_idx)
                        break;
                }
                mp_lidar_actor_active->SetVisibility(true);
                mp_lidar_actor_inactive->SetVisibility(true);

                for (int i = 0; i < m_sensor_module_arclengths.size(); i++)
                {
                    if (measurement.s == m_sensor_module_arclengths[i])
                    {
                        Eigen::Matrix4d sensor_frame = invertTransformation(m_measurements[m]->getOperatingPoint().pose.cast<double>());
                        Eigen::Matrix4d rot_x;
                        rot_x << 1, 0, 0, 0,
                            0, 0, -1, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 1;
                        sensor_frame = sensor_frame * rot_x;
                        vtkSmartPointer<vtkMatrix4x4> sensor_frame_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
                        for (int k = 0; k < 4; k++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                sensor_frame_vtk->SetElement(k, j, sensor_frame(k, j));
                            }
                        }
                        mp_sensor_module_actors[i]->SetUserMatrix(sensor_frame_vtk);
                        mp_sensor_module_actors[i]->SetVisibility(true);
                        break;
                    }
                }
            }
        }
        for (int i = lidar_display_idx; i < mp_lidar_points_active->GetNumberOfPoints(); i++)
        {
            mp_lidar_points_active->SetPoint(i, -1e2, -1e2, -1e2);
            mp_lidar_points_inactive->SetPoint(i, -1e2, -1e2, -1e2);
        }
        mp_lidar_points_active->Modified();
        mp_lidar_points_inactive->Modified();

        if (m_options.vicon.show)
        {
            // Ground truth render
            unsigned int meas_idx = 0;
            for (unsigned int m = 0; m < m_gt.size(); m++)
            {
                if (m_gt[m].t < m_time - 1e-2)
                    continue;
                else if (m_gt[m].t > m_time + 1e-2)
                    break;

                if (m_gt[m].type == Spacetime::SensorMeasurement::Type::Pose)
                {
                    vtkSmartPointer<vtkMatrix4x4> meas_frame_vtk = vtkSmartPointer<vtkMatrix4x4>::New();
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            meas_frame_vtk->SetElement(i, j, m_gt[m].value(i, j));
                        }
                    }
                    mp_gt[meas_idx]->SetUserMatrix(meas_frame_vtk);
                    mp_gt[meas_idx]->SetVisibility(true);
                    meas_idx++;

                    if (meas_idx == m_options.vicon.count)
                        break;
                }
            }
            for (unsigned int m = meas_idx; m < m_options.vicon.count; m++)
            {
                mp_gt[m]->SetVisibility(false);
            }
        }
        else
        {
            for (unsigned int m = 0; m < m_options.vicon.count; m++)
            {
                mp_gt[m]->SetVisibility(false);
            }
        }

        // Render interpolated covariances
        if (m_options.robot.show_covariances && m_state.interpolation_nodes[0].covarianceAvailable())
        {
            Eigen::EigenSolver<Eigen::MatrixXd> solver;

            // Covariance on interpolated states
            if (m_options.robot.show_interpolation && m_state.interpolation_nodes.size() > 0)
            {
                for (unsigned int i = m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1); i < (m_time_idx + 1) * ((m_options.topology.N - 1) * m_options.topology.Ms + 1); i++)
                {
                    unsigned int m = i - m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1);
                    if (!m_state.interpolation_nodes[i].getCovariance().isZero())
                    {
                        vtkSmartPointer<vtkMatrix4x4> sphere_frame_vtk;
                        Eigen::Vector3d s;
                        getSpherePose(m_state.interpolation_nodes[i], sphere_frame_vtk, s);

                        mp_ellipsoid_actors[m]->SetVisibility(true);
                        mp_ellipsoid_actors[m]->SetUserMatrix(sphere_frame_vtk);
                        mp_ellipsoid_actors[m]->SetScale(s(0), s(1), s(2));
                    }
                    else
                    {
                        mp_ellipsoid_actors[m]->SetVisibility(false);
                    }
                }
            }
            else
            {
                // Hide all ellipsoids
                for (unsigned int i = 0; i < mp_ellipsoid_actors.size(); i++)
                {
                    mp_ellipsoid_actors[i]->SetVisibility(false);
                }
            }

            // Covariances on estimation frames
            if (m_options.robot.show_frames && (m_state.estimation_nodes.size() > 0))
            {
                int estimation_time_idx = std::floor(m_time_idx / m_options.topology.Mt);
                for (unsigned int i = estimation_time_idx * m_options.topology.N; i < (estimation_time_idx + 1) * m_options.topology.N; i++)
                {
                    unsigned int m = i - estimation_time_idx * m_options.topology.N;
                    vtkSmartPointer<vtkMatrix4x4> sphere_frame_vtk;
                    Eigen::Vector3d s;

                    if (!m_state.estimation_nodes[i].getCovariance().isZero())
                    {
                        getSpherePose(m_state.estimation_nodes[i], sphere_frame_vtk, s);

                        mp_ellipsoid_actors_k[m]->SetVisibility(true);
                        mp_ellipsoid_actors_k[m]->SetUserMatrix(sphere_frame_vtk);
                        mp_ellipsoid_actors_k[m]->SetScale(s(0), s(1), s(2));
                    }
                    if ((size_t)i + m_options.topology.N < m_state.estimation_nodes.size())
                    {
                        if (m_options.robot.show_interpolation && !m_state.estimation_nodes[i + m_options.topology.N].getCovariance().isZero() && m_options.topology.Mt > 1)
                        {
                            getSpherePose(m_state.estimation_nodes[i + m_options.topology.N], sphere_frame_vtk, s);

                            mp_ellipsoid_actors_k1[m]->SetVisibility(true);
                            mp_ellipsoid_actors_k1[m]->SetUserMatrix(sphere_frame_vtk);
                            mp_ellipsoid_actors_k1[m]->SetScale(s(0), s(1), s(2));
                        }
                        else
                        {
                            mp_ellipsoid_actors_k1[m]->SetVisibility(false);
                        }
                    }
                }
            }
            else
            {
                // Hide all ellipsoids
                for (unsigned int i = 0; i < mp_ellipsoid_actors_k.size(); i++)
                {
                    mp_ellipsoid_actors_k[i]->SetVisibility(false);
                    mp_ellipsoid_actors_k1[i]->SetVisibility(false);
                }
            }
        }
        else
        {
            // Hide all ellipsoids
            for (unsigned int i = 0; i < mp_ellipsoid_actors.size(); i++)
            {
                mp_ellipsoid_actors[i]->SetVisibility(false);
            }
            // Hide all ellipsoids
            for (unsigned int i = 0; i < mp_ellipsoid_actors_k.size(); i++)
            {
                mp_ellipsoid_actors_k[i]->SetVisibility(false);
                mp_ellipsoid_actors_k1[i]->SetVisibility(false);
            }
        }

        // Draw map
        if (m_show_map && mp_map)
        {
            steam_icp::ArrayPoint3D points = mp_map->getPointCloudAtTime(m_time);
            // Update map points
            if (m_time_idx == 0)
            {
                for (unsigned int i = 0; i < mp_map->size(); i++)
                {
                    mp_map_points->SetPoint(i, 0, 0, 0); // Reset points at first time
                }
            }

            for (unsigned int i = 0; i < points.size(); i++)
            {
                mp_map_points->SetPoint(i, points[i].pt(0), points[i].pt(1), points[i].pt(2));
            }
            mp_map_points->Modified();
        }

        // Update scene
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();
    }

    void Visualizer::slotAnimationTimer()
    {
        if (m_reset)
        {
            info() << "Resetting animation";
            m_reset = false;
            m_time_idx = 0;
        }
        if (m_play)
        {
            m_last_time = std::chrono::high_resolution_clock::now();
            m_time_idx++;
        }
        if (m_time_idx >= ((m_topology_K - 1) * m_options.topology.Mt + 1))
        {
            // Press space to reset
            m_time_idx--;
            m_play = false;
            m_reset = true;
        }
        m_time = m_state.interpolation_nodes[m_time_idx * ((m_options.topology.N - 1) * m_options.topology.Ms + 1)].time;

        update();
        if (m_recording)
            screenshot();

        m_animation_timer.start(getDuration());
    }

    void Visualizer::initScene()
    {
        info() << "Visualizer::initScene()";
        // initialize background
        mp_ren->SetBackground(1., 1., 1.);

        m_infoLabel = new QLabel(this);
        m_infoLabel->setAlignment(Qt::AlignCenter);
        m_infoLabel->setStyleSheet("QLabel { background-color : white; color : black; font-size: 16px; font-weight: bold; }");
        m_infoLabel->setGeometry(this->width() / 2 - 150, 10, 300, 50);
        m_infoLabel->setVisible(false);

        // initialize lights
        initLights();

        // initialize floor
        initFloor();

        // initialize backbones
        initBackbones();

        // initialize robot frames
        initRobotFrames();

        // initialize world frame
        initWorldFrame();

        // initialize covariance ellipsoids
        initCovarianceEllipsoids();

        // initialize measurements
        initMeasurements();

        // initialize ground truth
        initGroundTruth();

        // initialize environment map
        initEnvironmentMap();

        // initialize lidar visualization
        initLidarPoints();

        // intialize sensor modules
        initSensorModules();

        // initialize camera view
        resetCameraView();

        // Update scene
        mp_renWin->Render();
    }

    void Visualizer::initLights()
    {
        vtkSmartPointer<vtkLight> light1 = vtkSmartPointer<vtkLight>::New();
        light1->SetLightTypeToSceneLight();
        light1->SetPosition(0, 0, 1);
        light1->SetFocalPoint(0, 0, 0);
        light1->SetColor(1.0, 1.0, 1.0);
        light1->SetIntensity(0.6);
        mp_ren->AddLight(light1);

        vtkSmartPointer<vtkLight> light2 = vtkSmartPointer<vtkLight>::New();
        light2->SetLightTypeToSceneLight();
        light2->SetPosition(-1, 1, 1);
        light2->SetFocalPoint(0, 0, 0);
        light2->SetColor(1.0, 1.0, 1.0);
        light2->SetIntensity(0.8);
        mp_ren->AddLight(light2);

        vtkSmartPointer<vtkLight> light3 = vtkSmartPointer<vtkLight>::New();
        light3->SetLightTypeToSceneLight();
        light3->SetPosition(-1, -1, -1);
        light3->SetFocalPoint(0, 0, 0);
        light3->SetColor(1.0, 1.0, 1.0);
        light3->SetIntensity(0.4);
        mp_ren->AddLight(light3);
    }

    void Visualizer::initFloor()
    {
        vtkSmartPointer<vtkRegularPolygonSource> floorPoly = vtkSmartPointer<vtkRegularPolygonSource>::New();
        floorPoly->SetNumberOfSides(4);
        floorPoly->SetNormal(0, 0, 1);
        floorPoly->SetCenter(0, 0, 0.0);
        floorPoly->SetRadius(0.5);

        vtkSmartPointer<vtkPolyDataMapper> floorMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        floorMapper->SetInputConnection(floorPoly->GetOutputPort());

        mp_floor = vtkSmartPointer<vtkActor>::New();
        mp_floor->SetMapper(floorMapper);
        mp_floor->GetProperty()->SetColor(0.9, 0.9, 0.9);
        mp_floor->GetProperty()->SetAmbient(0.3);
        mp_floor->GetProperty()->SetDiffuse(0.5);
        mp_floor->GetProperty()->SetSpecular(0.1);
        mp_floor->RotateZ(45.0);

        mp_ren->AddActor(mp_floor);
    }

    void Visualizer::initBackbones()
    {
        mp_backbone_points = vtkSmartPointer<vtkPoints>::New();
        for (unsigned int i = 0; i < (m_options.topology.N - 1) * m_options.topology.Ms + 1; i++)
        {
            mp_backbone_points->InsertPoint(i, 0, 0, 0);
        }

        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        polydata->SetPoints(mp_backbone_points);
        polydata->Allocate();

        for (unsigned int m = 0; m < (m_options.topology.N - 1) * m_options.topology.Ms; m++)
        {
            vtkIdType connectivity[2] = {m, m + 1};
            polydata->InsertNextCell(VTK_LINE, 2, connectivity);
        }

        // Create scalar array for colors
        vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetName("Colors");
        colors->SetNumberOfComponents(3);
        colors->SetNumberOfTuples(polydata->GetNumberOfCells());

        // Assign alternating colors: black and dark grey
        for (vtkIdType i = 0; i < polydata->GetNumberOfCells(); i++)
        {
            if (i % 2 == 0)
            {
                // Black
                colors->SetTuple3(i, 0, 0, 0);
            }
            else
            {
                // Dark grey
                colors->SetTuple3(i, 64, 64, 64);
            }
        }

        polydata->GetCellData()->SetScalars(colors);

        vtkSmartPointer<vtkTubeFilter> tubeFilter = vtkSmartPointer<vtkTubeFilter>::New();
        tubeFilter->SetInputData(polydata);
        tubeFilter->SetRadius(m_options.topology.radius);
        tubeFilter->SetNumberOfSides(50);

        vtkSmartPointer<vtkPolyDataMapper> backboneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        backboneMapper->SetInputConnection(tubeFilter->GetOutputPort());
        backboneMapper->SetScalarModeToUseCellData();

        mp_backbone_actor = vtkSmartPointer<vtkActor>::New();
        mp_backbone_actor->SetMapper(backboneMapper);
        mp_backbone_actor->GetProperty()->SetAmbient(0.3);
        mp_backbone_actor->GetProperty()->SetDiffuse(0.5);
        mp_backbone_actor->GetProperty()->SetSpecular(0.1);

        mp_ren->AddActor(mp_backbone_actor);
    }

    void Visualizer::initRobotFrames()
    {
        mp_axes_k.clear();
        mp_axes_k1.clear();
        mp_axes_interp.clear();
        for (unsigned int i = 0; i < m_options.topology.N; i++)
        {
            vtkSmartPointer<vtkAxesActor> robot_axes_k = createAxesActor(std::max(0.025, 0.25 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius));
            mp_axes_k.push_back(robot_axes_k);
            mp_ren->AddActor(robot_axes_k);

            vtkSmartPointer<vtkAxesActor> robot_axes_k1 = createAxesActor(std::max(0.025, 0.25 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius));
            mp_axes_k1.push_back(robot_axes_k1);
            mp_ren->AddActor(robot_axes_k1);
        }
        for (unsigned int i = 0; i < (m_options.topology.N - 1) * m_options.topology.Ms + 1; i++)
        {
            vtkSmartPointer<vtkAxesActor> robot_axes_interp = createAxesActor(std::max(0.025, 0.25 * m_options.topology.radius), std::max(0.015, 1.5 * m_options.topology.radius));
            mp_axes_interp.push_back(robot_axes_interp);
            mp_ren->AddActor(robot_axes_interp);
        }
    }

    void Visualizer::initWorldFrame()
    {
        vtkSmartPointer<vtkAxesActor> world_axes = createAxesActor(0.05, 0.01);
        mp_ren->AddActor(world_axes);
    }

    void Visualizer::initCovarianceEllipsoids()
    {
        mp_ellipsoid_actors.clear();
        mp_ellipsoid_actors_k.clear();
        mp_ellipsoid_actors_k1.clear();

        vtkSmartPointer<vtkSphereSource> ellipsoid_source = vtkSmartPointer<vtkSphereSource>::New();
        ellipsoid_source->SetCenter(0.0, 0.0, 0.0);
        ellipsoid_source->SetRadius(1);
        ellipsoid_source->SetPhiResolution(100);
        ellipsoid_source->SetThetaResolution(100);

        vtkSmartPointer<vtkPolyDataMapper> ellipsoid_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        ellipsoid_mapper->SetInputConnection(ellipsoid_source->GetOutputPort());

        for (unsigned int n = 0; n < (m_options.topology.N - 1) * m_options.topology.Ms + 1; n++)
        {
            vtkSmartPointer<vtkActor> ellipsoid_actor = createEllipsoidActor(ellipsoid_mapper);
            mp_ellipsoid_actors.push_back(ellipsoid_actor);
            mp_ren->AddActor(ellipsoid_actor);
        }

        for (unsigned int n = 0; n < m_options.topology.N; n++)
        {
            vtkSmartPointer<vtkActor> ellipsoid_actor_k = createEllipsoidActor(ellipsoid_mapper, true);
            mp_ellipsoid_actors_k.push_back(ellipsoid_actor_k);
            mp_ren->AddActor(ellipsoid_actor_k);

            vtkSmartPointer<vtkActor> ellipsoid_actor_k1 = createEllipsoidActor(ellipsoid_mapper, true);
            mp_ellipsoid_actors_k1.push_back(ellipsoid_actor_k1);
            mp_ren->AddActor(ellipsoid_actor_k1);
        }
    }

    void Visualizer::initMeasurements()
    {
        mp_aurora_meas.clear();

        vtkNew<vtkSphereSource> measSphereSource;
        measSphereSource->SetCenter(0.0, 0.0, 0.0);
        measSphereSource->SetRadius(0.003);
        measSphereSource->SetPhiResolution(100);
        measSphereSource->SetThetaResolution(100);

        vtkNew<vtkPolyDataMapper> sphereMapper;
        sphereMapper->SetInputConnection(measSphereSource->GetOutputPort());

        for (unsigned int i = 0; i < m_options.aurora.count; i++)
        {
            vtkSmartPointer<vtkAxesActor> meas = createAxesActor(0.5, 0.008);
            // vtkSmartPointer<vtkActor> meas = vtkSmartPointer<vtkActor>::New();
            // meas->SetMapper(sphereMapper);
            // meas->GetProperty()->SetColor(colors->GetColor3d("Cornsilk").GetData());
            mp_aurora_meas.push_back(meas);
            mp_ren->AddActor(meas);
        }
    }

    void Visualizer::initGroundTruth()
    {
        mp_gt.clear();
        for (unsigned int i = 0; i < m_options.vicon.count; i++)
        {
            vtkSmartPointer<vtkAxesActor> gt_meas = createAxesActor(std::max(0.025, 0.25 * m_options.topology.radius) * m_options.vicon.size, std::max(0.015, 1.5 * m_options.topology.radius) * m_options.vicon.size);
            mp_gt.push_back(gt_meas);
            mp_ren->AddActor(gt_meas);
        }
    }

    void Visualizer::initEnvironmentMap()
    {
        if (m_options.environment.path.empty())
            return;
        vtkNew<vtkSTLReader> reader;
        reader->SetFileName(m_options.environment.path.c_str());
        reader->Update();

        // Visualize

        vtkNew<vtkTransform> transform;
        transform->Scale(m_options.environment.scale, m_options.environment.scale, m_options.environment.scale); // Scale down by 1000x

        vtkNew<vtkTransformPolyDataFilter> transformFilter;
        transformFilter->SetInputConnection(reader->GetOutputPort());
        transformFilter->SetTransform(transform);
        transformFilter->Update();

        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(transformFilter->GetOutputPort());

        mp_environment_actor = vtkSmartPointer<vtkActor>::New();
        mp_environment_actor->SetMapper(mapper);
        mp_environment_actor->GetProperty()->SetAmbient(0.1); // Increased ambient lighting
        mp_environment_actor->GetProperty()->SetDiffuse(0.7); // Reduced diffuse
        mp_environment_actor->GetProperty()->SetDiffuseColor(
            colors->GetColor3d("LightSteelBlue").GetData());
        mp_environment_actor->GetProperty()->SetSpecular(0.5);       // Reduced specular
        mp_environment_actor->GetProperty()->SetSpecularPower(20.0); // Reduced specular power
        mp_environment_actor->SetVisibility(m_options.environment.show);
        mp_environment_actor->GetProperty()->SetOpacity(m_options.environment.opacity);
        mp_ren->AddActor(mp_environment_actor);

        // Get bounds from environment actor
        double *env_bounds = nullptr;
        double grid_spacing = m_options.estimator_options.icp_options.VOXEL_SIZE;

        if (mp_environment_actor && mp_environment_actor->GetVisibility())
        {
            env_bounds = mp_environment_actor->GetBounds();
        }
        else
        {
            // Fallback to default bounds if environment actor is not available
            static double default_bounds[6] = {-0.5, 0.5, -0.5, 0.5, -0.5, 0.5};
            env_bounds = default_bounds;
        }

        // Create 3D wireframe grid
        vtkSmartPointer<vtkPoints> grid_points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> grid_lines = vtkSmartPointer<vtkCellArray>::New();

        // Extract bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
        double x_min = env_bounds[0], x_max = env_bounds[1];
        double y_min = env_bounds[2], y_max = env_bounds[3];
        double z_min = env_bounds[4], z_max = env_bounds[5];

        // Calculate number of grid lines for each dimension
        int num_lines_x = static_cast<int>((x_max - x_min) / grid_spacing) + 1;
        int num_lines_y = static_cast<int>((y_max - y_min) / grid_spacing) + 1;
        int num_lines_z = static_cast<int>((z_max - z_min) / grid_spacing) + 1;

        vtkIdType point_id = 0;

        // Create horizontal lines (parallel to X-axis) in XY plane
        for (int i = 0; i < num_lines_y; i++)
        {
            for (int k = 0; k < num_lines_z; k++)
            {
                double y = y_min + i * grid_spacing;
                double z = z_min + k * grid_spacing;

                // Add start point
                grid_points->InsertNextPoint(x_min, y, z);
                vtkIdType start_id = point_id++;

                // Add end point
                grid_points->InsertNextPoint(x_max, y, z);
                vtkIdType end_id = point_id++;

                // Create line
                vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
                line->GetPointIds()->SetId(0, start_id);
                line->GetPointIds()->SetId(1, end_id);
                grid_lines->InsertNextCell(line);
            }
        }

        // Create vertical lines (parallel to Y-axis) in XY and YZ planes
        for (int i = 0; i < num_lines_x; i++)
        {
            for (int k = 0; k < num_lines_z; k++)
            {
                double x = x_min + i * grid_spacing;
                double z = z_min + k * grid_spacing;

                // Add start point
                grid_points->InsertNextPoint(x, y_min, z);
                vtkIdType start_id = point_id++;

                // Add end point
                grid_points->InsertNextPoint(x, y_max, z);
                vtkIdType end_id = point_id++;

                // Create line
                vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
                line->GetPointIds()->SetId(0, start_id);
                line->GetPointIds()->SetId(1, end_id);
                grid_lines->InsertNextCell(line);
            }
        }

        // Create depth lines (parallel to Z-axis) in XZ and YZ planes
        for (int i = 0; i < num_lines_x; i++)
        {
            for (int j = 0; j < num_lines_y; j++)
            {
                double x = x_min + i * grid_spacing;
                double y = y_min + j * grid_spacing;

                // Add start point
                grid_points->InsertNextPoint(x, y, z_min);
                vtkIdType start_id = point_id++;

                // Add end point
                grid_points->InsertNextPoint(x, y, z_max);
                vtkIdType end_id = point_id++;

                // Create line
                vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
                line->GetPointIds()->SetId(0, start_id);
                line->GetPointIds()->SetId(1, end_id);
                grid_lines->InsertNextCell(line);
            }
        }

        // Create polydata for the grid
        vtkSmartPointer<vtkPolyData> grid_polydata = vtkSmartPointer<vtkPolyData>::New();
        grid_polydata->SetPoints(grid_points);
        grid_polydata->SetLines(grid_lines);

        // Create mapper
        vtkSmartPointer<vtkPolyDataMapper> grid_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        grid_mapper->SetInputData(grid_polydata);

        // Create actor
        mp_grid_actor = vtkSmartPointer<vtkActor>::New();
        mp_grid_actor->SetMapper(grid_mapper);
        mp_grid_actor->GetProperty()->SetColor(0.5, 0.5, 0.5); // Gray color
        mp_grid_actor->GetProperty()->SetLineWidth(1.0);
        mp_grid_actor->GetProperty()->SetOpacity(0.3);

        mp_ren->AddActor(mp_grid_actor);
    }

    void Visualizer::initLidarPoints()
    {
        // Active points (green)
        mp_lidar_points_active = vtkSmartPointer<vtkPoints>::New();
        for (unsigned int i = 0; i < 64 * m_options.lidar.count; i++) // This is hacky
        {
            mp_lidar_points_active->InsertNextPoint(0, 0, 0);
        }

        vtkSmartPointer<vtkPolyData> polydata_active = vtkSmartPointer<vtkPolyData>::New();
        polydata_active->SetPoints(mp_lidar_points_active);

        vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter_active = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertexFilter_active->SetInputData(polydata_active);
        vertexFilter_active->Update();

        vtkSmartPointer<vtkPolyDataMapper> lidarMapper_active = vtkSmartPointer<vtkPolyDataMapper>::New();
        lidarMapper_active->SetInputConnection(vertexFilter_active->GetOutputPort());

        mp_lidar_actor_active = vtkSmartPointer<vtkActor>::New();
        mp_lidar_actor_active->SetMapper(lidarMapper_active);
        mp_lidar_actor_active->GetProperty()->SetColor(0, 1, 0);
        mp_lidar_actor_active->GetProperty()->SetPointSize(m_options.lidar.size);
        mp_lidar_actor_active->GetProperty()->SetAmbient(0.3);
        mp_lidar_actor_active->GetProperty()->SetDiffuse(0.5);
        mp_lidar_actor_active->GetProperty()->SetSpecular(0.1);
        mp_ren->AddActor(mp_lidar_actor_active);

        // Inactive lidar points (red)
        mp_lidar_points_inactive = vtkSmartPointer<vtkPoints>::New();
        for (unsigned int i = 0; i < 64 * m_options.lidar.count; i++) // This is hacky
        {
            mp_lidar_points_inactive->InsertNextPoint(0, 0, 0);
        }

        vtkSmartPointer<vtkPolyData> polydata_inactive = vtkSmartPointer<vtkPolyData>::New();
        polydata_inactive->SetPoints(mp_lidar_points_inactive);

        vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter_inactive = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertexFilter_inactive->SetInputData(polydata_inactive);
        vertexFilter_inactive->Update();

        vtkSmartPointer<vtkPolyDataMapper> lidarMapper_inactive = vtkSmartPointer<vtkPolyDataMapper>::New();
        lidarMapper_inactive->SetInputConnection(vertexFilter_inactive->GetOutputPort());

        mp_lidar_actor_inactive = vtkSmartPointer<vtkActor>::New();
        mp_lidar_actor_inactive->SetMapper(lidarMapper_inactive);
        mp_lidar_actor_inactive->GetProperty()->SetColor(1, 0, 0);
        mp_lidar_actor_inactive->GetProperty()->SetPointSize(m_options.lidar.size);
        mp_lidar_actor_inactive->GetProperty()->SetAmbient(0.3);
        mp_lidar_actor_inactive->GetProperty()->SetDiffuse(0.5);
        mp_lidar_actor_inactive->GetProperty()->SetSpecular(0.1);
        mp_ren->AddActor(mp_lidar_actor_inactive);
    }

    void Visualizer::initMap()
    {
        if (!mp_map)
            return;

        // Active points (green)
        mp_map_points = vtkSmartPointer<vtkPoints>::New();
        for (unsigned int i = 0; i < mp_map->size(); i++) // This is hacky
        {
            mp_map_points->InsertNextPoint(0, 0, 0);
        }

        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        polydata->SetPoints(mp_map_points);

        vtkSmartPointer<vtkVertexGlyphFilter> vertex_filter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertex_filter->SetInputData(polydata);
        vertex_filter->Update();

        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(vertex_filter->GetOutputPort());

        mp_map_actor = vtkSmartPointer<vtkActor>::New();
        mp_map_actor->SetMapper(mapper);
        mp_map_actor->GetProperty()->SetColor(0, 0, 1);
        mp_map_actor->GetProperty()->SetPointSize(m_options.lidar.size);
        mp_map_actor->GetProperty()->SetAmbient(0.3);
        mp_map_actor->GetProperty()->SetDiffuse(0.5);
        mp_map_actor->GetProperty()->SetSpecular(0.1);
        mp_ren->AddActor(mp_map_actor);
    }

    void Visualizer::initSensorModules()
    {
        mp_sensor_module_actors.clear();
        for (unsigned int i = 0; i < m_sensor_module_arclengths.size(); i++)
        {
            vtkSmartPointer<vtkAssembly> module_assembly = vtkSmartPointer<vtkAssembly>::New();
            vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkSmartPointer<vtkCylinderSource>::New();
            cylinderSource->SetRadius(m_options.topology.radius + 0.003);
            cylinderSource->SetHeight(0.015);
            cylinderSource->SetResolution(50);

            vtkSmartPointer<vtkPolyDataMapper> cylinderMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            cylinderMapper->SetInputConnection(cylinderSource->GetOutputPort());

            vtkSmartPointer<vtkActor> cylinderActor = vtkSmartPointer<vtkActor>::New();
            cylinderActor->SetMapper(cylinderMapper);
            cylinderActor->GetProperty()->SetColor(1.0, 0.0, 0.0); // Red
            cylinderActor->GetProperty()->SetAmbient(0.3);
            cylinderActor->GetProperty()->SetDiffuse(0.5);
            cylinderActor->GetProperty()->SetSpecular(0.1);

            module_assembly->AddPart(cylinderActor);

            {
                // create three small black cubes placed on the cylinder outer radius at 120° intervals
                double sphere_radius = 0.003;
                double radial_offset = m_options.topology.radius + 0.003;
                double angles[3] = {0.0 + vtkMath::Pi() / 3.0 * (i % 2 == 1), 2.0 * vtkMath::Pi() / 3.0 + vtkMath::Pi() / 3.0 * (i % 2 == 1), 4.0 * vtkMath::Pi() / 3.0 + vtkMath::Pi() / 3.0 * (i % 2 == 1)};

                for (int ci = 0; ci < 3; ++ci)
                {

                    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
                    sphereSource->SetCenter(0.0, 0.0, 0.0);
                    sphereSource->SetRadius(sphere_radius);
                    sphereSource->SetPhiResolution(20);
                    sphereSource->SetThetaResolution(20);
                    sphereSource->Update();

                    vtkSmartPointer<vtkPolyDataMapper> sphereMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
                    sphereMapper->SetInputConnection(sphereSource->GetOutputPort());

                    vtkSmartPointer<vtkActor> sphere_actor = vtkSmartPointer<vtkActor>::New();
                    sphere_actor->SetMapper(sphereMapper);
                    sphere_actor->GetProperty()->SetColor(0.0, 0.0, 0.0); // black
                    sphere_actor->GetProperty()->SetAmbient(0.3);
                    sphere_actor->GetProperty()->SetDiffuse(0.5);
                    sphere_actor->GetProperty()->SetSpecular(0.1);

                    double a = angles[ci];
                    double x = radial_offset * std::cos(a);
                    double y = 0.0; //radial_offset * std::sin(a);
                    double z = - radial_offset * std::sin(a); // center on cylinder mid-height
                    sphere_actor->SetPosition(x, y, z);

                    module_assembly->AddPart(sphere_actor);


                    // { // add a line from the center of the cylinder to the sphere
                    //     vtkSmartPointer<vtkPoints> linePoints = vtkSmartPointer<vtkPoints>::New();
                    //     linePoints->InsertNextPoint(0.0, 0.0, 0.0);
                    //     double lx = std::cos(a);
                    //     double lz = -std::sin(a);
                    //     linePoints->InsertNextPoint(lx, 0.0, lz);

                    //     vtkSmartPointer<vtkCellArray> lineCells = vtkSmartPointer<vtkCellArray>::New();
                    //     vtkSmartPointer<vtkLine> vline = vtkSmartPointer<vtkLine>::New();
                    //     vline->GetPointIds()->SetId(0, 0);
                    //     vline->GetPointIds()->SetId(1, 1);
                    //     lineCells->InsertNextCell(vline);

                    //     vtkSmartPointer<vtkPolyData> linePoly = vtkSmartPointer<vtkPolyData>::New();
                    //     linePoly->SetPoints(linePoints);
                    //     linePoly->SetLines(lineCells);

                    //     vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
                    //     lineMapper->SetInputData(linePoly);

                    //     vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
                    //     lineActor->SetMapper(lineMapper);
                    //     lineActor->GetProperty()->SetColor(1.0, 1.0, 0.0); // yellow
                    //     lineActor->GetProperty()->SetLineWidth(2.0);

                    //     module_assembly->AddPart(lineActor);
                    // }
                }
            }

            mp_sensor_module_actors.push_back(module_assembly);
            mp_ren->AddActor(module_assembly);
        }
    }

    void Visualizer::resetCameraView()
    {
        mp_ren->GetActiveCamera()->SetPosition(m_options.cameras[0].position(0), m_options.cameras[0].position(1), m_options.cameras[0].position(2));
        mp_ren->GetActiveCamera()->SetFocalPoint(m_options.cameras[0].focal_point(0), m_options.cameras[0].focal_point(1), m_options.cameras[0].focal_point(2));
        mp_ren->GetActiveCamera()->SetViewUp(m_options.cameras[0].view_up(0), m_options.cameras[0].view_up(1), m_options.cameras[0].view_up(2));
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();
    }

    vtkSmartPointer<vtkAxesActor> Visualizer::createAxesActor(double cylinderRadius, double totalLength)
    {
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
        axes->SetXAxisLabelText("");
        axes->SetYAxisLabelText("");
        axes->SetZAxisLabelText("");
        axes->SetShaftTypeToCylinder();
        axes->SetCylinderRadius(cylinderRadius);
        axes->SetTotalLength(totalLength, totalLength, totalLength);
        axes->SetVisibility(true);
        return axes;
    }

    vtkSmartPointer<vtkActor> Visualizer::createEllipsoidActor(vtkSmartPointer<vtkPolyDataMapper> mapper, bool red)
    {
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        if (red)
            actor->GetProperty()->SetColor(1, 0, 0);
        else
            actor->GetProperty()->SetColor(0, 0, 1);
        actor->GetProperty()->SetOpacity(0.15);
        actor->GetProperty()->SetAmbient(0.3);
        actor->GetProperty()->SetDiffuse(0.5);
        actor->GetProperty()->SetSpecular(0.1);
        actor->SetVisibility(false);
        return actor;
    }

    void Visualizer::screenshot()
    {
        std::filesystem::path screenshots_folder = std::filesystem::path(m_output_path + "screenshots");
        if (!std::filesystem::exists(screenshots_folder))
            std::filesystem::create_directory(screenshots_folder);

        // Take screenshot from current view
        std::filesystem::path current_view_folder = screenshots_folder / "current_view";
        if (!std::filesystem::exists(current_view_folder))
            std::filesystem::create_directories(current_view_folder);

        QPixmap pixmap = this->grab();
        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGBA8888);
        QImage flippedImage = image.mirrored(false, true); // Flip vertically to correct VTK coordinate system

        vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
        imageData->SetDimensions(flippedImage.width(), flippedImage.height(), 1);
        imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 4);

        // Copy the image data
        unsigned char *vtkPixels = static_cast<unsigned char *>(imageData->GetScalarPointer());
        memcpy(vtkPixels, flippedImage.constBits(), flippedImage.sizeInBytes());

        m_screenshot_writer->SetInputData(imageData);

        std::filesystem::path current_view_file = current_view_folder / (std::to_string(m_time_idx) + ".png");
        m_screenshot_writer->SetFileName(current_view_file.c_str());
        m_screenshot_writer->Write();

        if (m_verbose)
        {
            info() << "Screenshot saved for current view at " << current_view_file;
        }

        // Take screenshots from all configured camera angles
        for (size_t i = 1; i < m_options.cameras.size(); i++)
        {
            std::string subfolder_name = "camera_" + std::to_string(i);
            screenshotFromCamera(m_options.cameras[i], subfolder_name);
        }

        if (m_options.cameras.empty() && m_verbose)
        {
            info() << "No camera configurations defined. Only current view screenshot taken.";
        }
    }

    void Visualizer::screenshotFromCamera(const VisualizerOptions::CameraOptions &camera_config, const std::string &subfolder_name)
    {
        // Store current camera state
        double current_pos[3], current_focal[3], current_up[3];
        mp_ren->GetActiveCamera()->GetPosition(current_pos);
        mp_ren->GetActiveCamera()->GetFocalPoint(current_focal);
        mp_ren->GetActiveCamera()->GetViewUp(current_up);

        // Set camera to the specified configuration
        mp_ren->GetActiveCamera()->SetPosition(camera_config.position(0), camera_config.position(1), camera_config.position(2));
        mp_ren->GetActiveCamera()->SetFocalPoint(camera_config.focal_point(0), camera_config.focal_point(1), camera_config.focal_point(2));
        mp_ren->GetActiveCamera()->SetViewUp(camera_config.view_up(0), camera_config.view_up(1), camera_config.view_up(2));
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();

        // Create subfolder path
        std::filesystem::path screenshots_folder = std::filesystem::path(m_output_path + "screenshots");
        std::filesystem::path subfolder_path = screenshots_folder / subfolder_name;
        if (!std::filesystem::exists(subfolder_path))
            std::filesystem::create_directories(subfolder_path);

        // Take screenshot
        QPixmap pixmap = this->grab();
        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGBA8888);
        QImage flippedImage = image.mirrored(false, true); // Flip vertically to correct VTK coordinate system

        vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
        imageData->SetDimensions(flippedImage.width(), flippedImage.height(), 1);
        imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 4);

        // Copy the image data
        unsigned char *vtkPixels = static_cast<unsigned char *>(imageData->GetScalarPointer());
        memcpy(vtkPixels, flippedImage.constBits(), flippedImage.sizeInBytes());

        m_screenshot_writer->SetInputData(imageData);

        std::filesystem::path file_path = subfolder_path / (std::to_string(m_time_idx) + ".png");
        m_screenshot_writer->SetFileName(file_path.c_str());
        m_screenshot_writer->Write();

        // Restore original camera state
        mp_ren->GetActiveCamera()->SetPosition(current_pos);
        mp_ren->GetActiveCamera()->SetFocalPoint(current_focal);
        mp_ren->GetActiveCamera()->SetViewUp(current_up);
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();

        if (m_verbose)
        {
            info() << "Screenshot saved for camera: " << subfolder_name << " at " << file_path;
        }
    }
} // namespace Spacetime