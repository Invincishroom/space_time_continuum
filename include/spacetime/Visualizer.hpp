#pragma once

#include <chrono>
#include <sstream>
#include <functional>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <QObject>
#include <QVTKOpenGLNativeWidget.h>
#include <QTimer>
#include <QEvent>
#include <QPaintEvent>
#include <QDomDocument>
#include <QDomElement>
#include <QDomNodeList>
#include <QDomNode>
#include <QMouseEvent>
#include <QDebug>
#include <QString>
#include <QCoreApplication>
#include <QDateTime>
#include <QLabel>
#include <QSlider>
#include <QPainter>
#include <QPixmap>
#include <QVBoxLayout>
#include <QScrollArea>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkAxesActor.h>
#include <vtkAssembly.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkPoints.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkDoubleArray.h>
#include <vtkRegularPolygonSource.h>
#include <vtkLight.h>
#include <vtkOpenGLLight.h>
#include <vtkSmartPointer.h>
#include <vtkAxesActor.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkTransform.h>
#include <vtkPolyDataReader.h>
#include <QVTKInteractorAdapter.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkLine.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkTubeFilter.h>
#include <vtkLineSource.h>
#include <vtkPointData.h>
#include <vtkCubeSource.h>
#include <vtkSmartPointer.h>
#include <vtkSelectEnclosedPoints.h>
#include <vtkIntArray.h>
#include <vtkDataArray.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyLine.h>
#include <vtkSTLReader.h>
#include <vtkMath.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkMassProperties.h>
#include <vtkOBBTree.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>
#include <vtkDataSet.h>
#include <vtkInformation.h>
#include <vtkQuaternionInterpolator.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkConvexPointSet.h>
#include <vtkDataSetMapper.h>
#include <vtkGlyph3DMapper.h>
#include <vtkNamedColors.h>
#include <vtkPoints.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkShrinkFilter.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkCylinderSource.h>
#include <vtkMatrix4x4.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkSTLReader.h>
#include <vtkGenericDataObjectWriter.h>
#include <vtkUnstructuredGridReader.h>
#include "spacetime/types.hpp"
#include "spacetime/factors.hpp"
#include "spacetime/estimators/Estimator.hpp"
#include "spacetime/visualizer/PlotWidget.hpp"
#include "spacetime/visualizer/TimelineWidget.hpp"
#include "spacetime/visualizer/InfoWidget.hpp"

namespace Spacetime
{
    class Visualizer : public QVTKOpenGLNativeWidget
    {
        Q_OBJECT

        friend void onLeftButtonDown(vtkObject *caller, long unsigned int eventId, void *clientData, void *callData);

    public:
        Visualizer(VisualizerOptions options, std::string output_path = "");
        ~Visualizer();
        void sendMessage(QString message);
        void sendMessage(QString message, int duration);

        void animate(const Spacetime::Estimator::Results &result, const std::vector<std::shared_ptr<Factors::MeasurementFactor>> &measurements = {}, const std::vector<Spacetime::SensorMeasurement> &ground_truth = {});

    public slots:
        void slotAnimationTimer();
        void hideMessage();

    private:
        VisualizerOptions m_options;
        void update();
        void initScene();
        void getSpherePose(Spacetime::SystemState<double>::Node node, vtkSmartPointer<vtkMatrix4x4> &sphere_frame_vtk, Eigen::Vector3d &s);
        int getDuration();
        void screenshot();
        void screenshotFromCamera(const Spacetime::VisualizerOptions::CameraOptions &camera_config, const std::string &subfolder_name);

        // initialization functions
        void initLights();
        void initFloor();
        void initBackbones();
        void initRobotFrames();
        void initWorldFrame();
        void initCovarianceEllipsoids();
        void initMeasurements();
        void initGroundTruth();
        void initEnvironmentMap();
        void initLidarPoints();
        void initMap();
        void resetCameraView();
        void initSensorModules(); 

        std::vector<double> m_sensor_module_arclengths = {0.168, 0.345, 0.521};
        std::vector<vtkSmartPointer<vtkAssembly>> mp_sensor_module_actors;

        vtkSmartPointer<vtkAxesActor> createAxesActor(double cylinderRadius, double totalLength);
        vtkSmartPointer<vtkActor> createEllipsoidActor(vtkSmartPointer<vtkPolyDataMapper> mapper, bool red = false);

        vtkSmartPointer<vtkRenderWindow> mp_renWin;
        vtkSmartPointer<vtkRenderer> mp_ren;

        // Animation data
        Spacetime::SystemState<double> m_state;
        std::vector<std::shared_ptr<Factors::MeasurementFactor>> m_measurements = {};
        std::vector<Spacetime::SensorMeasurement> m_gt = {};

        // Stuff to visualize

        std::vector<vtkSmartPointer<vtkAxesActor>> mp_axes_k, mp_axes_k1, mp_axes_interp;
        vtkSmartPointer<vtkAxesActor> m_selected_frame;
        double m_selected_frame_arclength;
        std::vector<vtkSmartPointer<vtkAxesActor>> mp_aurora_meas;
        std::vector<vtkSmartPointer<vtkAxesActor>> mp_gt;

        vtkSmartPointer<vtkPoints> mp_backbone_points;
        vtkSmartPointer<vtkActor> mp_backbone_actor;
        vtkSmartPointer<vtkActor> mp_environment_actor;

        std::vector<vtkSmartPointer<vtkPoints>> mp_coupling_points;
        std::vector<vtkSmartPointer<vtkActor>> mp_coupling_actors;

        std::vector<vtkSmartPointer<vtkActor>> mp_joint_actors;

        vtkSmartPointer<vtkActor> mp_floor;
        vtkSmartPointer<vtkTextActor> mp_label;

        vtkSmartPointer<vtkPoints> mp_lidar_points_active, mp_lidar_points_inactive, mp_map_points;
        vtkSmartPointer<vtkActor> mp_lidar_actor_active, mp_lidar_actor_inactive, mp_map_actor, mp_grid_actor;
        std::shared_ptr<steam_icp::Map> mp_map;

        std::vector<vtkSmartPointer<vtkActor>> mp_ellipsoid_actors;
        std::vector<vtkSmartPointer<vtkActor>> mp_ellipsoid_actors_k, mp_ellipsoid_actors_k1;

        // Screenshot
        QTimer m_animation_timer;
        std::chrono::high_resolution_clock::time_point m_last_time;
        double m_time_diff;
        bool m_last_time_initialized = false; // Add a flag to check initialization

        // Display options
        double m_speed = 1.0;
        bool m_play = true, m_show_map = true, m_show_grid = false;

        vtkNew<vtkPNGWriter> m_screenshot_writer;
        bool m_recording = false;

        std::string m_output_path;

        QLabel *m_infoLabel;
        PlotWidget *m_plotWidget;
        InfoWidget *m_diagnosticsLabel;
        QTimer m_infoTimer;
        TimelineWidget *m_timelineWidget;

        unsigned int m_time_idx = 0;
        double m_time = 0.0, m_timestep;

        bool m_verbose = true;
        bool m_reset = false;
        int m_topology_K;
    };
} // namespace Spacetime