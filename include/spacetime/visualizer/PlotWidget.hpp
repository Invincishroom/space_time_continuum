#pragma once

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

#include "spacetime/types.hpp"

class PlotWidget : public QWidget
{
    Q_OBJECT

private:
    struct PlotData
    {
        // Interpolation nodes data (lines only)
        std::vector<double> interp_time;
        std::vector<std::vector<double>> interp_epsilon; // 6 components
        std::vector<std::vector<double>> interp_varpi;   // 6 components

        // Estimation nodes data (markers only)
        std::vector<double> estim_time;
        std::vector<std::vector<double>> estim_epsilon; // 6 components
        std::vector<std::vector<double>> estim_varpi;   // 6 components
    };

    PlotData m_plotData;
    QVBoxLayout *m_mainLayout;
    QScrollArea *m_scrollArea;
    QWidget *m_plotContainer;
    QVBoxLayout *m_plotLayout;

    std::vector<QLabel *> m_epsilonPlots; // Will hold plot images
    std::vector<QLabel *> m_varpiPlots;   // Will hold plot images
    double m_arclength;

    const std::vector<QString> m_componentNames = {
        "X", "Y", "Z", "Roll", "Pitch", "Yaw"};

    const std::vector<QColor> m_plotColors = {
        QColor(255, 0, 0),    // Red
        QColor(0, 255, 0),    // Green
        QColor(0, 0, 255),    // Blue
        QColor(255, 165, 0),  // Orange
        QColor(128, 0, 128),  // Purple
        QColor(255, 192, 203) // Pink
    };

    // Plot dimensions - will be calculated based on widget geometry
    int m_plotWidth;
    int m_plotHeight;
    static constexpr int WIDGET_PADDING = 20;
    static constexpr int HEADER_HEIGHT = 30;
    static constexpr int NUM_PLOTS = 12; // 6 epsilon + 6 varpi

    void calculatePlotDimensions()
    {
        // Calculate plot width based on widget width minus padding
        m_plotWidth = width() - WIDGET_PADDING;

        // Calculate plot height based on available height
        // Account for headers (2), padding, and space between plots
        int availableHeight = height() - (2 * HEADER_HEIGHT) - (WIDGET_PADDING * 2);
        int spaceBetweenPlots = 10 * (NUM_PLOTS - 1); // 10px between each plot
        m_plotHeight = (availableHeight - spaceBetweenPlots) / NUM_PLOTS;

        // Ensure minimum dimensions
        m_plotWidth = std::max(m_plotWidth, 300);
        m_plotHeight = std::max(m_plotHeight, 300);
    }

    void updatePlotSizes()
    {
        calculatePlotDimensions();

        // Update all plot label sizes
        for (auto *plot : m_epsilonPlots)
        {
            plot->setFixedSize(m_plotWidth, m_plotHeight);
        }
        for (auto *plot : m_varpiPlots)
        {
            plot->setFixedSize(m_plotWidth, m_plotHeight);
        }
    }

    void initializeUI()
    {
        m_mainLayout = new QVBoxLayout(this);
        m_mainLayout->setContentsMargins(0, 0, 0, 0);

        // Create scroll area - only vertical scrolling
        m_scrollArea = new QScrollArea(this);
        m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_scrollArea->setWidgetResizable(true);

        m_plotContainer = new QWidget();
        m_plotLayout = new QVBoxLayout(m_plotContainer);
        m_plotLayout->setContentsMargins(WIDGET_PADDING / 2, WIDGET_PADDING / 2, WIDGET_PADDING / 2, WIDGET_PADDING / 2);

        // Create epsilon plots
        QLabel *epsilonHeader = new QLabel("<b>Epsilon Components (Strain)</b>");
        epsilonHeader->setAlignment(Qt::AlignCenter);
        epsilonHeader->setMinimumHeight(HEADER_HEIGHT);
        m_plotLayout->addWidget(epsilonHeader);

        for (int i = 0; i < 6; ++i)
        {
            QLabel *plotLabel = new QLabel();
            plotLabel->setStyleSheet("border: 1px solid gray; background-color: white;");
            plotLabel->setAlignment(Qt::AlignCenter);
            plotLabel->setText(QString("Epsilon %1 (%2)").arg(i).arg(m_componentNames[i]));
            plotLabel->setScaledContents(false);
            m_epsilonPlots.push_back(plotLabel);
            m_plotLayout->addWidget(plotLabel, 0, Qt::AlignHCenter);
        }

        // Create varpi plots
        QLabel *varpiHeader = new QLabel("<b>Varpi Components (Velocity)</b>");
        varpiHeader->setAlignment(Qt::AlignCenter);
        varpiHeader->setMinimumHeight(HEADER_HEIGHT);
        m_plotLayout->addWidget(varpiHeader);

        for (int i = 0; i < 6; ++i)
        {
            QLabel *plotLabel = new QLabel();
            plotLabel->setStyleSheet("border: 1px solid gray; background-color: white;");
            plotLabel->setAlignment(Qt::AlignCenter);
            plotLabel->setText(QString("Varpi %1 (%2)").arg(i).arg(m_componentNames[i]));
            plotLabel->setScaledContents(false);
            m_varpiPlots.push_back(plotLabel);
            m_plotLayout->addWidget(plotLabel, 0, Qt::AlignHCenter);
        }

        // Add stretch to push everything to the top
        m_plotLayout->addStretch();

        m_scrollArea->setWidget(m_plotContainer);
        m_mainLayout->addWidget(m_scrollArea);
    }

    QPixmap createPlot(const std::vector<double> &interpTimeData, const std::vector<double> &interpValueData,
                       const std::vector<double> &estimTimeData, const std::vector<double> &estimValueData,
                       const QString &title, const QString &ylabel, const QColor &color)
    {
        QPixmap pixmap(m_plotWidth, m_plotHeight);
        pixmap.fill(Qt::white);

        // Check if we have any data to plot
        bool hasInterpData = !interpTimeData.empty() && !interpValueData.empty() && interpTimeData.size() == interpValueData.size();
        bool hasEstimData = !estimTimeData.empty() && !estimValueData.empty() && estimTimeData.size() == estimValueData.size();

        if (!hasInterpData && !hasEstimData)
        {
            QPainter painter(&pixmap);
            painter.drawText(pixmap.rect(), Qt::AlignCenter, "No data available");
            return pixmap;
        }

        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing);

        // Set margins - increased left margin for Y-axis labels
        int leftMargin = std::max(50, m_plotWidth / 8);
        int otherMargins = std::max(25, std::min(m_plotWidth, m_plotHeight) / 15);
        QRect plotRect(leftMargin, otherMargins, m_plotWidth - leftMargin - otherMargins,
                       m_plotHeight - 2 * otherMargins);

        // Find data bounds across both datasets
        std::vector<double> allTimeData, allValueData;
        if (hasInterpData)
        {
            allTimeData.insert(allTimeData.end(), interpTimeData.begin(), interpTimeData.end());
            allValueData.insert(allValueData.end(), interpValueData.begin(), interpValueData.end());
        }
        if (hasEstimData)
        {
            allTimeData.insert(allTimeData.end(), estimTimeData.begin(), estimTimeData.end());
            allValueData.insert(allValueData.end(), estimValueData.begin(), estimValueData.end());
        }

        auto minMaxTime = std::minmax_element(allTimeData.begin(), allTimeData.end());
        auto minMaxValue = std::minmax_element(allValueData.begin(), allValueData.end());

        double timeMin = *minMaxTime.first;
        double timeMax = *minMaxTime.second;
        double valueMin = *minMaxValue.first;
        double valueMax = *minMaxValue.second;

        // Add some padding to value range
        double valueRange = valueMax - valueMin;
        if (valueRange < 1e-12)
            valueRange = 1.0; // Avoid division by zero
        valueMin -= valueRange * 0.1;
        valueMax += valueRange * 0.1;
        valueRange = valueMax - valueMin; // Recalculate after padding

        // Determine appropriate scaling for Y-axis values
        double maxAbsValue = std::max(std::abs(valueMin), std::abs(valueMax));
        int commonExponent = 0;
        double scaleFactor = 1.0;

        if (maxAbsValue > 0)
        {
            if (maxAbsValue < 0.001 || maxAbsValue >= 10000)
            {
                commonExponent = static_cast<int>(std::floor(std::log10(maxAbsValue)));
                scaleFactor = std::pow(10.0, -commonExponent);
            }
        }

        // Draw axes
        painter.setPen(QPen(Qt::black, 1));
        painter.drawRect(plotRect);

        // Draw title
        QFont titleFont = painter.font();
        titleFont.setBold(true);
        titleFont.setPointSize(std::max(8, m_plotHeight / 20));
        painter.setFont(titleFont);
        painter.drawText(QRect(0, 5, m_plotWidth, 25), Qt::AlignCenter, title);

        // Draw scale factor if needed (top-left corner of plot area)
        if (commonExponent != 0)
        {
            QFont scaleFont = painter.font();
            scaleFont.setPointSize(std::max(6, m_plotHeight / 30));
            scaleFont.setBold(false);
            painter.setFont(scaleFont);
            QString scaleText = QString("×10^%1").arg(commonExponent);
            painter.setPen(QPen(Qt::black, 1));
            painter.drawText(QRect(plotRect.left() + 5, plotRect.top() - 18, 80, 15), Qt::AlignLeft, scaleText);
        }

        // Set font for tick labels
        QFont labelFont = painter.font();
        labelFont.setBold(false);
        labelFont.setPointSize(std::max(6, m_plotHeight / 30));
        painter.setFont(labelFont);

        // Draw grid lines with tick labels
        painter.setPen(QPen(Qt::lightGray, 1));

        // Vertical grid lines (time) with labels
        int numTimeGrids = std::max(3, std::min(6, m_plotWidth / 120));
        for (int i = 0; i <= numTimeGrids; ++i)
        {
            double t = timeMin + (timeMax - timeMin) * i / numTimeGrids;
            int x = plotRect.left() + static_cast<int>((t - timeMin) / (timeMax - timeMin) * plotRect.width());
            painter.drawLine(x, plotRect.top(), x, plotRect.bottom());

            // Time tick labels
            painter.setPen(QPen(Qt::black, 1));
            QString timeLabel = QString::number(t, 'g', 3);
            painter.drawText(QRect(x - 25, plotRect.bottom() + 5, 50, 15), Qt::AlignCenter, timeLabel);
            painter.setPen(QPen(Qt::lightGray, 1));
        }

        // Horizontal grid lines (values) with properly spaced labels
        int numValueGrids = std::max(3, std::min(6, m_plotHeight / 60));
        for (int i = 0; i <= numValueGrids; ++i)
        {
            double v = valueMin + (valueMax - valueMin) * i / numValueGrids;
            int y = plotRect.bottom() - static_cast<int>((v - valueMin) / (valueMax - valueMin) * plotRect.height());
            painter.drawLine(plotRect.left(), y, plotRect.right(), y);

            // Value tick labels - use consistent formatting
            painter.setPen(QPen(Qt::black, 1));
            double displayValue = v * scaleFactor;
            QString valueLabel;

            // Use consistent decimal places based on the range
            double scaledRange = valueRange * scaleFactor;
            int decimalPlaces = 2;
            if (scaledRange < 0.1)
                decimalPlaces = 3;
            else if (scaledRange > 100)
                decimalPlaces = 1;
            else if (scaledRange > 10)
                decimalPlaces = 2;

            // Format with fixed decimal places for consistency
            if (std::abs(displayValue) < 1000)
            {
                valueLabel = QString::number(displayValue, 'f', decimalPlaces);
            }
            else
            {
                valueLabel = QString::number(displayValue, 'e', 1);
            }

            // Draw label with sufficient space - use the full left margin
            QRect labelRect(5, y - 8, leftMargin - 10, 16);
            painter.drawText(labelRect, Qt::AlignRight | Qt::AlignVCenter, valueLabel);
            painter.setPen(QPen(Qt::lightGray, 1));
        }

        // Draw interpolation data as lines (no markers)
        if (hasInterpData && interpTimeData.size() > 1)
        {
            painter.setPen(QPen(color, 2));

            for (size_t i = 1; i < interpTimeData.size(); ++i)
            {
                int x1 = plotRect.left() + static_cast<int>((interpTimeData[i - 1] - timeMin) / (timeMax - timeMin) * plotRect.width());
                int y1 = plotRect.bottom() - static_cast<int>((interpValueData[i - 1] - valueMin) / (valueMax - valueMin) * plotRect.height());
                int x2 = plotRect.left() + static_cast<int>((interpTimeData[i] - timeMin) / (timeMax - timeMin) * plotRect.width());
                int y2 = plotRect.bottom() - static_cast<int>((interpValueData[i] - valueMin) / (valueMax - valueMin) * plotRect.height());

                painter.drawLine(x1, y1, x2, y2);
            }
        }

        // Draw estimation data as markers only (no lines)
        if (hasEstimData)
        {
            painter.setPen(QPen(color.darker(), 2));
            painter.setBrush(QBrush(color.darker()));
            int markerRadius = 1; // Larger markers for estimation nodes

            for (size_t i = 0; i < estimTimeData.size(); ++i)
            {
                int x = plotRect.left() + static_cast<int>((estimTimeData[i] - timeMin) / (timeMax - timeMin) * plotRect.width());
                int y = plotRect.bottom() - static_cast<int>((estimValueData[i] - valueMin) / (valueMax - valueMin) * plotRect.height());
                painter.drawEllipse(x - markerRadius, y - markerRadius, 2 * markerRadius, 2 * markerRadius);
            }
        }

        return pixmap;
    }

    void updatePlots()
    {
        if (m_plotData.interp_time.empty() && m_plotData.estim_time.empty())
            return;

        // Update epsilon plots
        for (int i = 0; i < 6; ++i)
        {
            QString title = QString("Epsilon %1 (%2)").arg(i).arg(m_componentNames[i]);
            QString ylabel = (i < 3) ? "Linear Strain" : "Angular Strain";

            // Get interpolation data for this component
            std::vector<double> interpData = (i < m_plotData.interp_epsilon.size()) ? m_plotData.interp_epsilon[i] : std::vector<double>();
            std::vector<double> estimData = (i < m_plotData.estim_epsilon.size()) ? m_plotData.estim_epsilon[i] : std::vector<double>();

            QPixmap plot = createPlot(m_plotData.interp_time, interpData,
                                      m_plotData.estim_time, estimData,
                                      title, ylabel, m_plotColors[i]);
            m_epsilonPlots[i]->setPixmap(plot);
        }

        // Update varpi plots
        for (int i = 0; i < 6; ++i)
        {
            QString title = QString("Varpi %1 (%2)").arg(i).arg(m_componentNames[i]);
            QString ylabel = (i < 3) ? "Linear Velocity" : "Angular Velocity";

            // Get interpolation data for this component
            std::vector<double> interpData = (i < m_plotData.interp_varpi.size()) ? m_plotData.interp_varpi[i] : std::vector<double>();
            std::vector<double> estimData = (i < m_plotData.estim_varpi.size()) ? m_plotData.estim_varpi[i] : std::vector<double>();

            QPixmap plot = createPlot(m_plotData.interp_time, interpData,
                                      m_plotData.estim_time, estimData,
                                      title, ylabel, m_plotColors[i]);
            m_varpiPlots[i]->setPixmap(plot);
        }
    }

    void resizeEvent(QResizeEvent *event) override
    {
        QWidget::resizeEvent(event);
        updatePlotSizes();
        // Regenerate plots with new dimensions if we have data
        if (!m_plotData.interp_time.empty())
        {
            updatePlots();
        }
    }

public:
    PlotWidget(QWidget *parent = nullptr) : QWidget(parent), m_plotWidth(600), m_plotHeight(200)
    {
        // Initialize data containers
        m_plotData.interp_epsilon.resize(6);
        m_plotData.interp_varpi.resize(6);
        m_plotData.estim_epsilon.resize(6);
        m_plotData.estim_varpi.resize(6);

        initializeUI();
        this->setVisible(false); 
    }

    double getArclength() const
    {
        return m_arclength;
    }

    void setGeometry(int x, int y, int w, int h)
    {
        QWidget::setGeometry(x, y, w, h);
        updatePlotSizes();
        // Regenerate plots with new dimensions if we have data
        if (!m_plotData.interp_time.empty())
        {
            updatePlots();
        }
    }

    void setGeometry(const QRect &rect)
    {
        QWidget::setGeometry(rect);
        updatePlotSizes();
        // Regenerate plots with new dimensions if we have data
        if (!m_plotData.interp_time.empty())
        {
            updatePlots();
        }
    }

    void updateData(const Spacetime::SystemState<double> &state, double arclength = -1.0)
    {
        // Clear previous data
        m_plotData.interp_time.clear();
        m_plotData.estim_time.clear();
        for (auto &component : m_plotData.interp_epsilon)
            component.clear();
        for (auto &component : m_plotData.interp_varpi)
            component.clear();
        for (auto &component : m_plotData.estim_epsilon)
            component.clear();
        for (auto &component : m_plotData.estim_varpi)
            component.clear();

        // Extract data from interpolation nodes
        for (const auto &node : state.interpolation_nodes)
        {
            // Filter by arclength if specified (use small tolerance for floating point comparison)
            if (arclength >= 0.0 && std::abs(node.arclength - arclength) > 1e-9)
                continue;

            m_plotData.interp_time.push_back(node.time);

            // Extract epsilon components (convert from DTYPE to double)
            for (int i = 0; i < 6; ++i)
            {
                m_plotData.interp_epsilon[i].push_back(node.epsilon(i));
            }

            // Extract varpi components (convert from DTYPE to double)
            for (int i = 0; i < 6; ++i)
            {
                m_plotData.interp_varpi[i].push_back(node.varpi(i));
            }
        }

        // Extract data from estimation nodes
        for (const auto &node : state.estimation_nodes)
        {
            // Filter by arclength if specified (use small tolerance for floating point comparison)
            if (arclength >= 0.0 && std::abs(node.arclength - arclength) > 1e-9)
                continue;

            m_plotData.estim_time.push_back(node.time);

            // Extract epsilon components (convert from DTYPE to double)
            for (int i = 0; i < 6; ++i)
            {
                m_plotData.estim_epsilon[i].push_back(node.epsilon(i));
            }

            // Extract varpi components (convert from DTYPE to double)
            for (int i = 0; i < 6; ++i)
            {
                m_plotData.estim_varpi[i].push_back(node.varpi(i));
            }
        }

        updatePlots();
    }

    void clearData()
    {
        m_plotData.interp_time.clear();
        m_plotData.estim_time.clear();
        for (auto &component : m_plotData.interp_epsilon)
            component.clear();
        for (auto &component : m_plotData.interp_varpi)
            component.clear();
        for (auto &component : m_plotData.estim_epsilon)
            component.clear();
        for (auto &component : m_plotData.estim_varpi)
            component.clear();

        // Clear plot displays
        for (auto *plot : m_epsilonPlots)
        {
            plot->clear();
            plot->setText("No data");
        }
        for (auto *plot : m_varpiPlots)
        {
            plot->clear();
            plot->setText("No data");
        }
    }
};