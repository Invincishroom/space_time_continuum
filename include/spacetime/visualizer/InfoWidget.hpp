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

class InfoWidget : public QLabel
{
    Q_OBJECT

private:
    struct DisplayItem
    {
        enum Type
        {
            VALUE,
            HEADER
        };
        Type type;
        std::string label;
        void *value;
        std::function<std::string()> stringifier; // Function to convert value to string

        DisplayItem(const std::string &l, void *v, std::function<std::string()> s)
            : type(VALUE), label(l), value(v), stringifier(s) {}

        DisplayItem(const std::string &headerText)
            : type(HEADER), label(headerText), value(nullptr) {}
    };

    std::vector<DisplayItem> m_displayItems;

    void updateDisplay()
    {
        QString displayText;
        for (const auto &item : m_displayItems)
        {
            if (item.type == DisplayItem::HEADER)
            {
                if (!displayText.isEmpty())
                    displayText += "<br><br>";
                displayText += "<b>" + QString::fromStdString(item.label) + "</b>";
            }
            else // VALUE
            {
                if (!displayText.isEmpty())
                    displayText += "<br>";

                displayText += QString::fromStdString(item.label);
                if (item.value || item.stringifier)
                {
                    displayText += ": ";

                    try
                    {
                        displayText += QString::fromStdString(item.stringifier());
                    }
                    catch (...)
                    {
                        displayText += "Error displaying value";
                    }
                }
            }
        }
        this->setText(displayText);
    }

    // SFINAE helper to check if type is streamable
    template <typename T>
    struct is_streamable
    {
    private:
        template <typename U>
        static auto test(int) -> decltype(std::declval<std::ostream &>() << std::declval<U>(), std::true_type{});

        template <typename>
        static std::false_type test(...);

    public:
        static constexpr bool value = decltype(test<T>(0))::value;
    };

public:
    InfoWidget(QWidget *parent = nullptr) : QLabel(parent)
    {
        this->setStyleSheet(
            "QLabel {"
            "background-color: rgba(0, 0, 0, 150);"
            "color: white;"
            "padding: 5px;"
            "border-radius: 5px;"
            "line-height: 1.2;"
            "}");
        this->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        this->setTextFormat(Qt::RichText); // Enable HTML formatting
        this->setVisible(false);
    }

    // Add a header at the current position
    void addHeader(const std::string &label)
    {
        m_displayItems.emplace_back(label);
        updateDisplay();
    }

    // Add a label with a lambda function that returns the value
    void addLabel(const std::string &label, std::function<std::string()> valueFunction)
    {
        m_displayItems.emplace_back(label, nullptr, valueFunction);
        updateDisplay();
    }

    // Template function to add diagnostic values of any streamable type
    template <typename T>
    typename std::enable_if<is_streamable<T>::value, void>::type
    addLabel(const std::string &label, T *value)
    {
        // Create a lambda that captures the pointer and converts to string
        auto stringifier = [value]() -> std::string
        {
            if (value == nullptr)
            {
                return "null";
            }

            std::ostringstream oss;
            oss << *value;
            return oss.str();
        };

        m_displayItems.emplace_back(label, static_cast<void *>(value), stringifier);
        updateDisplay();
    }

    void addLabel(const std::string &label)
    {
        m_displayItems.emplace_back(label, nullptr, nullptr);
        updateDisplay();
    }

    // Overload for non-streamable types (will cause compile error with helpful message)
    template <typename T>
    typename std::enable_if<!is_streamable<T>::value, void>::type
    addLabel(const std::string &label, T *value)
    {
        static_assert(is_streamable<T>::value,
                      "Type T must be streamable (support std::cout << value). "
                      "Please define operator<< for your type or use a streamable type.");
    }

    // Function to remove a diagnostic value by label
    void removeDiagnosticValue(const std::string &label)
    {
        m_displayItems.erase(
            std::remove_if(m_displayItems.begin(), m_displayItems.end(),
                           [&label](const DisplayItem &item)
                           {
                               return item.type == DisplayItem::VALUE && item.label == label;
                           }),
            m_displayItems.end());
        updateDisplay();
    }

    // Function to remove a header by label
    void removeHeader(const std::string &label)
    {
        m_displayItems.erase(
            std::remove_if(m_displayItems.begin(), m_displayItems.end(),
                           [&label](const DisplayItem &item)
                           {
                               return item.type == DisplayItem::HEADER && item.label == label;
                           }),
            m_displayItems.end());
        updateDisplay();
    }

    // Function to clear all diagnostic values and headers
    void clearAll()
    {
        m_displayItems.clear();
        this->setText("");
    }

    // Function to clear only diagnostic values (keep headers)
    void clearDiagnosticValues()
    {
        m_displayItems.erase(
            std::remove_if(m_displayItems.begin(), m_displayItems.end(),
                           [](const DisplayItem &item)
                           {
                               return item.type == DisplayItem::VALUE;
                           }),
            m_displayItems.end());
        updateDisplay();
    }

    // Function to refresh the display (call this when values change)
    void refresh()
    {
        updateDisplay();
    }
};