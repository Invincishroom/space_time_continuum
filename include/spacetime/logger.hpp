#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <streambuf>
#include <memory>

// ANSI color codes
#define COLOR_RED "\033[31m"
#define COLOR_ORANGE "\033[38;5;208m" // Bright orange
#define COLOR_YELLOW "\033[33m"
#define COLOR_WHITE "\033[37m"
#define COLOR_RESET "\033[0m"

enum class LogLevel
{
    ERROR,
    WARNING,
    DEBUG,
    INFO
};

// Mapping log level to color + label
inline std::string levelColor(LogLevel level)
{
    switch (level)
    {
    case LogLevel::ERROR:
        return COLOR_RED;
    case LogLevel::WARNING:
        return COLOR_ORANGE;
    case LogLevel::DEBUG:
        return COLOR_YELLOW;
    case LogLevel::INFO:
        return COLOR_WHITE;
    }
    return COLOR_RESET;
}

inline std::string levelLabel(LogLevel level)
{
    switch (level)
    {
    case LogLevel::ERROR:
        return "[ERROR]";
    case LogLevel::WARNING:
        return "[WARNING]";
    case LogLevel::DEBUG:
        return "[DEBUG]";
    case LogLevel::INFO:
        return "[INFO]";
    }
    return "[LOG]";
}

// --- Base stream-style logger ---
class Logger
{
public:
    Logger(LogLevel level) : level_(level)
    {
        std::cout << levelColor(level_) << levelLabel(level_) << " ";
        buffer_.str("");
        buffer_.clear();
    }

    ~Logger()
    {
        std::cout << buffer_.str() << COLOR_RESET << std::endl;
    }

    template <typename T>
    Logger &operator<<(const T &value)
    {
        buffer_ << value;
        return *this;
    }

    Logger &operator<<(std::ostream &(*manip)(std::ostream &))
    {
        manip(buffer_);
        return *this;
    }

private:
    LogLevel level_;
    std::ostringstream buffer_;
};

// --- Direct function-style log ---
inline void logMessage(LogLevel level, const std::string &msg)
{
    std::cout << levelColor(level) << levelLabel(level)
              << " " << msg << COLOR_RESET << std::endl;
}

// --- Stream-style factories ---
inline Logger error() { return Logger(LogLevel::ERROR); }
inline Logger warning() { return Logger(LogLevel::WARNING); }
inline Logger debug() { return Logger(LogLevel::DEBUG); }
inline Logger info() { return Logger(LogLevel::INFO); }

// --- Function-style helpers ---
inline void error(const std::string &msg) { logMessage(LogLevel::ERROR, msg); }
inline void warning(const std::string &msg) { logMessage(LogLevel::WARNING, msg); }
inline void debug(const std::string &msg) { logMessage(LogLevel::DEBUG, msg); }
inline void info(const std::string &msg) { logMessage(LogLevel::INFO, msg); }

// --- Redirected stream mode ---
class RedirectLogger
{
public:
    RedirectLogger(LogLevel level)
        : level_(level), old_buf_(std::cout.rdbuf(buffer_.rdbuf()))
    {
        std::cout << levelColor(level_) << levelLabel(level_) << std::endl;
    }

    void logReset()
    {
        std::cout.rdbuf(old_buf_);
        std::cout << buffer_.str() << COLOR_RESET << std::endl;
    }

private:
    LogLevel level_;
    std::ostringstream buffer_;
    std::streambuf *old_buf_;
};

// global redirector instance
inline std::unique_ptr<RedirectLogger> redirectInstance;

inline void errorStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::ERROR); }
inline void warningStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::WARNING); }
inline void debugStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::DEBUG); }
inline void infoStart() { redirectInstance = std::make_unique<RedirectLogger>(LogLevel::INFO); }

inline void logReset()
{
    if (redirectInstance)
    {
        redirectInstance->logReset();
        redirectInstance.reset();
    }
}
#endif // LOGGER_H