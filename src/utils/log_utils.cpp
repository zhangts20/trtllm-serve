#include "log_utils.h"

#include <cstring>
#include <ctime>
#include <iostream>

LogLevel Logger::mLogLevel = LOG_INFO;

void Logger::initLogLevel() {
    const char *log_level = std::getenv("LOG_LEVEL");
    if (log_level) {
        mLogLevel = getLogLevelFromString(std::string(log_level));
    }
}

void Logger::log(LogLevel log_level, const std::string &message, const char *filepath, int linenum) {
    if (log_level >= mLogLevel) {
        const char *filename = strrchr(filepath, '/') ? strrchr(filepath, '/') + 1 : filepath;
        std::cout << "[" << getTimestamp() << "] " << "[" << logLevelToString(log_level) << "]" << " " << filename
                  << " " << linenum << ": " << getLogColor(log_level) << message << "\033[0m" << std::endl;
    }
}

std::string Logger::logLevelToString(LogLevel level) {
    switch (level) {
    case LOG_DEBUG:
        return "D";
    case LOG_INFO:
        return "I";
    case LOG_WARNING:
        return "W";
    case LOG_ERROR:
        return "E";
    default:
        return "UNKNOWN";
    }
}

std::string Logger::getLogColor(LogLevel level) {
    switch (level) {
    case LOG_DEBUG:
        return "\033[34m";
    case LOG_INFO:
        return "\033[32m";
    case LOG_WARNING:
        return "\033[33m";
    case LOG_ERROR:
        return "\033[31m";
    default:
        return "UNKNOWN";
    }
}

std::string Logger::getTimestamp() {
    std::time_t now = std::time(nullptr);
    char buf[100] = {0};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

LogLevel Logger::getLogLevelFromString(const std::string &level) {
    if (level == "DEBUG")
        return LOG_DEBUG;
    if (level == "WARNING")
        return LOG_WARNING;
    if (level == "ERROR")
        return LOG_ERROR;
    return LOG_INFO;
}