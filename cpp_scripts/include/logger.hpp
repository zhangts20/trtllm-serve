#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include <cstring>
#include <ctime>
#include <iostream>
#include <string>

enum LogLevel {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
};

class Logger {
  public:
    static LogLevel mLogLevel;

    static void initLogLevel();

    static void log(LogLevel log_level, const std::string &message,
                    const char *filename, int linenum);

  private:
    static std::string logLevelToString(LogLevel log_level);

    static std::string getLogColor(LogLevel log_level);

    static std::string getTimestamp();

    static LogLevel getLogLevelFromString(const std::string &log_level);
};

#define LOG_DEBUG(message) Logger::log(LOG_DEBUG, message, __FILE__, __LINE__)
#define LOG_INFO(message) Logger::log(LOG_INFO, message, __FILE__, __LINE__)
#define LOG_WARNING(message)                                                   \
    Logger::log(LOG_WARNING, message, __FILE__, __LINE__)
#define LOG_ERROR(message) Logger::log(LOG_ERROR, message, __FILE__, __LINE__)

#endif // _LOGGER_HPP_