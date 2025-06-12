#ifndef _STRING_UTILS_H_
#define _STRING_UTILS_H_

#include "log_utils.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string dumpJson(json &body);

template <typename T> T getValueorDefault(json &params, const std::string &key, const T &default_value) {
    if (params.contains(key)) {
        try {
            return params[key].get<T>();
        } catch (const std::exception &e) {
            LOG_ERROR("Failed to parse parameters: " + std::string(e.what()));
        }
    }
    return default_value;
}

#endif
