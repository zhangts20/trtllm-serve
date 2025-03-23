#include "string_utils.h"

std::string dumpJson(json &body) {
    if (body.is_object()) {
        std::string result = "{";
        for (auto it = body.begin(); it != body.end(); ++it) {
            if (it != body.begin()) {
                result += ",";
            }
            result += "\"" + it.key() + "\":" + dumpJson(it.value());
        }
        result += "}";
        return result;
    } else if (body.is_array()) {
        std::string result = "[";
        for (size_t i = 0; i < body.size(); ++i) {
            if (i > 0)
                result += ",";
            result += dumpJson(body[i]);
        }
        result += "]";
        return result;
    } else if (body.is_string()) {
        return "\"" + body.get<std::string>() + "\"";
    } else if (body.is_number()) {
        std::ostringstream oss;
        if (body.is_number_float()) {
            float v = body.get<float>();
            oss << std::fixed << std::setprecision(4) << v;
        } else {
            oss << body;
        }
        return oss.str();
    } else if (body.is_boolean()) {
        return body.get<bool>() ? "true" : "false";
    } else if (body.is_null()) {
        return "null";
    }
    return "";
}
