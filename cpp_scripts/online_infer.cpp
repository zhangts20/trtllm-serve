#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include "session.hpp"

using ParamsType = std::variant<int, std::string, bool>;

template <typename T> T getValurOrDefault(json &params, const std::string &key, const T &default_value) {
    if (params.contains(key)) {
        try {
            return params[key].get<T>();
        } catch (const std::exception &e) {
            LOG_ERROR("Failed to parse parameters: " + std::string(e.what()));
        }
    }
    return default_value;
}

std::string dumpJson(json &response_body) {
    if (response_body.is_object()) {
        std::string result = "{";
        // Walk each item of dictionary
        for (auto it = response_body.begin(); it != response_body.end(); ++it) {
            if (it != response_body.begin())
                result += ",";
            result += "\"" + it.key() + "\":" + dumpJson(it.value());
        }
        result += "}";
        return result;
    } else if (response_body.is_array()) {
        std::string result = "[";
        for (size_t i = 0; i < response_body.size(); ++i) {
            if (i > 0)
                result += ",";
            result += dumpJson(response_body[i]);
        }
        result += "]";
        return result;
    } else if (response_body.is_string()) {
        return "\"" + response_body.get<std::string>() + "\"";
    } else if (response_body.is_number()) {
        std::ostringstream oss;
        if (response_body.is_number_float()) {
            float v = response_body.get<float>();
            oss << std::fixed << std::setprecision(4) << v;
        } else {
            oss << response_body;
        }
        return oss.str();
    } else if (response_body.is_boolean()) {
        return response_body.get<bool>() ? "true" : "false";
    } else if (response_body.is_null()) {
        return "null";
    }
    return "";
}

void handleRequests(const httplib::Request &req, httplib::Response &res, InferenceSession *inference_session) {
    try {
        json body_json = json::parse(req.body);
        std::cout << body_json.dump() << std::endl;
        // Check members
        if (!body_json.contains("inputs")) {
            LOG_ERROR("The key inputs does not exist");
        }
        if (!body_json.contains("parameters")) {
            LOG_ERROR("The key parameters does not exist");
        }
        // Parse inputs
        if (!body_json["inputs"].is_string()) {
            LOG_ERROR("The inputs must be a string");
        }
        std::string inputs = body_json["inputs"].get<std::string>();
        // Append requests
        int max_new_tokens = getValurOrDefault<int>(body_json["parameters"], "max_new_tokens", 17);
        int num_beams = getValurOrDefault<int>(body_json["parameters"], "num_beams", 1);
        inference_session->addRequests(inputs, true, max_new_tokens, num_beams);
    } catch (const json::parse_error &e) {
        LOG_ERROR("Invalid JSON format: " + std::string(e.what()));
    } catch (const std::exception &e) {
        LOG_ERROR("Runtime error: " + std::string(e.what()));
    }

    res.set_chunked_content_provider("text/event-stream", [inference_session](size_t offset, httplib::DataSink &sink) {
        while (true) {
            auto output_config = inference_session->serve();
            if (output_config.has_value()) {
                // Decode output tokens
                std::vector<std::string> generated_text;
                for (auto output_tokens : output_config->output_tokens) {
                    std::string text;
                    inference_session->tokenizer_session->decode(text, output_tokens);
                    generated_text.emplace_back(text);
                }
                json response_body = {{"request_id", output_config->request_id},
                                      {"output_tokens", output_config->output_tokens},
                                      {"output_logprobs", output_config->output_logprobs},
                                      {"finish_reason", output_config->finish_reason},
                                      {"generated_text", generated_text}};
                std::string return_str = dumpJson(response_body) + "\n";
                sink.write(return_str.data(), return_str.size());
                return true;
            } else {
                sink.done();
                return false;
            }
        }
    });
}

int main(int argc, char **argv, char **envp) {
    InputServerConfig input_server_config = parseServerArgs(argc, argv, envp);
    // Initialize tensorrt_llm plugins
    initTrtLlmPlugins();
    // Initialize InferenceSession
    InferenceSession inference_session;
    inference_session.initialize(input_server_config.engine_dir);

    httplib::Server server;

    server.Post("/generate_stream", [&](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Content-Type", "text/event-stream");
        handleRequests(req, res, &inference_session);
    });

    LOG_INFO("Server available at localhost:" + std::to_string(input_server_config.port));
    server.listen("localhost", input_server_config.port);
}