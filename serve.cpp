#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "args_utils.h"
#include "httplib.h"
#include "log_utils.h"
#include "nlohmann/json.hpp"
#include "string_utils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "trtllm_session.h"
#include <mpi.h>

using json = nlohmann::json;

void handleRequests(const httplib::Request &req, httplib::Response &res, InferenceSession *inference_session) {
    try {
        json body_json = json::parse(req.body);
        if (!body_json.contains("inputs")) {
            LOG_ERROR("The key inputs does not exist");
        }
        if (!body_json.contains("parameters")) {
            LOG_ERROR("The key parameters does not exist");
        }
        if (!body_json["inputs"].is_string()) {
            LOG_ERROR("The inputs must be a string");
        }
        std::string inputs = body_json["inputs"].get<std::string>();
        bool is_streaming = getValueorDefault<bool>(body_json["parameters"], "streaming", true);
        unsigned max_new_tokens = getValueorDefault<int>(body_json["parameters"], "max_new_tokens", 17);
        unsigned num_beams = getValueorDefault<int>(body_json["parameters"], "num_beams", 1);
        unsigned top_k = getValueorDefault<int>(body_json["parameters"], "top_k", 1);
        float top_p = getValueorDefault<float>(body_json["parameters"], "top_p", 0.9);
        SamplingParameters sampling_parameters = {is_streaming, max_new_tokens, num_beams, top_k, top_p};
        // The model_dir will not be used when adding requests，we set it to null here
        InputConfig input_config = {/* model_dir =*/std::nullopt,
                                    /* input_text =*/inputs,
                                    /* sampling_parameters =*/sampling_parameters};
        inference_session->addRequests(input_config);
    } catch (const json::parse_error &e) {
        LOG_ERROR("Invalid JSON format: " + std::string(e.what()));
    } catch (const std::exception &e) {
        LOG_ERROR("Runtime error: " + std::string(e.what()));
    }

    res.set_chunked_content_provider("text/event-stream", [inference_session](size_t offset, httplib::DataSink &sink) {
        while (true) {
            auto output_config = inference_session->serve();
            if (!output_config.has_value()) {
                const std::string ev_failed = "\n[FAILED]\n\n";
                sink.write(ev_failed.c_str(), ev_failed.size());
                break;
            }

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

            bool finished = std::any_of(output_config->finish_reason.begin(), output_config->finish_reason.end(),
                                        [](const std::string &reason) { return reason != "running"; });

            if (finished) {
                const std::string ev_done = "\n[DONE]\n\n";
                sink.write(ev_done.c_str(), ev_done.size());
                break;
            }
        }
        sink.done();

        return true;
    });
}

int main(int argc, char **argv, char **envp) {
    InputServerConfig input_server_config = parseServerArgs(argc, argv, envp);
    initTrtLlmPlugins();
    InferenceSession inference_session;
    if (!inference_session.initialize(input_server_config.model_dir)) {
        return -1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        return 0;
    }

    httplib::Server server;

    server.Post("/generate_stream", [&](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Content-Type", "text/event-stream");
        handleRequests(req, res, &inference_session);
    });

    LOG_INFO("Server available at 0.0.0.0:" + std::to_string(input_server_config.port));
    server.listen("0.0.0.0", input_server_config.port);
}