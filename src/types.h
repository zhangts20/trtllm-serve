#ifndef _TYPES_H_
#define _TYPES_H_

#include <string>
#include <vector>
#include <optional>

struct SamplingParameters {
    bool is_streaming;
    unsigned max_new_tokens;
    unsigned num_beams;
    unsigned top_k;
    float top_p;
};

struct InputConfig {
    std::optional<std::string> model_dir;
    std::string input_text;
    SamplingParameters sampling_parameters;
};

struct InputServerConfig {
    std::string model_dir;
    unsigned port;
};

#endif
