#include <iostream>
#include <fstream>
#include <string>

#include "rapidjson/document.h"

int main() {
    std::ifstream t("/data/zhangtaoshan/models/llama2-7b/config.json");
    std::string t_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    rapidjson::Document document;
    document.Parse(t_str.c_str());

    std::cout << document.HasMember("model_type") << std::endl;
    std::cout << document["model_type"].GetString() << std::endl;
    if (document["architectures"].IsArray()) {
        for (size_t i = 0; i < document["architectures"].Size(); ++i) {
            std::cout << "arch: " << document["architectures"][i].GetString() << std::endl;
        }
    }
}
