//
// Created by 刘鑫 on 2024/6/25.
//

#include "copilota.h"

#include <nlohmann/json.hpp>
#include "agent.h"

using namespace std;
using json = nlohmann::json;
static const string url = "http://localhost:11434";

#ifdef __cplusplus
extern "C" {
#endif

const char *agent(const char *question) {
    string prompt = question;

    auto data = ollama(url, question);

    if(!data.contains("response")) {
        return (char *) dalloc(0);;
    }
    string response = data["response"].get<string>();
    size_t size = response.length();
    char *buffer = (char *) dalloc(size);
    memcpy(buffer, response.c_str(), size);

    return buffer;
}

#ifdef __cplusplus
};
#endif
