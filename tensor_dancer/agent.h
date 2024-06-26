//
// Created by 刘鑫 on 2024/6/26.
//

#ifndef TENSOR_DANCER_AGENT_H
#define TENSOR_DANCER_AGENT_H

#include <nlohmann/json.hpp>
#include "httplib.h"
#include "src/dancer.h"
#include <cstring>

using namespace std;
using json = nlohmann::json;

json make_request(const string &content, json &context);

json ollama(const string &url, const string &question);

#endif //TENSOR_DANCER_AGENT_H
