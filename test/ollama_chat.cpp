//
// Created by 刘鑫 on 2024/6/25.
//

#include "nlohmann/json.hpp"
#include "httplib.h"
#include <iostream>
#include <string>
using namespace std;
using json = nlohmann::json;

static json ctx;

json make_request(string& content, json& context) {
    auto result = json::parse(R"(
{
    "stream": false,
    "system": "",
    "model": "phi3:mini"
}
)");

    result["prompt"] = content;
    if(!context.empty()){
        result["context"] = context;
    }

    return result;
}

int main(int argc, char** argv) {

    static const string url = "http://localhost:11434";

    httplib::Client cli(url);

    while (true) {
        cout << ">> ";
        string prompt;
        cin >> prompt;

        if(prompt.empty()){
            cin.clear();
            cout << endl;
            continue;
        }

        auto req = make_request(prompt, ctx);
        auto resp = cli.Post("/api/generate", req.dump(), "application/json");
        if(resp == nullptr){
            continue;
        }
        auto body = json::parse(resp->body);
        cout << body["response"].get<string>() << endl;
        if(body.contains("context")) {
            ctx = body["context"];
        }
    }

    return 0;
}
