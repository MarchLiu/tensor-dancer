//
// Created by 刘鑫 on 2024/6/25.
//

#include <iostream>
#include <string>
#include <format>
#include <sstream>
#include <nlohmann/json.hpp>
#include "httplib.h"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv) {
    static const string url = "http://localhost:11434";

    httplib::Client cli(url);

    auto resp = cli.Get("/api/tags");
    cout << resp->status << endl;

    auto data = json::parse(resp->body);

    auto models = data["models"];
    for(auto iter = models.begin(); iter != models.end(); iter++){
        cout<< iter.value()["name"] << endl;
    }

    return 0;
}