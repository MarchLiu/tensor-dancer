//
// Created by 刘鑫 on 2024/6/26.
//

#include "agent.h"
#include "insight.h"
#include "sstream"

static json ctx;

static const string prompt = R"(
# SQL Assistant

## Database Tables

The following are the user tables and fields in this database:

{{{}}}

## Objective

Generate SQL queries for PostgreSQL as described in the next section.

## Appeal
)";
static const string slot = "{{{}}}";

json make_request(const string &content, json &context) {

    vector<Table> tables = list_tables();
    stringstream doc_buf;
    for(const auto& table: tables) {
        doc_buf << table.toDoc();
    }
    string p = string(prompt);
    size_t pos = p.find(slot);
    p.replace(pos, slot.size(), doc_buf.str());
    p += content;

    json result = {
            {"stream", false},
            {"system", "your are a sql export"},
            {"model",  "phi3:mini"}
    };

    result["prompt"] = p;
    if (!context.empty()) {
        result["context"] = context;
    }
    return result;
}

json ollama(const string &url, const string &question) {

    auto request = make_request(question, ctx);
    httplib::Client cli(url);
    cli.set_read_timeout(60, 0); // 5 seconds

    string req = request.dump();

    auto resp = cli.Post("/api/generate", req, "application/json");
    if (resp == nullptr) {
        return json::parse("{}");
    }
    auto data = json::parse(resp->body);
    if (data == nullptr) {
        return json::parse("{}");
    }
    if (data.contains("context")) {
        ctx = data["context"];
    }
    return data;
}
