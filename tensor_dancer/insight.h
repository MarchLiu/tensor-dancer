//
// Created by 刘鑫 on 2024/6/26.
//

#ifndef TENSOR_DANCER_INSIGHT_H
#define TENSOR_DANCER_INSIGHT_H

#include <string>

using namespace std;

struct Column {
    Column(const string &name, const string &type, bool primaryKey);

    void setPrimaryKey(bool primaryKey);

    const string &getName() const;

    const string &getType() const;

    bool isPrimaryKey() const;

    string name;
    string type;
    bool primaryKey;
};

struct Table {
public:
    Table(const string &schema, const string &name);

    void setPrimaryKey(const string &column);

    void setNotPrimaryKey(const string &column);

    const string &getSchema() const;

    const string &getName1() const;

    const vector<Column> &getColumns() const;

    string toDoc() const;

    string schema;
    string name;
    vector<Column> columns;
};



string current_database();

vector<string> list_pk(const string &schema, const string &table);

vector<Column> list_columns(const string &schema, const string &table);

vector<Table> list_tables();

#endif //TENSOR_DANCER_INSIGHT_H
