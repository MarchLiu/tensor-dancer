//
// Created by 刘鑫 on 2024/6/26.
//

#include "insight.h"

extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"
}

#include <sstream>

static const string query_tables =
        "select schemaname, tablename from pg_catalog.pg_tables where schemaname != 'pg_catalog'";

static const string query_columns = R"(
SELECT column_name, data_type
  FROM information_schema.columns
 WHERE
    table_catalog = $1 AND
    table_schema = $2 AND
    table_name   = $3
)";


static const string query_pk = R"(
SELECT ccu.column_name
FROM information_schema.table_constraints as tc
    join information_schema.constraint_column_usage as ccu
        on tc.constraint_name = ccu.constraint_name
WHERE
    tc.table_catalog = $1 AND
    tc.table_schema = $2 AND
    tc.table_name = $3 and
    tc.constraint_type = 'PRIMARY KEY'
)";

Table::Table(const string &schema, const string &name) : schema(schema), name(name) {
    this->schema = schema;
    this->name = name;
    this->columns = vector<Column>();
}

void Table::setPrimaryKey(const string &column) {
    for (auto iter: this->columns) {
        if (iter.name == column) {
            iter.primaryKey = true;
            return;
        }
    }
}

const string &Table::getSchema() const {
    return schema;
}

const string &Table::getName1() const {
    return name;
}

const vector<Column> &Table::getColumns() const {
    return columns;
}

void Table::setNotPrimaryKey(const string &column) {
    for (auto iter: this->columns) {
        if (iter.name == column) {
            iter.primaryKey = false;
            return;
        }
    }
}

string Table::toDoc() const {
    stringstream buffer("### ");
    buffer << this->name << endl;
    buffer << "|column|type|primary key|" << endl;
    for (const auto &column: this->columns) {
        buffer << "|" << column.name << "|" << column.type << "|" << (column.isPrimaryKey() ? "t" : "f") << "|" << endl;
    }
    return buffer.str();
}

const string &Column::getName() const {
    return name;
}

const string &Column::getType() const {
    return type;
}

bool Column::isPrimaryKey() const {
    return primaryKey;
}

Column::Column(const string &name, const string &type, bool primaryKey) : name(name), type(type),
                                                                          primaryKey(primaryKey) {
    this->name = name;
    this->type = type;
    this->primaryKey = primaryKey;
}

void Column::setPrimaryKey(bool pk) {
    Column::primaryKey = pk;
}


string current_database() {
    string result;

    int ret = SPI_exec("select current_database()", 1);

    if (ret > 0 && SPI_tuptable != nullptr) {
        SPITupleTable * tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        HeapTuple tuple = tuptable->vals[0];
        result = SPI_getvalue(tuple, tupdesc, 1);
    }

    return result;
}

vector<Column> list_columns(const string &schema, const string &table) {
    vector<Column> result;
    string database = current_database();

    Oid argtypes[3] = {TEXTOID, TEXTOID, TEXTOID};
    Datum values[3] = {CStringGetTextDatum(database.c_str()),
                       CStringGetTextDatum(schema.c_str()),
                       CStringGetTextDatum(table.c_str())};

    long ret = SPI_execute_with_args(query_columns.c_str(),
                                     3,
                                     argtypes,
                                     values,
                                     nullptr,
                                     true,
                                     0);

    if (ret > 0 && SPI_tuptable != nullptr) {
        SPITupleTable * tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        for (uint64 i = 0; i < tuptable->numvals; i++) {
            HeapTuple tuple = tuptable->vals[i];

            string column_name = SPI_getvalue(tuple, tupdesc, 1);
            string column_type = SPI_getvalue(tuple, tupdesc, 2);
            Column column(column_name, column_type, false);
            result.push_back(column);
        }
    }

    return result;
}

vector<string> list_pk(const string &schema, const string &table) {
    vector<string> result;
    string database = current_database();

    Oid argtypes[3] = {TEXTOID, TEXTOID, TEXTOID};
    Datum values[3] = {CStringGetTextDatum(database.c_str()),
                       CStringGetTextDatum(schema.c_str()),
                       CStringGetTextDatum(table.c_str())};

    long ret = SPI_execute_with_args(query_pk.c_str(),
                                     3,
                                     argtypes,
                                     values,
                                     nullptr,
                                     true,
                                     0);

    if (ret > 0 && SPI_tuptable != nullptr) {
        SPITupleTable * tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        for (uint64 i = 0; i < tuptable->numvals; i++) {
            HeapTuple tuple = tuptable->vals[i];

            string column_name = SPI_getvalue(tuple, tupdesc, 1);
            result.push_back(column_name);
        }
    }

    return result;
}


vector<Table> list_tables() {
    vector<Table> result;

    long ret = SPI_exec(query_tables.c_str(), 0);

    if (ret > 0 && SPI_tuptable != nullptr) {
        SPITupleTable * tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        for (uint64 i = 0; i < tuptable->numvals; i++) {
            HeapTuple tuple = tuptable->vals[i];

            string schema_name = SPI_getvalue(tuple, tupdesc, 1);
            string table_name = SPI_getvalue(tuple, tupdesc, 2);
            Table table(schema_name, table_name);
            vector<Column> columns = list_columns(schema_name, table_name);
            table.columns = columns;

            vector<string> pk_list = list_pk(schema_name, table_name);
            for (const auto &pk: pk_list) {
                table.setPrimaryKey(pk);
            }

            result.push_back(table);
        }
    }

    return result;
}
