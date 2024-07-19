"""
将指定的文件写入到数据库中的bytea字段
"""
import json
import sys
import psycopg2
from psycopg2 import sql

db_params = {
    'dbname': 'cve',
    'host': 'localhost',
}

mfile = sys.argv[1]
with open(mfile, 'rb') as f:
    blob = f.read()

query = sql.SQL("INSERT INTO pca (content, meta) VALUES (%s, %s)")

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

try:
    meta = {
        "type": "matrix",
        "category": "pca",
        "samples": "cve",
        "row": 256,
        "column": 4096,
    }
    # 执行SQL语句
    cur.execute(query, (blob, json.dumps(meta)))

    # 提交事务
    conn.commit()
    print("数据写入成功。")
except Exception as e:
    # 发生错误时回滚事务
    conn.rollback()
    print("数据写入失败：", e)
finally:
    # 关闭游标和连接
    cur.close()
    conn.close()