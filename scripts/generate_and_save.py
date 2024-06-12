'''
generate test dataset in pg

you need create extension vector at first;

the table is ```
create table items
(
    id            bigserial
        primary key,
    content       text,
    embedding     vector(4096),
    norm_256  vector(256),
    norm_512  vector(512),
    norm_1024 vector(1024)
);
```

requirements:
 * sqlalchemy
 * psycopg2
 * requests

$un it as:

```
python generate_and_save.py xxx.txt
```

The script will insert content, origin vector and reduced vectors into items table line by line.

'''
import os.path
import sys

import sqlalchemy
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
import re, json
import requests

engine = create_engine("postgresql+psycopg2://localhost/pgv")

# doc_path = "/Users/mars/jobs/blue-pro/postgresql-16.1/doc/src/sgml"
path = sys.argv[1]

exp = re.compile(r"<para>(.*?)</para>")
ollama_url = "http://localhost:11434/api/embeddings"


def embedding(segment, model='llama3:8b'):
    doc = {
        "model": model,
        "prompt": segment
    }
    resp = requests.post(ollama_url, json=doc)
    return resp.json()["embedding"]


session_maker = sessionmaker(bind=engine)


def parse_doc(doc, encoding='gb18030'):
    with open(doc, encoding=encoding) as f, session_maker() as session:
        for l in f:
            line = l.strip()
            if not line:
                continue
            content = line.strip()
            checkt = session.execute(text("select id from items where content=:content"),
                                     {"content": content}).scalar()
            if checkt is not None:
                continue

            vector = embedding(content)
            try:
                session.execute(text("insert into items(content, embedding) "
                                     "values(:line, :emb)"
                                     "on conflict do nothing"),
                                {"emb": json.dumps(vector), "line": content})
                print(line)
                session.commit()
            except sqlalchemy.exc.OperationalError as err:
                session.rollback()
                print(err)
                # for segment in line.split():
                #     vector = embedding(segment)
                #     session.execute(text("insert into items(content, embedding) "
                #                          "values(:line, :emb)"
                #                          "on conflict do nothing"),
                #                     {"emb": json.dumps(vector), "line": segment})
                #     session.commit()
                #     print(segment)


if os.path.isdir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                p = os.path.join(root, file)
                try:
                    parse_doc(p, encoding="gb18030")
                except UnicodeDecodeError as err:
                    print(p)
                    print(err)
                    parse_doc(p, encoding='utf16')

elif os.path.isfile(path):
    parse_doc(path)
else:
    print("invalid path")
    sys.exit(1)
