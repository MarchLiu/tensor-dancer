'''
generate test dataset in pg

you need create extension vector at first;

the table is ```
create table items
(
    id        serial primary key,
    content   text,
    embedding vector(4096),
    indexed vector(256),
    meta jsonb default '{}'::jsonb
);

create index on cve using gin(meta);
create index on cve using hnsw(indexed vector_l2_ops);

create table pca
(
    id   serial primary key,
    meta jsonb default '{}'::jsonb,
    content bytea
)
create index on pca using gin(meta);

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

engine = create_engine("postgresql+psycopg2://localhost/cve")

# doc_path = "/Users/mars/jobs/blue-pro/postgresql-16.1/doc/src/sgml"
path = sys.argv[1]

exp = re.compile(r"<para>(.*?)</para>")
ollama_url = "http://localhost:11434/api/embeddings"


def embedding(segment, model='dolphin-llama3:256k'):
    doc = {
        "model": model,
        "prompt": segment
    }
    resp = requests.post(ollama_url, json=doc)
    return resp.json()["embedding"]


session_maker = sessionmaker(bind=engine)


def parse_doc(doc, encoding='gb18030', tablename='items'):
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
                session.execute(text(f"insert into {tablename}(content, embedding) "
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


def all_in_one(path, doc, encoding='ascii', tablename='items'):
    with open(doc, encoding=encoding) as f, session_maker() as session:
        meta = json.dumps({
            "filename": doc,
            "path": path
        })
        checkt = session.execute(text(f"select id from {tablename} where meta @> :meta"),
                                 {"meta": meta}).scalar()
        if checkt is not None:
            return

        content = f.read()

        vector = embedding(content)
        try:
            session.execute(text(f"insert into {tablename}(content, meta, embedding) "
                                 "values(:line, :meta, :emb)"
                                 "on conflict do nothing"),
                            {"emb": json.dumps(vector),
                             "meta": meta,
                             "line": content})
            print(file)
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
            if file.endswith(".md") and (file.strip() != 'README.md'):
                p = os.path.join(root, file)
                try:
                    # parse_doc(p, encoding="gb18030")
                    all_in_one(root, p, tablename='cve')
                except UnicodeDecodeError as err:
                    print(p)
                    print(err)
                    # parse_doc(p, encoding='utf16')
                    all_in_one(root, p, encoding='utf8', tablename='cve')

elif os.path.isfile(path):
    parse_doc(path)
else:
    print("invalid path")
    sys.exit(1)
