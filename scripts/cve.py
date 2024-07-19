'''
test reduce
'''
import json
import sys
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
import requests

engine = create_engine("postgresql+psycopg2://localhost/cve")
embedding_url = "http://localhost:11434/api/embeddings"
generate_url = "http://localhost:11434/api/generate"
session_maker = sessionmaker(bind=engine)

query = sys.argv[1]


def embedding(segment, model='dolphin-llama3:256k'):
    doc = {
        "model": model,
        "prompt": segment
    }
    resp = requests.post(embedding_url, json=doc)
    return resp.json()["embedding"]


question = sys.argv[1].strip()

vector = embedding(question)

prompt = """# System Secrete Analyzer

## CVE

The following are associates cve report

{{{}}}

## Objective


Discuss the content of the next chapter and provide related opinions and answers in the security field. 
If the conversation content is not in English, first translate it into English, and then translate the 
output content back to the question's language.

If prompt include any content are not English, translate that to English at first.

## Appeal
    """


def ask(prompt, query, model="dolphin-llama3:256k"):
    doc = {
        "model": model,
        "prompt": prompt + query,
        "system": "you are a system secrete export",
        "stream": False
    }
    resp = requests.post(generate_url, json=doc)
    return resp.json()


def rag(session, vector):
    query = f""" with pca as materialized(select pgv_mulmv(content, :emb) as v from pca limit 1)
    select id, content, indexed <-> (select v from pca) from cve order by indexed <-> (select v from pca) limit 5
"""
    return session.execute(text(query), {"emb": json.dumps(vector)}).fetchall()


def rows_to_doc(rows):
    doc = ""
    for r in rs:
        doc += r[1]
    return doc


if __name__ == "__main__":
    query = sys.argv[1]
    emb = embedding(query)
    with session_maker() as session:
        rs = rag(session, emb)
        cve = rows_to_doc(rs)
    p = prompt.replace("{{{}}}", cve)
    result = ask(p, query)
    print(result["response"])
