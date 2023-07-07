import json

from elasticsearch import Elasticsearch
from tqdm import tqdm

es_client = Elasticsearch("http://localhost:9200")

print("Try Deleting...")
try:
    es_client.indices.delete(index="opinion")
except Exception as e:
    print(f"Error: {e}")

print("Try Create...")
try:
    es_client.indices.create(index="opinion")
except Exception as e:
    print(f"Error: {e}")

if __name__ == '__main__':
    actions = []
    for row in tqdm(json.load(open("./report_2022_new.json", "r", encoding="utf-8"))):
        resp = es_client.index(index="opinion", id=row["id"], document=row)
        print(resp.body)

    result = es_client.count(index="opinion")
    print(result.body['count'])
