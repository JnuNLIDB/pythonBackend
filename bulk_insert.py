import json

from elasticsearch import Elasticsearch
from tqdm import tqdm

es_client = Elasticsearch("http://localhost:9200")

es_client.indices.create(index="opinion")

if __name__ == '__main__':
    actions = []
    for row in tqdm(json.load(open("./report_2022_new.json", "r", encoding="utf-8"))):
        action = {"index": {"_index": "opinion", "_id": int(row["id"])}}
        actions.append(action)
        actions.append(row)
    es_client.bulk(index="opinion", operations=actions)

    result = es_client.count(index="opinion")
    print(result.body['count'])
