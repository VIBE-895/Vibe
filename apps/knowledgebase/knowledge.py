# Created by guxu at 10/19/24
import chromadb
import os
import logging
from typing import List

from .. import PROJECT_BASE_PATH
from ..utils.utils import gen_id

DATABASE_PATH = os.path.join(PROJECT_BASE_PATH, 'persistent_clients')


class Knowledge():
    def __init__(self, client_path, collection_name):
        self.client_name = client_path
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(DATABASE_PATH, client_path))
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

    def gen_id(self, data):
        return [gen_id() for _ in range(len(data))]

    def add(self, data: List[str], ids=None):
        self.collection.add(
            documents=data,
            ids=self.gen_id(data) if ids is None else ids,
        )

    def print_knowledge(self):
        logging.info(self.chroma_client.list_collections())
        all_result = self.chroma_client.get_collection(name="fitness").get()
        logging.info(all_result)

    def query(self, query, top_k=10):
        if not self.chroma_client:
            raise RuntimeError("chroma client is None")

        result = self.collection.query(
            query_texts=query,
            n_results=top_k,
            # where_document={"$contains": fitness_type}
        )
        return result["documents"][0]


