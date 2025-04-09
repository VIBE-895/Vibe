# Created by guxu at 4/3/25
import shutil

from apps.knowledgebase.knowledge import Knowledge
from apps import PROJECT_BASE_PATH
import logging
import os
import pytest

client_path = os.path.join(PROJECT_BASE_PATH, 'test', 'unittest', 'knowledgebase', 'persistent_clients')
os.makedirs(client_path, exist_ok=True)

@pytest.fixture(autouse=True)
def romove_existing_client():
    if os.path.exists(client_path):
        shutil.rmtree(client_path)
    yield
    if os.path.exists(client_path):
        shutil.rmtree(client_path)

def test_constructor():
    knowledge = Knowledge(client_path, collection_name="test1")
    if not knowledge.collection_name == "test1":
        raise RuntimeError("Knowledge constructor failed")
    logging.info("Knowledge constructor passed")

def test_add():
    knowledge = Knowledge(client_path, collection_name="test2")
    knowledge.add([
        "Collections are where you'll store your embeddings, documents, and any additional metadata. Collections index your embeddings and documents, and enable efficient retrieval and filtering. You can create a collection with a name:",
        "Chroma will store your text and handle embedding and indexing automatically. You can also customize the embedding model. You must provide unique string IDs for your documents."
    ])
    assert knowledge.collection.count() == 2

def test_search():
    knowledge = Knowledge(client_path, collection_name="test3")
    knowledge.add([
        "Collections are where you'll store your embeddings, documents, and any additional metadata. Collections index your embeddings and documents, and enable efficient retrieval and filtering. You can create a collection with a name:",
        "Chroma will store your text and handle embedding and indexing automatically. You can also customize the embedding model. You must provide unique string IDs for your documents."
    ])
    res = knowledge.query(query="chroma collection")
    print(type(res))
    print(res)

