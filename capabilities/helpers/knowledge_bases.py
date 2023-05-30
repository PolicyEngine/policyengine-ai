from typing import Iterable, Callable
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import chromadb
from capabilities.helpers.text_splitters import llm_split

class KnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def add(self, value: str, splitter: Callable[[str], Iterable[str]] = llm_split):
        raise NotImplementedError

    def search(self, query: str, top_n: int = 1) -> Iterable[str]:
        raise NotImplementedError


    def partition_and_embed(self, value: str, split_fn: Callable[[str], Iterable[str]]):
        values_to_add = []
        splits = split_fn(value)
        for partitioned_value in set(splits):
            values_to_add.append(partitioned_value)
        embeddings_to_add = self.model.encode(values_to_add)
        return values_to_add, embeddings_to_add

class NumPyKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self.data = []
        self.embeddings = []
        self.embeddings_array = None
        super().__init__()

    def save(self, path: str):
        np.save(path, self.embeddings_array)
        pd.DataFrame({"data": self.data}).to_csv(
            path + ".csv.gz", index=False, compression="gzip"
        )

    def load(self, path: str):
        self.embeddings_array = np.load(path)
        self.data = pd.read_csv(path + ".csv.gz")["data"].tolist()

    def add(self, value: str, split_fn: Callable[[str], Iterable[str]]):
        values_to_add, embeddings_to_add = self.partition_and_embed(value, split_fn)

        self.data.extend(values_to_add)
        self.embeddings.extend(embeddings_to_add)
        if self.embeddings_array is None:
            self.embeddings_array = np.array(self.embeddings)
        else:
            self.embeddings_array = np.vstack(
                (self.embeddings_array, np.array(embeddings_to_add))
            )

    def search(self, query: str, top_n: int = 1) -> Iterable[str]:
        query_embedding = self.model.encode(query)
        similarity = util.dot_score(query_embedding, self.embeddings_array)[0]
        top_n_idx = np.argsort(similarity)[-top_n:]
        for idx in top_n_idx:
            yield self.data[idx]

class ChromaKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self.client = chroma_client = chromadb.Client()
        self.collection = chroma_client.create_collection(name="tmp")
        super().__init__()

    def save(self, name:str):
        self.collection.modify(name=name)
        self.collection.persist()

    def load(self, name:str):
        self.collection = self.client.get_collection(name=name)

    def add(self, value: str, split_fn: Callable[[str], Iterable[str]]):
        values_all, embeddings_all = self.partition_and_embed(value, split_fn=split_fn)

        values_to_add = []
        embeddings_to_add = [] 
        ids = []
        for v, e in zip(values_all, embeddings_all):
            id = str(hash(v))

            existing_ids = self.collection.get(ids=[id])["ids"]
            if existing_ids:
                continue
            
            values_to_add.append(v)
            embeddings_to_add.append(e)
            ids.append(id)

        self.collection.add(
            embeddings=embeddings_to_add,
            documents=values_to_add,
            ids=ids
        )

    
    def search(self, query: str, top_n: int = 1) -> Iterable[str]:
        query_embedding = self.model.encode(query) 
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )['documents'][0]

        
