from typing import List, Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class RetrieverFactory:
    def __init__(self, embedding_model_name: str, chunks: List[Document]):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        if not chunks:
            raise ValueError("RetrieverFactory: got 0 chunks. Cannot build index.")

        vectordb = FAISS.from_documents(chunks, self.embedding_model)
        vectordb.save_local("faiss_index")
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 8})

        print("RetriverFactory: Vector store with", vectordb.index.ntotal, "embeddings")

    def getRetriever(self):
        return self.retriever

    def getEmbeddingModel(self) -> HuggingFaceEmbeddings:
        return self.embedding_model

    def embedQuery(self, query: str) -> Union[List[float], None]:
        if not self.embedding_model:
            return None
        return self.embedding_model.embed_query("Hello World!")
