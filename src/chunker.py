from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self):
        self.chunks = []

    def getChunks(self) -> List[Document]:
        return self.chunks

    def addRawDocuments(self, raw_documents: List[Document]) -> bool:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=30, length_function=len
            )

            chunks = text_splitter.split_documents(raw_documents)

            self.chunks.extend(chunks)

            print(
                f"Chunker: Loaded {len(raw_documents)} raw documents into {len(chunks)} chunks."
            )
            return True
        except Exception as e:
            print(
                f"Chunker: loading chunks from {len(raw_documents)} raw documents failed: {e}"
            )
            return False
