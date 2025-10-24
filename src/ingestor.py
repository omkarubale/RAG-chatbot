import glob
from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)


class Ingestor:
    def __init__(self):
        self.raw_docs = []
        self.source_pdfs = []
        self.source_urls = []
        self.source_texts = []

    def getRawDocs(self):
        return self.raw_docs

    def getSources(self):
        return {
            "pdfs": self.source_pdfs,
            "urls": self.source_urls,
            "texts": self.source_texts,
        }

    def addDocuments(self, files_path: str) -> bool:
        try:
            pdf_paths = glob.glob(files_path)
            docs = []

            for file_path in pdf_paths:
                docs.extend(PyPDFLoader(file_path).load())
                self.source_pdfs.append(file_path)

            self.raw_docs.extend(docs)

            print(
                f"Ingestor: Loaded {len(docs)} PDF pages from {len(pdf_paths)} files in {files_path}."
            )
            return True
        except Exception as e:
            print(f"Ingestor: loading documents from path: {files_path} failed: {e}")
            return False

    def addTextDocumentsByPattern(self, files_path: str, pattern: str) -> bool:
        try:
            loader = DirectoryLoader(
                files_path,
                glob=pattern,
                loader_cls=TextLoader,
                show_progress=True,
                use_multithreading=True,
            )
            docs = loader.load()
            self.raw_docs.extend(docs)

            for d in docs:
                src = d.metadata.get("source", None)
                if src and src not in self.source_texts:
                    self.source_texts.append(src)

            print(
                f"Ingestor: Loaded {len(docs)} PDF pages from {files_path} path matching {pattern}."
            )
            return True
        except Exception as e:
            print(f"Ingestor: loading documents from path: {files_path} failed: {e}")
            return False

    def addWebpages(self, url) -> bool:
        try:
            docs = WebBaseLoader([url]).load()
            self.raw_docs.extend(docs)

            self.source_urls.append(url)

            print(
                f"Ingestor: Loaded {len(docs)} documents from webpage contents {url}."
            )
            return True
        except Exception as e:
            print(f"Ingestor: loading webpage from url: {url} failed: {e}")
            return False
