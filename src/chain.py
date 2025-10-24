import streamlit as st

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.retriever_factory import RetrieverFactory


SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""


class Chain:
    def __init__(
        self, llm_model_name: str, embedding_model_name: str, chunks: List[Document]
    ):
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            raise ValueError("Chain: API key not found. Cannot initialize LLM Model.")

        llm = ChatGroq(
            api_key=api_key,
            model=llm_model_name,
            temperature=0.1,
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self._getRetriever(embedding_model_name, chunks),
            combine_docs_chain_kwargs={"prompt": self._getPromptTemplate()},
            return_source_documents=True,
        )

    def _getRetriever(self, embedding_model_name: str, chunks: List[Document]):
        return RetrieverFactory(embedding_model_name, chunks).getRetriever()

    def _getPromptTemplate(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context", "question"], template=SYSTEM_TEMPLATE
        )

    def getChain(self) -> ConversationalRetrievalChain:
        return self.chain

    def prompt(self, question: str, chat_history: List[str]) -> dict:
        result = self.chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))

        return result
