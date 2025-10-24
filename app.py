import os
import streamlit as st

from src.chain import Chain
from src.chunker import Chunker
from src.ingestor import Ingestor

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

URLS = [
    # --- BigCommerce – shipping & refunds ---
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
]


@st.cache_resource(show_spinner=True)
def initChain():
    # disable tokenizer parallelism as we are running on streamlit
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ingestor = Ingestor()

    ingestor.addDocuments("data/Everstorm_*.pdf")

    for url in URLS:
        ingestor.addWebpages(url)

    chunker = Chunker()
    chunker.addRawDocuments(ingestor.getRawDocs())

    chain = Chain("llama-3.3-70b-versatile", "thenlper/gte-small", chunker.getChunks())

    return chain


chain = initChain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("What is in your mind?")
if question:
    with st.spinner("Thinking..."):
        response = chain.prompt(question, st.session_state.history)

for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(bot)
