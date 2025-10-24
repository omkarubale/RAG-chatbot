import os
import pathlib

# disable run on save to stop streamlit watcher from traversing pytorch files
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
# disable tokenizer parallelism as we are running on streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USER_AGENT"] = "Om-RAG-Chatbot/0.1"
import streamlit as st

from src.chain import Chain
from src.chunker import Chunker
from src.ingestor import Ingestor

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

URLS = [
    # --- BigCommerce – shipping & refunds ---
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
]


@st.cache_resource(show_spinner=True)
def initChain() -> tuple[Chain, Ingestor]:
    ingestor = Ingestor()

    ingestor.addDocuments("data/Everstorm_*.pdf")

    for url in URLS:
        ingestor.addWebpages(url)

    chunker = Chunker()
    chunker.addRawDocuments(ingestor.getRawDocs())

    chain = Chain("llama-3.3-70b-versatile", "thenlper/gte-small", chunker.getChunks())

    return chain, ingestor.getSources()


chain, sources = initChain()

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Data sources")

if sources.get("pdfs"):
    st.sidebar.subheader("PDF docs")

    for pdf_path in sources["pdfs"]:
        # show just the filename for cleanliness
        filename = pathlib.Path(pdf_path).name
        st.sidebar.markdown(f"- **{filename}**")

        # read the file bytes so Streamlit can offer it
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.sidebar.download_button(
            label="Download",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=f"download-{filename}",
        )

if sources.get("urls"):
    st.sidebar.subheader("URLs")
    for url in sources["urls"]:
        st.sidebar.markdown(f"- {url}")

if sources.get("texts"):
    st.sidebar.subheader("Text files")
    for txt in sources["texts"]:
        st.sidebar.markdown(f"- `{txt}`")

question = st.chat_input("What is in your mind?")
if question:
    with st.spinner("Thinking..."):
        response = chain.prompt(question, st.session_state.history)

for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(bot)
