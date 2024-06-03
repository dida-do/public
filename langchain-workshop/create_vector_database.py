# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %%
import torch
# Check if CUDA is available
# it is not requires to use cuda.
torch.cuda.is_available()

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from transformers import (
    AutoTokenizer,
)
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from qdrant_client import models

import qdrant_client
import platform

def check_os()->str:
    """Check the OS of the system"""
    os_name = platform.system()
    return os_name


# %%
if check_os() == "Windows":
    embedding_path="huggingface_models\\BAAI\\bge-large-en-v1.5"
else:
    embedding_path="huggingface_models/BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
emb_tokenizer = AutoTokenizer.from_pretrained(embedding_path)


# %%
def _measure_length_with_tokenizer(text: str) -> int:
    """Measure the length of the text with tokenizer.

    Args:
        text: the text to be measured.

    Returns:
        The number of tokens of the input text.
    """
    return len(emb_tokenizer.encode(text))


# %%
# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    length_function=_measure_length_with_tokenizer,
    separators=["\n\n", "\n", ".", "?", ":", ",", ":", "!", " ", ""],
)
docs = []

# %%
paper_paths = list(Path("llm-papers").glob("*.pdf"))

# %%
docs = []
if check_os() == "Windows":
    title_splitter = "\\"
else:
    title_splitter = "/"
for paper in paper_paths:
    paper_pages = PyPDFLoader(str(paper)).load()
    title = paper_pages[0].metadata["source"]
    normalized_title = " ".join(
        title.split(title_splitter)[1][:-4].replace("_", " ").split(".")[-1].split()
    )
    content = ""
    for page in paper_pages:
        content += page.page_content + "\n"
    paper_pages[0].page_content = content
    paper_pages[0].metadata = {"source": normalized_title}
    docs.append(paper_pages[0])


# %%
splits = text_splitter.split_documents(docs)

# %%
chunk_index = 0
current_title = ""
for split in splits:
    if current_title != split.metadata["source"]:
        chunk_index = 0
        current_title = split.metadata["source"]
    split.metadata["chunk"] = chunk_index
    split.metadata = {"metadata":split.metadata}
    chunk_index += 1

# %%
splits[0].metadata

# %% [markdown]
# # Create a local qdrant database

# %%
qdrant = Qdrant.from_documents(
    splits,
    embeddings,
    path="./qdrant-database",
    collection_name="llm_papers",
)

# %%
qdrant.search("self-rag",search_type="similarity",k=5)
