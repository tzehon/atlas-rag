import streamlit as st
import gcsfs
import os, pymongo
import time

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.chat_engine import ContextChatEngine

st.set_page_config(page_title="Atlas with RAG", page_icon=":computer:")

st.title("Powered by MongoDB Atlas 👋")

openai_api_key = st.text_input("OpenAI API Key :key:", key="api_key", type="password")

conn_string = st.text_input("MongoDB Atlas Connection String :thread:", type="password", help="mongodb+srv://<username>:<password>@xxx.yyy.mongodb.net/?retryWrites=true&w=majority")

col1, col2 = st.columns(2)

with col1:
    db = st.text_input("Database :file_cabinet:", help="Doesn't have to exist beforehand")

with col2:
    coll = st.text_input("Collection :books:", help="Doesn't have to exist beforehand")

col3, col4, col5 = st.columns(3)

with col3:
    proj_id = st.text_input("GCP Project ID :cloud:", help="Project ID pls, not Project Name!")

with col4:
    bucket = st.text_input("GCS Bucket :bucket:", help="Include a trailing slash", placeholder="bucket-name/")

with col5:
    access_token = st.text_input("GCP Access Token :key:", type="password", placeholder="Optional - for private buckets", help="If GCS bucket is private, retrieve token by issuing 'gcloud auth login' and 'gcloud auth print-access-token'")

all_fields_filled = (
    openai_api_key != '' and
    conn_string != '' and
    db != '' and
    coll != '' and
    proj_id != '' and
    bucket != ''
)

def configure_models():
    Settings.llm = OpenAI(model="gpt-4", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.chunk_size = 100
    Settings.chunk_overlap = 10

def vector_store(conn_string, db, coll):
    mongodb_client = pymongo.MongoClient(conn_string)
    mongodb_coll = mongodb_client[db][coll]
    atlas_vector_search = MongoDBAtlasVectorSearch(
        mongodb_client,
        db_name = db,
        collection_name = coll,
        index_name = f'{db}_{coll}_index'
    )
    vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_search)
    return mongodb_coll, vector_store_context

def vector_embeddings():
    return ""

def load_data(proj_id, bucket):
    if access_token:
        gcs_fs = gcsfs.GCSFileSystem(project=proj_id, token=access_token)
    else:
        gcs_fs = gcsfs.GCSFileSystem(project=proj_id)
    sample_data = SimpleDirectoryReader(
        input_dir=bucket,
        fs=gcs_fs
    ).load_data()
    return sample_data

if all_fields_filled:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if st.button('Init'):
        with st.spinner('Configuring chunking, embedding and LLM models...'):
            configure_models()
            st.success('Chunking, embedding and LLM models configured!')

        with st.spinner('Loading files...'):
            sample_data = load_data(proj_id, bucket)
            st.success('Files loaded!')

        with st.spinner('Instantiating Vector Store...'):
            mongodb_coll, vector_store_context = vector_store(conn_string, db, coll)
            st.success('Vector Store instantiated!')

        with st.spinner('Storing as vector embeddings...'):
            if 'vector_store_index' not in st.session_state:
                st.session_state.vector_store_index = VectorStoreIndex.from_documents(
                    sample_data, storage_context=vector_store_context, show_progress=True
                )
            else:
                vector_store_index = st.session_state.vector_store_index
            st.success('Vector embeddings stored!')

        with st.spinner('Creating vector search index...'):
            mongo_index_def = {
                'name': f'{db}_{coll}_index',
                'definition': {
                    'mappings': {
                        'dynamic': True,
                        'fields': {
                            'embedding': {
                                'type': 'knnVector',
                                'dimensions': 1536,
                                'similarity': 'cosine'
                            }
                        }
                    }
                }
            }

            mongodb_coll.create_search_index(mongo_index_def)
            st.success('Search index is building! Checkout the Atlas UI 😊')
else:
    st.write('Please fill out all fields.')

def response_generator(prompt):
    vector_store_index = st.session_state.vector_store_index
    # Instantiate Atlas Vector Search as a retriever
    vector_store_retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=20)
    chat_engine = ContextChatEngine.from_defaults(
        retriever=vector_store_retriever,
        system_prompt="You are a chatbot, able to answer from the context shared and also the prior knowledge of your LLM")
    # Prompt the LLM
    with st.spinner(text="Thinking..."):
        streaming_response = chat_engine.stream_chat(prompt)
    for text in streaming_response.response_gen:
        yield text
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me something!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})