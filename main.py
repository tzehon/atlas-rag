import streamlit as st
import gcsfs
import getpass, os, pymongo, pprint
import inspect
import random
import time

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter, FilterOperator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo.errors import OperationFailure

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.set_page_config(page_title="Atlas with RAG", page_icon=":computer:")

st.title("Powered by MongoDB Atlas")

openai_api_key = st.text_input("OpenAI API Key", key="api_key", type="password")

conn_string = st.text_input("MongoDB Atlas Connection String", type="password")

st.write('Database: llamaindex_db')
st.write('Collection: test')
st.write('Vector index: vector_index')

col1, col2 = st.columns(2)

with col1:
    proj_id = st.text_input("GCP Project ID")

with col2:
    bucket = st.text_input("GCS Bucket")

all_fields_filled = (
    openai_api_key != '' and
    conn_string != '' and
    proj_id != '' and
    bucket != ''
)

if all_fields_filled:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if st.button('Init'):
        with st.spinner('Configuring chunking and selecting embedding model...'):
            Settings.llm = OpenAI()
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
            Settings.chunk_size = 100
            Settings.chunk_overlap = 10
            st.success('Chunking and embedding model configured!')

        with st.spinner('Loading files...'):
            gcs_fs = gcsfs.GCSFileSystem(project=proj_id)
            # st.write(gcs_fs.ls(bucket))
            sample_data = SimpleDirectoryReader(
                input_dir=bucket,
                fs=gcs_fs
            ).load_data()
            st.success('Files loaded!')

        with st.spinner('Instantiating Vector Store...'):
            mongodb_client = pymongo.MongoClient(conn_string)
            mongodb_coll = mongodb_client['llamaindex_db']['test']
            atlas_vector_search = MongoDBAtlasVectorSearch(
                mongodb_client,
                db_name = "llamaindex_db",
                collection_name = "test",
                index_name = "vector_index"
            )
            vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_search)
            st.success('Vector Store instantiated!')

        with st.spinner('Storing as vector embeddings...'):
            vector_store_index = VectorStoreIndex.from_documents(
                sample_data, storage_context=vector_store_context, show_progress=True
            )
            st.success('Vector embeddings stored!')

        with st.spinner('Creating vector search index...'):
            mongo_index_def = {
                'name': 'vector_index',
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
            st.success('Search index is building!')
else:
    st.write('Please fill out all fields.')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})