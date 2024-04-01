import streamlit as st
import gcsfs
import os, pymongo, pprint

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter, FilterOperator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

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
            if 'vector_store_index' not in st.session_state:
                st.session_state.vector_store_index = VectorStoreIndex.from_documents(
                    sample_data, storage_context=vector_store_context, show_progress=True
                )
            else:
                vector_store_index = st.session_state.vector_store_index
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

def response_generator(prompt):
    vector_store_index = st.session_state.vector_store_index
    # Instantiate Atlas Vector Search as a retriever
    vector_store_retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=5)
    # Pass the retriever into the query engine
    query_engine = RetrieverQueryEngine(retriever=vector_store_retriever)
    # Prompt the LLM
    response = query_engine.query(prompt)
    print(response)
    print("\nSource documents: ")
    pprint.pprint(response.source_nodes)
    return response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_markdown(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})