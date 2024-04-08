[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atlas-rag-aq5fzeohyxbu8ghrzrv53p.streamlit.app/)

## Overview

This app is designed to demonstrate the power of Retrieval-Augmented Generation (RAG) architecture. It enables users to bring their own OpenAI API keys, MongoDB Atlas clusters, and PDF documents stored in Google Cloud Storage (GCS) buckets. Through MongoDB Atlas's vector search capabilities, the app provides a unified way to quickly query, retrieve semantically similar results, and interact with information extracted from PDFs, using OpenAI's LLMs.

### Features

- **Hosted Version Available**: Access the app immediately through [https://atlas-rag-aq5fzeohyxbu8ghrzrv53p.streamlit.app/](https://atlas-rag-aq5fzeohyxbu8ghrzrv53p.streamlit.app/) without any setup required.
- **Local Setup Option**: For those preferring to run the app on their own system, local setup instructions are provided.
- **Custom Resources**: Use your own OpenAI API keys, MongoDB Atlas clusters, and GCS buckets.
- **Data Vectorization and Storage**: Automates the vectorization of PDF content for efficient retrieval and storage in MongoDB Atlas.
- **Semantic Search**: Leverages MongoDB Atlas's vector search for semantically similar results.
- **Integration with OpenAI LLM**: Utilizes queries and retrieved results in conjunction with OpenAI language models for dynamic interaction.

## Getting Started

You can interact with the application through its hosted version or by setting it up locally on your machine.

### Hosted Version

Simply visit [https://atlas-rag-aq5fzeohyxbu8ghrzrv53p.streamlit.app/](https://atlas-rag-aq5fzeohyxbu8ghrzrv53p.streamlit.app/) to start using the app. No setup required!

### Local Installation

#### Prerequisites

- Python 3.10+
- venv
- OpenAI API key
- MongoDB Atlas cluster
- Google Cloud Platform account with a GCS bucket

#### Setup

1. Clone the repository:
   ```
   git clone https://github.com/tzehon/atlas-rag.git
   ```
2. Navigate to the app directory:
   ```
   cd atlas-rag
   ```
3. Create a virtual environment:
   ```
   python -m venv .venv
   ```
4. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On Unix or MacOS: `source .venv/bin/activate`
5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

After setup, run the application locally with:
```
streamlit run main.py
```
Then follow the on-screen instructions to interact with the app.

## How It Works

The application works by vectorizing PDF documents stored in GCS buckets and using MongoDB Atlas to store and retrieve these vectors. Users can submit queries that the system processes to find semantically similar documents. These results, along with the initial query, are then passed to OpenAI's language models for further interaction and enrichment.