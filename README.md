# DocQnA - Advanced Document Q&A System

AI-powered document question-answering system with GraphRAG, multiple embeddings, and GCP integration.

## Features

- **Dual RAG Modes**: Standard RAG, GraphRAG, or Hybrid approach
- **Multiple Embeddings**: OpenAI, Sentence Transformers, Vertex AI
- **Vector Stores**: ChromaDB and FAISS support
- **GCP Integration**: Cloud Storage, BigQuery, Document AI, Cloud Logging
- **Knowledge Graphs**: NetworkX + Neo4j for entity relationships
- **Interactive UI**: Streamlit-based interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. GCP Setup:
   - Create a GCP project
   - Enable APIs: Cloud Storage, BigQuery, Document AI, Cloud Logging
   - Create service account and download JSON key
   - Create a Cloud Storage bucket

4. (Optional) Neo4j Setup:
```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:latest
```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Architecture

- **app.py**: Main Streamlit interface
- **utils/rag.py**: Standard RAG with ChromaDB/FAISS
- **utils/graph_rag.py**: GraphRAG with entity extraction
- **utils/embeddings.py**: Multi-model embedding support
- **utils/gcp_services.py**: GCP service integrations
- **utils/document_processor.py**: PDF processing and chunking

## GCP Services Used

1. **Cloud Storage**: Document storage
2. **Cloud Logging**: Query logging
3. **BigQuery**: Analytics storage
4. **Document AI**: Advanced PDF processing
5. **Vertex AI**: Embeddings (textembedding-gecko)
