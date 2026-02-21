from typing import List, Dict
import chromadb
import faiss
import numpy as np
from utils.embeddings import EmbeddingManager
import openai
import config

class RAGSystem:
    def __init__(self, embedding_model: str = "openai"):
        self.embedding_manager = EmbeddingManager()
        self.embedding_model = embedding_model
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.faiss_index = None
        self.chunks = []
    
    def index_documents(self, chunks: List[str], use_faiss: bool = False):
        self.chunks = chunks
        embeddings = self.embedding_manager.get_embeddings(chunks, self.embedding_model)
        
        if use_faiss:
            self._index_with_faiss(embeddings)
        else:
            self._index_with_chroma(chunks, embeddings)
    
    def _index_with_chroma(self, chunks: List[str], embeddings: List[List[float]]):
        self.collection = self.chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
    
    def _index_with_faiss(self, embeddings: List[List[float]]):
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_manager.get_embeddings([query], self.embedding_model)[0]
        
        if self.faiss_index:
            return self._retrieve_faiss(query_embedding, top_k)
        else:
            return self._retrieve_chroma(query, top_k)
    
    def _retrieve_chroma(self, query: str, top_k: int) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents'][0]
    
    def _retrieve_faiss(self, query_embedding: List[float], top_k: int) -> List[str]:
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.faiss_index.search(query_array, top_k)
        return [self.chunks[i] for i in indices[0]]
    
    def answer_question(self, query: str, top_k: int = 3) -> str:
        context_chunks = self.retrieve(query, top_k)
        context = "\n\n".join(context_chunks)
        
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer questions based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        return response.choices[0].message.content
