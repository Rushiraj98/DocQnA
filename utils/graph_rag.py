from typing import List, Dict
import networkx as nx
from neo4j import GraphDatabase
import openai
import config
from utils.embeddings import EmbeddingManager

class GraphRAGSystem:
    def __init__(self, embedding_model: str = "openai"):
        self.embedding_manager = EmbeddingManager()
        self.embedding_model = embedding_model
        self.graph = nx.Graph()
        self.neo4j_driver = None
        self.chunks = []
        
        if config.NEO4J_URI:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
                )
            except:
                pass
    
    def build_knowledge_graph(self, chunks: List[str]):
        self.chunks = chunks
        
        for i, chunk in enumerate(chunks):
            entities = self._extract_entities(chunk)
            
            for entity in entities:
                self.graph.add_node(entity, chunk_id=i, text=chunk)
            
            for j in range(len(entities)):
                for k in range(j + 1, len(entities)):
                    self.graph.add_edge(entities[j], entities[k], chunk_id=i)
        
        if self.neo4j_driver:
            self._sync_to_neo4j()
    
    def _extract_entities(self, text: str) -> List[str]:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract key entities (people, places, concepts) from text. Return as comma-separated list."},
                {"role": "user", "content": text}
            ]
        )
        
        entities_str = response.choices[0].message.content
        return [e.strip() for e in entities_str.split(",") if e.strip()]
    
    def _sync_to_neo4j(self):
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            for node in self.graph.nodes(data=True):
                session.run(
                    "CREATE (n:Entity {name: $name, chunk_id: $chunk_id, text: $text})",
                    name=node[0],
                    chunk_id=node[1].get('chunk_id', -1),
                    text=node[1].get('text', '')
                )
            
            for edge in self.graph.edges(data=True):
                session.run(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    CREATE (a)-[:RELATED_TO {chunk_id: $chunk_id}]->(b)
                    """,
                    source=edge[0],
                    target=edge[1],
                    chunk_id=edge[2].get('chunk_id', -1)
                )
    
    def retrieve_with_graph(self, query: str, top_k: int = 3) -> List[str]:
        query_entities = self._extract_entities(query)
        
        relevant_chunks = set()
        for entity in query_entities:
            if entity in self.graph:
                neighbors = list(self.graph.neighbors(entity))
                for neighbor in neighbors[:top_k]:
                    chunk_id = self.graph.nodes[neighbor].get('chunk_id')
                    if chunk_id is not None and chunk_id < len(self.chunks):
                        relevant_chunks.add(self.chunks[chunk_id])
        
        return list(relevant_chunks)[:top_k]
    
    def answer_question(self, query: str, top_k: int = 3) -> str:
        context_chunks = self.retrieve_with_graph(query, top_k)
        
        if not context_chunks:
            return "No relevant information found in the knowledge graph."
        
        context = "\n\n".join(context_chunks)
        
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer questions using the knowledge graph context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        return response.choices[0].message.content
    
    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()
