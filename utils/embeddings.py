from typing import List
import openai
from sentence_transformers import SentenceTransformer
from vertexai.language_models import TextEmbeddingModel
import config

class EmbeddingManager:
    def __init__(self):
        self.models = {}
    
    def get_embeddings(self, texts: List[str], model_type: str = "openai") -> List[List[float]]:
        if model_type == "openai":
            return self._openai_embeddings(texts)
        elif model_type == "sentence-transformers":
            return self._sentence_transformer_embeddings(texts)
        elif model_type == "vertex":
            return self._vertex_embeddings(texts)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.embeddings.create(
            input=texts,
            model=config.EMBEDDING_MODELS["openai"]
        )
        return [item.embedding for item in response.data]
    
    def _sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        if "sentence-transformers" not in self.models:
            self.models["sentence-transformers"] = SentenceTransformer(
                config.EMBEDDING_MODELS["sentence-transformers"]
            )
        model = self.models["sentence-transformers"]
        embeddings = model.encode(texts)
        return embeddings.tolist()
    
    def _vertex_embeddings(self, texts: List[str]) -> List[List[float]]:
        model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODELS["vertex"])
        embeddings = []
        for text in texts:
            embedding = model.get_embeddings([text])[0]
            embeddings.append(embedding.values)
        return embeddings
