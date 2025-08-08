from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingService:
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = SentenceTransformer("upskyy/e5-large-korean")
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            raise e
    
    def encode_text(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            print(f"임베딩 오류: {e}")
            raise e
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        try:
            embeddings = self.model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            print(f"배치 임베딩 오류: {e}")
            raise e
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            print(f"유사도 계산 오류: {e}")
            raise e

embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service