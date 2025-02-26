from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
from ragflow_sdk.modules.chunk import Chunk


class Attack_Type(Enum):
    RAG_POISONING = "RAG_POISONING"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    HALLUCINATION = "HALLUCINATION"

    def __init__(self, value: str):
        self.parameters: Dict[str, Any] = {}

    def set_parameters(self, params: Dict[str, Any]):
        self.parameters = params

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters


class Defense_Type(Enum):
    PARAPHRASING = "PARAPHRASING"
    PERPLEXITY_DETECTION = "PERPLEXITY_DETECTION"
    KNOWLEDGE_EXPANSION = "KNOWLEDGE_EXPANSION"

    def __init__(self, value: str):
        self.parameters: Dict[str, Any] = {}

    def set_parameters(self, params: Dict[str, Any]):
        self.parameters = params

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters


@dataclass
class ChunkWrapper:
    id: str
    content: str
    important_keywords: List[str]
    questions: List[str]
    create_time: str
    create_timestamp: float
    dataset_id: str
    document_name: str
    document_id: str
    available: bool

    @classmethod
    def from_rag_chunk(cls, chunk: Chunk):
        return cls(
            id=chunk.id,
            content=chunk.content,
            important_keywords=chunk.important_keywords,
            questions=chunk.questions,
            create_time=chunk.create_time,
            create_timestamp=chunk.create_timestamp,
            dataset_id=chunk.dataset_id,
            document_name=chunk.document_name,
            document_id=chunk.document_id,
            available=chunk.available,
        )

    def to_json(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the ChunkWrapper."""
        return asdict(self)
