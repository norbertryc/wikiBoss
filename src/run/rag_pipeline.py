import os
from dotenv import load_dotenv
from config import HUGGING_FACE_MODEL, ENV_PATH
from src.vector_store import QdrantManager
from src.embedding_engine import EmbeddingEngine
from src.assistant import WikiBoss


def rag_pipeline(
        embedding_model: dict = HUGGING_FACE_MODEL,
        qdrant_collection_name: str = "on_md_headers_e5_full",
        device: str = "cuda",
        half_precision: bool = False
):
    load_dotenv(dotenv_path=ENV_PATH)
    groq_api_key = os.getenv("GROQ_API_KEY")

    embedding_engine = EmbeddingEngine(embedding_model,
                                       device=device,
                                       use_fp16=half_precision)
    qdrant = QdrantManager(embedding_engine=embedding_engine,
                           collection_name=qdrant_collection_name)
    assistant = WikiBoss(qdrant_manager=qdrant,
                         api_key=groq_api_key)

    print("Zadaj pytanie (wpisz 'quit' aby wyjść)")
    while True:
        question = input("Twoje pytanie: ").strip()

        if question.lower() == "quit":
            break

        if not question:
            continue

        try:
            answer = assistant.invoke(question)
            print(f"\nWikiBoss: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    rag_pipeline()