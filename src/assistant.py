import logging
from groq import Groq
from config import config
from .vector_store import QdrantManager

logger = logging.getLogger(__name__)

class WikiBoss:
    """
    Basic RAG interface for Polish Wikipedia using Groq.
    """

    def __init__(
            self,
            qdrant_manager: QdrantManager,
            api_key: str = None,
            model: str = config.primary_model,
            fallback_model: str = config.fallback_model,
            enable_fallback: bool = True
    ):
        """
        Initialize WikiBoss RAG system.

        Args:
            qdrant_manager: QdrantManager instance for document retrieval
            api_key: Groq API key
            model: Primary LLM model name (default: llama-3.3-70b-versatile)
            enable_fallback: Enable automatic fallback to secondary model on rate limits
        """
        self.qdrant_manager = qdrant_manager
        self.client = Groq(api_key=api_key)
        self.model = model
        self.fallback_model = fallback_model if enable_fallback else None
        self._temperature = 0.2
        self._top_k = 5

        logger.info(f"WikiBoss initialized with model: {model}")
        if enable_fallback:
            logger.info(f"Fallback model enabled: {self.fallback_model}")

    @property
    def temperature(self) -> float:
        """
        LLM temperature parameter (0.0 = deterministic, 1.0 = creative).

        Returns:
            float: Current temperature value
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """
        Set LLM temperature with validation.

        Args:
            value: Temperature value between 0 and 1

        Raises:
            ValueError: If value not in [0, 1] range
        """
        if not 0 <= value <= 1.:
            raise ValueError("Temperature must be between 0 and 1")
        self._temperature = value

    @property
    def top_k(self) -> int:
        """
        Number of documents to retrieve from vector store.

        Returns:
            int: Current top_k value
        """
        return self._top_k

    @top_k.setter
    def top_k(self, value: int):
        """
        Set number of documents to retrieve with validation.

        Args:
            value: Number of documents between 1 and 25

        Raises:
            ValueError: If value not in [1, 25] range
        """
        if not 1 <= value <= 25:
            raise ValueError("Top k must be between 1 and 20")
        self._top_k = value

    @staticmethod
    def build_message(question: str, context: str) -> list:
        """
        Build messages for Groq API

        Args:
            question: User question
            context: Formatted context

        Returns:
            List of message dicts
        """
        system_prompt = {
            "role": "system",
            "content": (
                "Jesteś pomocnym asystentem AI odpowiadającym na pytania na podstawie polskiej Wikipedii.\n\n"
                "ZASADY:\n"
                "1. Odpowiadaj TYLKO na podstawie podanych fragmentów artykułów\n"
                "2. Bądź zwięzły i konkretny\n"
                "3. Jeśli fragmenty nie zawierają odpowiedzi, powiedz: 'Przepraszam, nie znalazłem odpowiedzi w dostępnych źródłach'\n\n"
            )
        }
        user_query = {
            "role": "user",
            "content": (
                f"Kontekst (fragmenty artykułów):\n"
                f"{context}\n\n"
                f"Pytanie: {question}"
            )
        }

        return [system_prompt, user_query]

    def _retrieve(self, question: str, top_k: int) -> list[dict]:
        """
        Retrieve relevant documents from Qdrant

        Args:
            question: Query text
            top_k: Number of documents

        Returns:
            List of retrieved documents with metadata
        """
        try:
            results = self.qdrant_manager.retrieve(
                query=question,
                limit=top_k
            )
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _generate( self, question: str, context: str, model: str
                   ) -> str:
        """
        Generate answer using Groq

        Args:
            question: User question
            context: Formatted context
            model: Model to use

        Returns:
            Generated answer
        """
        messages = self.build_message(question, context)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self._temperature
        )

        return response.choices[0].message.content

    def _generate_with_fallback(
            self,
            question: str,
            context: str
    ) -> tuple[str, str]:
        """
        Generate answer with automatic fallback

        Args:
            question: User question
            context: Formatted context

        Returns:
            Tuple of (answer, model_used)
        """
        try:
            answer = self._generate(question, context, self.model)
            return answer, self.model

        except Exception as e:
            if ("429" in str(e)) and self.fallback_model:
                logger.warning(f"Rate limit on {self.model}, using fallback: {self.fallback_model}")

                try:
                    answer = self._generate(question, context, self.fallback_model)
                    return answer, self.fallback_model
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise

            logger.error(f"Generation failed: {e}")
            raise


    def query(self,
              question: str,
              return_sources: bool = True
              ) -> dict:
        """
        Main RAG query method

        Args:
            question: User question
            return_sources: Include source documents in response

        Returns:
            dict with 'answer' and optionally 'sources', 'contexts', 'model_used'
        """
        retrieval = self._retrieve(question, self.top_k)

        if not retrieval:
            return {
                "question": question,
                "answer": "Przepraszam, nie znalazłem żadnych relevantnych źródeł.",
                "sources": [],
                "model_used": None
            }

        context = "\n".join([result["payload"]["text"] for result in retrieval])
        answer, model_used = self._generate_with_fallback(question, context)

        results = {
            "question": question,
            "answer": answer
        }

        if return_sources:
            results["model_used"] = model_used
            results["sources"] = retrieval
            results["context"] = context

        return results

    def invoke(self, question: str) -> str:
        """
        Simple invoke - returns just the answer string.

        Args:
            question: User question

        Returns:
            str: Generated answer
        """
        result = self.query(question, return_sources=False)
        return result["answer"]
