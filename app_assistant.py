import os
from config import Config
from groq import Groq
from src.retriever import Retriever
from src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from langchain_core.documents import Document


cfg = Config()

class Assistant:
    """
    Assistant wraps retrieval + LLM generation.
    It uses a retriever to get relevant documents and a Groq LLM to generate answers.
    """

    def __init__(self, cfg: Config): 
        self.cfg = cfg 
        self.retriever = Retriever(qdrant_url=cfg.qdrant_url, 
                                   collection_name=cfg.base_collection_name, 
                                   embedding_model=cfg.embedding_model, 
                                   top_k=cfg.top_k) 
        
        self.client = self._initialize_client()
    
    def _convert_to_documents(self, results) -> list[Document]:
        """ Convert Qdrant search results (ScoredPoint) to LangChain Documents. 
        """ 
        documents: list[Document] = [] 
        for hit in results: documents.append( 
            Document( 
                page_content=hit.payload.get("content", ""), 
                metadata={ "title": hit.payload.get("title", ""), "score": hit.score, 
                }, 
            ) 
        ) 
            
        return documents

    def _initialize_client(self):
        """
        Create and return a Groq client using the API key from Config or env.
        Raises ValueError if no key is available.
        """
        api_key = self.cfg.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in Config or environment")
        return Groq(api_key=cfg.groq_api_key)

    def _build_context(self, docs, max_chars=None):
        """
        Build context from retrieved documents.
        If the context is too long for the LLM, trim it with max_chars.
        """
        texts = [d.page_content for d in docs]
        context = "\n".join(texts)

        if max_chars and len(context) > max_chars:
            context = context[:max_chars]

        return context


    def generate_answer(self, query: str) -> str:
        """
        Retrieve top_k documents from the retriever and generate an answer using Groq LLM.

        """
        raw_results = self.retriever.retrieve( query, top_k=cfg.top_k)
        docs = self._convert_to_documents(raw_results)
        if not docs:
            return "No documents found in the retriever."

        context_text = self._build_context(docs, max_chars=self.cfg.llm_max_context_chars)
        
        prompt = USER_PROMPT_TEMPLATE.format( context=context_text, question=query)

        response = self.client.chat.completions.create( 
            messages=[ {"role": "system", "content": SYSTEM_PROMPT}, 
                      {"role": "user", "content": prompt}, 
                      ], 
                      model=self.cfg.llm_model_name, 
        )

        return response.choices[0].message.content


if __name__ == "__main__": 
    cfg = Config() 
    assistant = Assistant(cfg) 
    print("Interactive mode. Press Ctrl+C to exit.\n") 
    
    for question in iter(input, ""): 
        answer = assistant.generate_answer(question) 
        print("Answer:\n", answer, "\n")