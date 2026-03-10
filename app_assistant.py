import os
import json

from config import Config
from groq import Groq
from src.retriever import Retriever
from src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, MULTI_HOP_PLANNER_PROMPT
from langchain_core.documents import Document

from src.vector_store import QdrantManager

cfg = Config()

class Assistant:
    """
    Assistant wraps retrieval + LLM generation.
    It uses a retriever to get relevant documents and a Groq LLM to generate answers.
    """

    def __init__(self, cfg: Config): 
        self.cfg = cfg 
        qdrant_manager = QdrantManager(url=cfg.qdrant_url)
        self.retriever = Retriever( 
            qdrant_manager=qdrant_manager, 
            collection_name=cfg.base_collection_name, 
            embedding_model=cfg.embedding_model, 
            top_k=cfg.top_k )

        self.client = self._initialize_client()

    def _initialize_client(self):
        """
        Create and return a Groq client using the API key from Config or env.
        Raises ValueError if no key is available.
        """
        api_key = self.cfg.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in Config or environment")
        return Groq(api_key=cfg.groq_api_key)
    
    
    def _convert_to_documents(self, results) -> list[Document]:
        """ Convert Qdrant search results (ScoredPoint) to LangChain Documents. 
        """ 
        documents: list[Document] = [] 
        for hit in results: documents.append( 
            Document( 
                page_content=hit.payload.get("text", ""), 
                metadata={ "title": hit.payload.get("title", ""), "score": hit.score, 
                }, 
            ) 
        ) 
            
        return documents


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
    
    def _plan_query(self, query: str) -> list[str]: 
        """
        Use LLM to decide whether the query requires multi-hop decomposition.
        Returns a list of subquestions if multi-hop, otherwise an empty list.
        """
        prompt = MULTI_HOP_PLANNER_PROMPT.replace("{query}", query)  
        response = self.client.chat.completions.create(  
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."}, 
                {"role": "user", "content": prompt},
            ],
            model=self.cfg.llm_model_name,
            temperature=0,  
        )

        content = response.choices[0].message.content.strip() 

        try:
            data = json.loads(content)  

            if data.get("multi_hop"):
                return data.get("subquestions", [])

            return []

        except Exception:
            return []


    def generate_answer(self, query: str) -> str:
        """
        Retrieve top_k documents from the retriever and generate an answer using Groq LLM.

        """
        subquestions = self._plan_query(query)

        if not subquestions:
            raw_results = self.retriever.retrieve(query)
            docs = self._convert_to_documents(raw_results)
            if not docs:
                return "No documents found for the query."
            
            prompt_context = self._build_context(docs, max_chars=self.cfg.llm_max_context_chars_single)

        else:
            combined_contexts = [] 

            for subq in subquestions:
                raw_results = self.retriever.retrieve(subq)
                docs = self._convert_to_documents(raw_results)
                if  docs:
                    ctx = self._build_context(docs, max_chars=self.cfg.llm_max_context_chars_per_subq)
                    combined_contexts.append(ctx)
            if not combined_contexts:
                return "No documents found for any of the subquestions."
            
            prompt_context = "\n".join(combined_contexts)
        
        prompt = USER_PROMPT_TEMPLATE.format( context=prompt_context, question=query)

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