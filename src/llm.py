import os
import sys
from groq import Groq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .retriever import retriever   

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")

client = Groq(api_key=GROQ_API_KEY)

def ask_about(query: str, top_k: int = 5) -> str:
    """Retrieve documents and generate an answer with Groq LLM"""
    docs = retriever.get_relevant_documents(query)[:top_k]
    if not docs:
        return "No documents found in the retriever."
    
    context_text = "\n".join([d.page_content for d in docs])
    prompt = (
    f"Answer the question using the following text as context. "
    f"You can also use your own knowledge if needed:\n\n"
    f"{context_text}\n\nQuestion: {query}"
)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = "Kim była Maria Skłodowska-Curie?"
    answer = ask_about(query)
    print("Answer:\n", answer)
