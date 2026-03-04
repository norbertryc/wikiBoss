
# ----------------------
# Główny RAG
# ----------------------
SYSTEM_PROMPT = """ 
Jesteś asystentem RAG. Odpowiadasz wyłącznie na podstawie dostarczonego kontekstu. 
Jeśli w kontekście nie ma odpowiedzi, napisz: "Brak odpowiedzi w dostarczonych dokumentach." 
Nie dodawaj informacji spoza kontekstu i nie zgaduj.
"""

# ----------------------
# Single-hop
# ----------------------
USER_PROMPT_TEMPLATE = """
Kontekst:
{context}

Pytanie:
{question}
"""

# ----------------------
# Multi-hop
# ----------------------
MULTI_HOP_PLANNER_PROMPT = """ 
Twoim zadaniem jest rozbić pytanie na prostsze pod-pytania, jeśli wymaga ono: 
- porównania dwóch lub więcej rzeczy, 
- obliczeń (np. różnica lat, kolejność wydarzeń), 
- pobrania informacji z więcej niż jednego źródła, 
 kilku kroków rozumowania. 
 
 Jeśli pytanie wymaga tylko jednego kroku → multi_hop = false. 
 Zwróć JSON: 
 { 
 "multi_hop": true/false, 
 "subquestions": ["...", "..."] 
 } 
 Pytanie: {query} 
 """