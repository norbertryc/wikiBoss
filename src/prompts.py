
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
Zdecyduj, czy poniższe pytanie wymaga rozbicia na kilka
niezależnych pytań faktograficznych.

Jeśli NIE wymaga rozbicia, zwróć dokładnie:
{
  "multi_hop": false,
  "subquestions": []
}

Jeśli WYMAGA rozbicia (np. porównanie dwóch osób), 
zwróć dokładnie:
{
  "multi_hop": true,
  "subquestions": ["pytanie 1", "pytanie 2"]
}

Zwróć WYŁĄCZNIE poprawny JSON. Nie dodawaj żadnego innego tekstu.

Pytanie:
{query}
"""