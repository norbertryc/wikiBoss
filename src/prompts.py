
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
# ----------------------
# Multi-hop
# ----------------------
MULTI_HOP_PLANNER_PROMPT = """ 
Twoim zadaniem jest zdecydować, czy pytanie wymaga wielu kroków (multi-hop),
a jeśli tak – rozbić je na prostsze pod-pytania.

Zawsze ustaw "multi_hop": true, jeśli pytanie:
- porównuje dwie lub więcej osób,
- zawiera sformułowania typu:
  - "kto żył dłużej",
  - "kto był starszy w momencie śmierci",
  - "kto urodził się wcześniej",
  - "kto zmarł wcześniej",
  - "o ile lat",
- wymaga obliczeń na podstawie dat (urodzenia, śmierci, długości życia).

W takich przypadkach wykonaj:
1. Zidentyfikuj osoby, których dotyczy pytanie.
2. Dla każdej osoby wygeneruj pod-pytania w formie zapytań do wyszukania:
   - "Data urodzenia (ur.) [OSOBA]"
   - "Data śmierci (zm.) [OSOBA]" 
3. Dodaj końcowe pod-pytanie:
   - "Na podstawie dat oblicz odpowiedź na pytanie główne."

Jeśli pytanie NIE wymaga porównań ani obliczeń (np. dotyczy tylko osiągnięć jednej osoby,
jej biografii, wpływu na epokę itp.) → ustaw "multi_hop": false
i pozostaw "subquestions" jako pustą listę.

Zwróć JSON:
{
 "multi_hop": true/false,
 "subquestions": ["...", "..."]
}

Pytanie: {query}
"""
