
from typing import List, Dict
import re
from collections import Counter


# ==========================
#   Podział na zdania
# ==========================
_sentence_splitter_re = re.compile(r'(?<=[.!?。！？])\s+')

def sentence_tokenize(text: str) -> List[str]:
    """Dzieli tekst na zdania (prosty regex, usuwa puste)."""
    if not text:
        return []
    s = text.replace("\n", " ").replace("\r", " ")
    sentences = [p.strip() for p in _sentence_splitter_re.split(s) if p.strip()]
    return sentences


# ==========================
# Pomocnicze funkcje
# ==========================
def chunk_score(chunk: str, query: str) -> int:
    """Zlicza dopasowania słów zapytania w danym chunku (prosty scoring)."""
    q_tokens = [w.lower() for w in re.findall(r"\w+", query)]
    tokens = [w.lower() for w in re.findall(r"\w+", chunk)]
    cnt = Counter(tokens)
    return sum(cnt[q] for q in q_tokens)


def make_chunks_from_sentences(sentences: List[str], chunk_size: int = 4, overlap: int = 1) -> List[str]:
    """
    Łączy zdania w chunki o zadanym rozmiarze z overlapem.
    """
    chunks = []
    n = len(sentences)
    if n == 0:
        return chunks

    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        chunk = " ".join(sentences[i:end])
        chunks.append(chunk)
        if end == n:
            break
        i = i + chunk_size - overlap
    return chunks


# ==========================
# CHUNKOWANIE POD RAG
# ==========================
def make_base_chunks(articles: List[Dict], chunk_size: int = 4, overlap: int = 1) -> List[Dict]:
    """
    Tworzy chunki do indeksowania RAG.
    Każdy chunk zawiera metadane o źródle.
    """
    all_chunks = []
    for article_id, art in enumerate(articles):
        sentences = sentence_tokenize(art["text"])
        chunks = make_chunks_from_sentences(sentences, chunk_size, overlap)
        for idx, ch in enumerate(chunks):
            all_chunks.append({
                "article_id": article_id,
                "title": art["title"],
                "url": art.get("url", ""),
                "chunk_id": idx,
                "chunk_text": ch,
            })
    print(f"Generated {len(all_chunks)} chunks from {len(articles)} articles.")
    return all_chunks


# ==========================
# DOKŁADNE SKANOWANIE
# ==========================
def scan_article_in_detail(article_text: str, query: str, chunk_size: int = 3, overlap: int = 2) -> List[Dict]:
    """
    Dokładne chunkowanie jednego artykułu pod kątem zapytania.
    Zwraca tylko chunki, które zawierają istotne słowa z query.
    """
    sentences = sentence_tokenize(article_text)
    chunks = make_chunks_from_sentences(sentences, chunk_size, overlap)
    results = []

    for idx, chunk in enumerate(chunks):
        score = chunk_score(chunk, query)
        if score > 0:
            results.append({
                "chunk_id": idx,
                "chunk_text": chunk,
                "score": score
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ==========================
# MULTI-SOURCE MERGE
# ==========================
def merge_chunk_results(chunk_sets: List[List[Dict]]) -> List[Dict]:
    """
    Łączy wiele list wyników chunków (np. z różnych artykułów)
    i sortuje globalnie po score malejąco.
    """
    merged = []
    for chunk_list in chunk_sets:
        merged.extend(chunk_list)
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged


# ==========================
# DEMO (lokalne testy)
# ==========================
if __name__ == "__main__":
    sample_articles = [
        {
            "title": "Albert Einstein",
            "url": "https://pl.wikipedia.org/wiki/Albert_Einstein",
            "text": (
                "Albert Einstein był fizykiem teoretycznym. "
                "Urodził się w 1879 roku w Ulm w Niemczech. "
                "Studiował na Politechnice w Zurychu. "
                "W 1905 roku opublikował pięć artykułów naukowych, które zmieniły fizykę. "
                "Otrzymał Nagrodę Nobla w 1921 roku. "
                "Zmarł w 1955 roku w Princeton."
            )
        }
    ]

    print("=== Tworzenie chunków RAG ===")
    base_chunks = make_base_chunks(sample_articles, chunk_size=3, overlap=1)
    for c in base_chunks:
        print(c)

    print("\n=== Dokładne skanowanie ===")
    query = "studia Einsteina w Zurychu i artykuły 1905"
    detailed = scan_article_in_detail(sample_articles[0]["text"], query)
    for d in detailed:
        print(d)
