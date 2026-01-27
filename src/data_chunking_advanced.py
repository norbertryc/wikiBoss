from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Dict
import spacy


# ============================================
#   Sentence tokenizer 
# ============================================

# This model provides a better understanding of sentence structure in the Polish language.
nlp = spacy.load("pl_core_news_sm")

def sentence_tokenize(text: str) -> List[str]:
    """
    Splits text into sentences using the spaCy language model.
    This helps avoid incorrect splits (e.g., abbreviations, initials, numbers).
    """

    if not text:
        return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ============================================
#   Split by headers
# ============================================

def split_into_sections(text: str) -> List[str]:
    """
Splits the article text into sections based on headers (#, ##, ###).
If there is remaining text after the last header (without a header),
it is extracted as a separate section.
"""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    docs = md_splitter.split_text(text)
    sections = [d.page_content.strip() for d in docs if d.page_content.strip()]

    if sections:
        last_section_text = sections[-1]
        last_section_end = text.rfind(last_section_text)
        remaining_text = text[last_section_end + len(last_section_text):].strip()

        if remaining_text:
            # jeśli coś zostało po ostatnim nagłówku → osobna sekcja
            sections.append(remaining_text)

    return sections


# ============================================
#   RecursiveCharacterTextSplitter
# ============================================

def split_section_into_chunks(section_text: str,
                              chunk_size: int = 800,
                              chunk_overlap: int = 100) -> List[str]:
    """
    Splits a section of Wikipedia text into smaller, logical fragments.
    First divides the text into sentences, then applies RecursiveCharacterTextSplitter.
    """
    # Split by sentences first
    sentences = sentence_tokenize(section_text)
    joined_text = "\n".join(sentences)

    # Split into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,           
    chunk_overlap= chunk_overlap,        
    separators=["\n\n", "\n", ". ", "? ", "! "],  
    )
    chunks = text_splitter.split_text(joined_text)
    return chunks


# ============================================
#   Pipeline for whole article
# ============================================

def chunk_wikipedia_article(article: Dict,
                            chunk_size: int = 600,
                            chunk_overlap: int = 100) -> List[Dict]:
    all_chunks = []
    sections = split_into_sections(article["text"])
    global_chunk_id = 1  

    for sec_id, section in enumerate(sections):
        section_chunks = split_section_into_chunks(section, chunk_size, chunk_overlap)
        for ch in section_chunks:
            all_chunks.append({
                "article_id": article.get("id", 0),
                "title": article["title"],
                "url": article.get("url", ""),
                "section_id": sec_id,
                "chunk_id": global_chunk_id, 
                "chunk_text": ch
            })
            global_chunk_id += 1  
    return all_chunks


