from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

    def chunk_text(self, text: str) -> list[dict]:
        chunks = []
        for idx, ch in enumerate(self.text_splitter.split_text(text)):
            chunks.append({
                "chunk_index": idx,
                "chunk_text": ch,
            })
        return chunks
