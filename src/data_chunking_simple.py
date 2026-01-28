from langchain_text_splitters import CharacterTextSplitter


class Chunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str) -> list[dict]:
        """
        Splits text into chunks.

        Returns:
            list of dicts:
            {
                "chunk_index": int,
                "chunk_text": str
            }
        """
        paragraphs = text.split("\n\n")

        chunks: list[dict] = []
        chunk_index = 0

        for para in paragraphs:
            for ch in self.text_splitter.split_text(para):
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": ch,
                })
                chunk_index += 1

        return chunks
