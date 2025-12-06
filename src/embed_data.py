from qdrant_client import QdrantClient, models as qd_models
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from .utils import track_progress_and_time, DataLoader
from .logging_config import logger
from config import HUGGING_FACE_MODEL


base_model = Transformer(HUGGING_FACE_MODEL)
pooling = Pooling(base_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[base_model, pooling],
                            device="cuda",
                            model_kwargs={"use_fast": True})


class QdrantIndexer(DataLoader):
    """"""
    def __init__(self,
                 *args,
                 embedding_model: SentenceTransformer = model,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collections = []

    def create_collection(self, name: str):
        """"""
        if not self.qdrant_client.collection_exists(name):

            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=qd_models.VectorParams(
                    size=self.embedding_model.get_sentence_embedding_dimension(),
                    distance=qd_models.Distance.COSINE
                ),
                optimizers_config=qd_models.OptimizersConfigDiff(
                    indexing_threshold=0
                )
            )
        else:
            logger.error(f"Collection {name} already exists.")

        return name

    def update_collection(self, collection_name: str):
        """"""
        self.qdrant_client.update_collection(
            collection_name=collection_name,
            optimizers_config=qd_models.OptimizersConfigDiff(
                indexing_threshold=20000
            ),
        )

    def create_point(self, chunk: str, chunk_id: int, metadata: dict = None):
        """"""
        point = qd_models.PointStruct(
            id=chunk_id,
            vector=self.embedding_model.encode(chunk).tolist(),
            payload={
                "text": chunk,
                **metadata
            }
        )
        return point

    @track_progress_and_time("Embedding/indexing")
    def _upload_points_gen(self, collection_name: str, upload_batch: int):
        """"""
        self.create_collection(collection_name)
        logger.info(f"Created collection {collection_name}")

        counter_points = 0
        points = []

        for article_idx, article in enumerate(self.articles, start=1):
            article_metadata = {
                "wiki_id": article["id"],
                "title": article["title"],
                "categories": article["categories"],
                "url": article["url"],
            }

            for chunk_idx, chunk in enumerate(article["chunks"], start=1):
                chunk_id = int(f"{article["id"]}{chunk_idx}")
                try:
                    if isinstance(chunk, dict):
                        point = self.create_point(
                            chunk["text"],
                            chunk_id=chunk_id,
                            metadata={**article_metadata, **chunk["metadata"]},
                        )
                    elif isinstance(chunk, str):
                        point = self.create_point(chunk, chunk_id=chunk_id, metadata=article_metadata)
                    else:
                        continue

                    points.append(point)
                    counter_points += 1

                    if len(points) == upload_batch:
                        self.qdrant_client.upload_points(
                            collection_name=collection_name,
                            points=points,
                        )
                        points.clear()

                except Exception as e:
                    logger.warning(
                        f"Error processing chunk {chunk_idx} of article "
                        f"'{article['title']}' (wiki_id={article['id']}): {e}"
                    )

            yield article

        if points:
            self.qdrant_client.upload_points(
                collection_name=collection_name,
                points=points,
            )

        self.update_collection(collection_name)
        logger.info(f"Indexed collection {collection_name} with {counter_points} points")

    def upload_points(self, collection_name: str, upload_batch: int = 20000) -> None:
        """"""
        for _ in self._upload_points_gen(collection_name, upload_batch):
            pass

    def delete_collection(self, collection_name: str):
        """"""
        self.qdrant_client.delete_collection(collection_name)
        logger.info(f"Deleted collection {collection_name}")
