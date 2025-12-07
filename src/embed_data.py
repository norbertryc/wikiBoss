from qdrant_client import QdrantClient, models as qd_models
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

import time

from .utils import track_progress_and_time, DataLoader
from .logging_config import logger
from config import HUGGING_FACE_MODEL


base_model = Transformer(HUGGING_FACE_MODEL)
pooling = Pooling(base_model.get_word_embedding_dimension())
model = SentenceTransformer(
    modules=[base_model, pooling],
    device="cuda",
    model_kwargs={"use_fast": True}
)
model.half()


class QdrantIndexer(DataLoader):
    """"""
    def __init__(self,
                 *args,
                 embedding_model: SentenceTransformer = model,
                 encoding_batch_size: int = 512,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collections = []
        self.encoding_batch_size = encoding_batch_size
        
        # detailed performance time
        self.time_encoding = 0
        self.time_conversion = 0
        self.time_point_creation = 0
        self.time_upload = 0
        self.time_data_prep = 0

    def create_collection(self, name: str):
        """Create Qdrant collection"""
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
        """Re-enable indexing after upload"""
        self.qdrant_client.update_collection(
            collection_name=collection_name,
            optimizers_config=qd_models.OptimizersConfigDiff(
                indexing_threshold=20000
            ),
        )

    def create_points_batch(self, texts, chunk_ids, metadatas):
        """"""
        t0 = time.perf_counter()
        embeddings = self.embedding_model.encode(texts,
                                                 batch_size=self.encoding_batch_size)

        self.time_encoding += time.perf_counter() - t0

        t0 = time.perf_counter()
        embeddings_list = [list(emb) for emb in embeddings]
        self.time_conversion += time.perf_counter() - t0

        t0 = time.perf_counter()
        points = [
            qd_models.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={"text": text, **metadata}
            )
            for chunk_id, embedding, text, metadata in zip(chunk_ids, embeddings_list, texts, metadatas)
        ]
        self.time_point_creation += time.perf_counter() - t0
        
        return points

    @track_progress_and_time("Embedding/indexing")
    def _upload_points_generator(self, collection_name: str, upload_batch: int):
        """"""
        self.create_collection(collection_name)
        logger.info(f"Created collection {collection_name}")

        self.time_encoding = 0
        self.time_conversion = 0
        self.time_point_creation = 0
        self.time_upload = 0
        self.time_data_prep = 0
        
        counter_points = 0
        texts_buffer = []
        ids_buffer = []
        metadata_buffer = []
        points_for_upload = []
        
        t_total_start = time.perf_counter()

        for article_idx, article in enumerate(self.articles, start=1):
            t_prep = time.perf_counter()
            
            article_metadata = {
                "wiki_id": article["id"],
                "title": article["title"],
                "categories": article["categories"],
                "url": article["url"],
            }
            
            self.time_data_prep += time.perf_counter() - t_prep

            for chunk_idx, chunk in enumerate(article["chunks"], start=1):
                t_prep = time.perf_counter()
                chunk_id = int(f"{article['id']}{chunk_idx}")
                self.time_data_prep += time.perf_counter() - t_prep
                
                try:
                    # collect for batch
                    if isinstance(chunk, dict):
                        texts_buffer.append(chunk["text"])
                        ids_buffer.append(chunk_id)
                        metadata_buffer.append({**article_metadata, **chunk["metadata"]})
                    elif isinstance(chunk, str):
                        texts_buffer.append(chunk)
                        ids_buffer.append(chunk_id)
                        metadata_buffer.append(article_metadata)
                    else:
                        continue

                    # process encoding batch
                    if len(texts_buffer) >= self.encoding_batch_size:
                        batch_points = self.create_points_batch(
                            texts_buffer,
                            ids_buffer,
                            metadata_buffer
                        )
                        points_for_upload.extend(batch_points)
                        counter_points += len(batch_points)
                        
                        texts_buffer.clear()
                        ids_buffer.clear()
                        metadata_buffer.clear()

                    # upload batch
                    if len(points_for_upload) >= upload_batch:
                        t_upload = time.perf_counter()
                        self.qdrant_client.upload_points(
                            collection_name=collection_name,
                            points=points_for_upload,
                        )
                        self.time_upload += time.perf_counter() - t_upload
                        points_for_upload.clear()

                except Exception as e:
                    logger.warning(
                        f"Error processing chunk {chunk_idx} of article "
                        f"'{article['title']}' (wiki_id={article['id']}): {e}"
                    )

            yield article

        # process and upload remaining data
        if texts_buffer:
            batch_points = self.create_points_batch(
                texts_buffer,
                ids_buffer,
                metadata_buffer
            )
            points_for_upload.extend(batch_points)
            counter_points += len(batch_points)

        if points_for_upload:
            t_upload = time.perf_counter()
            self.qdrant_client.upload_points(
                collection_name=collection_name,
                points=points_for_upload,
            )
            self.time_upload += time.perf_counter() - t_upload

        time_total = time.perf_counter() - t_total_start
        time_other = time_total - (
            self.time_encoding + 
            self.time_conversion + 
            self.time_point_creation + 
            self.time_upload + 
            self.time_data_prep
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"PERFORMANCE DETAILS - {counter_points} points")
        logger.info(f"{'='*60}")
        logger.info(f"Total time:          {time_total:8.2f}s  (100.0%)")
        logger.info(f"├─ Encoding (GPU):   {self.time_encoding:8.2f}s  ({self.time_encoding/time_total*100:5.1f}%)")
        logger.info(f"├─ Conversion:       {self.time_conversion:8.2f}s  ({self.time_conversion/time_total*100:5.1f}%)")
        logger.info(f"├─ Point creation:   {self.time_point_creation:8.2f}s  ({self.time_point_creation/time_total*100:5.1f}%)")
        logger.info(f"├─ Qdrant upload:    {self.time_upload:8.2f}s  ({self.time_upload/time_total*100:5.1f}%)")
        logger.info(f"├─ Data prep:        {self.time_data_prep:8.2f}s  ({self.time_data_prep/time_total*100:5.1f}%)")
        logger.info(f"└─ Other (I/O etc):  {time_other:8.2f}s  ({time_other/time_total*100:5.1f}%)")
        logger.info(f"{'='*60}")
        logger.info(f"Throughput: {counter_points/time_total:.1f} points/sec")
        logger.info(f"{'='*60}\n")
        
        self.update_collection(collection_name)
        logger.info(f"Indexed collection {collection_name} with {counter_points} points")

    def upload_points(self, collection_name: str, upload_batch: int = 20000) -> None:
        """Upload all points"""
        for _ in self._upload_points_generator(collection_name, upload_batch):
            pass

    def delete_collection(self, collection_name: str):
        """Delete collection by name"""
        self.qdrant_client.delete_collection(collection_name)
        logger.info(f"Deleted collection {collection_name}")
