import time
from qdrant_client import QdrantClient, models as qd_models

from .utils import track_progress_and_time, DataLoader
from .logging_config import logger
from .embedding_engine import EmbeddingEngine
from config import QDRANT_CONFIG, HNSW_CONFIG


class QdrantManager(DataLoader):
    """
    Manages Qdrant vector database operations for Wikipedia article chunks.

    Args:
        embedding_engine (EmbeddingEngine): Engine for encoding text to embeddings. Set to None
            for browse/delete operations only.
        encoding_batch_size (int): Batch size for encoding operations (default: 512)
        collection_name (str): Provide for retrieval operations.
        *args, **kwargs: Passed to DataLoader parent class

    Attributes:
        engine (EmbeddingEngine): Embedding engine instance
        qdrant_client (QdrantClient): Client for Qdrant operations
        encoding_batch_size (int): Batch size for encoding
        time_encoding (float): Time spent on GPU encoding
        time_conversion (float): Time spent converting embeddings to lists
        time_point_creation (float): Time spent creating point structures
        time_upload (float): Time spent uploading to Qdrant
        time_data_prep (float): Time spent on data preparation
    """

    def __init__(self,
                 *args,
                 input_path: str = None,
                 embedding_engine: EmbeddingEngine = None,
                 encoding_batch_size: int = 512,
                 collection_name: str = None,
                 **kwargs
                 ):

        if input_path is None:
            logger.warning("'input_path' not specified - limited to operations on existing vector stores")
        if embedding_engine is None:
            logger.warning("'embedding_engine' not provided - limited to collection browse/delete operations")

        super().__init__(*args, input_path=input_path, **kwargs)
        self.engine = embedding_engine
        self.qdrant_client = QdrantClient(**QDRANT_CONFIG)
        self.encoding_batch_size = encoding_batch_size
        self.collection_name = collection_name

        # detailed performance time
        self.time_encoding = 0
        self.time_conversion = 0
        self.time_point_creation = 0
        self.time_upload = 0
        self.time_data_prep = 0

    def create_collection(self,
                          name: str,
                          indexing_threshold: int = 0,
                          hnsw_config_kwargs: dict = HNSW_CONFIG
                          ) -> str:
        """
        Creates Qdrant collection with specified configuration.

        Args:
            name (str): Collection name
            indexing_threshold (int): Number of vectors before indexing starts (default: 0)
            hnsw_config_kwargs (dict): HNSW index configuration parameters

        Returns:
            str: Created collection name
        """
        if not self.qdrant_client.collection_exists(name):
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=qd_models.VectorParams(
                    size=self.engine.model.get_sentence_embedding_dimension(),
                    distance=qd_models.Distance.COSINE
                ),
                optimizers_config=qd_models.OptimizersConfigDiff(
                    indexing_threshold=indexing_threshold
                ),
                hnsw_config=qd_models.HnswConfigDiff(
                    **hnsw_config_kwargs
            ))
            logger.info(f"Created collection {name}")
        else:
            logger.error(f"Collection {name} already exists.")

        return name

    def update_collection(self, collection_name: str):
        """
        Re-enables indexing after bulk upload with standard threshold.

        Args:
            collection_name (str): Name of collection to update
        """
        self.qdrant_client.update_collection(
            collection_name=collection_name,
            optimizers_config=qd_models.OptimizersConfigDiff(
                indexing_threshold=20000
            ),
        )

    def _create_points_batch(self, texts, chunk_ids, metadatas):
        """
        Creates batch of Qdrant points from texts with embeddings.

        Args:
            texts (list[str]): Text chunks to encode
            chunk_ids (list[int]): Unique IDs for chunks
            metadatas (list[dict]): Metadata dictionaries for each chunk

        Returns:
            list[PointStruct]: Qdrant point structures ready for upload
        """
        t0 = time.perf_counter()
        embeddings = self.engine.model.encode(texts, batch_size=self.encoding_batch_size)
        self.time_encoding += time.perf_counter() - t0

        t0 = time.perf_counter()
        embeddings_list = [list(emb) for emb in embeddings]
        self.time_conversion += time.perf_counter() - t0
        time.sleep(0.1) # GPU temperature control

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

    @track_progress_and_time("Embedding/loading to vectorstore")
    def _upload_points_generator(self, collection_name: str, upload_batch: int):
        """
        Generator that encodes chunks and uploads them in batches to Qdrant.
        Tracks detailed performance metrics.

        Args:
            collection_name (str): Target collection name
            upload_batch (int): Number of points to accumulate before upload

        Yields:
            dict: Processed article or None if chunks were None
        """
        self.create_collection(collection_name)

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

        logger.info("Start encoding chunks...")
        t_total_start = time.perf_counter()

        for article_idx, article in enumerate(self.articles, start=1):
            if article["chunks"] is not None:
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
                            batch_points = self._create_points_batch(
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
            else:
                yield None

        # process and upload remaining data
        if texts_buffer:
            batch_points = self._create_points_batch(
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

        logger.info(f"PERFORMANCE DETAILS - {counter_points} points")
        logger.info(f"Total time:          {time_total:8.2f}s  (100.0%)")
        logger.info(f"├─ Encoding (GPU):   {self.time_encoding:8.2f}s  ({self.time_encoding / time_total * 100:5.1f}%)")
        logger.info(
            f"├─ Conversion:       {self.time_conversion:8.2f}s  ({self.time_conversion / time_total * 100:5.1f}%)")
        logger.info(
            f"├─ Point creation:   {self.time_point_creation:8.2f}s  ({self.time_point_creation / time_total * 100:5.1f}%)")
        logger.info(f"├─ Qdrant upload:    {self.time_upload:8.2f}s  ({self.time_upload / time_total * 100:5.1f}%)")
        logger.info(
            f"├─ Data prep:        {self.time_data_prep:8.2f}s  ({self.time_data_prep / time_total * 100:5.1f}%)")
        logger.info(f"└─ Other (I/O etc):  {time_other:8.2f}s  ({time_other / time_total * 100:5.1f}%)")
        logger.info(f"Throughput: {counter_points / time_total:.1f} points/sec")

        self.update_collection(collection_name)
        logger.info(f"Indexed collection {collection_name} with new {counter_points} points")

    def upload_points(self, collection_name: str, upload_batch: int = 10000) -> None:
        """
        Uploads all article chunks to specified Qdrant collection.

        Args:
            collection_name (str): Target collection name
            upload_batch (int): Number of points per upload batch (default: 10000)
        """
        for _ in self._upload_points_generator(collection_name, upload_batch):
            pass

    def delete_collections(self, collection_name: str|list):
        """
        Deletes specified Qdrant collection(s).

        Args:
            collection_name (str|list): Collection name or list of names to delete
        """
        if isinstance(collection_name, str):
            collection_name = [collection_name]
        for collection in collection_name:
            self.qdrant_client.delete_collection(collection)
            logger.info(f"Deleted collection {collection}")

    def retrieve(self, query: str, limit: int) -> list[dict]:
        """"""
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=self.engine.encode(self.engine.query_prefix + query).tolist(),
            limit=limit
        ).points

        logger.info(f"Retrieved {len(hits)} documents for query: '{query}...'")

        return [
            {
                "payload": hit.payload,
                "id": hit.id,
                "score": hit.score
            }
            for hit in hits
        ]

    def list_collections(self) -> list:
        """"""
