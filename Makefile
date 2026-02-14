start:
	sudo docker run --rm -d --name qdrant -p 6333:6333 -v $(PWD)/qdrant_storage:/qdrant/storage qdrant/qdrant

stop:
	sudo docker stop qdrant || true
	sudo docker rm qdrant || true
