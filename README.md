# wikiBoss
AI assistantat operating on wikipedia.

```bash
# run qdrant from docker image
make start

# check options
python main_data.py --help

# example chunking + loading qdrant collection
python main_data.py \
--no-download \
--no-clean \
--chunking-strategy on_md_headers
```
