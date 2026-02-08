# Loading DJL embedding models

The embedding script calls `POST /predictions/<model_name>` with a JSON array of strings. Models must be available in DJL before running the pipeline.

## Quick start

From the **lucene-test-data** repo root:

```bash
# One command: starts DJL, downloads data, generates embeddings
./run-embeddings.sh
```

Or step by step:

```bash
# 1. Start DJL with BGE-M3 (auto-loads from bge-m3-model/)
start-embedding-docker/start-djl.sh     # first time
start-embedding-docker/restart-djl.sh   # restart and wait for READY

# 2. (Optional) Also load MiniLM via management API
start-embedding-docker/load-models.sh

# 3. Generate embeddings
python scripts/generate_embeddings/generate_embeddings.py \
  -c scripts/generate_embeddings/config.example.yaml
```

## How BGE-M3 loads

BGE-M3 (1024d) auto-loads when the container starts via files in `bge-m3-model/`:

- **`serving.properties`** -- tells DJL to use the Python engine with `BAAI/bge-m3`
- **`model.py`** -- custom handler that loads the model via `transformers`, does mean pooling + L2 normalization, returns dense 1024d embeddings

The directory is mounted at `/opt/ml/model/bge-m3/` in `docker-compose.yml`. DJL auto-discovers it at startup (`Initial Models: ALL`) and names it **`bge_m3`** (underscores, from directory name).

The prediction endpoint is: `POST http://localhost:8091/predictions/bge_m3`

### torch.load safety bypass

The DJL 0.36.0 image ships torch < 2.6. The `transformers` library blocks `torch.load` for `pytorch_model.bin` due to CVE-2025-32434 (no safetensors available for bge-m3). The custom `model.py` monkey-patches `check_torch_load_is_safe` since we load from a trusted source (HuggingFace). When the DJL image upgrades torch to >= 2.6, this patch becomes a no-op.

## MiniLM (optional, 384d)

MiniLM is loaded via the management API (not auto-load):

```bash
start-embedding-docker/load-models.sh
```

This registers **all-MiniLM-L6-v2** at `/predictions/all-MiniLM-L6-v2`. To use it for embeddings, override the model in config or CLI:

```bash
python scripts/generate_embeddings/generate_embeddings.py \
  -c scripts/generate_embeddings/config.example.yaml \
  --model-name all-MiniLM-L6-v2 --dim 384
```

## Docker setup

- **Image:** `deepjavalibrary/djl-serving:0.36.0-pytorch-gpu` (or `-cpu`)
- **Port:** host 8091 -> container 8080
- **GPU detection:** `start-djl.sh` and `restart-djl.sh` auto-detect via `nvidia-smi`
- **Model store:** `/opt/ml/model` with `Initial Models: ALL`

## Check status

```bash
curl -s http://localhost:8091/models
```

When the model shows `"status": "READY"`, you can run the embedding script.

## Troubleshooting

- **Container exits immediately:** Check `docker compose logs` in `start-embedding-docker/`. Common: model loading failure (see below).
- **Model fails to load:** The first startup downloads ~2.3 GB from HuggingFace. Ensure network access and sufficient disk space.
- **OOM on GPU:** BGE-M3 uses ~2 GB VRAM (fp16). Reduce `batch_size` in config if inference OOMs on long documents.
- **Prediction returns 404:** Model name is `bge_m3` (underscore), not `bge-m3` (hyphen).
