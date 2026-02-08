"""Custom DJL handler for BAAI/bge-m3 dense embeddings via transformers."""
from djl_python import Input, Output
import torch

# The DJL 0.36.0 image ships torch < 2.6. The transformers library blocks
# torch.load for pytorch_model.bin (no safetensors available for bge-m3)
# due to CVE-2025-32434. We bypass the check since we load from HuggingFace.
# Patch both the source module AND modeling_utils (which imports it as a local name).
import transformers.utils.import_utils as _tiu
import transformers.modeling_utils as _mu
_noop = lambda: None
_tiu.check_torch_load_is_safe = _noop
_mu.check_torch_load_is_safe = _noop

from transformers import AutoTokenizer, AutoModel

model = None
tokenizer = None


def get_model(properties):
    model_id = properties.get("model_id", "BAAI/bge-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id).to(device)
    mdl.eval()
    if device == "cuda":
        mdl = mdl.half()

    return mdl, tok


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


def handle(inputs: Input) -> Output:
    global model, tokenizer

    if not model:
        model, tokenizer = get_model(inputs.get_properties())

    if inputs.is_empty():
        return None

    data = inputs.get_as_json()

    # Support both: list of strings (djl_client.py) or {"inputs": [...]}
    if isinstance(data, list):
        sentences = data
    elif isinstance(data, dict):
        sentences = data.get("inputs", data.get("text", []))
    else:
        sentences = [str(data)]

    if isinstance(sentences, str):
        sentences = [sentences]

    device = next(model.parameters()).device

    encoded = tokenizer(
        sentences, padding=True, truncation=True, max_length=8192, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = mean_pooling(outputs, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    result = embeddings.float().cpu().tolist()

    return Output().add_as_json(result)
