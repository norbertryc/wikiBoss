import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class PolishBertCasedEmbedder:
    def __init__(self, model_name="dkleczek/bert-base-polish-cased-v1", device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_batch(self, texts, batch_size=100):
        
        texts = sorted(texts, key=len)
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding BERT (CASED)"):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state  # [B, seq, 768]

                # Mean pooling
                mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)

                embeddings = (summed / counts).cpu()

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).numpy()