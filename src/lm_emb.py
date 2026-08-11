from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from openai import OpenAI

pretrained_repo = "/mnt/data/wangshu/all-roberta-large-v1"

pretrained_repo = {
    "nomic-embed-text-v1": "/mnt/data/grouph_share/transformer_llm/nomic-embed-text-v1"
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo, trust_remote_code=True):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(
            pretrained_repo,
            trust_remote_code=trust_remote_code,
            cache_dir=pretrained_repo,
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def load_sbert(model_name):

    print("loading model...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo[model_name])
    model = Sentence_Transformer(pretrained_repo[model_name], trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.bert_model)

    model.eval()
    print(f"Using device: {device}")
    print("finished loading model")
    return model, tokenizer, device


def sber_text2embedding(model, tokenizer, device, text, batch_size=16):
    try:
        encoding = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset(
            input_ids=encoding.input_ids, attention_mask=encoding.attention_mask
        )

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Placeholder for storing the embeddings
        all_embeddings = []

        # Iterate through batches
        with torch.no_grad():

            for batch in dataloader:
                # Move batch to the appropriate device
                batch = {key: value.to(device) for key, value in batch.items()}

                # Forward pass
                embeddings = model(
                    input_ids=batch["input_ids"], att_mask=batch["att_mask"]
                )

                # Append the embeddings to the list
                all_embeddings.append(embeddings)

        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    except:
        # 从模型 config 推导维度，避免硬编码 1024（nomic 是 768）
        try:
            if hasattr(model, "bert_model"):
                emb_dim = model.bert_model.config.hidden_size
            else:
                emb_dim = model.config.hidden_size
        except Exception:
            emb_dim = 768
        return torch.zeros((0, emb_dim))

    return all_embeddings


def text_to_embedding_batch(
    model, tokenizer, device, texts, batch_size=32, embedding_dim=1024
):
    """
    Encode a list of texts into embeddings using a specified model and tokenizer.

    Args:
        model: The model used for embedding.
        tokenizer: The tokenizer corresponding to the model.
        texts: List of texts to be encoded.
        batch_size: Number of samples per batch (default: 32).
        embedding_dim: Dimension of the output embeddings (default: 1024).

    Returns:
        Tensor of shape (n_samples, embedding_dim) containing the embeddings.
    """
    if isinstance(model, nn.DataParallel):
        # 如果是 DataParallel，访问内部的原始模型
        if hasattr(model.module, "bert_model"):
            embedding_dim = model.module.bert_model.config.hidden_size
        else:
            embedding_dim = 768  # 默认值
    else:
        # 如果不是 DataParallel，直接检查模型
        if hasattr(model, "bert_model"):
            embedding_dim = model.bert_model.config.hidden_size
        else:
            embedding_dim = 768  # 默认值

    try:
        # Tokenize the texts
        encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset(
            input_ids=encoding.input_ids, attention_mask=encoding.attention_mask
        )

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Placeholder for storing the embeddings
        all_embeddings = []

        # Iterate through batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding batches"):
                # Move batch to the appropriate device
                batch = {key: value.to(device) for key, value in batch.items()}

                # Forward pass
                embeddings = model(
                    input_ids=batch["input_ids"], att_mask=batch["att_mask"]
                )

                # Append the embeddings to the list
                all_embeddings.append(embeddings)

        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    except Exception as e:
        print(f"Error during embedding: {e}")
        return []

    # 逐条列表（每条一个 768 维 list），与云端 openai_embedding 返回格式一致，
    # 避免 2D tensor 直接赋 DataFrame 列触发 pandas "Expected a 1D array" 错误。
    return all_embeddings.tolist()


def load_contriever():
    print("Loading contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def contriever_text2embedding(model, tokenizer, device, text, batch_size=16):

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    try:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["att_mask"]
                )
                embeddings = mean_pooling(outputs[0], batch["att_mask"])
                all_embeddings.append(embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0).cpu()
    except:
        all_embeddings = torch.zeros((0, 1024))

    return all_embeddings


def openai_embedding(input_text, api_key, api_base, engine="text-embedding-ada-002"):
    """
    Generate an embedding for the input text using OpenAI's API.

    Args:
        input_text (str): The text for which to generate the embedding.
        api_key (str): Your OpenAI API key.
        api_base (str): The base URL of the OpenAI API.
        engine (str): The embedding model engine to use (default: 'text-embedding-ada-002').

    Returns:
        list: The embedding vector for the input text.
    """
    try:
        # Set up OpenAI API credentials and base URL
        client = OpenAI(api_key=api_key, base_url=api_base)

        # Request the embedding from the OpenAI API
        kwargs = {"input": input_text, "model": engine}
        if "embedding-3" in engine:
            kwargs["dimensions"] = 1024
        response = client.embeddings.create(**kwargs)

        # Extract the embedding from the response
        embedding = response.data[0].embedding
        return embedding

    except Exception as e:
        raise RuntimeError(
            f"Failed to generate embedding with model={engine}, api_base={api_base}: {e}"
        ) from e


# 本地 embedding 模型缓存（进程级：mp.Pool 每个 worker 各持有一份）
_local_embedder_cache = {}


def get_local_embedder(model_name="nomic-embed-text-v1"):
    """加载（并缓存）本地 sbert 风格 embedding 模型，避免每个查询重复加载。"""
    if model_name not in _local_embedder_cache:
        _local_embedder_cache[model_name] = load_sbert(model_name)
    return _local_embedder_cache[model_name]


def embed_query(text, args):
    """
    将单个查询文本编码为 float32 向量（统一入口，自动分流本地/云端）。

    - args.embedding_local 为 True 时：用本地 nomic（768 维），模型进程内只加载一次
    - 否则：回退到云端 OpenAI 兼容 embedding API，行为与之前一致
    """
    if getattr(args, "embedding_local", False):
        model, tokenizer, device = get_local_embedder(args.embedding_model_local)
        emb = text_to_embedding_batch(model, tokenizer, device, [text])
        # text_to_embedding_batch 现在返回逐条 list，emb[0] 是 list
        return np.asarray(emb[0], dtype=np.float32)
    return np.asarray(
        openai_embedding(
            text, args.embedding_api_key, args.embedding_api_base, args.embedding_model
        ),
        dtype=np.float32,
    )


load_model = {
    "sbert": load_sbert,
    "contriever": load_contriever,
}


load_text2embedding = {
    "sbert": sber_text2embedding,
    "contriever": contriever_text2embedding,
}


if __name__ == "__main__":
    test_text = "What is the capital of France?"
    model, tokenizer, device = load_sbert("nomic-embed-text-v1")
    test_text = [test_text]
    test_embedding = text_to_embedding_batch(model, tokenizer, device, test_text)
    print("the shape of the embedding is: ", len(test_embedding))
    print("the first 10 elements of the embedding are: ", test_embedding[:10])
