"""
Fine-tuning de RoBERTa para análisis de sentimientos (binario: negative/positive).

Requisitos (instalar antes de ejecutar, por ejemplo en Colab o local):
    pip install torch transformers datasets scikit-learn tqdm

Entrada esperada:
    Un CSV con columnas: 'review' (texto), 'sentiment' (valores 'positive' o 'negative')
    Por defecto: data.csv en el mismo directorio.

Ejecución (ejemplos):
    python fine_tune_roberta_sentiment.py --data_path data.csv --epochs 1
    python fine_tune_roberta_sentiment.py --data_path data.csv --save_dir ./clf_sentiment

El script:
  1) Carga y preprocesa datos (codifica etiquetas, train/test split)
  2) Tokeniza con RoBERTa (roberta-base)
  3) Crea Dataset y DataLoader
  4) Define el modelo entrenable (cabeza de clasificación simple)
  5) Entrena (CrossEntropy + Adam)
  6) Evalúa (accuracy, precision, recall, f1, classification_report)
  7) Guarda modelo y tokenizer
  8) Expone una función de inferencia (predict) y un ejemplo de uso
"""

import os
import argparse
import random
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup

# =====================
# Utils
# =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniques = sorted(list(set(labels)))
    # En este caso esperamos 'negative', 'positive'
    # pero lo hacemos robusto a orden/variantes
    label2id = {lbl: i for i, lbl in enumerate(uniques)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


# =====================
# Dataset
# =====================

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item


# =====================
# Modelo
# =====================

class RobertaForSentiment(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size  # 768 para roberta-base
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Tomamos el embedding del primer token (<s>)
        pooled = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden)
        x = self.pre_classifier(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}


# =====================
# Entrenamiento / Validación
# =====================

def train_one_epoch(model, dataloader, optimizer, scheduler, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    n_examples = 0
    n_correct = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"]
        logits = out["logits"]

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        n_correct += (preds == labels).sum().item()
        n_examples += labels.size(0)

    epoch_loss = running_loss / max(1, n_examples)
    epoch_acc = n_correct / max(1, n_examples)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    n_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
            logits = out["logits"]

            running_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            n_examples += labels.size(0)

    avg_loss = running_loss / max(1, n_examples)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(all_labels, all_preds, digits=4)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
        "y_true": all_labels,
        "y_pred": all_preds
    }


# =====================
# Guardado y Predicción
# =====================

def save_model_and_tokenizer(model, tokenizer, save_dir: str, id2label: Dict[int, str], label2id: Dict[str, int]):
    os.makedirs(save_dir, exist_ok=True)
    # Guardar pesos de la cabeza + backbone
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    # Guardar config mínima
    config = {
        "id2label": {int(k): v for k, v in id2label.items()},
        "label2id": label2id
    }
    with open(os.path.join(save_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    # Guardar tokenizer (vocab y merges)
    tokenizer.save_pretrained(save_dir)


def load_label_maps(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    id2label = {int(k): v for k, v in config["id2label"].items()}
    label2id = config["label2id"]
    return id2label, label2id


def predict(model, tokenizer, texts: List[str], device, max_len: int, id2label: Dict[int, str]) -> List[str]:
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]
            pred_id = int(torch.argmax(logits, dim=1).item())
            preds.append(id2label[pred_id])
    return preds


# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning RoBERTa para análisis de sentimientos")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Ruta al CSV con columnas 'review' y 'sentiment'")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Nombre del modelo base de Hugging Face")
    parser.add_argument("--max_len", type=int, default=256, help="Longitud máxima de tokens")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Proporción de warmup del total de steps")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="clf_sentiment", help="Directorio para guardar modelo y tokenizer")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # 1) Carga de datos
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(
            f"No se encontró '{args.data_path}'. Crea un CSV con columnas 'review','sentiment'."
        )

    df = pd.read_csv(args.data_path)
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError("El CSV debe tener columnas: 'review' y 'sentiment'")

    # Limpieza básica
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    texts = df["review"].astype(str).tolist()
    raw_labels = df["sentiment"].astype(str).str.lower().tolist()

    # 2) Codificación de etiquetas
    label2id, id2label = prepare_label_maps(raw_labels)
    y = [label2id[lbl] for lbl in raw_labels]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 4) Datasets y DataLoaders
    train_ds = SentimentDataset(X_train, y_train, tokenizer, args.max_len)
    test_ds = SentimentDataset(X_test, y_test, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(test_ds, batch_size=args.valid_batch_size, shuffle=False, num_workers=0)

    # 5) Modelo
    model = RobertaForSentiment(args.model_name, num_labels=len(label2id), dropout=0.2).to(device)

    # Optimizador y Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    # 6-7) Entrenamiento
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train - loss: {train_loss:.4f} | acc: {train_acc:.4f}")

        # 8) Evaluación intermedia
        eval_metrics = evaluate(model, valid_loader, device)
        print(f"Valid - loss: {eval_metrics['loss']:.4f} | acc: {eval_metrics['accuracy']:.4f} | f1_w: {eval_metrics['f1_weighted']:.4f}")
        print("\nClassification Report (Valid):\n", eval_metrics["classification_report"])

    # 9) Guardado
    save_model_and_tokenizer(model, tokenizer, args.save_dir, id2label=id2label, label2id=label2id)
    with open(os.path.join(args.save_dir, "metrics_valid.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": eval_metrics["accuracy"],
                "precision_weighted": eval_metrics["precision_weighted"],
                "recall_weighted": eval_metrics["recall_weighted"],
                "f1_weighted": eval_metrics["f1_weighted"],
            },
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"\nModelo y tokenizer guardados en: {args.save_dir}")

    # 10) Ejemplo de uso de inferencia
    example_texts = [
        "I absolutely loved this movie, the acting was brilliant and the story was touching.",
        "This was a terrible experience, I would not recommend it to anyone."
    ]
    preds = predict(model, tokenizer, example_texts, device, args.max_len, id2label)
    for t, p in zip(example_texts, preds):
        print(f"[PRED] {p} :: {t}")


if __name__ == "__main__":
    main()
