"""
train.py — Training and evaluation loops
"""
import time
import torch
import torch.nn as nn

from model import MiniTransformer
from utils import compute_accuracy, count_params
from data import VOCAB_SIZE, MAX_LEN


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for tokens, mask, labels in loader:
        tokens, mask, labels = tokens.to(device), mask.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits.detach().cpu(), labels.cpu())

    n = len(loader)
    return total_loss / n, total_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for tokens, mask, labels in loader:
            tokens, mask, labels = tokens.to(device), mask.to(device), labels.to(device)
            logits = model(tokens, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_acc += compute_accuracy(logits.cpu(), labels.cpu())

    n = len(loader)
    return total_loss / n, total_acc / n


def train_model(config, train_loader, val_loader, device, epochs=15):
    """Build and train one model variant."""
    model = MiniTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_layers=config['num_layers'],
        max_len=MAX_LEN,
        dropout=config['dropout'],
        use_positional_encoding=config['use_pe']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*55}")
    print(f"Training: {config['name']}")
    print(f"  Heads={config['num_heads']}  Layers={config['num_layers']}  PE={config['use_pe']}")
    print(f"  Parameters: {count_params(model):,}")
    print(f"{'='*55}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

    elapsed = time.time() - start
    print(f"\n✅ Training time: {elapsed/60:.2f} min")
    return model, history, elapsed