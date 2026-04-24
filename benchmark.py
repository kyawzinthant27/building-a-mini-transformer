"""
benchmark.py — Run all 4 model variants and produce the benchmark table
"""
import torch
import torch.nn as nn
import pandas as pd

from data import get_dataloaders
from train import train_model, evaluate
from utils import set_seed, count_params, plot_history, plot_all_models


SEED = 42
DATA_DIR = "."   # adjust if CSVs are elsewhere
EPOCHS = 15


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=32)

    # ── Model configs ────────────────────────────────────────
    configs = [
        {'name': 'Model A', 'embed_dim': 64, 'num_heads': 1, 'ff_dim': 128,
         'num_layers': 1, 'dropout': 0.1, 'lr': 0.001, 'use_pe': True},
        {'name': 'Model B', 'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
         'num_layers': 1, 'dropout': 0.1, 'lr': 0.001, 'use_pe': True},
        {'name': 'Model C', 'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
         'num_layers': 1, 'dropout': 0.1, 'lr': 0.001, 'use_pe': False},
        {'name': 'Model D', 'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
         'num_layers': 2, 'dropout': 0.1, 'lr': 0.001, 'use_pe': True},
    ]

    # ── Train all ────────────────────────────────────────────
    models, histories, times = [], [], []
    for cfg in configs:
        set_seed(SEED)
        model, history, t = train_model(cfg, train_loader, val_loader, device, epochs=EPOCHS)
        models.append(model)
        histories.append(history)
        times.append(t)
        plot_history(history, cfg['name'], save_path=f"{cfg['name'].replace(' ', '_')}_curves.png")

    # ── Combined plot ────────────────────────────────────────
    names = [f"{c['name']} (PE={'Yes' if c['use_pe'] else 'No'}, "
             f"H={c['num_heads']}, L={c['num_layers']})"
             for c in configs]
    plot_all_models(histories, names, save_path="all_models_comparison.png")

    # ── Build benchmark table ────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    results = []
    for cfg, mdl, hist, t in zip(configs, models, histories, times):
        _, test_acc = evaluate(mdl, test_loader, criterion, device)
        results.append({
            'Model': cfg['name'],
            'PE': 'Yes' if cfg['use_pe'] else 'No',
            'Heads': cfg['num_heads'],
            'Layers': cfg['num_layers'],
            'Val Acc': f"{max(hist['val_acc']):.4f}",
            'Test Acc': f"{test_acc:.4f}",
            'Train Time': f"{t/60:.2f} min",
            'Params': f"{count_params(mdl):,}",
        })

    df = pd.DataFrame(results)
    print("\n" + "="*75)
    print("BENCHMARK TABLE")
    print("="*75)
    print(df.to_string(index=False))
    df.to_csv("benchmark_results.csv", index=False)
    print("\n✅ Saved benchmark_results.csv")


if __name__ == "__main__":
    main()