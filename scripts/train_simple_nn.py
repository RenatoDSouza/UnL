#!/usr/bin/env python3
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    cache_path = Path("data/femnist_digits_0_1.npz")
    artifacts_dir = Path("/home/rdias/.gemini/antigravity-ide/brain/8368cd32-b5f9-4ad8-a15b-d57ff6984d89")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Error: {cache_path} not found.")
        return

    # 1. Load and prepare data
    print("Carregando os dados de cache...")
    data = np.load(cache_path)
    x_raw = data["x"].astype(np.float32) / 255.0  # Normalize to [0, 1]
    y_raw = data["y"].astype(np.float32)

    # Flatten the images from (n, 28, 28) to (n, 784)
    x_flat = x_raw.reshape(x_raw.shape[0], -1)

    print(f"Dataset total: {x_flat.shape[0]} amostras.")
    
    # 2. Train-test split (80% / 20%)
    seed = 42
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_flat))
    split_idx = int(0.8 * len(x_flat))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    x_train, y_train = x_flat[train_indices], y_raw[train_indices]
    x_test, y_test = x_flat[test_indices], y_raw[test_indices]
    
    # Convert to PyTorch Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo selecionado: {device}")
    
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Define a simple neural network (MLP)
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 64),
                nn.ReLU(),
                nn.Linear(64, 1) # Output logits for binary classification
            )
            
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = SimpleMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 4. Training loop
    epochs = 5
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    print("\nIniciando o treinamento da rede neural...")
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Evaluate on test set
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                running_test_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_test += (preds == batch_y).sum().item()
                total_test += batch_y.size(0)

        epoch_test_loss = running_test_loss / total_test
        epoch_test_acc = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)

        print(f"Época {epoch}/{epochs} | Loss Treino: {epoch_train_loss:.4f} | Acc Treino: {epoch_train_acc*100:.2f}% | Loss Teste: {epoch_test_loss:.4f} | Acc Teste: {epoch_test_acc*100:.2f}%")

    elapsed_time = time.perf_counter() - start_time
    print(f"\nTreinamento concluído em {elapsed_time:.2f} segundos.")

    # 5. Plotting results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(range(1, epochs + 1), train_losses, label="Treino", marker="o", color="#3498DB")
    ax1.plot(range(1, epochs + 1), test_losses, label="Teste", marker="s", color="#E74C3C")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (Entropia Cruzada)")
    ax1.set_title("Curva de Perda (Loss)")
    ax1.set_xticks(range(1, epochs + 1))
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(range(1, epochs + 1), [acc * 100 for acc in train_accs], label="Treino", marker="o", color="#3498DB")
    ax2.plot(range(1, epochs + 1), [acc * 100 for acc in test_accs], label="Teste", marker="s", color="#2ECC71")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia (%)")
    ax2.set_title("Curva de Acurácia")
    ax2.set_xticks(range(1, epochs + 1))
    ax2.set_ylim(min(test_accs)*100 - 5, 102)
    ax2.legend()

    plt.suptitle("Treinamento de Rede Neural Simples (MLP) - Dígitos 0 e 1 do FEMNIST", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    output_path = artifacts_dir / "nn_training_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training plot successfully to: {output_path}")

if __name__ == "__main__":
    main()
