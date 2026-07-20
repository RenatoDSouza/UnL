#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    cache_path = Path("data/femnist_digits_0_1.npz")
    artifacts_dir = Path("/home/rdias/.gemini/antigravity-ide/brain/8368cd32-b5f9-4ad8-a15b-d57ff6984d89")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Error: {cache_path} not found.")
        return

    # Load cache
    data = np.load(cache_path)
    x = data["x"] # Shape: (n, 28, 28)
    y = data["y"] # Shape: (n,)

    print(f"Loaded {len(x)} total samples of digits 0 and 1.")

    # Select 5 samples: 2 of class 0 and 3 of class 1 (or random)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(x), size=5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        img = x[idx]
        label = y[idx]
        ax = axes[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Amostra #{idx}\nRótulo: {label} (Dígito: {label})", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Exemplos de Imagens do Treinamento (Dígitos 0 e 1 do FEMNIST)", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    output_path = artifacts_dir / "digits_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample plot successfully to: {output_path}")

if __name__ == "__main__":
    main()
