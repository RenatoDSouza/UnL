import os
import numpy as np
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

def label_to_char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))
    return '?'

def main():
    print("Carregando primeira partição do FEMNIST via flwr-datasets...")
    
    # Carregar partição 0 usando o writer_id
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
    )
    partition = fds.load_partition(partition_id=0, split="train")

    # Extrair imagens e rótulos
    x = np.array([np.asarray(row["image"]) for row in partition], dtype=np.uint8)
    y = np.array([row["character"] for row in partition], dtype=np.int64)

    print("=== Informações do Dataset ===")
    print(f"Quantidade total de imagens nesta partição: {len(x)}")
    print(f"Formato de X (imagens): {x.shape} (esperado: n_amostras, 28, 28)")
    print(f"Formato de Y (rótulos): {y.shape} (esperado: n_amostras,)")
    print(f"Tipo de dados de X: {x.dtype}")
    print(f"Faixa de valores em X: Min={x.min()}, Max={x.max()}")
    
    # Contar distribuição de labels
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Total de classes únicas presentes nesta partição: {len(unique_labels)} (máximo original é 62)")
    print("Primeiros 10 rótulos na partição (índice: rótulo -> caractere):")
    for idx, (lbl, count) in enumerate(zip(unique_labels[:10], counts[:10])):
        print(f"  Classe {lbl:2d} ({label_to_char(lbl)}): {count} amostras")

    # Plotar uma grade de imagens aleatórias
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Amostras do FEMNIST (Partição 0)", fontsize=16)
    
    # Seed fixa para repetibilidade da visualização
    rng = np.random.default_rng(42)
    indices = rng.choice(len(x), size=16, replace=False)
    
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        img = x[idx]
        lbl = y[idx]
        char = label_to_char(lbl)
        
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Idx: {idx}\nClass: {lbl} ('{char}')", fontsize=9)
        ax.axis("off")
        
    plt.tight_layout()
    output_png = "data/femnist_sample_plot.png"
    plt.savefig(output_png)
    print(f"\nVisualização salva com sucesso em: {output_png}")

if __name__ == "__main__":
    main()
