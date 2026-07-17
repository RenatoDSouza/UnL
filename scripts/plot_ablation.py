#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot comparative results of unlearning baselines.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to unlearning_summary.json. Auto-detects if not provided."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output plot PNG. Defaults to outputs/plots/ablation_comparison.png."
    )
    args = parser.parse_args()

    # Auto-detect input path
    input_path = None
    if args.input:
        input_path = Path(args.input)
    else:
        # Try default paths
        candidates = [
            Path("experiments/qfl_unlearning_qfi/outputs/unlearning_summary.json"),
            Path("validacao/unlearning_outputs/angle/unlearning_summary.json"),
            Path("validacao/unlearning_outputs/reupload/unlearning_summary.json"),
            Path("validacao/unlearning_outputs/iqp/unlearning_summary.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                input_path = candidate
                print(f"Auto-detected summary file at: {input_path}")
                break

    if not input_path or not input_path.exists():
        print("Error: Could not find any unlearning_summary.json file.")
        print("Please run the unlearning experiment or smoke validation first, or specify --input.")
        return

    # Auto-detect output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "plots" / "ablation_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We look for baselines_aggregated first
    aggregated = data.get("baselines_aggregated", {})
    if not aggregated:
        print("Error: 'baselines_aggregated' not found in summary file.")
        return

    # Choose the first encoding mode available
    encoding = list(aggregated.keys())[0]
    print(f"Plotting results for encoding mode: {encoding}")
    modes_data = aggregated[encoding]

    modes = ["no_unlearning", "shap_only", "qfi_only", "shap_qfi", "retrain_complete"]
    # Filter modes that are actually in the data
    modes = [m for m in modes if m in modes_data]

    if not modes:
        print("Error: None of the target modes found in the aggregated baselines.")
        return

    # Extract metrics
    forget_acc = []
    retain_acc = []
    mia_auc = []

    for mode in modes:
        metrics = modes_data[mode]
        # Check for _mean keys first, otherwise fallback to direct values
        forget_acc.append(metrics.get("forget_accuracy_after_mean", metrics.get("forget_accuracy_after", metrics.get("forget_accuracy", 0.0))))
        retain_acc.append(metrics.get("retain_accuracy_after_mean", metrics.get("retain_accuracy_after", metrics.get("retain_accuracy", 0.0))))
        mia_auc.append(metrics.get("mia_auc_after_mean", metrics.get("mia_auc_after", metrics.get("mia_auc", 0.5))))

    # Setup the plot layout (2 subplots: Accuracies and MIA AUC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {
        "no_unlearning": "#E74C3C",       # Red
        "shap_only": "#F39C12",           # Orange
        "qfi_only": "#3498DB",            # Blue
        "shap_qfi": "#9B59B6",            # Purple
        "retrain_complete": "#2ECC71"     # Green
    }
    
    mode_labels = {
        "no_unlearning": "Original (No Unlearning)",
        "shap_only": "SHAP Only",
        "qfi_only": "QFI Only (Dense)",
        "shap_qfi": "SHAP + QFI (Híbrido)",
        "retrain_complete": "Retrained Reference"
    }

    x = np.arange(len(modes))
    width = 0.35

    # Plot 1: Accuracies (Forget vs Retain)
    rects1 = ax1.bar(x - width/2, forget_acc, width, label='Forget Set (Excluído)', color='#34495E')
    rects2 = ax1.bar(x + width/2, retain_acc, width, label='Retain Set (Mantidos)', color='#BDC3C7')
    
    ax1.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
    ax1.set_title('Acurácia por Método de Desaprendizado', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([mode_labels.get(m, m) for m in modes], rotation=25, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left')
    
    # Add values on top of bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # Plot 2: MIA AUC (Membership Inference Leakage)
    bar_colors = [colors.get(m, "#7F8C8D") for m in modes]
    rects3 = ax2.bar(x, mia_auc, width*1.5, color=bar_colors, edgecolor='black', alpha=0.85)
    ax2.axhline(0.5, color='gray', linestyle='--', label='Ideal (Apenas Acaso / 0.5)')
    
    ax2.set_ylabel('ROC-AUC do MIA', fontsize=12, fontweight='bold')
    ax2.set_title('Invasão de Privacidade (MIA AUC)\n[Menor é melhor, ideal = 0.5]', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([mode_labels.get(m, m) for m in modes], rotation=25, ha='right')
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    # Add values on top of bars
    for rect in rects3:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(f"Análise Comparativa de Desaprendizado ({encoding.upper()} Encoding)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
