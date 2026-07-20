#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from statistics import mean, pstdev
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def main():
    checkpoints_dir = Path("experiments/qfl_unlearning_qfi/outputs/checkpoints")
    plots_dir = Path("experiments/qfl_unlearning_qfi/outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoints_dir.exists():
        LOGGER.error(f"Checkpoints directory {checkpoints_dir} does not exist.")
        return

    # Load all completed checkpoints
    runs = []
    checkpoint_files = list(checkpoints_dir.glob("unlearning_state_*_seed*.json"))
    LOGGER.info(f"Found {len(checkpoint_files)} checkpoint JSON files.")

    for path in checkpoint_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "completed":
                summary = data.get("summary")
                if summary:
                    runs.append(summary)
                    LOGGER.info(f"Loaded completed run: encoding={summary.get('encoding')}, seed={summary.get('seed')}")
            else:
                LOGGER.info(f"Skipped running/incomplete checkpoint: {path.name}")
        except Exception as e:
            LOGGER.error(f"Error loading {path.name}: {e}")

    if not runs:
        LOGGER.error("No completed runs found to plot.")
        return

    # Group runs by encoding
    grouped = {}
    for run in runs:
        enc = str(run.get("encoding", "unknown"))
        grouped.setdefault(enc, []).append(run)

    # For each encoding, aggregate and plot
    for encoding, encoding_runs in grouped.items():
        LOGGER.info(f"Aggregating and plotting for encoding: {encoding} ({len(encoding_runs)} seeds completed)")
        
        # Build baseline rows
        rows = []
        for run in encoding_runs:
            seed = run["seed"]
            baselines = run.get("summary", {}).get("baselines", {})
            for mode, metrics in baselines.items():
                if not isinstance(metrics, dict):
                    continue
                row = {"seed": seed, "encoding": encoding, "mode": mode}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[key] = float(value)
                rows.append(row)

        # Aggregate metrics
        modes = ["no_unlearning", "shap_only", "qfi_only", "shap_qfi", "retrain_complete"]
        modes_data = {}
        
        # Group rows by mode
        by_mode = {}
        for r in rows:
            by_mode.setdefault(r["mode"], []).append(r)

        for mode in modes:
            if mode not in by_mode:
                continue
            metric_dicts = by_mode[mode]
            keys = sorted({k for d in metric_dicts for k in d if k not in ("seed", "encoding", "mode")})
            stats = {"num_seeds": len(metric_dicts)}
            for key in keys:
                values = [d[key] for d in metric_dicts if key in d]
                stats[f"{key}_mean"] = mean(values)
                stats[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
            modes_data[mode] = stats

        # Filter modes that have data
        active_modes = [m for m in modes if m in modes_data]
        if not active_modes:
            LOGGER.warning(f"No active modes found for encoding {encoding}.")
            continue

        forget_acc = []
        retain_acc = []
        mia_auc = []

        for mode in active_modes:
            stats = modes_data[mode]
            forget_acc.append(stats.get("forget_accuracy_after_mean", stats.get("forget_accuracy_after", stats.get("forget_accuracy", 0.0))))
            retain_acc.append(stats.get("retain_accuracy_after_mean", stats.get("retain_accuracy_after", stats.get("retain_accuracy", 0.0))))
            mia_auc.append(stats.get("mia_auc_after_mean", stats.get("mia_auc_after", stats.get("mia_auc", 0.5))))

        # Plot setup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
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

        x = np.arange(len(active_modes))
        width = 0.35

        # Plot 1: Accuracies (Forget vs Retain)
        rects1 = ax1.bar(x - width/2, forget_acc, width, label='Forget Set (Excluído)', color='#34495E')
        rects2 = ax1.bar(x + width/2, retain_acc, width, label='Retain Set (Mantidos)', color='#BDC3C7')
        
        ax1.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
        ax1.set_title('Acurácia por Método de Desaprendizado', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels([mode_labels.get(m, m) for m in active_modes], rotation=25, ha='right')
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

        # Plot 2: MIA AUC
        bar_colors = [colors.get(m, "#7F8C8D") for m in active_modes]
        rects3 = ax2.bar(x, mia_auc, width*1.5, color=bar_colors, edgecolor='black', alpha=0.85)
        ax2.axhline(0.5, color='gray', linestyle='--', label='Ideal (Apenas Acaso / 0.5)')
        
        ax2.set_ylabel('ROC-AUC do MIA', fontsize=12, fontweight='bold')
        ax2.set_title('Invasão de Privacidade (MIA AUC)\n[Menor é melhor, ideal = 0.5]', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels([mode_labels.get(m, m) for m in active_modes], rotation=25, ha='right')
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

        title_suffix = f"({len(encoding_runs)} seeds)"
        plt.suptitle(f"Análise Comparativa de Desaprendizado ({encoding.upper()} Encoding) - Parcial {title_suffix}", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = plots_dir / f"ablation_comparison_{encoding}_partial.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.info(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
