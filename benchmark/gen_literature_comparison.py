import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "paper" / "assets"
RESULTS_JSON = ASSETS / "final_study_results.json"
ASSETS.mkdir(parents=True, exist_ok=True)

P = {
    "bg": "#ffffff",
    "panel": "#f7f4ea",
    "text": "#1d1a16",
    "border": "#d2c6a8",
    "green": "#3f7d20",
    "blue": "#2f6e84",
    "amber": "#bf7a1f",
    "red": "#b64926",
}


def load_crop_ai_point() -> tuple[str, float]:
    payload = json.loads(RESULTS_JSON.read_text())
    best_name = payload["research_best_model"]
    rows = [row for row in payload["results"] if row["model_name"] == best_name and row["dataset_variant"] == "raw"]
    best_acc = rows[0]["test_accuracy"] * 100.0
    label = f"Crop AI\n{best_name.replace(' ', chr(10))}"
    return label, best_acc


def main() -> None:
    crop_ai_label, crop_ai_acc = load_crop_ai_point()
    labels = [
        crop_ai_label,
        "Shastri et al.\n2025",
        "Alam et al.\n2025",
        "Stracqualursi\n2025",
        "Afzal et al.\n2025",
    ]
    values = [crop_ai_acc, 99.27, 99.54, 99.80, 98.00]
    colors = [P["blue"], P["green"], P["green"], P["amber"], P["red"]]

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    bars = ax.bar(labels, values, color=colors, edgecolor=P["border"], linewidth=1.0)
    ax.set_ylim(98.0, 100.2)
    ax.set_ylabel("Reported Accuracy (%)", color=P["text"])
    ax.set_title("Crop AI vs Recent Published Crop-Recommendation Results", color=P["text"], fontweight="bold")
    ax.tick_params(axis="x", labelcolor=P["text"])
    ax.tick_params(axis="y", colors=P["text"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", color=P["text"], fontsize=10, fontweight="bold")
    ax.grid(axis="y", color=P["border"], linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(P["border"])

    fig.tight_layout()
    fig.savefig(ASSETS / "literature_comparison.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


if __name__ == "__main__":
    main()
