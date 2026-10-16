from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def pareto_curve(filename: str = "time_vs_map.png", dpi: int = 200):
    """
    Two stacked plots in one figure: specialization (top) and
    generalization (bottom). One colour per method; the x-axis (training time)
    is linear and metric, so expensive configurations sit visibly far right.
    Saves the figure into a 'plots/' folder next to this script file.
    """
    # --- Dati: per ogni metodo, (tempo [min], specializzazione icra, generalizzazione FIM_in) ---
    data = {
        "Joint retrain": {
            "color": "#1f77b4",  # blu
            "points": [
                (19.1, 36.8, 38.0),   # 2%
                (23.8, 35.5, 51.2),   # 10%
                (47.4, 38.1, 56.3),   # 50%
                (76.2, 37.7, 61.3),   # 100%
            ],
        },
        "Fine-tuning": {
            "color": "#d62728",  # rosso
            "points": [
                (0.7, 22.6, 48.8),    # 10%
                (2.8, 33.2, 29.5),    # 50%
                (5.4, 35.2, 33.1),    # 100%
            ],
        },
        "Replay": {
            "color": "#2ca02c",  # verde
            "points": [
                (5.8, 34.7, 42.1),    # buffer 2%
                (7.2, 34.0, 49.6),    # buffer 10%
                (14.2, 33.6, 54.9),   # buffer 50%
            ],
        },
    }

    # due pannelli impilati, stesso asse x
    fig, (ax_spec, ax_gen) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for method, info in data.items():
        color = info["color"]
        pts = sorted(info["points"], key=lambda p: p[0])
        times = [p[0] for p in pts]
        spec  = [p[1] for p in pts]
        gen   = [p[2] for p in pts]

        ax_spec.plot(times, spec, color=color, linestyle="-", marker="o",
                     linewidth=2, markersize=7, label=method)
        ax_gen.plot(times, gen, color=color, linestyle="-", marker="o",
                    linewidth=2, markersize=7, label=method)

    # --- Pannello superiore: specializzazione ---
    ax_spec.set_title("Specialization (icra)")
    ax_spec.set_ylabel("mAP50-95 (%)")
    ax_spec.set_ylim(0, 70)
    ax_spec.grid(True, linestyle="--", alpha=0.4)
    ax_spec.legend(title="Method", loc="lower right")

    # --- Pannello inferiore: generalizzazione ---
    ax_gen.set_title("Generalization (FIM_inside)")
    ax_gen.set_xlabel("Training time (minutes)")
    ax_gen.set_ylabel("mAP50-95 (%)")
    ax_gen.set_xlim(0, 80)
    ax_gen.set_ylim(0, 70)
    ax_gen.grid(True, linestyle="--", alpha=0.4)
    ax_gen.legend(title="Method", loc="lower right")

    fig.suptitle("Training time vs mAP50-95", fontsize=14)
    plt.tight_layout()

    # --- Cartella 'plots/' allo stesso livello di questo file ---
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / filename

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    pareto_curve()