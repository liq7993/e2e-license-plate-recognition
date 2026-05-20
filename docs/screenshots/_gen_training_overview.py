"""
从 runs/detect/train/results.csv 生成一张更"答辩友好"的训练曲线总览图，
存到 docs/screenshots/training_overview.png。

跑法：
    python docs/screenshots/_gen_training_overview.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示也能跑
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CSV = REPO / "runs" / "detect" / "train" / "results.csv"
OUT = HERE / "training_overview.png"


def main() -> None:
    df = pd.read_csv(CSV)
    epoch = df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(
        "YOLOv8 License-Plate Detection — Training Overview (20 epochs)",
        fontsize=14,
        fontweight="bold",
    )

    # ---- 左上：训练 loss ----
    ax = axes[0, 0]
    ax.plot(epoch, df["train/box_loss"], label="box_loss", linewidth=2)
    ax.plot(epoch, df["train/cls_loss"], label="cls_loss", linewidth=2)
    ax.plot(epoch, df["train/dfl_loss"], label="dfl_loss", linewidth=2)
    ax.set_title("Training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # ---- 右上：验证 loss ----
    ax = axes[0, 1]
    ax.plot(epoch, df["val/box_loss"], label="box_loss", linewidth=2)
    ax.plot(epoch, df["val/cls_loss"], label="cls_loss", linewidth=2)
    ax.plot(epoch, df["val/dfl_loss"], label="dfl_loss", linewidth=2)
    ax.set_title("Validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # ---- 左下：P / R / mAP ----
    ax = axes[1, 0]
    ax.plot(epoch, df["metrics/precision(B)"], label="Precision", linewidth=2)
    ax.plot(epoch, df["metrics/recall(B)"], label="Recall", linewidth=2)
    ax.plot(epoch, df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=2)
    ax.plot(
        epoch,
        df["metrics/mAP50-95(B)"],
        label="mAP@0.5:0.95",
        linewidth=2,
    )
    ax.set_title("Validation metrics")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # ---- 右下：终态指标卡 ----
    ax = axes[1, 1]
    ax.axis("off")
    last = df.iloc[-1]
    rows = [
        ("Final epoch", f"{int(last['epoch'])}"),
        ("Precision (B)", f"{last['metrics/precision(B)']:.4f}"),
        ("Recall (B)", f"{last['metrics/recall(B)']:.4f}"),
        ("mAP@0.5", f"{last['metrics/mAP50(B)']:.4f}"),
        ("mAP@0.5:0.95", f"{last['metrics/mAP50-95(B)']:.4f}"),
        ("train/box_loss", f"{last['train/box_loss']:.4f}"),
        ("val/box_loss", f"{last['val/box_loss']:.4f}"),
        (
            "Total wall-clock",
            f"{last['time'] / 3600:.2f} h ({last['time']:.0f} s)",
        ),
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.55, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    # 加粗表头
    for col in (0, 1):
        cell = table[(0, col)]
        cell.set_facecolor("#3b4252")
        cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Final-epoch summary", pad=20)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
