"""그림 스타일 공통 설정 (README/노트북용 정적 그림)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LOW, C_HIGH = "#2a78d6", "#eb6834"      # ≤50K / >50K  (고정 순서, 두 색만 사용)
C_NEUTRAL, C_ACCENT = "#9a9a94", "#1baf7a"
PALETTE = [C_LOW, C_HIGH, C_ACCENT, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })
    for f in ["NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]:
        if any(ff.name == f for ff in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    plt.rcParams["axes.unicode_minus"] = False


from pathlib import Path
FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

def save_show(name: str):
    """현재 그림을 results/figures/<name>.png 로 저장한 뒤 표시한다 (README에서 참조)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png")
    plt.show()
