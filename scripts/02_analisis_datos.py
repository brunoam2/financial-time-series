import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_PATH, FIGURES_PATH

data_file = PROCESSED_DATA_PATH / "combined_data.csv"
df = pd.read_csv(data_file, index_col=0, parse_dates=True)

target = "SPY_Close"
corr_target = df.corr()[target].drop(target).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=corr_target.values, y=corr_target.index, hue=corr_target.index, dodge=False, palette="coolwarm", legend=False)
plt.title(f"Correlación con la variable objetivo: {target}")
plt.xlabel("Coeficiente de correlación")
plt.ylabel("Variables")
plt.tight_layout()
plt.savefig(FIGURES_PATH / f"correlation_with_{target}.png")
plt.close()