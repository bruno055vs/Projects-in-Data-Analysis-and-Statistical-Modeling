# ============================================================
# GRÁFICOS BAYESIANOS (versão simplificada via bootstrap)
# ============================================================
# Este código gera um PDF com as distribuições posteriores
# dos parâmetros da Normal e Lognormal:
# μ, σ (Normal) e μ_log, σ_log (Lognormal)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats
import pandas as pd
import os

# ------------------------------------------------------------
# Função para remover outliers (método do IQR)
# ------------------------------------------------------------
def remove_outliers(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return arr[(arr >= lower) & (arr <= upper)]

# ------------------------------------------------------------
# BLOCO PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. Carregar o dataset
    df = pd.read_csv("electricity.csv")

    # Seleciona a variável de interesse
    data = df["nswprice"].dropna().values
    data_clean = remove_outliers(data)
    data_pos = data_clean[data_clean > 0]  # para o modelo Lognormal

    # --------------------------------------------------------
    # 2. Bootstrap para simular distribuições posteriores
    # (aproximação bayesiana sem usar PyMC)
    # --------------------------------------------------------
    n_boot = 2000
    rng = np.random.default_rng(42)

    # Normal
    boot_mu, boot_sigma = [], []
    for _ in range(n_boot):
        sample = rng.choice(data_clean, size=len(data_clean), replace=True)
        m, s = stats.norm.fit(sample)
        boot_mu.append(m)
        boot_sigma.append(s)

    # Lognormal
    boot_mu_log, boot_sigma_log = [], []
    for _ in range(n_boot):
        sample = rng.choice(data_pos, size=len(data_pos), replace=True)
        s_fit, loc_fit, scale_fit = stats.lognorm.fit(sample, floc=0)
        boot_sigma_log.append(s_fit)
        boot_mu_log.append(np.log(scale_fit))

    # --------------------------------------------------------
    # 3. Criar PDF com os gráficos
    # --------------------------------------------------------
    output = "graficos_bayesianos_somente.pdf"
    with PdfPages(output) as pdf:
        # Título principal
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Gráficos Bayesianos", ha='center', va='center',
                 fontsize=26, weight='bold')
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # μ (Normal)
        plt.figure(figsize=(8, 5))
        plt.hist(boot_mu, bins=50, color="skyblue", density=True)
        plt.title("Posterior Bayesiano (aprox.) - μ (Modelo Normal)")
        plt.xlabel("μ")
        plt.ylabel("Densidade")
        pdf.savefig()
        plt.close()

        # σ (Normal)
        plt.figure(figsize=(8, 5))
        plt.hist(boot_sigma, bins=50, color="orange", density=True)
        plt.title("Posterior Bayesiano (aprox.) - σ (Modelo Normal)")
        plt.xlabel("σ")
        plt.ylabel("Densidade")
        pdf.savefig()
        plt.close()

        # μ_log (Lognormal)
        plt.figure(figsize=(8, 5))
        plt.hist(boot_mu_log, bins=50, color="lightgreen", density=True)
        plt.title("Posterior Bayesiano (aprox.) - μ_log (Modelo Lognormal)")
        plt.xlabel("μ_log")
        plt.ylabel("Densidade")
        pdf.savefig()
        plt.close()

        # σ_log (Lognormal)
        plt.figure(figsize=(8, 5))
        plt.hist(boot_sigma_log, bins=50, color="purple", alpha=0.7, density=True)
        plt.title("Posterior Bayesiano (aprox.) - σ_log (Modelo Lognormal)")
        plt.xlabel("σ_log")
        plt.ylabel("Densidade")
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------
    # 4. Finalização
    # --------------------------------------------------------





