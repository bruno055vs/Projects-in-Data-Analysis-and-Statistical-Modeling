# analise_electricity.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognorm

# --- Config ---
os.makedirs("resultados", exist_ok=True)
sns.set(style="whitegrid")

# --- 1. Carregar dados ---
df = pd.read_csv("electricity.csv")
# Escolha a variável a analisar (mude se quiser)
col = "nswprice"
data_all = df[col].dropna().astype(float)

# Função utilitária para salvar figuras
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join("resultados", name), dpi=150)
    plt.close(fig)

# --- 2. Plot: Histograma + KDE ---
fig = plt.figure(figsize=(7,4))
sns.histplot(data_all, bins=40, stat="density", kde=True)
plt.title(f"Histograma + KDE — {col}")
plt.xlabel(col)
plt.ylabel("Densidade")
save_fig(fig, f"kde_hist_{col}.png")

# --- 3. Estimativas por MM (Method of Moments) e MLE ---
# Normal:
mm_norm_mu = data_all.mean()
mm_norm_sigma = data_all.std(ddof=1)

mle_norm_mu, mle_norm_sigma = norm.fit(data_all)  # MLE for normal

# Lognormal: only positive values
data_pos = data_all[data_all > 0]
if len(data_pos) == 0:
    raise RuntimeError("Sem valores positivos para ajuste lognormal.")

# MM for lognormal: use log-moments
logx = np.log(data_pos)
mm_lognorm_mu = logx.mean()        # mu for underlying normal
mm_lognorm_sigma = logx.std(ddof=1)

# MLE for lognormal (fix loc=0 for standard 2-parameter lognormal)
mle_lognorm_shape, mle_lognorm_loc, mle_lognorm_scale = lognorm.fit(data_pos, floc=0)
# scipy's lognorm: shape = sigma, scale = exp(mu)

# Print summary of parameter estimates
print("=== Parâmetros estimados ===")
print(f"Normal - MM: mu={mm_norm_mu:.6f}, sigma={mm_norm_sigma:.6f}")
print(f"Normal - MLE: mu={mle_norm_mu:.6f}, sigma={mle_norm_sigma:.6f}")
print()
print(f"Lognormal (dados > 0) - MM (log-space): mu_log={mm_lognorm_mu:.6f}, sigma_log={mm_lognorm_sigma:.6f}")
print(f"Lognormal - MLE (scipy): shape={mle_lognorm_shape:.6f}, loc={mle_lognorm_loc:.6f}, scale={mle_lognorm_scale:.6f}")
print()

# --- Helper: log-likelihoods, AIC, BIC ---
def ll_norm(params, data):
    mu, sigma = params
    return np.sum(norm.logpdf(data, mu, sigma))

def ll_lognorm(params, data_pos):  # params: shape, loc, scale
    shape, loc, scale = params
    return np.sum(lognorm.logpdf(data_pos, shape, loc, scale))

def compute_aic_bic(ll, k, n):
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return aic, bic

# Log-likelihoods and AIC/BIC
n = len(data_all)
ll_norm_mle = ll_norm((mle_norm_mu, mle_norm_sigma), data_all)
aic_norm, bic_norm = compute_aic_bic(ll_norm_mle, k=2, n=n)

# For lognormal: compute using positive data only
n_pos = len(data_pos)
ll_lognorm_mle = ll_lognorm((mle_lognorm_shape, mle_lognorm_loc, mle_lognorm_scale), data_pos)
aic_lognorm, bic_lognorm = compute_aic_bic(ll_lognorm_mle, k=3, n=n_pos)

print("=== AIC / BIC (MLE) ===")
print(f"Normal:  AIC={aic_norm:.3f}, BIC={bic_norm:.3f}")
print(f"Lognorm: AIC={aic_lognorm:.3f}, BIC={bic_lognorm:.3f}")

# --- 4. QQ-plots para cada ajuste (MM and MLE) ---
# We'll show QQ for Normal(MM), Normal(MLE), Lognormal(MM), Lognormal(MLE)
# 4 plots saved separately.

# Normal QQ (MLE)
fig = plt.figure(figsize=(6,4))
stats.probplot(data_all, dist="norm", sparams=(mle_norm_mu, mle_norm_sigma), plot=plt)
plt.title(f"QQ-plot Normal (MLE) - {col}")
save_fig(fig, f"qq_normal_mle_{col}.png")

# Normal QQ (MM) - same as MLE in normal case (MM gives sample mean/std but MLE for normal equals MM)
fig = plt.figure(figsize=(6,4))
stats.probplot(data_all, dist="norm", sparams=(mm_norm_mu, mm_norm_sigma), plot=plt)
plt.title(f"QQ-plot Normal (MM) - {col}")
save_fig(fig, f"qq_normal_mm_{col}.png")

# Lognormal QQ (MM): transform data_pos by log and compare to normal(mu_log, sigma_log)
# create theoretical quantiles by applying exp to normal quantiles
# We'll generate probplot against lognormal via transforming
fig = plt.figure(figsize=(6,4))
# For probplot with a user distribution, we can transform: take log(data_pos) and compare to normal(mu_log, sigma_log)
stats.probplot(np.log(data_pos), dist="norm", sparams=(mm_lognorm_mu, mm_lognorm_sigma), plot=plt)
plt.title(f"QQ-plot Lognormal (MM, log-data) - {col}")
save_fig(fig, f"qq_lognorm_mm_{col}.png")

# Lognormal QQ (MLE): compare log(data_pos) vs normal with implied mu,sigma from MLE
# mle_lognorm_scale = exp(mu_mle), so mu_mle = log(scale)
mu_mle_lognorm = np.log(mle_lognorm_scale)
sigma_mle_lognorm = mle_lognorm_shape
fig = plt.figure(figsize=(6,4))
stats.probplot(np.log(data_pos), dist="norm", sparams=(mu_mle_lognorm, sigma_mle_lognorm), plot=plt)
plt.title(f"QQ-plot Lognormal (MLE, log-data) - {col}")
save_fig(fig, f"qq_lognorm_mle_{col}.png")

# --- 5. Resíduos padronizados (quantile residuals) ---
# For a fitted continuous distribution F, quantile residuals: z_i = Phi^{-1}(F(x_i;theta))
def quantile_residuals_norm(data, mu, sigma):
    u = norm.cdf(data, mu, sigma)
    # avoid exactly 0/1
    u = np.clip(u, 1e-10, 1-1e-10)
    z = norm.ppf(u)
    return z

def quantile_residuals_lognorm(data_pos, shape, loc, scale):
    u = lognorm.cdf(data_pos, shape, loc, scale)
    u = np.clip(u, 1e-10, 1-1e-10)
    z = norm.ppf(u)  # transform uniform->normal
    return z

# Residuals for normal (MLE)
resid_norm_mle = quantile_residuals_norm(data_all, mle_norm_mu, mle_norm_sigma)
# Residuals for lognormal (MLE) - only positive data; create array aligned with data_all by NaN for non-positive
resid_lognorm_mle_pos = quantile_residuals_lognorm(data_pos, mle_lognorm_shape, mle_lognorm_loc, mle_lognorm_scale)
# Create aligned array:
resid_lognorm_mle = pd.Series(index=data_all.index, dtype=float)
resid_lognorm_mle.loc[data_pos.index] = resid_lognorm_mle_pos
resid_lognorm_mle = resid_lognorm_mle.values

# Plot histogram of residuals
fig = plt.figure(figsize=(6,4))
sns.histplot(resid_norm_mle, bins=40, stat="density", kde=True)
plt.title("Resíduos padronizados (Normal MLE)")
save_fig(fig, f"residuos_normal_mle_{col}.png")

fig = plt.figure(figsize=(6,4))
sns.histplot(resid_lognorm_mle_pos, bins=40, stat="density", kde=True)
plt.title("Resíduos padronizados (Lognormal MLE) — dados > 0")
save_fig(fig, f"residuos_lognorm_mle_{col}.png")

# --- 6. Identificar possíveis outliers ---
# Usaremos duas abordagens: (A) IQR e (B) resíduos padronizados (|z|>3)
# A) IQR on original data
q1, q3 = np.percentile(data_all, [25,75])
iqr = q3 - q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr
outliers_iqr = data_all[(data_all < lower) | (data_all > upper)]

# B) Quantile residuals threshold (normal residuals)
outliers_resid = data_all[np.abs(resid_norm_mle) > 3]

print()
print("=== Outliers identificados ===")
print(f"IQR method: {len(outliers_iqr)} pontos (exemplos):")
print(outliers_iqr.head().to_string(index=False))
print()
print(f"Residuals method (|z|>3): {len(outliers_resid)} pontos (exemplos):")
print(outliers_resid.head().to_string(index=False))

# Save a scatter to show outliers on distribution
fig = plt.figure(figsize=(7,4))
sns.histplot(data_all, bins=40, stat="density", color="lightgray")
plt.scatter(outliers_iqr, np.zeros_like(outliers_iqr), color="red", label="outliers IQR", zorder=5)
plt.scatter(outliers_resid, np.zeros_like(outliers_resid)+0.01, color="purple", label="outliers resid", zorder=6)
plt.legend()
plt.title("Outliers indicados sobre histograma")
save_fig(fig, f"outliers_{col}.png")

# --- 7. Avaliar efeito da remoção dos outliers ---
data_no_out_iqr = data_all[~data_all.isin(outliers_iqr)]
data_no_out_resid = data_all[~data_all.isin(outliers_resid)]

def reestimate_and_plot(data_subset, tag):
    # Re-estimate MLE normal and lognormal (lognormal only if >0)
    mu_m, sigma_m = norm.fit(data_subset)
    print(f"\nSubset ({tag}): n={len(data_subset)}, Normal MLE mu={mu_m:.6f}, sigma={sigma_m:.6f}")
    # Lognormal on positives
    data_subset_pos = data_subset[data_subset > 0]
    if len(data_subset_pos)>0:
        shape_m, loc_m, scale_m = lognorm.fit(data_subset_pos, floc=0)
        print(f"           Lognormal MLE shape={shape_m:.6f}, scale={scale_m:.6f}")
    else:
        shape_m, loc_m, scale_m = (np.nan, np.nan, np.nan)
        print("           Lognormal não aplicável (sem valores positivos)")

    # Plot KDE + fitted curves
    fig = plt.figure(figsize=(7,4))
    xs = np.linspace(data_subset.min(), data_subset.max(), 300)
    sns.kdeplot(data_subset, fill=True, label="KDE (dados)")
    plt.plot(xs, norm.pdf(xs, mu_m, sigma_m), label="Normal MLE")
    if not np.isnan(shape_m):
        plt.plot(xs, lognorm.pdf(xs, shape_m, loc_m, scale_m), label="Lognormal MLE")
    plt.title(f"KDE + Ajustes ({tag}) - {col}")
    plt.legend()
    save_fig(fig, f"reajuste_{tag}_{col}.png")

# Re-estimate and plot:
reestimate_and_plot(data_no_out_iqr, "sem_outliers_IQR")
reestimate_and_plot(data_no_out_resid, "sem_outliers_resid")

# Compare numeric change in parameters (before vs after)
print("\n=== Comparação de parâmetros antes vs depois (Normal MLE) ===")
mu_before, sigma_before = mle_norm_mu, mle_norm_sigma
mu_after_iqr, sigma_after_iqr = norm.fit(data_no_out_iqr)
mu_after_resid, sigma_after_resid = norm.fit(data_no_out_resid)
print(f"Antes: mu={mu_before:.6f}, sigma={sigma_before:.6f}")
print(f"Apos remoção IQR: mu={mu_after_iqr:.6f}, sigma={sigma_after_iqr:.6f}")
print(f"Apos remoção Resid: mu={mu_after_resid:.6f}, sigma={sigma_after_resid:.6f}")

# Recompute AICs on subsets (if desired)
ll_norm_iqr = ll_norm((mu_after_iqr, sigma_after_iqr), data_no_out_iqr)
aic_norm_iqr, bic_norm_iqr = compute_aic_bic(ll_norm_iqr, 2, len(data_no_out_iqr))
print("\nAIC Normal - antes vs depois (IQR):")
print(f"Antes: AIC={aic_norm:.3f}, Depois IQR: AIC={aic_norm_iqr:.3f}")

# Save a summary CSV of metrics
summary = {
    "model": ["Normal_MLE", "Lognorm_MLE"],
    "AIC_before": [aic_norm, aic_lognorm],
    "BIC_before": [bic_norm, bic_lognorm],
    "n_before": [n, n_pos]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv("resultados/summary_metrics.csv", index=False)

print("\nTudo salvo em pasta 'resultados/'. Arquivos principais:")
for f in os.listdir("resultados"):
    print(" -", f)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# === Carregar dados ===
df = pd.read_csv("electricity.csv")
vars_num = ["nswprice", "vicprice"]
dados = df[vars_num]

# === Remover valores ausentes ===
dados = dados.dropna()

# === Escolher variável para exemplo ===
col = "nswprice"
data = dados[col]
data = data[data > 0]  # lognormal não aceita zeros

# === Ajuste Normal e Lognormal ===
mu_norm, sigma_norm = norm.fit(data)
shape_logn, loc_logn, scale_logn = lognorm.fit(data, floc=0)

# === Resíduos padronizados ===
residuos_norm = (data - mu_norm) / sigma_norm
residuos_logn = (np.log(data) - np.log(scale_logn)) / shape_logn

# === Remover outliers (|z| > 3) ===
limite = 3
residuos_norm_sem_out = residuos_norm[(residuos_norm > -limite) & (residuos_norm < limite)]
residuos_logn_sem_out = residuos_logn[(residuos_logn > -limite) & (residuos_logn < limite)]

# === Plotar resíduos sem outliers ===
plt.figure(figsize=(7, 4))
sns.histplot(residuos_norm_sem_out, bins=30, stat="density", color="lightblue", alpha=0.6)
sns.kdeplot(residuos_norm_sem_out, color="blue")
plt.title("Resíduos padronizados (Normal MLE) — sem outliers")
plt.xlabel("Resíduo")
plt.ylabel("Densidade")
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(residuos_logn_sem_out, bins=30, stat="density", color="lightblue", alpha=0.6)
sns.kdeplot(residuos_logn_sem_out, color="blue")
plt.title("Resíduos padronizados (Lognormal MLE) — sem outliers")
plt.xlabel("Resíduo")
plt.ylabel("Densidade")
plt.show()

# === Mostrar resumo ===
print(f"Nº de resíduos removidos (Normal): {len(residuos_norm) - len(residuos_norm_sem_out)}")
print(f"Nº de resíduos removidos (Lognormal): {len(residuos_logn) - len(residuos_logn_sem_out)}")
