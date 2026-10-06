"""
ANÁLISE EXPLORATÓRIA (pós-hoc) — divergência de vocabulário.

Decidida DEPOIS de ver os resultados principais; não substitui o teste confirmatório.
Pergunta: a diferença entre A (BM25) e B (E5) depende de quanto as palavras da consulta
aparecem literalmente no título + descrição do dataset certo?

Sobreposição = (palavras distintas da consulta presentes no texto do alvo) / (palavras distintas da consulta),
usando exatamente a mesma tokenização do método A (minúsculas, sem acento, [a-z0-9]+).

Grupos (fixados antes de rodar esta análise):
  G0 "nenhuma"  : sobreposição = 0   (divergência total de vocabulário)
  G1 "parcial"  : 0 < sobreposição < 1
  G2 "todas"    : sobreposição = 1   (todas as palavras da consulta estão no texto)

Em cada grupo: MRR A e B, diferença com IC 95% bootstrap, teste de aleatorização pareado,
Hit@10 e McNemar mid-p — os mesmos procedimentos do tp_busca.py.
6 testes no total -> limiar de Bonferroni = 0,05 / 6 ≈ 0,0083 (relatado junto).

Uso:  python exploratoria.py            (lê a pasta saida\\)
      python exploratoria.py --pasta saida_AUTOTESTE_sintetico
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tp_busca import CONFIG, mcnemar, preparar, tokenizar, log

GRUPOS = ["G0 nenhuma", "G1 parcial", "G2 todas"]


def sobreposicao(consulta: str, texto: str) -> float:
    q = set(tokenizar(consulta))
    if not q:
        return np.nan
    return len(q & set(tokenizar(texto))) / len(q)


def grupo(s: float) -> str:
    return GRUPOS[0] if s == 0 else (GRUPOS[2] if s == 1 else GRUPOS[1])


def analisar_grupo(g: pd.DataFrame, rng) -> dict:
    from scipy import stats
    k = CONFIG["top_k_hit"]
    d = (g["rr_B"] - g["rr_A"]).to_numpy()
    out = {"n": len(g), "MRR_A": g["rr_A"].mean(), "MRR_B": g["rr_B"].mean(), "dif_B_menos_A": d.mean(),
           "B_melhor_pior_empate": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
           "mediana_pos_A": g["pos_A"].median(), "mediana_pos_B": g["pos_B"].median()}
    if len(g) >= 2 and np.any(d != 0):
        out["p_aleatorizacao"] = stats.permutation_test(
            (g["rr_B"].to_numpy(), g["rr_A"].to_numpy()), lambda x, y: np.mean(x - y),
            permutation_type="samples", n_resamples=CONFIG["n_permutacoes"],
            alternative="two-sided", random_state=rng).pvalue
        ic = stats.bootstrap((d,), np.mean, n_resamples=CONFIG["n_bootstrap"], method="percentile",
                             confidence_level=0.95, random_state=rng).confidence_interval
        out["IC95_bootstrap_dif"] = [ic.low, ic.high]
    else:
        out["p_aleatorizacao"], out["IC95_bootstrap_dif"] = 1.0, [0.0, 0.0]
    hA, hB = g[f"hit{k}_A"].astype(bool), g[f"hit{k}_B"].astype(bool)
    out[f"Hit@{k}_A"], out[f"Hit@{k}_B"] = hA.mean(), hB.mean()
    out["mcnemar_midp"] = mcnemar(int((hB & ~hA).sum()), int((hA & ~hB).sum()))
    return out


def figura(res: dict, pasta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = CONFIG["top_k_hit"]
    rot = [f"{g}\n(n={res[g]['n']})" for g in GRUPOS]
    x = np.arange(len(GRUPOS))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (met, tit) in enumerate([("MRR", "MRR"), (f"Hit@{k}", f"Hit@{k}")]):
        ax[i].bar(x - .2, [res[g][f"{met}_A"] for g in GRUPOS], .4, label="A: BM25", color="#555")
        ax[i].bar(x + .2, [res[g][f"{met}_B"] for g in GRUPOS], .4, label="B: semântica (E5)", color="#aaa")
        ax[i].set_xticks(x, rot); ax[i].set_ylim(0, 1); ax[i].set_title(tit); ax[i].legend(loc="lower right")
    fig.suptitle("EXPLORATÓRIO — palavras da consulta presentes no título + descrição do dataset certo")
    fig.tight_layout(); fig.savefig(pasta / "fig_exploratoria.png", dpi=200); plt.close(fig)


def main(pasta: Path) -> None:
    meta = pd.read_csv(pasta / "redu_metadados.csv", dtype=str)
    df, docs, _, _, _ = preparar(meta)
    texto = dict(zip(df["global_id"], docs))
    r = pd.read_csv(pasta / "ranks.csv")
    r["sobreposicao"] = [sobreposicao(c, texto[g]) for c, g in zip(r["consulta"], r["global_id"])]
    r = r.dropna(subset=["sobreposicao"])
    r["grupo"] = r["sobreposicao"].map(grupo)
    rng = np.random.default_rng(CONFIG["seed"])
    res = {g: analisar_grupo(r[r["grupo"] == g], rng) for g in GRUPOS}
    res["_nota"] = ("ANÁLISE EXPLORATÓRIA pós-hoc; 6 testes -> limiar de Bonferroni "
                    f"{CONFIG['alpha'] / 6:.4f}. Não substitui o resultado confirmatório.")
    r.to_csv(pasta / "exploratoria_por_consulta.csv", index=False, encoding="utf-8")
    (pasta / "exploratoria.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    figura(res, pasta)
    log(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default="saida")
    main(Path(ap.parse_args().pasta))
