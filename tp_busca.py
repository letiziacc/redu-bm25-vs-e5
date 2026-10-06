# -*- coding: utf-8 -*-
"""
Desenho (busca de item conhecido / known-item search):
  - Índice: título + descrição de TODOS os datasets publicados.
  - Consultas: as palavras-chave de cada dataset que tem >= 2 palavras-chave.
  - Resposta certa: o próprio dataset de onde saíram as palavras-chave.
  - Métrica por consulta: reciprocal rank (1/posição). Pareado: mesma consulta em A e B.

Etapas:
    python tp_busca.py coletar      # Search API do REDU -> saida/redu_metadados.csv
    python tp_busca.py experimento  # A e B para todas as consultas -> saida/ranks.csv
    python tp_busca.py analisar     # testes, IC, figuras -> saida/resultados.json
    python tp_busca.py autoteste    # tudo offline com dados SINTÉTICOS (só para testar o código)

Opção: --b-tfidf  usa TF-IDF como método B (plano B se o modelo semântico não puder ser baixado).
"""
from __future__ import annotations

import argparse
import html
import json
import platform
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração fixada ANTES de ver resultados (pré-registro)
# ---------------------------------------------------------------------------
CONFIG = {
    "base_url": "https://redu.unicamp.br",
    "per_page": 1000,                      # máximo documentado da Search API
    "pausa_s": 1.0,
    "min_keywords": 2,                     # datasets com menos palavras-chave não geram consulta
    "bm25_k1": 1.2, "bm25_b": 0.75,        # valores padrão do BM25 no Lucene/Solr
    "modelo_B": "intfloat/multilingual-e5-small",  # MIT; exige prefixos "query: "/"passage: "
    "top_k_hit": 10,
    "alpha": 0.05,
    "n_permutacoes": 100_000,
    "n_bootstrap": 10_000,
    "seed": 20260923,
}
SAIDA = Path("saida")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def limpar(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 1) COLETA — Search API (parâmetros documentados: q, type, fq, per_page, start, metadata_fields)
# ---------------------------------------------------------------------------
def http_get_json(url: str, params: dict) -> dict:
    import requests
    r = requests.get(url, params=params, timeout=90,
                     headers={"User-Agent": "TP-Metodologia-Unicamp (uso academico)"})
    r.raise_for_status()
    return r.json()


def extrair_keywords(item: dict) -> list[str]:
    """Lê citation:keyword -> keywordValue (o vocabulário, ex. LCSH, fica em outro subcampo)."""
    kws = []
    try:
        campos = item["metadataBlocks"]["citation"]["fields"]
    except (KeyError, TypeError):
        return kws
    for f in campos:
        if f.get("typeName") == "keyword":
            for v in f.get("value", []):
                kv = v.get("keywordValue", {}).get("value")
                if kv and kv.strip():
                    kws.append(kv.strip())
    return kws


def cmd_coletar(get=http_get_json, pausa=True) -> None:
    SAIDA.mkdir(exist_ok=True)
    url = f"{CONFIG['base_url']}/api/search"
    linhas, start, total = [], 0, None
    while True:
        params = {"q": "*", "type": "dataset", "fq": "publicationStatus:Published",
                  "metadata_fields": "citation:keyword",
                  "per_page": CONFIG["per_page"], "start": start}
        data = get(url, params)["data"]
        total = data["total_count"]
        for it in data["items"]:
            linhas.append({"global_id": it.get("global_id"), "titulo": limpar(it.get("name", "")),
                           "descricao": limpar(it.get("description", "")),
                           "keywords": json.dumps(extrair_keywords(it), ensure_ascii=False),
                           "publicado_em": it.get("published_at")})
        start += CONFIG["per_page"]
        log(f"{min(start, total)}/{total} datasets")
        if start >= total:
            break
        if pausa:
            time.sleep(CONFIG["pausa_s"])
    df = pd.DataFrame(linhas).drop_duplicates("global_id")
    if len(df) != total:
        log(f"AVISO: {len(df)} únicos de total_count={total}")
    df.to_csv(SAIDA / "redu_metadados.csv", index=False, encoding="utf-8")
    meta = {"coletado_em": datetime.now().isoformat(), "total_count": total, "n_unicos": len(df),
            "n_com_keywords": int((df["keywords"].map(json.loads).map(len) > 0).sum())}
    (SAIDA / "coleta_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(json.dumps(meta))


# ---------------------------------------------------------------------------
# 2) PREPARAÇÃO do corpus e das consultas
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int], list[str], dict]:
    rel = {"n_coletados": len(df)}
    df = df.copy()
    df["titulo"] = df["titulo"].fillna("").map(limpar)
    df["descricao"] = df["descricao"].fillna("").map(limpar)
    vazio = (df["titulo"] + df["descricao"]).str.strip() == ""
    rel["excl_sem_texto"] = int(vazio.sum())
    df = df[~vazio].reset_index(drop=True)
    docs = (df["titulo"] + ". " + df["descricao"]).tolist()     # palavras-chave NÃO entram no índice
    alvos, consultas, n_poucas = [], [], 0
    for i, kws in enumerate(df["keywords"].fillna("[]").map(json.loads)):
        unicas = list(dict.fromkeys(k for k in kws if norm(k)))
        if len({norm(k) for k in unicas}) >= CONFIG["min_keywords"]:
            alvos.append(i); consultas.append(" ".join(unicas))
        else:
            n_poucas += 1
    rel.update({"n_corpus": len(docs), "n_consultas": len(consultas),
                "sem_consulta_menos_de_2_keywords": n_poucas})
    return df, docs, alvos, consultas, rel


# ---------------------------------------------------------------------------
# 3) MÉTODOS
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"[a-z0-9]+")


def tokenizar(s: str) -> list[str]:
    return TOKEN.findall(norm(s))


class BM25:
    """BM25 no estilo Lucene/Solr: idf = ln(1 + (N - df + 0.5)/(df + 0.5)).
    Sem stemming e sem stopwords (corpus bilíngue pt/en); declarado como limitação."""

    def __init__(self, docs: list[str], k1: float, b: float):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer(tokenizer=tokenizar, lowercase=False, token_pattern=None)
        tf = self.cv.fit_transform(docs).tocsc().astype(np.float64)
        N = tf.shape[0]
        dfreq = np.diff(tf.indptr)
        self.idf = np.log1p((N - dfreq + 0.5) / (dfreq + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm_len = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm_len[tf.row])
        from scipy.sparse import csr_matrix
        self.W = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, consulta: str) -> np.ndarray:
        q = self.cv.transform([consulta])          # termos repetidos contam 1 vez
        q.data[:] = 1.0
        q = q.multiply(self.idf).tocsr()
        return np.asarray(self.W @ q.T.toarray()).ravel()


class Semantico:
    def __init__(self, docs: list[str]):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(CONFIG["modelo_B"])
        self.D = self.m.encode(["passage: " + d for d in docs], batch_size=32,
                               normalize_embeddings=True, show_progress_bar=True)

    def scores_lote(self, consultas: list[str]) -> np.ndarray:
        Q = self.m.encode(["query: " + q for q in consultas], batch_size=64,
                          normalize_embeddings=True, show_progress_bar=True)
        return Q @ self.D.T


class TFIDF:
    """Plano B / autoteste: TF-IDF com cosseno."""
    def __init__(self, docs: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(tokenizer=tokenizar, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.D = self.v.fit_transform(docs)

    def scores_lote(self, consultas: list[str]) -> np.ndarray:
        return (self.v.transform(consultas) @ self.D.T).toarray()


def posicao(scores: np.ndarray, alvo: int) -> float:
    """Posição do alvo; empates recebem a posição média (desempate aleatório esperado)."""
    s = scores[alvo]
    maiores = int((scores > s).sum())
    iguais = int((scores == s).sum())      # inclui o próprio alvo
    return maiores + (iguais + 1) / 2


def cmd_experimento(b_tfidf: bool = False) -> None:
    df = pd.read_csv(SAIDA / "redu_metadados.csv", dtype=str)
    df, docs, alvos, consultas, rel = preparar(df)
    log(json.dumps(rel))
    t0 = time.perf_counter(); A = BM25(docs, CONFIG["bm25_k1"], CONFIG["bm25_b"])
    posA = [posicao(A.scores(q), a) for q, a in zip(consultas, alvos)]
    tA = time.perf_counter() - t0
    t0 = time.perf_counter(); B = TFIDF(docs) if b_tfidf else Semantico(docs)
    S = B.scores_lote(consultas)
    posB = [posicao(S[i], a) for i, a in enumerate(alvos)]
    tB = time.perf_counter() - t0
    r = pd.DataFrame({"global_id": df.loc[alvos, "global_id"].values, "consulta": consultas,
                      "pos_A": posA, "pos_B": posB})
    r["rr_A"], r["rr_B"] = 1 / r["pos_A"], 1 / r["pos_B"]
    k = CONFIG["top_k_hit"]
    r[f"hit{k}_A"], r[f"hit{k}_B"] = r["pos_A"] <= k, r["pos_B"] <= k
    r.to_csv(SAIDA / "ranks.csv", index=False, encoding="utf-8")
    amb = {"data": datetime.now().isoformat(), "python": sys.version.split()[0],
           "sistema": platform.platform(), "config": CONFIG, "preparacao": rel,
           "metodo_B": "TF-IDF (plano B)" if b_tfidf else CONFIG["modelo_B"],
           "tempo_total_A_s": tA, "tempo_total_B_s": tB}
    try:
        import sentence_transformers, sklearn
        amb.update({"sentence_transformers": sentence_transformers.__version__, "sklearn": sklearn.__version__})
    except ImportError:
        pass
    (SAIDA / "ambiente.json").write_text(json.dumps(amb, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{len(r)} consultas; MRR A={r['rr_A'].mean():.3f}  B={r['rr_B'].mean():.3f}")


# ---------------------------------------------------------------------------
# 4) ANÁLISE
#  Principal: teste de aleatorização pareado (troca de sinais) na diferença de MRR,
#  recomendado para avaliação de busca por Smucker, Allan & Carterette (CIKM 2007).
#  Concordância: t pareado e Wilcoxon signed-rank. Secundário: McNemar mid-p em Hit@10
#  (Fagerland et al., 2013); o p exato também é relatado.
# ---------------------------------------------------------------------------
def mcnemar(b: int, c: int) -> dict:
    """McNemar para dados binários pareados, a partir dos pares discordantes
    b (só B acerta) e c (só A acerta). Sob H0, X ~ Binomial(n = b + c, p = 0,5).
      - exato (condicional):  p = min(1, 2 * P(X <= k)),                k = min(b, c)
      - mid-p (PRINCIPAL):    p = min(1, 2 * (P(X < k) + 0,5 * P(X = k)))
    O exato é conservador; Fagerland, Lydersen & Laake (2013, BMC Med Res Methodol 13:91)
    recomendam o mid-p. O exato é relatado apenas para transparência."""
    from scipy.stats import binom
    n, k = b + c, min(b, c)
    if n == 0:
        return {"b_so_B_acha": b, "c_so_A_acha": c, "n_discordantes": 0, "p_midp": 1.0, "p_exato": 1.0}
    p_exato = min(1.0, 2 * binom.cdf(k, n, 0.5))
    p_midp = min(1.0, 2 * (binom.cdf(k - 1, n, 0.5) + 0.5 * binom.pmf(k, n, 0.5)))
    return {"b_so_B_acha": b, "c_so_A_acha": c, "n_discordantes": n, "p_midp": p_midp, "p_exato": p_exato}


def cmd_analisar() -> None:
    from scipy import stats
    r = pd.read_csv(SAIDA / "ranks.csv")
    k = CONFIG["top_k_hit"]
    d = (r["rr_B"] - r["rr_A"]).to_numpy()
    rng = np.random.default_rng(CONFIG["seed"])

    perm = stats.permutation_test((r["rr_B"].to_numpy(), r["rr_A"].to_numpy()),
                                  lambda x, y: np.mean(x - y), permutation_type="samples",
                                  n_resamples=CONFIG["n_permutacoes"], alternative="two-sided",
                                  random_state=rng)
    boot = stats.bootstrap((d,), np.mean, n_resamples=CONFIG["n_bootstrap"], method="percentile",
                           confidence_level=0.95, random_state=rng)
    t = stats.ttest_rel(r["rr_B"], r["rr_A"])
    w = stats.wilcoxon(r["rr_B"], r["rr_A"], zero_method="wilcox")
    hA, hB = r[f"hit{k}_A"].astype(bool), r[f"hit{k}_B"].astype(bool)
    mcn = mcnemar(int((hB & ~hA).sum()), int((hA & ~hB).sum()))
    nao_achados = r[(r["pos_A"] > k) & (r["pos_B"] > k)]
    nao_achados.to_csv(SAIDA / "baixa_encontrabilidade.csv", index=False, encoding="utf-8")

    res = {
        "n_consultas": len(r),
        "MRR_A": r["rr_A"].mean(), "MRR_B": r["rr_B"].mean(), "dif_MRR_B_menos_A": d.mean(),
        "IC95_bootstrap_dif": [boot.confidence_interval.low, boot.confidence_interval.high],
        "principal_aleatorizacao": {"p": perm.pvalue, "n_permutacoes": CONFIG["n_permutacoes"]},
        "concordancia_t_pareado": {"t": t.statistic, "p": t.pvalue},
        "concordancia_wilcoxon": {"W": w.statistic, "p": w.pvalue, "n_empates_zero": int((d == 0).sum())},
        "consultas_B_melhor_pior_empate": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
        f"Hit@{k}_A": hA.mean(), f"Hit@{k}_B": hB.mean(),
        f"secundario_mcnemar_midp_hit{k}": mcn,
        "mediana_posicao_A": r["pos_A"].median(), "mediana_posicao_B": r["pos_B"].median(),
        "n_nao_encontrados_por_nenhum_top10": len(nao_achados),
        "decisao": ("rejeita H0" if perm.pvalue < CONFIG["alpha"] else "não rejeita H0") + f" (alfa={CONFIG['alpha']})",
    }
    (SAIDA / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    figuras(r, res, k)
    log(json.dumps(res, indent=2, ensure_ascii=False, default=float))


def figuras(r, res, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    amb = json.loads((SAIDA / "ambiente.json").read_text(encoding="utf-8"))
    nB = "B: TF-IDF" if "TF-IDF" in amb["metodo_B"] else "B: semântica (E5)"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(2)
    ax[0].bar(x - .2, [res["MRR_A"], res["MRR_B"]], .4, label="MRR", color="#555")
    ax[0].bar(x + .2, [res[f"Hit@{k}_A"], res[f"Hit@{k}_B"]], .4, label=f"Hit@{k}", color="#aaa")
    ax[0].set_xticks(x, ["A: BM25", nB]); ax[0].set_ylim(0, 1); ax[0].legend()
    ax[0].set_title(f"{res['n_consultas']} consultas — p = {res['principal_aleatorizacao']['p']:.3g}")
    lim = max(r["pos_A"].max(), r["pos_B"].max())
    ax[1].scatter(r["pos_A"], r["pos_B"], s=6, alpha=.4, color="#333")
    ax[1].plot([1, lim], [1, lim], "k--", lw=.8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Posição do dataset certo em A"); ax[1].set_ylabel("Posição do dataset certo em B")
    ax[1].set_title("Abaixo da diagonal: B encontrou melhor")
    fig.tight_layout(); fig.savefig(SAIDA / "fig_resultados.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------------------
# AUTOTESTE (dados sintéticos no formato documentado da Search API)
# ---------------------------------------------------------------------------
def api_falsa(n=800, seed=0):
    rng = np.random.default_rng(seed)
    temas = [("liga titanio manufatura aditiva microestrutura", "titanium alloys additive manufacturing martensite"),
             ("solo agricultura irrigacao milho produtividade", "soil irrigation maize crop yield"),
             ("proteina gene expressao celula cancer", "gene expression protein cancer cells"),
             ("entrevista memoria historia oral arquivo", "oral history interviews archive memory"),
             ("rede sensores energia consumo eficiencia", "sensor networks energy efficiency")]
    itens = []
    for i in range(n):
        pt, en = temas[i % len(temas)]
        palavras = (pt + " " + en).split()
        desc = " ".join(rng.choice(palavras, 25)) + f" amostra{i} lote{i % 37}"
        kws = list(dict.fromkeys(rng.choice(en.split(), 3))) + ([f"lote{i % 37}"] if i % 2 else [f"amostra{i}"])
        campos = [] if i % 10 == 0 else [{"typeName": "keyword", "multiple": True, "typeClass": "compound",
                   "value": [{"keywordValue": {"typeName": "keywordValue", "value": k},
                              "keywordVocabulary": {"typeName": "keywordVocabulary", "value": "LCSH"}} for k in kws]}]
        itens.append({"type": "dataset", "name": f"Dataset {i}: " + " ".join(rng.choice(pt.split(), 3)),
                      "global_id": f"doi:FAKE/{i:05d}", "description": f"<p>{desc}</p>",
                      "published_at": "2024-01-01T00:00:00Z",
                      "metadataBlocks": {"citation": {"displayName": "Citation Metadata", "fields": campos}}})

    def get(url, params):
        assert url.endswith("/api/search") and params["metadata_fields"] == "citation:keyword"
        s, p = params["start"], params["per_page"]
        return {"status": "OK", "data": {"total_count": len(itens), "start": s, "items": itens[s:s + p]}}
    return get


def cmd_autoteste() -> None:
    global SAIDA
    from scipy import stats
    SAIDA = Path("saida_AUTOTESTE_sintetico"); CONFIG["per_page"] = 300; CONFIG["n_permutacoes"] = 20_000
    log("== (1) coleta com API simulada (paginação + metadata_fields)")
    cmd_coletar(get=api_falsa(), pausa=False)
    df = pd.read_csv(SAIDA / "redu_metadados.csv")
    assert len(df) == 800 and df["keywords"].map(json.loads).map(len).eq(0).sum() == 80
    log("== (2) experimento (B = TF-IDF, substituto offline)")
    cmd_experimento(b_tfidf=True)
    log("== (3) análise")
    cmd_analisar()
    log("== (4) verificações independentes")
    # (4a) BM25 conferido à mão num corpus mínimo
    docs = ["gato gato cao", "cao peixe", "passaro"]
    bm = BM25(docs, 1.2, 0.75)
    N, dl, avg = 3, np.array([3, 2, 1]), 2.0
    idf_gato = np.log(1 + (N - 1 + .5) / (1 + .5))
    esperado = idf_gato * 2 * 2.2 / (2 + 1.2 * (1 - .75 + .75 * dl[0] / avg))
    assert np.isclose(bm.scores("gato")[0], esperado) and bm.scores("gato")[1] == 0
    log(f"BM25 confere com o cálculo manual ({esperado:.4f})")
    # (4b) posição com empates
    assert posicao(np.array([3., 1., 3., 0.]), 0) == 1.5 and posicao(np.array([0., 0., 0.]), 1) == 2.0
    log("Posição com empates correta (média)")
    # (4c) teste de aleatorização: sistemas idênticos -> p = 1; comparação com t pareado em dados grandes
    r = pd.read_csv(SAIDA / "ranks.csv")
    p_ident = stats.permutation_test((r["rr_A"].to_numpy(), r["rr_A"].to_numpy()), lambda x, y: np.mean(x - y),
                                     permutation_type="samples", n_resamples=2000, random_state=0).pvalue
    assert p_ident == 1.0
    log("Sistemas idênticos -> p = 1 (sem falso positivo)")
    # (4d) McNemar: exato confere com o exemplo da documentação do Pingouin (b=8, c=40 -> p_exact ~ 0,000003)
    m = mcnemar(8, 40)
    assert np.isclose(m["p_exato"], 2 * stats.binom.cdf(8, 48, 0.5)) and round(m["p_exato"], 6) == 0.000003
    # mid-p: fórmula alternativa equivalente 2*(P(X<=k) - 0,5*P(X=k)); mid-p <= exato; b=c -> p=1
    alt = 2 * (stats.binom.cdf(8, 48, .5) - .5 * stats.binom.pmf(8, 48, .5))
    assert np.isclose(m["p_midp"], alt) and m["p_midp"] <= m["p_exato"]
    assert np.isclose(mcnemar(10, 10)["p_midp"], 1.0) and mcnemar(0, 0)["p_midp"] == 1.0  # isclose: arredondamento
    log(f"McNemar confere (exato={m['p_exato']:.2e}, mid-p={m['p_midp']:.2e}; b=c -> p=1)")
    log("AUTOTESTE OK. Números SINTÉTICOS, sem valor científico.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["coletar", "experimento", "analisar", "autoteste"])
    ap.add_argument("--b-tfidf", action="store_true", help="plano B: TF-IDF como método B")
    a = ap.parse_args()
    {"coletar": cmd_coletar, "analisar": cmd_analisar, "autoteste": cmd_autoteste}.get(
        a.etapa, lambda: cmd_experimento(a.b_tfidf))()
