#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compara a cobertura genômica por grupo (Tetrapoda) entre 3 regiões:

- Neotropics
- Europe
- NorthAmerica

Usa os arquivos gerados pelo pipeline multi-região:

    region_outputs/<Região>/sampling_by_group_neotropics_unique.csv

Saída:
    plot_sampling_by_group_3regions.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajuste o caminho base se necessário
PATH_BASE = Path(".")
BASE_OUTDIR = PATH_BASE / "region_outputs"

# Regiões esperadas (pastas geradas pelo pipeline)
REGIONS = ["Europe", "NorthAmerica", "Neotropics"]

def load_region_group_sampling(region_name: str) -> pd.DataFrame:
    """
    Lê o CSV de cobertura por grupo para uma região, adicionando coluna 'region'.
    Espera colunas:
        - group
        - sampling_proportion  (0–1)

    Usa arquivos no formato:
        region_outputs/<Região>/sampling_by_group_<Região>_unique.csv
    """
    group_csv_name = f"sampling_by_group_{region_name}_unique.csv"
    path = BASE_OUTDIR / region_name / group_csv_name

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para {region_name}: {path}")

    df = pd.read_csv(path)
    if "group" not in df.columns or "sampling_proportion" not in df.columns:
        raise ValueError(
            f"Arquivo {path} não tem as colunas esperadas "
            f"('group', 'sampling_proportion'). Colunas: {list(df.columns)}"
        )

    df["region"] = region_name
    df["coverage_pct"] = df["sampling_proportion"] * 100.0
    return df[["group", "region", "coverage_pct"]]

def plot_group_comparison_3regions():
    # Carrega todos
    dfs = []
    for reg in REGIONS:
        try:
            dfs.append(load_region_group_sampling(reg))
        except Exception as e:
            print(f"⚠ Problema ao carregar região {reg}: {e}")
    if not dfs:
        print("❌ Nenhuma região pôde ser carregada. Abortando.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Garante ordem consistente dos grupos
    groups = sorted(df_all["group"].unique())
    regions = REGIONS  # mantém a ordem declarada lá em cima

    # Pivot para ter tabela group × region
    pivot = df_all.pivot(index="group", columns="region", values="coverage_pct")
    pivot = pivot.reindex(index=groups, columns=regions)

    # ----- Gráfico horizontal -----
    y = np.arange(len(groups))  # posição central de cada grupo no eixo Y
    n_regions = len(regions)
    height = 0.8 / n_regions    # altura de cada barrinha dentro do “bloco” do grupo

    plt.figure(figsize=(9, 5))

    for i, reg in enumerate(regions):
        if reg not in pivot.columns:
            continue
        vals = pivot[reg].values

        # desloca cada região em torno da posição central y
        offset = (i - (n_regions - 1) / 2) * height
        ys = y + offset

        plt.barh(ys, vals, height=height, label=reg)

        # rótulos numéricos à direita de cada barra
        for yi, xi in zip(ys, vals):
            if np.isnan(xi):
                continue
            plt.text(
                xi + 0.5,       # um pouquinho à direita da barra
                yi,
                f"{xi:.1f}",
                va="center",
                ha="left",
                fontsize=7,
            )

    plt.yticks(y, groups)
    plt.gca().invert_yaxis()
    plt.xlabel("Species with genomes (%)")
    plt.ylabel("Tetrapod group")
    plt.title("Genomic coverage by group across regions")
    plt.legend(title="Region", fontsize=8, title_fontsize=9)
    plt.tight_layout()

    out_png = PATH_BASE / "plot_sampling_by_group_3regions_horizontal.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"📊 Figura comparativa entre regiões (horizontal) salva em: {out_png}")


if __name__ == "__main__":
    plot_group_comparison_3regions()
