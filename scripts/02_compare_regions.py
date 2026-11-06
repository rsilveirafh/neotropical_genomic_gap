#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compara a cobertura genômica por grupo (Tetrapoda) entre 3 regiões:

- Europe
- NorthAmerica
- Neotropics

Usa os arquivos gerados pelo pipeline multi-região:

    region_outputs/<Região>/sampling_by_group_<Região>_unique.csv

Cada arquivo deve conter as colunas:

    group
    n_species_total
    n_species_with_genome
    sampling_proportion

Os rótulos das barras mostram:

    percentual (número de espécies com genoma / número total de espécies)

Casos em que n_species_total = 0 são tratados como ausentes/não aplicáveis
(NA), e não como 100% de cobertura.

Saídas:

    plot_sampling_by_group_3regions_horizontal.png
    plot_sampling_by_group_3regions_horizontal.pdf
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Ajuste o caminho base se necessário
import sys
from pathlib import Path

# Os prints usam emoji; garante UTF-8 no terminal do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Caminhos relativos à raiz do repositório (o script roda de qualquer pasta)
REPO_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_DIR / "data" / "inputs"
OUTPUT_DIR = REPO_DIR / "outputs"
PATH_BASE = OUTPUT_DIR
BASE_OUTDIR = PATH_BASE / "region_outputs"

# Regiões esperadas, na ordem em que aparecerão na legenda e no gráfico
REGIONS = ["Europe", "NorthAmerica", "Neotropics"]


REQUIRED_COLUMNS = {
    "group",
    "n_species_total",
    "n_species_with_genome",
}


def load_region_group_sampling(region_name: str) -> pd.DataFrame:
    """Lê e valida o CSV de cobertura por grupo para uma região."""

    group_csv_name = f"sampling_by_group_{region_name}_unique.csv"
    path = BASE_OUTDIR / region_name / group_csv_name

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para {region_name}: {path}")

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Arquivo {path} não tem as colunas necessárias: "
            f"{sorted(missing_columns)}. Colunas encontradas: {list(df.columns)}"
        )

    # Garante que as contagens sejam numéricas.
    df["n_species_total"] = pd.to_numeric(
        df["n_species_total"], errors="coerce"
    )
    df["n_species_with_genome"] = pd.to_numeric(
        df["n_species_with_genome"], errors="coerce"
    )

    if df[["n_species_total", "n_species_with_genome"]].isna().any().any():
        raise ValueError(
            f"Arquivo {path} contém valores não numéricos ou ausentes nas "
            "colunas de contagem."
        )

    if (df[["n_species_total", "n_species_with_genome"]] < 0).any().any():
        raise ValueError(f"Arquivo {path} contém contagens negativas.")

    invalid_counts = df["n_species_with_genome"] > df["n_species_total"]
    if invalid_counts.any():
        problematic_groups = df.loc[invalid_counts, "group"].astype(str).tolist()
        raise ValueError(
            f"Arquivo {path} contém grupos com mais espécies com genoma do que "
            f"espécies totais: {problematic_groups}"
        )

    df["region"] = region_name

    # Recalcula a cobertura diretamente das contagens.
    # Quando o grupo não ocorre na região (0/0), usa NaN em vez de 100%.
    df["coverage_pct"] = np.where(
        df["n_species_total"] > 0,
        100.0 * df["n_species_with_genome"] / df["n_species_total"],
        np.nan,
    )

    return df[
        [
            "group",
            "region",
            "n_species_total",
            "n_species_with_genome",
            "coverage_pct",
        ]
    ]


def plot_group_comparison_3regions() -> None:
    """Gera o gráfico horizontal de cobertura genômica entre regiões."""

    dfs = []
    for region in REGIONS:
        try:
            dfs.append(load_region_group_sampling(region))
        except Exception as exc:
            print(f"⚠ Problema ao carregar região {region}: {exc}")

    if not dfs:
        print("❌ Nenhuma região pôde ser carregada. Abortando.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Garante ordem consistente dos grupos e das regiões.
    groups = sorted(df_all["group"].dropna().unique())
    regions = REGIONS

    coverage = df_all.pivot(
        index="group", columns="region", values="coverage_pct"
    ).reindex(index=groups, columns=regions)

    n_total = df_all.pivot(
        index="group", columns="region", values="n_species_total"
    ).reindex(index=groups, columns=regions)

    n_genomes = df_all.pivot(
        index="group", columns="region", values="n_species_with_genome"
    ).reindex(index=groups, columns=regions)

    y = np.arange(len(groups))
    n_regions = len(regions)
    height = 0.8 / n_regions

    fig_height = max(5.0, 1.0 + 0.9 * len(groups))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    for i, region in enumerate(regions):
        values = coverage[region].to_numpy(dtype=float)
        totals = n_total[region].to_numpy(dtype=float)
        genomes = n_genomes[region].to_numpy(dtype=float)

        offset = (i - (n_regions - 1) / 2) * height
        ys = y + offset

        # NaN é convertido apenas para desenhar barras de comprimento zero.
        # O rótulo abaixo deixa explícito que o caso é NA.
        plot_values = np.nan_to_num(values, nan=0.0)
        ax.barh(ys, plot_values, height=height, label=region)

        for yi, pct, total, genome_count in zip(
            ys, values, totals, genomes
        ):
            if np.isnan(total) or np.isnan(genome_count):
                continue

            total_int = int(total)
            genomes_int = int(genome_count)

            if total_int == 0:
                label = "NA (0/0)"
                x_position = 0.6
            else:
                # separador de milhar em inglês (4,636), como pede a GMB
                label = f"{pct:.1f}% ({genomes_int:,}/{total_int:,})"
                #label = f"{pct:.1f}%"
                x_position = pct + 0.6

            ax.text(
                x_position,
                yi,
                label,
                va="center",
                ha="left",
                fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlabel("Species with genomes (%)")
    ax.set_ylabel("Tetrapod group")
    ax.set_title("Genomic coverage by group across regions")
    ax.set_xlim(0, 60)
    ax.legend(title="Region", fontsize=12, title_fontsize=12)

    fig.tight_layout()

    out_png = PATH_BASE / "plot_sampling_by_group_3regions_horizontal.png"
    out_pdf = PATH_BASE / "plot_sampling_by_group_3regions_horizontal.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    # Figura 1 do artigo: TIFF 600 dpi, LZW, RGB (formato pedido pela GMB)
    out_tif = PATH_BASE / "plot_sampling_by_group_3regions_horizontal.tif"
    fig.savefig(out_tif, dpi=600, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    Image.open(out_tif).convert("RGB").save(out_tif, compression="tiff_lzw", dpi=(600, 600))
    print(f"   TIF: {out_tif}")

    print(f"📊 Figura PNG salva em: {out_png}")
    print(f"📄 Figura PDF salva em: {out_pdf}")


if __name__ == "__main__":
    plot_group_comparison_3regions()
