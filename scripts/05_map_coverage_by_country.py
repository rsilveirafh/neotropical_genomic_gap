#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gera um painel 3x2 com mapas de cobertura genômica por país
para os seis grupos de Tetrapoda na região Neotropical.

Entrada esperada (gerada por 01_coverage_by_region.py, em outputs/):
    region_outputs/Neotropics/sampling_country_by_group_Neotropics.csv

Colunas esperadas no CSV:
    - iso2
    - group
    - sampling_proportion

Saídas:
    region_outputs/Neotropics/map_country_group_3x2_Neotropics_quantiles.png
    region_outputs/Neotropics/map_country_group_3x2_Neotropics_quantiles.pdf
    region_outputs/Neotropics/map_country_group_3x2_Neotropics_quantiles.tif (600 dpi)

Observação:
    A escala é baseada em classes fixas de cobertura (%):
        0–8.9
        8.9–20.0
        20.0–29.2
        29.2–42.8
        42.8–100.0
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from shapely.geometry import box


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

import sys
from pathlib import Path

# Os prints usam emoji; garante UTF-8 no terminal do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Caminhos relativos à raiz do repositório (o script roda de qualquer pasta)
REPO_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_DIR / "data" / "inputs"
OUTPUT_DIR = REPO_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "region_outputs" / "Neotropics" / "sampling_country_by_group_Neotropics.csv"

OUT_DIR = OUTPUT_DIR / "region_outputs" / "Neotropics"
OUT_PNG = OUT_DIR / "map_country_group_3x2_Neotropics_quantiles.png"
OUT_PDF = OUT_DIR / "map_country_group_3x2_Neotropics_quantiles.pdf"
OUT_TIF = OUT_DIR / "map_country_group_3x2_Neotropics_quantiles.tif"  # Figura 4 (formato pedido pela GMB)

NEOTROPICAL_ISO2 = {
    "MX", "BZ", "CR", "GT", "HN", "NI", "PA", "SV",
    "CU", "DO", "HT", "PR", "JM", "TT", "BB", "BS",
    "AG", "LC", "GD", "VC", "DM",
    "AR", "BO", "BR", "CL", "CO", "EC", "GF",
    "GY", "PE", "PY", "SR", "UY", "VE",
}

GROUPS_ORDER = [
    "Amphibia", "Aves",
    "Mammalia", "Lepidosauria",
    "Testudines", "Crocodylia",
]

QUANTILE_LABELS = [
    "0.0–8.9%",
    "8.9–20.0%",
    "20.0–29.2%",
    "29.2–42.8%",
    "42.8–100.0%",
]
QUANTILE_EDGES_PCT = [0.0, 8.9, 20.0, 29.2, 42.8, 100.0]

ISO2_TO_ISO3 = {
    "MX": "MEX", "BZ": "BLZ", "CR": "CRI", "GT": "GTM", "HN": "HND",
    "NI": "NIC", "PA": "PAN", "SV": "SLV",
    "CU": "CUB", "DO": "DOM", "HT": "HTI", "PR": None,
    "JM": "JAM", "TT": "TTO", "BB": "BRB", "BS": "BHS",
    "AG": "ATG", "LC": "LCA", "GD": "GRD", "VC": "VCT", "DM": "DMA",
    "AR": "ARG", "BO": "BOL", "BR": "BRA", "CL": "CHL", "CO": "COL",
    "EC": "ECU", "GF": "FRA", "GY": "GUY", "PE": "PER", "PY": "PRY",
    "SR": "SUR", "UY": "URY", "VE": "VEN",
}


# ---------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------

def load_world():
    """Carrega camada de países da Natural Earth usando ADM0_A3."""
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    print(f"🌍 Carregando países de: {url}")

    world = gpd.read_file(url)
    world["iso3"] = world["ADM0_A3"].astype(str).str.upper().str.strip()

    print(f"   → {len(world)} polígonos carregados.")
    return world


def validate_input(df: pd.DataFrame) -> bool:
    required = {"iso2", "group", "sampling_proportion"}
    missing = required.difference(df.columns)

    if missing:
        print(f"❌ ERRO: colunas ausentes no CSV: {sorted(missing)}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        return False

    return True


def assign_fixed_quantile_classes(series: pd.Series) -> pd.Series:
    """
    Converte sampling_proportion (0–1) em classes fixas 1..5,
    com base nos intervalos percentuais definidos em QUANTILE_EDGES_PCT.
    """
    pct = series * 100.0
    bins = np.array(QUANTILE_EDGES_PCT, dtype=float)

    classes = np.digitize(pct, bins[1:-1], right=False) + 1
    classes = np.clip(classes, 1, 5)

    out = pd.Series(classes, index=series.index, dtype="float")
    out[series.isna()] = np.nan
    return out


def build_french_guiana_from_france(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Cria uma feição separada para GF a partir da geometria da França,
    recortando de fato apenas a parte sul-americana e removendo FRA.
    """
    france = world[world["iso3"] == "FRA"].copy()
    if france.empty:
        print("⚠ França (FRA) não encontrada na camada base.")
        return world

    # bounding box aproximada da Guiana Francesa
    gf_bbox = box(-60, 1, -50, 8)

    # recorte real da geometria
    france["geometry"] = france.geometry.intersection(gf_bbox)
    gf = france[~france.geometry.is_empty].copy()

    if gf.empty:
        print("⚠ Recorte espacial de GF a partir de FRA retornou vazio.")
        return world

    # dissolve para virar uma feição só
    gf = gf.dissolve().reset_index(drop=True)
    gf = gpd.GeoDataFrame(gf, geometry="geometry", crs=world.crs)

    gf["iso3"] = "GUF"
    gf["ADM0_A3"] = "GUF"

    # remove FRA original e adiciona GUF
    world_no_france = world[world["iso3"] != "FRA"].copy()
    world2 = pd.concat([world_no_france, gf], ignore_index=True)
    world2 = gpd.GeoDataFrame(world2, geometry="geometry", crs=world.crs)

    print("   → Feição de GF criada a partir da geometria da França.")
    return world2

def plot_maps():
    print("\n📌 Diretório atual:", Path.cwd())
    print("🔎 Verificando arquivo de entrada:", INPUT_CSV)

    if not INPUT_CSV.exists():
        print(f"❌ ERRO: Arquivo não encontrado: {INPUT_CSV}")
        return

    print("📥 Lendo CSV...")
    df = pd.read_csv(INPUT_CSV)

    if not validate_input(df):
        return

    df["iso2"] = df["iso2"].astype(str).str.upper().str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    df["sampling_proportion"] = pd.to_numeric(df["sampling_proportion"], errors="coerce")

    df = df[df["iso2"].isin(NEOTROPICAL_ISO2)].copy()

    if df.empty:
        print("❌ ERRO: Nenhum país neotropical encontrado no CSV.")
        return

    print(f"   → {len(df)} linhas válidas após filtro neotropical.")

    df["coverage_class"] = assign_fixed_quantile_classes(df["sampling_proportion"])
    df["iso3"] = df["iso2"].map(ISO2_TO_ISO3)
    df.loc[df["iso2"] == "GF", "iso3"] = "GUF"

    if df["iso3"].isna().any():
        missing = sorted(df.loc[df["iso3"].isna(), "iso2"].unique())
        print(f"⚠ Alguns ISO2 ficaram sem mapeamento iso3: {missing}")

    # camada de países
    world = load_world()
    world = build_french_guiana_from_france(world)

    neotropical_iso3 = {
        ISO2_TO_ISO3[x]
        for x in NEOTROPICAL_ISO2
        if x in ISO2_TO_ISO3 and ISO2_TO_ISO3[x] is not None
    }
    neotropical_iso3.discard("FRA")
    neotropical_iso3.add("GUF")

    world_nt = world[world["iso3"].isin(neotropical_iso3)].copy()
    world_nt = world_nt[world_nt["iso3"] != "FRA"].copy()

    if world_nt.empty:
        print("❌ ERRO: Nenhum país neotropical encontrado na camada de mapa.")
        return

    print(f"   → {len(world_nt)} países/territórios encontrados na camada de mapa.")

    # colormap discreto
    n_classes = 5
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_classes)))
    bounds = np.arange(0.5, n_classes + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = world_nt.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    pad_x = dx * 0.05
    pad_y = dy * 0.08

    print("🗺 Gerando figura 3x2...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    for ax, grp in zip(axes, GROUPS_ORDER):
        sub = df[df["group"] == grp][["iso2", "iso3", "sampling_proportion", "coverage_class"]].copy()
        gdf = world_nt.merge(sub, on="iso3", how="left")

        # fundo sem dados
        gdf.plot(
            ax=ax,
            color="lightgrey",
            edgecolor="black",
            linewidth=0.4,
            zorder=1,
        )

        # tudo exceto GF
        gdf_main = gdf[(gdf["coverage_class"].notna()) & (gdf["iso3"] != "GUF")].copy()
        if not gdf_main.empty:
            gdf_main.plot(
                column="coverage_class",
                cmap=cmap,
                norm=norm,
                linewidth=0.4,
                ax=ax,
                edgecolor="black",
                zorder=2,
            )

        # GF por último, mesmo contorno do resto
        gdf_gf = gdf[(gdf["coverage_class"].notna()) & (gdf["iso3"] == "GUF")].copy()
        if not gdf_gf.empty:
            gdf_gf.plot(
                column="coverage_class",
                cmap=cmap,
                norm=norm,
                linewidth=0.4,
                ax=ax,
                edgecolor="black",
                zorder=5,
            )

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_title(grp, fontsize=16)
        ax.set_axis_off()

    # colorbar à direita
    cax = fig.add_axes([0.90, 0.18, 0.022, 0.60])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        boundaries=bounds,
        ticks=np.arange(1, n_classes + 1),
    )
    cbar.set_ticklabels(QUANTILE_LABELS)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Relative genomic coverage", fontsize=16)

    # Sem legenda "No data": a legenda da Figura 4 já diz que os países sem dados ficam em cinza.

    plt.tight_layout(rect=[0.02, 0.05, 0.88, 0.96])

    print("💾 Salvando arquivos...")
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_TIF, dpi=600, bbox_inches="tight")
    plt.close()

    print("✅ Figura salva em:")
    print(f"   PNG: {OUT_PNG}")
    print(f"   PDF: {OUT_PDF}")

    # TIFF em RGB, com compressão LZW (600 dpi)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    Image.open(OUT_TIF).convert("RGB").save(OUT_TIF, compression="tiff_lzw", dpi=(600, 600))
    print(f"   TIF: {OUT_TIF}\n")


if __name__ == "__main__":
    plot_maps()