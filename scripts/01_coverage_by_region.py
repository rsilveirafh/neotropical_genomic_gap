#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline de cobertura genômica por região (versão OFFLINE).

Entradas esperadas (em data/inputs/):

- families_seen_live.csv
    Colunas necessárias:
        iso2, speciesKey, group, count (count não é usada)

- species_country_filtered_clean.tsv
    Colunas necessárias:
        usageKey, parent_name, iso2, species (species opcional)

- species_family_cache.csv
    Colunas necessárias:
        speciesKey, family

- famscan_all_assemblies_filtered.tsv
    Colunas esperadas (pelo menos):
        Organism Name, Assembly Level, Assembly Release Date

- representativity_by_group.csv
    Colunas esperadas:
        group, n_species_with_genome, n_gbif_species
    (cobertura_global = n_species_with_genome / n_gbif_species)

Saídas (em outputs/region_outputs/<Região>/):

Por região:
- sampling_by_group_<Região>_unique.csv
- sampling_by_family_<Região>_unique.csv
- sampling_country_by_group_<Região>.csv
- sampling_country_by_family_<Região>.csv

Figuras:
- plot_sampling_by_group_<Região>.png
- plot_sampling_by_group_neotropics_vs_global.png  (apenas para Neotropics)
- plot_sampling_by_family_<GROUP>.png
- plot_top10_families_by_group_2x2_<Região>.png
- heatmap_country_group_<Região>.png
- scatter_family_richness_vs_coverage_<GROUP>_<Região>.png
- assembly_composition_by_group_<Região>.png
- cumulative_genomes_over_time_<Região>.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Os prints usam emoji; garante UTF-8 no terminal do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# 1. Parâmetros gerais
# =============================================================================

# Caminhos relativos à raiz do repositório (o script roda de qualquer pasta)
REPO_DIR = Path(__file__).resolve().parents[1]
PATH_BASE = REPO_DIR / "data" / "inputs"
OUTPUT_DIR = REPO_DIR / "outputs"

# Países da região Neotropical (ISO2)
NEOTROPICAL_ISO2 = {
    # Mesoamérica
    "MX", "BZ", "CR", "GT", "HN", "NI", "PA", "SV",
    # Caribe
    "CU", "DO", "HT", "PR", "JM", "TT", "BB", "BS",
    "AG", "LC", "GD", "VC", "DM",
    # América do Sul
    "AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF",
    "GY", "PE", "PY", "SR", "UY", "VE",
}

# Europa (ajuste se quiser outra definição)
EUROPE_ISO2 = {
    "AL","AD","AM","AT","AZ","BY","BE","BA","BG","HR","CY","CZ","DK","EE","FI",
    "FR","GE","DE","GR","HU","IS","IE","IT","KZ","XK","LV","LI","LT","LU","MT",
    "MD","MC","ME","NL","MK","NO","PL","PT","RO","RU","SM","RS","SK","SI","ES",
    "SE","CH","TR","UA","VA","GB","UK"
}

# América do Norte (versão simples)
NORTH_AMERICA_ISO2 = {
    "US", "CA", "MX",  # principais
    "GL", "BM"         # opcionais
}

# Catálogo de regiões
REGION_DEFS = {
    "Neotropics": NEOTROPICAL_ISO2,
    "Europe": EUROPE_ISO2,
    "NorthAmerica": NORTH_AMERICA_ISO2,
}

# Grupos de Tetrapoda
TETRAPOD_GROUPS = [
    "Amphibia",
    "Aves",
    "Mammalia",
    "Lepidosauria",
    "Testudines",
    "Crocodylia",
]

BASE_OUTDIR = OUTPUT_DIR / "region_outputs"
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

# Será ajustado dentro de run_region_pipeline
OUTDIR = BASE_OUTDIR


# =============================================================================
# 2. Funções de utilidade
# =============================================================================

def load_inputs():
    """Carrega os arquivos de entrada principais, garantindo tipos."""
    families = pd.read_csv(PATH_BASE / "families_seen_live.csv")
    sp_gen = pd.read_csv(PATH_BASE / "species_country_filtered_clean.tsv", sep="\t")
    fam_cache = pd.read_csv(PATH_BASE / "species_family_cache.csv")

    # Garantir tipos inteiros para IDs de espécie
    families["speciesKey"] = families["speciesKey"].astype("Int64")
    sp_gen["usageKey"] = sp_gen["usageKey"].astype("Int64")
    fam_cache["speciesKey"] = fam_cache["speciesKey"].astype("Int64")

    return families, sp_gen, fam_cache


def load_assemblies():
    """
    Lê o arquivo famscan_all_assemblies_filtered.tsv com info de nível de montagem.
    Retorna um DataFrame ou None se não encontrar.
    """
    path = PATH_BASE / "famscan_all_assemblies_filtered.tsv"
    if not path.exists():
        print("⚠ Arquivo 'famscan_all_assemblies_filtered.tsv' não encontrado. "
              "Gráficos por família e séries temporais serão gerados sem info de montagem/data.")
        return None

    assemblies = pd.read_csv(path, sep="\t")
    return assemblies


def simplify_assembly_level(raw):
    """
    Converte o texto original de 'Assembly Level' em uma das três categorias:
    'Contig', 'Scaffold' ou 'Chromosome'.

    'Complete Genome' é tratado como 'Chromosome'.
    """
    if not isinstance(raw, str):
        return None
    lvl = raw.strip().lower()
    if "contig" in lvl:
        return "Contig"
    if "scaffold" in lvl:
        return "Scaffold"
    if "chromosome" in lvl or "complete genome" in lvl:
        return "Chromosome"
    return None


def add_family_assembly_breakdown(
    fam_sampling_df: pd.DataFrame,
    sp_gen: pd.DataFrame,
    fam_cache: pd.DataFrame,
    assemblies: pd.DataFrame | None,
    region_iso2: set,
) -> pd.DataFrame:
    """
    Adiciona ao DataFrame de famílias (fam_sampling_df) três colunas:

    - n_genome_contig
    - n_genome_scaffold
    - n_genome_chromosome

    representando, para cada (group, family), quantas espécies com genoma
    na região têm como melhor montagem: contig, scaffold ou chromosome.

    Se o arquivo de assemblies ou as colunas necessárias não existirem,
    retorna fam_sampling_df inalterado.
    """
    if assemblies is None:
        return fam_sampling_df

    if "species" not in sp_gen.columns:
        print("⚠ Coluna 'species' não encontrada em species_country_filtered_clean.tsv. "
              "Não será possível cruzar assemblies por espécie. "
              "Gráficos por família serão gerados sem categorias de montagem.")
        return fam_sampling_df

    # 1) Espécies da região com genoma (sp_gen + fam_cache)
    sp_nt = sp_gen[
        (sp_gen["iso2"].isin(region_iso2)) &
        (sp_gen["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    sp_nt = sp_nt.merge(
        fam_cache,
        left_on="usageKey",
        right_on="speciesKey",
        how="left",
        validate="many_to_one",
    )

    if "family" not in sp_nt.columns:
        print("⚠ Coluna 'family' não encontrada após merge com species_family_cache.csv. "
              "Não será possível calcular categorias de montagem por família.")
        return fam_sampling_df

    sp_nt = sp_nt.dropna(subset=["family", "species"])

    # Um registro por espécie + grupo + família (independente do país)
    sp_nt_unique = (
        sp_nt[["species", "parent_name", "family"]]
        .drop_duplicates()
        .rename(columns={"parent_name": "group"})
    )

    # 2) Melhor nível de montagem por espécie no arquivo de assemblies
    if "Organism Name" not in assemblies.columns or "Assembly Level" not in assemblies.columns:
        print("⚠ Colunas 'Organism Name' ou 'Assembly Level' não encontradas em "
              "famscan_all_assemblies_filtered.tsv. "
              "Gráficos por família serão gerados sem categorias de montagem.")
        return fam_sampling_df

    asm = assemblies.copy()
    asm["assembly_category"] = asm["Assembly Level"].apply(simplify_assembly_level)
    asm["species"] = asm["Organism Name"].astype(str).str.strip()

    # Ranking para escolher a "melhor" montagem por espécie
    rank_map = {"Contig": 1, "Scaffold": 2, "Chromosome": 3}
    asm["level_rank"] = asm["assembly_category"].map(rank_map).fillna(0)

    asm_best = (
        asm.sort_values("level_rank")
        .drop_duplicates(subset=["species"], keep="last")
        [["species", "assembly_category"]]
    )

    # 3) Junta espécies da região + melhor nível de montagem
    sp_asm = sp_nt_unique.merge(
        asm_best,
        on="species",
        how="left",
    )

    sp_asm = sp_asm.dropna(subset=["assembly_category"])

    if sp_asm.empty:
        print("⚠ Nenhuma espécie da região com genoma pôde ser associada a um nível "
              "de montagem. Gráficos por família serão gerados sem categorias.")
        return fam_sampling_df

    agg = (
        sp_asm
        .groupby(["group", "family", "assembly_category"])["species"]
        .nunique()
        .rename("n_species_with_genome_asm")
        .reset_index()
    )

    pivot = agg.pivot_table(
        index=["group", "family"],
        columns="assembly_category",
        values="n_species_with_genome_asm",
        fill_value=0,
    ).reset_index()

    for cat in ["Contig", "Scaffold", "Chromosome"]:
        if cat not in pivot.columns:
            pivot[cat] = 0

    pivot = pivot.rename(columns={
        "Contig": "n_genome_contig",
        "Scaffold": "n_genome_scaffold",
        "Chromosome": "n_genome_chromosome",
    })

    fam_extended = fam_sampling_df.merge(
        pivot,
        on=["group", "family"],
        how="left",
    )

    for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]:
        if col not in fam_extended.columns:
            fam_extended[col] = 0
        else:
            fam_extended[col] = fam_extended[col].fillna(0).astype(int)

    return fam_extended


# =============================================================================
# 3. Cobertura regional por grupo (genérica por região)
# =============================================================================

def compute_group_sampling(
    families: pd.DataFrame,
    sp_gen: pd.DataFrame,
    region_iso2: set,
    region_label: str,
) -> pd.DataFrame:
    """
    Calcula a proporção de espécies com genoma por grupo em uma região.
    """
    fam_nt = families[
        (families["iso2"].isin(region_iso2)) &
        (families["group"].isin(TETRAPOD_GROUPS))
    ].copy()

    denom = (
        fam_nt
        .groupby("group")["speciesKey"]
        .nunique()
        .rename("n_species_total")
        .reset_index()
    )

    sp_nt = sp_gen[
        (sp_gen["iso2"].isin(region_iso2)) &
        (sp_gen["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    num = (
        sp_nt
        .groupby("parent_name")["usageKey"]
        .nunique()
        .rename("n_species_with_genome")
        .reset_index()
        .rename(columns={"parent_name": "group"})
    )

    sampling = denom.merge(num, on="group", how="left")
    sampling["n_species_with_genome"] = sampling["n_species_with_genome"].fillna(0).astype(int)
    sampling["sampling_proportion"] = (
        sampling["n_species_with_genome"] / sampling["n_species_total"]
    )

    sampling = sampling.sort_values("group").reset_index(drop=True)

    out_csv = OUTDIR / f"sampling_by_group_{region_label}_unique.csv"
    sampling.to_csv(out_csv, index=False)
    print("✅ Cobertura regional por grupo salva em:", out_csv)

    return sampling


# =============================================================================
# 4. Cobertura regional por família dentro de cada grupo
# =============================================================================

def compute_family_sampling(
    families: pd.DataFrame,
    sp_gen: pd.DataFrame,
    fam_cache: pd.DataFrame,
    region_iso2: set,
    region_label: str,
) -> pd.DataFrame:
    """
    Calcula a proporção de espécies com genoma por família dentro de cada grupo
    em uma região, usando species_family_cache.csv.
    """

    fam_nt = families[
        (families["iso2"].isin(region_iso2)) &
        (families["group"].isin(TETRAPOD_GROUPS))
    ].copy()

    fam_nt = fam_nt.merge(
        fam_cache,
        on="speciesKey",
        how="left",
        validate="many_to_one",
    )

    denom = (
        fam_nt
        .dropna(subset=["family"])
        .drop_duplicates(subset=["speciesKey", "family", "group"])
        .groupby(["group", "family"])["speciesKey"]
        .nunique()
        .rename("n_species_total")
        .reset_index()
    )

    sp_nt = sp_gen[
        (sp_gen["iso2"].isin(region_iso2)) &
        (sp_gen["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    sp_nt = sp_nt.merge(
        fam_cache,
        left_on="usageKey",
        right_on="speciesKey",
        how="left",
        validate="many_to_one",
    )

    num = (
        sp_nt
        .dropna(subset=["family"])
        .drop_duplicates(subset=["usageKey", "family", "parent_name"])
        .groupby(["parent_name", "family"])["usageKey"]
        .nunique()
        .rename("n_species_with_genome")
        .reset_index()
        .rename(columns={"parent_name": "group"})
    )

    fam_sampling = denom.merge(num, on=["group", "family"], how="left")
    fam_sampling["n_species_with_genome"] = (
        fam_sampling["n_species_with_genome"].fillna(0).astype(int)
    )
    fam_sampling["sampling_proportion"] = (
        fam_sampling["n_species_with_genome"] / fam_sampling["n_species_total"]
    )

    fam_sampling = fam_sampling.sort_values(["group", "family"]).reset_index(drop=True)

    out_csv = OUTDIR / f"sampling_by_family_{region_label}_unique.csv"
    fam_sampling.to_csv(out_csv, index=False)
    print("✅ Cobertura regional por família salva em:", out_csv)

    return fam_sampling


# =============================================================================
# 5. Cobertura por país × grupo
# =============================================================================

def compute_country_group_sampling(
    families: pd.DataFrame,
    sp_gen: pd.DataFrame,
    region_iso2: set,
    region_label: str,
) -> pd.DataFrame:
    """
    Calcula, para cada país da região e grupo:
    - n_species_total: espécies únicas observadas no país (por grupo)
    - n_species_with_genome: espécies com genoma que ocorrem no país (por grupo)
    - sampling_proportion: proporção de cobertura por grupo e país
    """

    fam_nt = families[
        (families["iso2"].isin(region_iso2)) &
        (families["group"].isin(TETRAPOD_GROUPS))
    ].copy()

    denom = (
        fam_nt
        .groupby(["iso2", "group"])["speciesKey"]
        .nunique()
        .rename("n_species_total")
        .reset_index()
    )

    sp_nt = sp_gen[
        (sp_gen["iso2"].isin(region_iso2)) &
        (sp_gen["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    num = (
        sp_nt
        .groupby(["iso2", "parent_name"])["usageKey"]
        .nunique()
        .rename("n_species_with_genome")
        .reset_index()
        .rename(columns={"parent_name": "group"})
    )

    df_country_group = denom.merge(num, on=["iso2", "group"], how="left")
    df_country_group["n_species_with_genome"] = (
        df_country_group["n_species_with_genome"].fillna(0).astype(int)
    )
    df_country_group["sampling_proportion"] = (
        df_country_group["n_species_with_genome"] / df_country_group["n_species_total"]
    )

    df_country_group = df_country_group.sort_values(["iso2", "group"]).reset_index(drop=True)

    out_csv = OUTDIR / f"sampling_country_by_group_{region_label}.csv"
    df_country_group.to_csv(out_csv, index=False)
    print("✅ Cobertura por país × grupo salva em:", out_csv)

    return df_country_group


# =============================================================================
# 6. Cobertura por país × família
# =============================================================================

def compute_country_family_sampling(
    families: pd.DataFrame,
    sp_gen: pd.DataFrame,
    fam_cache: pd.DataFrame,
    region_iso2: set,
    region_label: str,
) -> pd.DataFrame:
    """
    Calcula, para cada país da região, grupo e família:
    - n_species_total
    - n_species_with_genome
    - sampling_proportion
    """

    fam_nt = families[
        (families["iso2"].isin(region_iso2)) &
        (families["group"].isin(TETRAPOD_GROUPS))
    ].copy()

    fam_nt = fam_nt.merge(
        fam_cache,
        on="speciesKey",
        how="left",
        validate="many_to_one",
    )

    denom = (
        fam_nt
        .dropna(subset=["family"])
        .drop_duplicates(subset=["iso2", "speciesKey", "group", "family"])
        .groupby(["iso2", "group", "family"])["speciesKey"]
        .nunique()
        .rename("n_species_total")
        .reset_index()
    )

    sp_nt = sp_gen[
        (sp_gen["iso2"].isin(region_iso2)) &
        (sp_gen["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    sp_nt = sp_nt.merge(
        fam_cache,
        left_on="usageKey",
        right_on="speciesKey",
        how="left",
        validate="many_to_one",
    )

    num = (
        sp_nt
        .dropna(subset=["family"])
        .drop_duplicates(subset=["iso2", "usageKey", "parent_name", "family"])
        .groupby(["iso2", "parent_name", "family"])["usageKey"]
        .nunique()
        .rename("n_species_with_genome")
        .reset_index()
        .rename(columns={"parent_name": "group"})
    )

    df_country_fam = denom.merge(
        num, on=["iso2", "group", "family"], how="left"
    )
    df_country_fam["n_species_with_genome"] = (
        df_country_fam["n_species_with_genome"].fillna(0).astype(int)
    )
    df_country_fam["sampling_proportion"] = (
        df_country_fam["n_species_with_genome"] / df_country_fam["n_species_total"]
    )

    df_country_fam = df_country_fam.sort_values(["iso2", "group", "family"]).reset_index(drop=True)

    out_csv = OUTDIR / f"sampling_country_by_family_{region_label}.csv"
    df_country_fam.to_csv(out_csv, index=False)
    print("✅ Cobertura por país × família salva em:", out_csv)

    return df_country_fam


# =============================================================================
# 7. Gráficos básicos
# =============================================================================

def plot_group_sampling(sampling_df: pd.DataFrame, region_label: str):
    """Gráfico de barras horizontal da proporção de espécies com genoma por grupo."""
    plt.figure(figsize=(8, 5))

    y = sampling_df["group"]
    x = sampling_df["sampling_proportion"] * 100.0

    plt.barh(y, x)

    for xi, yi in zip(x, y):
        plt.text(xi + 0.3, yi, f"{xi:.1f}%", va="center", fontsize=9)

    plt.xlabel("Species with genomes (%)")
    plt.ylabel("Group")
    plt.title(f"Genomic coverage of Tetrapod groups – {region_label}")
    plt.tight_layout()

    out_png = OUTDIR / f"plot_sampling_by_group_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Gráfico por grupo salvo em:", out_png)


def plot_group_sampling_neotropics_vs_global(
    sampling_neotropics: pd.DataFrame,
    global_group_stats: pd.DataFrame | None,
):
    """
    Gera uma figura comparando a cobertura (%) por grupo:
    - Neotropics (sampling_neotropics)
    - Global (global_group_stats)
    """
    if global_group_stats is None or global_group_stats.empty:
        print("⚠ Dados globais ausentes/vazios. Pulando figura Neotropical vs Global.")
        return

    df = sampling_neotropics[["group", "sampling_proportion"]].copy()
    df = df.merge(global_group_stats[["group", "sampling_global"]], on="group", how="inner")

    if df.empty:
        print("⚠ Não há interseção de grupos entre Neotropics e dados globais.")
        return

    groups = df["group"].tolist()
    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(8, 5))

    neo_pct = df["sampling_proportion"] * 100.0
    glob_pct = df["sampling_global"] * 100.0

    plt.bar(x - width / 2, neo_pct, width, label="Neotropics")
    plt.bar(x + width / 2, glob_pct, width, label="Global")

    plt.xticks(x, groups, rotation=45, ha="right")
    plt.ylabel("Species with genomes (%)")
    plt.title("Genomic coverage by group: Neotropics vs Global")

    for xi, yi in zip(x - width / 2, neo_pct):
        plt.text(xi, yi + 0.5, f"{yi:.1f}", ha="center", va="bottom", fontsize=7)
    for xi, yi in zip(x + width / 2, glob_pct):
        plt.text(xi, yi + 0.5, f"{yi:.1f}", ha="center", va="bottom", fontsize=7)

    plt.legend(fontsize=8)
    plt.tight_layout()

    out_png = OUTDIR / "plot_sampling_by_group_neotropics_vs_global.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Gráfico Neotropics vs Global salvo em:", out_png)


def plot_family_sampling(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    min_species: int = 10,
):
    """
    Gera gráficos de barras horizontais da proporção de espécies com genoma
    por família, separados por grupo. Salva um PNG por grupo.

    Se as colunas de categorias de montagem (n_genome_contig, n_genome_scaffold,
    n_genome_chromosome) estiverem presentes, o gráfico é empilhado por nível
    de montagem. Caso contrário, mantém o gráfico simples com uma barra única.

    min_species: número mínimo de espécies na família para incluí-la.
    """
    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    for grp in TETRAPOD_GROUPS:
        sub = fam_sampling_df[
            (fam_sampling_df["group"] == grp) &
            (fam_sampling_df["n_species_total"] >= min_species)
        ].copy()

        if sub.empty:
            print(f"⚠ Nenhuma família com >= {min_species} espécies em {grp}. Pulando.")
            continue

        sub = sub.sort_values("sampling_proportion", ascending=False)

        plt.figure(figsize=(10, max(4, 0.3 * len(sub))))
        y = sub["family"]

        if has_assembly_cols:
            total = sub["n_species_total"].astype(float)

            x_contig = (sub["n_genome_contig"] / total) * 100.0
            x_scaffold = (sub["n_genome_scaffold"] / total) * 100.0
            x_chrom = (sub["n_genome_chromosome"] / total) * 100.0

            x_contig = x_contig.fillna(0.0)
            x_scaffold = x_scaffold.fillna(0.0)
            x_chrom = x_chrom.fillna(0.0)

            plt.barh(y, x_contig, label="Contig")
            plt.barh(y, x_scaffold, left=x_contig, label="Scaffold")
            plt.barh(y, x_chrom, left=x_contig + x_scaffold, label="Chromosome")

            x_total = sub["sampling_proportion"] * 100.0

            for xi, yi, n_with, n_total in zip(
                x_total, y, sub["n_species_with_genome"], sub["n_species_total"]
            ):
                plt.text(
                    xi + 0.3,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=8
                )

            plt.legend(title="Assembly level", fontsize=8, title_fontsize=9)

        else:
            x = sub["sampling_proportion"] * 100.0
            plt.barh(y, x)
            for xi, yi, n_with, n_total in zip(
                x, y, sub["n_species_with_genome"], sub["n_species_total"]
            ):
                plt.text(
                    xi + 0.3,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=8
                )

        plt.xlabel("Species with genomes (%)")
        plt.ylabel("Family")
        plt.title(f"Genomic coverage by family in {grp} ({region_label})")
        plt.tight_layout()

        out_png = OUTDIR / f"plot_sampling_by_family_{grp}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"📊 Gráfico por família ({grp}) salvo em:", out_png)


def plot_top10_families_panels(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    n_top: int = 10,
    min_species: int = 10,
):
    """
    Figura 2x2 com as top N famílias (% de espécies com genoma)
    para Amphibia, Aves, Mammalia, Lepidosauria.
    """
    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    groups_for_panels = ["Amphibia", "Aves", "Mammalia", "Lepidosauria"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for idx, grp in enumerate(groups_for_panels):
        ax = axes[idx]

        sub = fam_sampling_df[
            (fam_sampling_df["group"] == grp) &
            (fam_sampling_df["n_species_total"] >= min_species)
        ].copy()

        if sub.empty:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", fontsize=10)
            ax.set_title(grp)
            ax.set_yticks([])
            continue

        sub = sub.sort_values("sampling_proportion", ascending=False).head(n_top)

        y_labels = sub["family"]
        y_pos = np.arange(len(sub))

        if has_assembly_cols:
            total = sub["n_species_total"].astype(float)

            x_contig = (sub["n_genome_contig"] / total) * 100.0
            x_scaffold = (sub["n_genome_scaffold"] / total) * 100.0
            x_chrom = (sub["n_genome_chromosome"] / total) * 100.0

            x_contig = x_contig.fillna(0.0)
            x_scaffold = x_scaffold.fillna(0.0)
            x_chrom = x_chrom.fillna(0.0)

            ax.barh(y_pos, x_contig, label="Contig")
            ax.barh(y_pos, x_scaffold, left=x_contig, label="Scaffold")
            ax.barh(y_pos, x_chrom, left=x_contig + x_scaffold, label="Chromosome")

            x_total = sub["sampling_proportion"] * 100.0

            for xi, yi, n_with, n_total in zip(
                x_total, y_pos,
                sub["n_species_with_genome"],
                sub["n_species_total"]
            ):
                ax.text(
                    xi + 0.3,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=7
                )
        else:
            x = sub["sampling_proportion"] * 100.0
            ax.barh(y_pos, x)
            for xi, yi, n_with, n_total in zip(
                x, y_pos,
                sub["n_species_with_genome"],
                sub["n_species_total"]
            ):
                ax.text(
                    xi + 0.3,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=7
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_title(grp)

    if has_assembly_cols:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Assembly levels",
                   fontsize=8, title_fontsize=9,
                   loc="upper right")

    fig.supylabel("Family")
    fig.supxlabel("Species with genomes (%)")
    fig.suptitle(
        f"Top 10 families by genomic coverage in each group ({region_label})",
        fontsize=12
    )
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

    out_png = OUTDIR / f"plot_top10_families_by_group_2x2_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Figura 2x2 de top 10 famílias por grupo salva em:", out_png)



def plot_richness_vs_coverage_scatter(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    min_species: int = 10,
):
    """
    Gráficos de dispersão por grupo:
    - X: riqueza total da família (n_species_total)
    - Y: % de espécies com genoma
    - Cor: fração dos genomas em nível Chromosome (entre as espécies com genoma)
    """
    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    for grp in TETRAPOD_GROUPS:
        sub = fam_sampling_df[
            (fam_sampling_df["group"] == grp) &
            (fam_sampling_df["n_species_total"] >= min_species)
        ].copy()

        if sub.empty:
            print(f"⚠ Nenhuma família com >= {min_species} espécies em {grp} para o scatter. Pulando.")
            continue

        x = sub["n_species_total"]
        y = sub["sampling_proportion"] * 100.0

        plt.figure(figsize=(7, 5))

        if has_assembly_cols:
            with_genome = sub["n_species_with_genome"].replace(0, np.nan)
            frac_chrom = (sub["n_genome_chromosome"] / with_genome).fillna(0.0)
            sc = plt.scatter(x, y, c=frac_chrom, s=40)
            cbar = plt.colorbar(sc)
            cbar.set_label("Fraction of genomes at Chromosome level")
        else:
            plt.scatter(x, y, s=40)

        for xi, yi, fam in zip(x, y, sub["family"]):
            plt.text(
                xi,
                yi,
                fam,
                fontsize=7,
                alpha=0.7,
                rotation=25,
                ha="left",
                va="bottom"
            )

        plt.xlabel("Species richness in region")
        plt.ylabel("Species with genomes (%)")
        plt.title(f"Richness × genomic coverage – {grp} ({region_label})")
        plt.tight_layout()

        out_png = OUTDIR / f"scatter_family_richness_vs_coverage_{grp}_{region_label}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"📊 Scatter riqueza × cobertura ({grp}, {region_label}) salvo em:", out_png)


def plot_assembly_composition_by_group(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
):
    """
    Gráfico de barras empilhadas por grupo mostrando, entre as espécies com genoma:
    - % dos genomas em Contig
    - % em Scaffold
    - % em Chromosome
    (agregado sobre todas as famílias de cada grupo)
    """
    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    if not has_assembly_cols:
        print("⚠ Não há colunas de nível de montagem nas famílias. "
              "Pulando gráfico de composição de montagem por grupo.")
        return

    rows = []
    for grp in TETRAPOD_GROUPS:
        sub = fam_sampling_df[fam_sampling_df["group"] == grp].copy()
        if sub.empty:
            continue

        n_contig = sub["n_genome_contig"].sum()
        n_scaf = sub["n_genome_scaffold"].sum()
        n_chrom = sub["n_genome_chromosome"].sum()
        total = n_contig + n_scaf + n_chrom

        if total == 0:
            continue

        rows.append({
            "group": grp,
            "pct_contig": 100.0 * n_contig / total,
            "pct_scaffold": 100.0 * n_scaf / total,
            "pct_chromosome": 100.0 * n_chrom / total,
        })

    if not rows:
        print("⚠ Nenhum grupo com genomas para o gráfico de composição de montagem.")
        return

    df = pd.DataFrame(rows).set_index("group")

    plt.figure(figsize=(7, 5))
    x_pos = np.arange(len(df.index))

    bottom_contig = df["pct_contig"]
    bottom_scaf = bottom_contig + df["pct_scaffold"]

    plt.bar(x_pos, df["pct_contig"], label="Contig")
    plt.bar(x_pos, df["pct_scaffold"], bottom=bottom_contig, label="Scaffold")
    plt.bar(x_pos, df["pct_chromosome"], bottom=bottom_scaf, label="Chromosome")

    plt.xticks(x_pos, df.index, rotation=45, ha="right")
    plt.ylabel("Share of genomes (%)")
    plt.title(f"Assembly-level composition of genomes by group ({region_label})")
    plt.legend(title="Assembly level", fontsize=8, title_fontsize=9)
    plt.tight_layout()

    out_png = OUTDIR / f"assembly_composition_by_group_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Gráfico de composição de montagem por grupo salvo em:", out_png)


def plot_cumulative_genomes_over_time(
    sp_gen: pd.DataFrame,
    assemblies: pd.DataFrame,
    region_iso2: set,
    region_label: str,
):
    """
    Série temporal mostrando o número acumulado de ESPÉCIES com genoma ao longo do tempo,
    para espécies da região (iso2 ∈ region_iso2) de Tetrapoda.

    - Cada espécie é contada uma única vez, na data do seu PRIMEIRO assembly.
    - Homo sapiens é excluído explicitamente.
    """
    if assemblies is None:
        print("⚠ Sem arquivo de assemblies carregado. Não será possível gerar o gráfico "
              "de genomas acumulados no tempo.")
        return

    date_col = "Assembly Release Date"
    if date_col not in assemblies.columns:
        print(f"⚠ Coluna '{date_col}' não encontrada em famscan_all_assemblies_filtered.tsv. "
              f"Colunas disponíveis: {list(assemblies.columns)}")
        return

    print(f"🕒 Usando coluna de data '{date_col}' para o gráfico cumulativo de genomas ({region_label}).")

    if "species" not in sp_gen.columns:
        print("⚠ Coluna 'species' não encontrada em species_country_filtered_clean.tsv. "
              "Não será possível cruzar com assemblies para o gráfico cumulativo.")
        return

    sp_nt = sp_gen[
        (sp_gen["iso2"].isin(region_iso2)) &
        (sp_gen["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    if sp_nt.empty:
        print(f"⚠ Nenhuma espécie de Tetrapoda encontrada em sp_gen para {region_label}. "
              "Pulando gráfico cumulativo de genomas.")
        return

    sp_nt_unique = (
        sp_nt[["species", "parent_name"]]
        .drop_duplicates()
        .rename(columns={"parent_name": "group"})
    )

    if "Organism Name" not in assemblies.columns:
        print("⚠ Coluna 'Organism Name' não encontrada em famscan_all_assemblies_filtered.tsv. "
              "Não será possível cruzar assemblies por espécie.")
        return

    asm = assemblies.copy()
    asm["species"] = asm["Organism Name"].astype(str).str.strip()

    asm_neo = asm.merge(
        sp_nt_unique,
        on="species",
        how="inner",
    )

    if asm_neo.empty:
        print(f"⚠ Nenhum assembly pôde ser associado a espécies da região ({region_label}). "
              "Pulando gráfico cumulativo de genomas.")
        return

    asm_neo[date_col] = pd.to_datetime(asm_neo[date_col], errors="coerce")
    asm_neo = asm_neo.dropna(subset=[date_col])

    if asm_neo.empty:
        print("⚠ Todas as datas de assembly ficaram inválidas após conversão. "
              "Verifique o formato da coluna de data.")
        return

    asm_neo = asm_neo[asm_neo["species"] != "Homo sapiens"]

    species_first = (
        asm_neo
        .groupby(["group", "species"])[date_col]
        .min()
        .reset_index()
        .rename(columns={date_col: "first_date"})
    )

    df = (
        species_first
        .groupby(["group", "first_date"])
        .size()
        .rename("n_new_species")
        .reset_index()
    )

    df = df.sort_values(["group", "first_date"])
    df["cum_species"] = df.groupby("group")["n_new_species"].cumsum()

    df_total = (
        species_first
        .groupby("first_date")
        .size()
        .rename("n_new_species")
        .reset_index()
        .sort_values("first_date")
    )
    df_total["cum_species"] = df_total["n_new_species"].cumsum()

    plt.figure(figsize=(8, 5))

    for grp in TETRAPOD_GROUPS:
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        plt.plot(sub["first_date"], sub["cum_species"], label=grp, linewidth=1.5)

    plt.plot(
        df_total["first_date"],
        df_total["cum_species"],
        label="Total (all groups)",
        linewidth=2.0,
        linestyle="--",
    )

    plt.xlabel("Genome release date")
    plt.ylabel("Cumulative number of species with genomes")
    plt.title(f"Cumulative number of species with genomes over time ({region_label}, Tetrapoda)")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_png = OUTDIR / f"cumulative_genomes_over_time_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Gráfico de espécies com genoma acumuladas no tempo salvo em:", out_png)


def plot_country_group_heatmap(
    df_country_group: pd.DataFrame,
    region_label: str,
    n_quantiles: int = 5,
):
    """
    Gera um heatmap país × grupo usando classes de quantis da proporção
    de espécies com genoma (%), em vez de valores absolutos.

    Espera df_country_group com colunas:
        - iso2
        - group
        - sampling_proportion (0–1)
    """

    # Configuração de tamanhos de fonte
    title_fs = 14
    label_fs = 14
    tick_fs = 15

    # Pivot: linhas = países (iso2), colunas = grupos, valores = proporção
    pivot = df_country_group.pivot(
        index="iso2",
        columns="group",
        values="sampling_proportion",
    )

    # Ordena países alfabeticamente (iso2)
    pivot = pivot.sort_index()

    # Converte proporção para percentuais
    data_pct = pivot.values * 100.0

    # Vetor "flat" com todos os valores válidos (sem NaN)
    flat = data_pct[~np.isnan(data_pct)]
    if flat.size == 0:
        print(f"⚠ Não há dados válidos para o heatmap país × grupo ({region_label}).")
        return

    # ---------------------------------------------
    # Cálculo das classes de quantis
    # ---------------------------------------------
    qs = np.linspace(0, 1, n_quantiles + 1)
    quantile_edges = np.quantile(flat, qs)

    # Remove bordas duplicadas (quando a distribuição é muito estreita)
    quantile_edges = np.unique(quantile_edges)

    # Se sobrar menos de 3 bordas, a distribuição é muito estreita: usar escala contínua
    if len(quantile_edges) < 3:
        print("⚠ Distribuição muito estreita; usando escala contínua padrão.")
        fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(pivot))))

        # Usa matriz mascarada para NaN aparecerem em branco
        data_masked = np.ma.masked_invalid(data_pct)
        im = ax.imshow(data_masked, aspect="auto", interpolation="nearest")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(
            pivot.columns,
            rotation=45,
            ha="right",
            fontsize=tick_fs,
        )

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=tick_fs)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=tick_fs)
        cbar.set_label("Species with genomes (%)", fontsize=label_fs)

        ax.set_title(
            f"Genomic coverage by country × group ({region_label})",
            fontsize=title_fs,
        )
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)

        plt.tight_layout()
        out_png = OUTDIR / f"heatmap_country_group_{region_label}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print("📊 Heatmap país × grupo (contínuo) salvo em:", out_png)
        return

    # ---------------------------------------------
    # Mapeia cada célula para um índice de quantil
    # ---------------------------------------------
    # Matriz para armazenar a classe de quantil (1..n_quantiles), NaN se não tiver dado
    quantile_indices = np.full_like(data_pct, np.nan, dtype=float)

    for i in range(data_pct.shape[0]):
        for j in range(data_pct.shape[1]):
            val = data_pct[i, j]
            if np.isnan(val):
                continue
            # np.digitize retorna o índice do intervalo em quantile_edges
            idx = np.digitize(val, quantile_edges, right=True)
            # Garante que fique entre 1 e n_quantiles
            idx = max(1, min(idx, n_quantiles))
            quantile_indices[i, j] = idx

    # Usa matriz mascarada para NaNs não aparecerem coloridos
    quantile_masked = np.ma.masked_invalid(quantile_indices)

    # Colormap discreto: uma cor por quantil
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_quantiles)))
    bounds = np.arange(0.5, n_quantiles + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(pivot))))

    im = ax.imshow(
        quantile_masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    # Eixo X: grupos
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        pivot.columns,
        rotation=45,
        ha="right",
        fontsize=tick_fs,
    )

    # Eixo Y: países (iso2)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=tick_fs)

    # Barra de cores com ticks em 1..n_quantiles (classes discretas)
    cbar = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=range(1, n_quantiles + 1))
    cbar.ax.tick_params(labelsize=tick_fs)

    # Constrói labels de quantis a partir das bordas
    labels = []
    for k in range(n_quantiles):
        lo_idx = min(k, len(quantile_edges) - 2)
        hi_idx = lo_idx + 1
        lo = quantile_edges[lo_idx]
        hi = quantile_edges[hi_idx]
        labels.append(f"{lo:.1f}–{hi:.1f}%")

    if len(labels) < n_quantiles:
        labels += [""] * (n_quantiles - len(labels))

    cbar.set_ticklabels(labels[:n_quantiles])
    cbar.set_label(
        "Relative coverage (quantile classes of % species with genomes)",
        fontsize=label_fs,
    )

    ax.set_title(
        f"Genomic coverage by country × group (quantiles, {region_label})",
        fontsize=title_fs,
    )
    ax.tick_params(axis="both", which="both", labelsize=tick_fs)

    plt.tight_layout()
    out_png = OUTDIR / f"heatmap_country_group_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Heatmap país × grupo (quantis) salvo em:", out_png)

def plot_top10_families_by_n_genomes_panels(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    n_top: int = 10,
    min_species: int = 10,
):
    """
    Figura 2x2 com as top N famílias por N ABSOLUTO de espécies com genoma
    para Amphibia, Aves, Mammalia, Lepidosauria.

    Lógica:
    - Filtra famílias com n_species_total >= min_species.
    - Ordena por n_species_with_genome (decrescente).
    - Usa barras horizontais com comprimento = número de espécies com genoma.
    - Se tiver colunas de assembly (contig/scaffold/chromosome), faz barra empilhada
      com os N de espécies em cada nível de montagem.
    """
    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    groups_for_panels = ["Amphibia", "Aves", "Mammalia", "Lepidosauria"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for idx, grp in enumerate(groups_for_panels):
        ax = axes[idx]

        sub = fam_sampling_df[
            (fam_sampling_df["group"] == grp) &
            (fam_sampling_df["n_species_total"] >= min_species)
        ].copy()

        if sub.empty:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", fontsize=10)
            ax.set_title(grp)
            ax.set_yticks([])
            continue

        # Ordena por número absoluto de espécies com genoma
        sub = sub.sort_values("n_species_with_genome", ascending=False).head(n_top)

        y_labels = sub["family"]
        y_pos = np.arange(len(sub))

        if has_assembly_cols:
            x_contig = sub["n_genome_contig"].fillna(0).astype(int)
            x_scaffold = sub["n_genome_scaffold"].fillna(0).astype(int)
            x_chrom = sub["n_genome_chromosome"].fillna(0).astype(int)

            ax.barh(y_pos, x_contig, label="Contig")
            ax.barh(y_pos, x_scaffold, left=x_contig, label="Scaffold")
            ax.barh(y_pos, x_chrom, left=x_contig + x_scaffold, label="Chromosome")

            x_total = sub["n_species_with_genome"].astype(int)

            for xi, yi in zip(x_total, y_pos):
                ax.text(
                    xi + 0.3,
                    yi,
                    f"{xi}",
                    va="center",
                    fontsize=7
                )
        else:
            x_total = sub["n_species_with_genome"].astype(int)
            ax.barh(y_pos, x_total)
            for xi, yi in zip(x_total, y_pos):
                ax.text(
                    xi + 0.3,
                    yi,
                    f"{xi}",
                    va="center",
                    fontsize=7
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_title(grp)

    if has_assembly_cols:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Assembly levels",
                   fontsize=8, title_fontsize=9,
                   loc="upper right")

    fig.supylabel("Family")
    fig.supxlabel("Number of species with genomes")
    fig.suptitle(
        f"Top families by number of species with genomes in each group ({region_label})",
        fontsize=12
    )
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

    out_png = OUTDIR / f"plot_top10_families_by_n_genomes_2x2_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Figura 2x2 de top famílias por N de espécies com genoma salva em:", out_png)

def plot_top_families_by_N_3x2(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    n_top: int = 10,
    min_species: int = 10,
):
    """
    Figura 3x2 (6 painéis), com os grupos:

        Amphibia       | Aves
        Crocodylia     | Lepidosauria
        Mammalia       | Testudines

    Cada painel mostra as top N famílias por N ABSOLUTO de espécies com genoma,
    com barras empilhadas Contig/Scaffold/Chromosome (se disponíveis).

    Regras:
      - Famílias precisam ter >= threshold específico do grupo.
      - Testudines → min 2 spp ; Crocodylia → min 1 spp ; demais → min_species.
      - Eixo X é padronizado pelo máximo global de N.
    """

    groups_order = [
        "Amphibia", "Aves",
        "Crocodylia", "Lepidosauria",
        "Mammalia", "Testudines"
    ]

    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    # Thresholds especiais
    group_min_species = {
        "Testudines": 2,
        "Crocodylia": 1,
    }

    # Eixo X padronizado
    global_max_N = fam_sampling_df["n_species_with_genome"].max()
    if pd.isna(global_max_N) or global_max_N <= 0:
        global_max_N = 1.0

    # Criar figura 3×2
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    axes = axes.flatten()

    for ax, grp in zip(axes, groups_order):

        sub_group = fam_sampling_df[fam_sampling_df["group"] == grp].copy()
        N_total_genomes = int(sub_group["n_species_with_genome"].sum())
        N_total_species = int(sub_group["n_species_total"].sum())

        this_min = group_min_species.get(grp, min_species)

        sub = sub_group[sub_group["n_species_total"] >= this_min].copy()

        if sub.empty or N_total_genomes == 0:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", fontsize=12)
            ax.set_yticks([])
            ax.set_title(f"{grp} (N = 0)")
            ax.set_xlim(0, global_max_N)
            continue

        # Top N por N absoluto
        top = (
            sub.sort_values("n_species_with_genome", ascending=False)
               .head(n_top)
        )
        top = top.sort_values("n_species_with_genome", ascending=False)

        y = np.arange(len(top))
        fam_labels = top["family"].tolist()
        x_total = top["n_species_with_genome"].astype(int)

        if has_assembly_cols:
            x_contig = top["n_genome_contig"].fillna(0).astype(int)
            x_scaf   = top["n_genome_scaffold"].fillna(0).astype(int)
            x_chrom  = top["n_genome_chromosome"].fillna(0).astype(int)

            ax.barh(y, x_contig, label="Contig")
            ax.barh(y, x_scaf,   left=x_contig, label="Scaffold")
            ax.barh(y, x_chrom,  left=x_contig + x_scaf, label="Chromosome")

            for xi, yi, n_with, n_total in zip(
                x_total, y, top["n_species_with_genome"], top["n_species_total"]
            ):
                ax.text(xi + 0.3, yi, f"{n_with}/{n_total}", va="center", fontsize=10.5)

        else:
            ax.barh(y, x_total)
            for xi, yi, n_total in zip(x_total, y, top["n_species_total"]):
                ax.text(xi + 0.3, yi, f"{xi}/{n_total}", va="center", fontsize=10.5)

        ax.set_yticks(y)
        ax.set_yticklabels(fam_labels, fontsize=12)

        ax.set_xlim(0, global_max_N)
        ax.tick_params(axis="x", labelsize=14)
        
        # -------------------------------
        # Remover labels do eixo X das primeiras 2 linhas
        # (somente a última linha mostra os valores)
        # -------------------------------
        if ax not in axes[4:6]:       # painéis 0–3 → escondem X
            ax.tick_params(axis="x", labelbottom=False)        

        ax.set_title(f"{grp} (N={N_total_genomes}/{N_total_species})", fontsize=13)

    # Legenda global
    if has_assembly_cols:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Assembly level",
                   fontsize=10, title_fontsize=12, loc="upper right")

    fig.suptitle(
        f"Top families by number of species with genomes ({region_label})",
        fontsize=15
    )

    plt.tight_layout(rect=[0.03, 0.03, 0.96, 0.95])

    out_png = OUTDIR / f"plot_top_families_by_N_3x2_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("📊 Figura 3×2 salva em:", out_png)

def plot_top_families_by_N_3x2_local_axes(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    n_top: int = 10,
    min_species: int = 10,
):
    """
    Figura 3x2 (6 painéis), com os grupos:

        Amphibia       | Aves
        Crocodylia     | Lepidosauria
        Mammalia       | Testudines

    Versão com:
      - Eixo X LOCAL para cada grupo (baseado no máximo de N daquele grupo).
      - SOMENTE uma legenda global de Assembly level.
      - Altura padrão das barras (sem espaços estranhos entre elas).
      - Recuo do texto (n_genomas/total) proporcional ao eixo X,
        para não ficar exagerado em grupos com poucos genomas.
    """

    groups_order = [
        "Amphibia", "Aves",
        "Crocodylia", "Lepidosauria",
        "Mammalia", "Testudines",
    ]

    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    group_min_species = {
        "Testudines": 2,
        "Crocodylia": 1,
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for ax, grp in zip(axes, groups_order):

        sub_group = fam_sampling_df[fam_sampling_df["group"] == grp].copy()
        N_total_genomes = int(sub_group["n_species_with_genome"].sum())
        N_total_species = int(sub_group["n_species_total"].sum())

        this_min = group_min_species.get(grp, min_species)

        sub = sub_group[sub_group["n_species_total"] >= this_min].copy()

        if sub.empty or N_total_genomes == 0:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", fontsize=12)
            ax.set_yticks([])
            ax.set_title(f"{grp} (N=0/{N_total_species})", fontsize=13)
            ax.set_xlim(0, 1)
            continue

        # Top N por N absoluto
        top = (
            sub.sort_values("n_species_with_genome", ascending=False)
               .head(n_top)
        )
        top = top.sort_values("n_species_with_genome", ascending=False)

        y = np.arange(len(top))
        fam_labels = top["family"].tolist()
        x_total = top["n_species_with_genome"].astype(int)

        if has_assembly_cols:
            x_contig = top["n_genome_contig"].fillna(0).astype(int)
            x_scaf   = top["n_genome_scaffold"].fillna(0).astype(int)
            x_chrom  = top["n_genome_chromosome"].fillna(0).astype(int)

            # altura padrão (não especifico height → usa ~0.8)
            h1 = ax.barh(y, x_contig, label="Contig")
            h2 = ax.barh(y, x_scaf,   left=x_contig, label="Scaffold")
            h3 = ax.barh(y, x_chrom,  left=x_contig + x_scaf, label="Chromosome")

            # guardar uma única vez para legenda global
            if legend_handles is None:
                legend_handles = [h1, h2, h3]
                legend_labels = ["Contig", "Scaffold", "Chromosome"]

        else:
            ax.barh(y, x_total)

        # ---- eixo X local + offset proporcional ----
        local_max = x_total.max()
        if pd.isna(local_max) or local_max <= 0:
            local_max = 1.0

        ax.set_xlim(0, local_max * 1.15)
        ax.tick_params(axis="x", labelsize=12)

        # offset proporcional ao tamanho do eixo (2% do max)
        offset = 0.02 * local_max
        if offset < 0.1:
            offset = 0.1  # mínimo para não colar na barra

        # textos n_genomas/total com offset proporcional
        for xi, yi, n_with, n_total in zip(
            x_total, y, top["n_species_with_genome"], top["n_species_total"]
        ):
            ax.text(xi + offset, yi, f"{n_with}/{n_total}",
                    va="center", fontsize=10.5)

        ax.set_yticks(y)
        ax.set_yticklabels(fam_labels, fontsize=12)

        ax.set_title(f"{grp} (N={N_total_genomes}/{N_total_species})", fontsize=13)

    # Legenda global única
    if has_assembly_cols and legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            title="Assembly level",
            fontsize=10,
            title_fontsize=12,
            loc="upper right",
        )

    fig.suptitle(
        f"Top families by number of species with genomes (local axes, {region_label})",
        fontsize=15,
    )

    plt.tight_layout(rect=[0.03, 0.03, 0.96, 0.95])

    out_png = OUTDIR / f"plot_top_families_by_N_3x2_local_axes_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("📊 Figura 3×2 (eixos locais, espaçamento harmonizado) salva em:", out_png)

def plot_families_coverage_and_N_panels(
    fam_sampling_df: pd.DataFrame,
    region_label: str,
    n_top: int = 10,
    min_species: int = 10,
):
    """
    Figura 6x2 com, para cada grupo de Tetrapoda (todas entradas em TETRAPOD_GROUPS):

    Linha = grupo (Amphibia, Aves, Crocodylia, Mammalia, Lepidosauria, Testudines)
      - coluna 0: top N famílias por PORCENTAGEM de cobertura (% espécies com genoma),
                  com barras empilhadas Contig/Scaffold/Chromosome, em ordem
                  DECRESCENTE (maior cobertura em cima).
      - coluna 1: top N famílias por N ABSOLUTO de espécies com genoma,
                  também com barras empilhadas Contig/Scaffold/Chromosome,
                  em ordem DECRESCENTE (mais genomas em cima).

    Regras:
      - só considera famílias com n_species_total >= limiar de cada grupo;
      - limiar padrão = min_species, mas Testudines e Crocodylia usam
        thresholds menores para não desaparecerem.
      - rótulo da linha: "<Grupo> (N = total de espécies com genoma no grupo)".
    """

    has_assembly_cols = all(
        col in fam_sampling_df.columns
        for col in ["n_genome_contig", "n_genome_scaffold", "n_genome_chromosome"]
    )

    groups_for_rows = TETRAPOD_GROUPS  # já inclui todos os 6 grupos
    n_rows = len(groups_for_rows)
    n_cols = 2
    
    # ------------------------------------------------------------------
    # Determinar limites globais de X (para uniformizar os gráficos)
    # ------------------------------------------------------------------
    global_max_pct = (fam_sampling_df["sampling_proportion"] * 100).max()
    global_max_N = fam_sampling_df["n_species_with_genome"].max()

    # Valores de fallback caso tudo seja NaN/zero
    if pd.isna(global_max_pct) or global_max_pct <= 0:
        global_max_pct = 1.0
    if pd.isna(global_max_N) or global_max_N <= 0:
        global_max_N = 1.0

    # thresholds específicos por grupo (ajusta se quiser outros valores)
    group_min_species = {
        "Testudines": 2,
        "Crocodylia": 1,
        # demais grupos usam min_species padrão
    }

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 2.8 * n_rows),
        sharex=False,
    )

    if n_rows == 1:
        axes = np.array([axes])

    for i, grp in enumerate(groups_for_rows):
        ax_cov = axes[i, 0]
        ax_abs = axes[i, 1]

        # Todas as famílias do grupo
        all_grp = fam_sampling_df[fam_sampling_df["group"] == grp].copy()

        # N total de espécies com genoma (somando todas as famílias do grupo)
        N_total_genomes = int(all_grp["n_species_with_genome"].sum())

        # threshold específico por grupo
        this_min_species = group_min_species.get(grp, min_species)

        # Filtra famílias com n_species_total >= limiar do grupo
        sub = all_grp[all_grp["n_species_total"] >= this_min_species].copy()

        if sub.empty or N_total_genomes == 0:
            ax_cov.text(0.5, 0.5, "No data",
                        ha="center", va="center", fontsize=12)
            ax_abs.text(0.5, 0.5, "No data",
                        ha="center", va="center", fontsize=12)
            ax_cov.set_yticks([])
            ax_abs.set_yticks([])

            ax_cov.set_ylabel(f"{grp} (N = 0)", fontsize=12, rotation=90)
            ax_cov.set_xlim(0, 1)
            ax_abs.set_xlim(0, 1)
            continue

        # ======================================================
        # COLUNA ESQUERDA: top N por porcentagem de cobertura (DECRESCENTE)
        # ======================================================
        cov_top = (
            sub.sort_values("sampling_proportion", ascending=False)
               .head(n_top)
        )
        # mantém decrescente → maiores em cima
        cov_top = cov_top.sort_values("sampling_proportion", ascending=False)

        y_cov = np.arange(len(cov_top))
        fam_labels_cov = cov_top["family"].tolist()

        if has_assembly_cols:
            total = cov_top["n_species_total"].astype(float)

            x_contig = (cov_top["n_genome_contig"] / total) * 100.0
            x_scaffold = (cov_top["n_genome_scaffold"] / total) * 100.0
            x_chrom = (cov_top["n_genome_chromosome"] / total) * 100.0

            x_contig = x_contig.fillna(0.0)
            x_scaffold = x_scaffold.fillna(0.0)
            x_chrom = x_chrom.fillna(0.0)

            ax_cov.barh(y_cov, x_contig, label="Contig")
            ax_cov.barh(y_cov, x_scaffold, left=x_contig, label="Scaffold")
            ax_cov.barh(
                y_cov,
                x_chrom,
                left=x_contig + x_scaffold,
                label="Chromosome",
            )

            x_total_pct = cov_top["sampling_proportion"] * 100.0

            for xi, yi, n_with, n_total in zip(
                x_total_pct,
                y_cov,
                cov_top["n_species_with_genome"],
                cov_top["n_species_total"],
            ):
                ax_cov.text(
                    xi + 0.35,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=8,
                )
        else:
            x_total_pct = cov_top["sampling_proportion"] * 100.0
            ax_cov.barh(y_cov, x_total_pct)
            for xi, yi, n_with, n_total in zip(
                x_total_pct,
                y_cov,
                cov_top["n_species_with_genome"],
                cov_top["n_species_total"],
            ):
                ax_cov.text(
                    xi + 0.35,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=8,
                )

        ax_cov.set_yticks(y_cov)
        ax_cov.set_yticklabels(fam_labels_cov, fontsize=12)

        ax_cov.set_ylabel(f"{grp} (N = {N_total_genomes})",
                          fontsize=14, rotation=90)

        if i == n_rows - 1:
            ax_cov.set_xlabel("Genomic coverage (%)", fontsize=16)

        ax_cov.set_xlim(0, global_max_pct)
        ax_cov.tick_params(axis="x", labelsize=14, width=2, length=5)
        
        # ======================================================
        # COLUNA DIREITA: top N por N ABSOLUTO (DECRESCENTE)
        # ======================================================
        abs_top = (
            sub.sort_values("n_species_with_genome", ascending=False)
               .head(n_top)
        )
        abs_top = abs_top.sort_values("n_species_with_genome", ascending=False)

        y_abs = np.arange(len(abs_top))
        fam_labels_abs = abs_top["family"].tolist()
        x_total_abs = abs_top["n_species_with_genome"].astype(int)

        if has_assembly_cols:
            x_contig_abs = abs_top["n_genome_contig"].fillna(0).astype(int)
            x_scaffold_abs = abs_top["n_genome_scaffold"].fillna(0).astype(int)
            x_chrom_abs = abs_top["n_genome_chromosome"].fillna(0).astype(int)

            ax_abs.barh(y_abs, x_contig_abs, label="Contig")
            ax_abs.barh(
                y_abs,
                x_scaffold_abs,
                left=x_contig_abs,
                label="Scaffold",
            )
            ax_abs.barh(
                y_abs,
                x_chrom_abs,
                left=x_contig_abs + x_scaffold_abs,
                label="Chromosome",
            )

            # rótulo: n_with / n_total para CADA família
            for xi, yi, n_with, n_total in zip(
                x_total_abs,
                y_abs,
                abs_top["n_species_with_genome"],
                abs_top["n_species_total"],
            ):
                ax_abs.text(
                    xi + 0.35,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=8,
                )            
        else:
            ax_abs.barh(y_abs, x_total_abs)
            for xi, yi, n_total in zip(
                x_total_abs, y_abs, abs_top["n_species_total"]
            ):
                n_with = xi  # aqui xi já é n_species_with_genome
                ax_abs.text(
                    xi + 0.35,
                    yi,
                    f"{n_with}/{n_total}",
                    va="center",
                    fontsize=8,
                )
                
        ax_abs.set_yticks(y_abs)
        ax_abs.set_yticklabels(fam_labels_abs, fontsize=12)

        if i == n_rows - 1:
            ax_abs.set_xlabel("Species with genomes", fontsize = 18)

        ax_abs.set_xlim(0, global_max_N)
        ax_abs.tick_params(axis="x", labelsize=14, width=2, length=5)
        
        # ---------------------------------------------------------
        # Remover rótulos do eixo X nas linhas de cima
        # (deixa eixo X completo só na última linha)
        # ---------------------------------------------------------
        if i < n_rows - 1:
            ax_cov.tick_params(axis="x", labelbottom=False)
            ax_abs.tick_params(axis="x", labelbottom=False)


    # Títulos das colunas
    axes[0, 0].set_title("Genomic coverage (%)", fontsize=17)
    axes[0, 1].set_title("Species with genomes", fontsize=17)

    if has_assembly_cols:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Assembly level",
            fontsize=13,
            title_fontsize=15,
            loc="upper right",
        )

    fig.suptitle(
        f"Family-level genomic coverage and assembly composition by group ({region_label})",
        fontsize=13,
    )

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

    out_png = OUTDIR / f"plot_families_coverage_vs_N_6x2_{region_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("📊 Figura 6x2 (cobertura % vs N absoluto, com Contig/Scaffold/Chromosome) salva em:", out_png)

# =============================================================================
# 8. Leitura de estatísticas globais
# =============================================================================

def load_global_group_stats():
    """
    Lê a tabela representativity_by_group.csv contendo as métricas globais.

    Espera colunas:
        - group
        - n_species_with_genome
        - n_gbif_species

    Retorna DataFrame com:
        - group
        - sampling_global  (proporção 0–1)
    """
    path = PATH_BASE / "representativity_by_group.csv"
    if not path.exists():
        print("⚠ Arquivo global 'representativity_by_group.csv' não encontrado. "
              "Figura Neotropical vs Global não será gerada.")
        return None

    # Arquivo é CSV com vírgula
    df = pd.read_csv(path)

    required_cols = {"group", "n_species_with_genome", "n_gbif_species"}
    missing = required_cols.difference(df.columns)
    if missing:
        print("⚠ Colunas esperadas não encontradas em representativity_by_group.csv:")
        print("   Faltando:", missing)
        print("   Colunas disponíveis:", list(df.columns))
        return None

    df["sampling_global"] = df["n_species_with_genome"] / df["n_gbif_species"]

    return df


# =============================================================================
# 9. Função para rodar pipeline por região
# =============================================================================

def run_region_pipeline(
    region_name: str,
    region_iso2: set,
    families: pd.DataFrame,
    sp_gen: pd.DataFrame,
    fam_cache: pd.DataFrame,
    assemblies: pd.DataFrame | None,
    global_group_stats: pd.DataFrame | None,
):
    """Roda o pipeline completo para uma região específica."""
    global OUTDIR

    print("\n" + "=" * 70)
    print(f"▶ Rodando pipeline para região: {region_name}")
    print("=" * 70)

    OUTDIR = BASE_OUTDIR / region_name
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Cobertura regional por grupo
    sampling_groups = compute_group_sampling(families, sp_gen, region_iso2, region_name)

    # 2) Cobertura regional por família
    sampling_families = compute_family_sampling(families, sp_gen, fam_cache, region_iso2, region_name)

    # 2b) Assemblies por família
    sampling_families = add_family_assembly_breakdown(
        sampling_families,
        sp_gen,
        fam_cache,
        assemblies,
        region_iso2,
    )

    # 3) Cobertura por país × grupo
    sampling_country_group = compute_country_group_sampling(families, sp_gen, region_iso2, region_name)

    # 4) Cobertura por país × família
    _ = compute_country_family_sampling(families, sp_gen, fam_cache, region_iso2, region_name)

    # 5) Gráficos principais
    plot_group_sampling(sampling_groups, region_name)

    # Comparação Neotropics vs Global (só faz sentido para a região Neotropics)
    if global_group_stats is not None and region_name == "Neotropics":
        plot_group_sampling_neotropics_vs_global(sampling_groups, global_group_stats)

    plot_family_sampling(sampling_families, region_name, min_species=10)
    plot_top10_families_panels(sampling_families, region_name, n_top=10, min_species=10)
    plot_top10_families_by_n_genomes_panels(sampling_families, region_name, n_top=10, min_species=10)
    plot_top_families_by_N_3x2(sampling_families, region_name, n_top=10, min_species=10)
    plot_top_families_by_N_3x2_local_axes(sampling_families, region_name, n_top=10, min_species=10)
    plot_country_group_heatmap(sampling_country_group, region_name)
    plot_families_coverage_and_N_panels(sampling_families,region_name,n_top=10,min_species=10)

    # 6) Gráficos extras
    plot_richness_vs_coverage_scatter(sampling_families, region_name, min_species=10)
    plot_assembly_composition_by_group(sampling_families, region_name)

    # 7) Gráfico cumulativo de genomas ao longo do tempo
    if assemblies is not None:
        plot_cumulative_genomes_over_time(sp_gen, assemblies, region_iso2, region_name)

    print(f"✅ Pipeline concluído para região: {region_name}")


# =============================================================================
# 10. main
# =============================================================================

def main():
    print("🔄 Lendo arquivos de entrada...")
    families, sp_gen, fam_cache = load_inputs()

    print("🔄 Lendo arquivo de assemblies...")
    assemblies = load_assemblies()

    print("🔄 Lendo estatísticas globais por grupo...")
    global_group_stats = load_global_group_stats()

    # Escolha aqui quais regiões rodar
    regions_to_run = ["Neotropics", "Europe", "NorthAmerica"]

    for region_name in regions_to_run:
        region_iso2 = REGION_DEFS[region_name]
        run_region_pipeline(
            region_name=region_name,
            region_iso2=region_iso2,
            families=families,
            sp_gen=sp_gen,
            fam_cache=fam_cache,
            assemblies=assemblies,
            global_group_stats=global_group_stats,
        )

    print("\n✅ Pipeline multi-region concluído.")


if __name__ == "__main__":
    main()
