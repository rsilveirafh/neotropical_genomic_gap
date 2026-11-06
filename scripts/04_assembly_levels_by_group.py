#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tabela S5: nível de montagem (melhor montagem por espécie) das espécies neotropicais, por grupo.
Na Neotrópica, espécie = chave do GBIF (usageKey), como na Tabela S1; o total bate com a S1.
Também gera a mesma contagem para o mundo todo (assembly_levels_by_group_global.csv).

Entradas: data/inputs/famscan_all_assemblies_filtered.tsv e species_country_filtered_clean.tsv
Saídas:   outputs/assembly_levels_by_group_neotropics.csv (Tabela S5)
          outputs/assembly_levels_by_group_global.csv
"""

import sys
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =========================
# Parâmetros gerais
# =========================

# Caminhos relativos à raiz do repositório (o script roda de qualquer pasta)
REPO_DIR = Path(__file__).resolve().parents[1]
PATH_BASE = REPO_DIR / "data" / "inputs"
OUTPUT_DIR = REPO_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Países da região Neotropical (ISO2)
NEOTROPICAL_ISO2 = set([
    # Mesoamérica
    "MX", "BZ", "CR", "GT", "HN", "NI", "PA", "SV",
    # Caribe
    "CU", "DO", "HT", "PR", "JM", "TT", "BB", "BS",
    "AG", "LC", "GD", "VC", "DM",
    # América do Sul
    "AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF",
    "GY", "PE", "PY", "SR", "UY", "VE",
])

# Grupos focais (Tetrapoda)
TETRAPOD_GROUPS = [
    "Amphibia",
    "Aves",
    "Mammalia",
    "Lepidosauria",
    "Testudines",
    "Crocodylia",
]


# =========================
# Funções auxiliares
# =========================

def simplify_assembly_level(raw: str):
    """
    Mapeia o texto de 'Assembly Level' para:
    - 'Contig'
    - 'Scaffold'
    - 'Chromosome'

    'Complete Genome' entra em 'Chromosome'.
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


def load_assemblies(path: Path) -> pd.DataFrame:
    """Lê famscan_all_assemblies_filtered.tsv."""
    df = pd.read_csv(path, sep="\t")
    # Normaliza nome da espécie
    df["species"] = df["Organism Name"].astype(str).str.strip()
    # Categoria de montagem simplificada
    df["assembly_category"] = df["Assembly Level"].apply(simplify_assembly_level)
    return df


def get_best_assembly_per_species(assemblies: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada espécie, retém APENAS o melhor nível de montagem.

    Usa a coluna:
    - 'assembly_category' (Contig / Scaffold / Chromosome)
    - 'parent_name' (grupo: Amphibia, Aves, etc.)

    Retorna um DataFrame com colunas:
    - species
    - group
    - assembly_category
    """
    rank_map = {"Contig": 1, "Scaffold": 2, "Chromosome": 3}

    asm = assemblies.copy()
    asm["level_rank"] = asm["assembly_category"].map(rank_map).fillna(0)

    # pega o melhor (maior rank) por espécie
    asm_best = (
        asm.sort_values("level_rank")
           .drop_duplicates(subset=["species"], keep="last")
           .rename(columns={"parent_name": "group"})
    )

    # mantemos só o essencial
    asm_best = asm_best[["species", "group", "assembly_category"]]

    # Se quiser restringir só aos tetrapod groups definidos:
    asm_best = asm_best[asm_best["group"].isin(TETRAPOD_GROUPS)]

    return asm_best


def load_neotropical_species(path_sp: Path) -> pd.DataFrame:
    """
    Lê species_country_filtered_clean.tsv e devolve
    as espécies que ocorrem na Neotrópica, com o grupo.

    Espera colunas:
    - usageKey
    - parent_name
    - iso2
    - species
    """
    sp = pd.read_csv(path_sp, sep="\t")
    # usageKey = 1 é o reino Animalia no GBIF (match falho de nome, ocorrência
    # em ~250 países). Fica fora, como no script 01.
    sp = sp[sp["usageKey"] != 1].copy()

    sp_nt = sp[
        (sp["iso2"].isin(NEOTROPICAL_ISO2)) &
        (sp["parent_name"].isin(TETRAPOD_GROUPS))
    ].copy()

    # Um registro por (chave GBIF, nome NCBI, grupo). A mesma usageKey pode
    # aparecer com mais de um nome (ex.: subespécies), então a contagem final
    # é feita por usageKey, como na Tabela S1.
    sp_nt_unique = (
        sp_nt[["usageKey", "species", "parent_name"]]
        .drop_duplicates()
        .rename(columns={"parent_name": "group"})
    )

    return sp_nt_unique


def count_assembly_levels_by_group_global(asm_best: pd.DataFrame) -> pd.DataFrame:
    """
    Conta, GLOBALMENTE, quantas espécies em cada grupo
    têm melhor montagem em Contig / Scaffold / Chromosome.
    """
    counts = (
        asm_best
        .groupby(["group", "assembly_category"])["species"]
        .nunique()
        .rename("n_species")
        .reset_index()
    )

    pivot = counts.pivot_table(
        index="group",
        columns="assembly_category",
        values="n_species",
        fill_value=0,
    ).reset_index()

    # garante colunas
    for col in ["Contig", "Scaffold", "Chromosome"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # total de espécies com genoma por grupo
    pivot["n_species_with_genome"] = (
        pivot[["Contig", "Scaffold", "Chromosome"]].sum(axis=1)
    )

    return pivot


def count_assembly_levels_by_group_neotropics(
    asm_best: pd.DataFrame,
    sp_nt_unique: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conta, APENAS para espécies Neotropicais, quantas espécies em cada grupo
    têm melhor montagem em Contig / Scaffold / Chromosome.

    asm_best: saída de get_best_assembly_per_species()
    sp_nt_unique: saída de load_neotropical_species()
    """
    # restringe aos species que ocorrem na Neotrópica; o grupo vem do GBIF
    # (mesma fonte da Tabela S1)
    rank_map = {"Contig": 1, "Scaffold": 2, "Chromosome": 3}
    neo_best = sp_nt_unique.merge(
        asm_best[["species", "assembly_category"]],
        on="species",
        how="inner",
    )

    if neo_best.empty:
        print("⚠ Nenhuma espécie de asm_best pôde ser associada à Neotrópica.")
        return pd.DataFrame()

    # Melhor montagem por chave GBIF dentro de cada grupo (uma espécie = uma
    # usageKey por grupo, exatamente como o numerador da Tabela S1)
    neo_best["level_rank"] = neo_best["assembly_category"].map(rank_map)
    neo_best = (
        neo_best.sort_values("level_rank")
        .drop_duplicates(subset=["group", "usageKey"], keep="last")
    )

    counts = (
        neo_best
        .groupby(["group", "assembly_category"])["usageKey"]
        .nunique()
        .rename("n_species")
        .reset_index()
    )

    pivot = counts.pivot_table(
        index="group",
        columns="assembly_category",
        values="n_species",
        fill_value=0,
    ).reset_index()

    for col in ["Contig", "Scaffold", "Chromosome"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["n_species_with_genome"] = (
        pivot[["Contig", "Scaffold", "Chromosome"]].sum(axis=1)
    )

    return pivot


# =========================
# main
# =========================

def main():
    path_asm = PATH_BASE / "famscan_all_assemblies_filtered.tsv"
    path_sp = PATH_BASE / "species_country_filtered_clean.tsv"

    print("🔄 Lendo assemblies...")
    asm = load_assemblies(path_asm)

    print("🔄 Escolhendo melhor montagem por espécie...")
    asm_best = get_best_assembly_per_species(asm)

    print("🔄 Lendo espécies Neotropicais (GBIF)...")
    sp_nt_unique = load_neotropical_species(path_sp)

    print("🔄 Contando níveis de montagem por grupo (GLOBAL)...")
    df_global = count_assembly_levels_by_group_global(asm_best)
    df_global.to_csv(OUTPUT_DIR / "assembly_levels_by_group_global.csv", index=False)
    print("✅ Salvo em assembly_levels_by_group_global.csv")
    print(df_global)

    print("🔄 Contando níveis de montagem por grupo (NEOTRÓPICA)...")
    df_neo = count_assembly_levels_by_group_neotropics(asm_best, sp_nt_unique)
    if not df_neo.empty:
        df_neo.to_csv(OUTPUT_DIR / "assembly_levels_by_group_neotropics.csv", index=False)
        print("✅ Salvo em assembly_levels_by_group_neotropics.csv")
        print(df_neo)
    else:
        print("⚠ Tabela Neotrópica vazia, nada salvo.")

    print("🏁 Concluído.")


if __name__ == "__main__":
    main()
