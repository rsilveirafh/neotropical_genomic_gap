#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tabela S5: nível de montagem dos genomas das espécies neotropicais, por grupo.

Para cada espécie (chave do GBIF, `usageKey`) que ocorre em pelo menos um país neotropical
e tem genoma, toma a MELHOR montagem disponível no NCBI:
    Contig < Scaffold < Chromosome (Complete Genome conta como Chromosome)
e conta quantas espécies de cada grupo ficam em cada nível.

Usa a mesma definição de Neotrópico e a mesma unidade (usageKey) da Tabela S1
(01_coverage_by_region.py), de modo que n_species_with_genome bate com a S1.

Entradas (data/inputs/):
    species_country_filtered_clean.tsv   usageKey, parent_name, iso2, species
    famscan_all_assemblies_filtered.tsv  Organism Name, Assembly Level

Saída:
    outputs/region_outputs/Neotropics/assembly_levels_by_group_Neotropics.csv

Nota: a versão anterior desta tabela (enviada à revista em 2025) incluía as Ilhas Malvinas (FK)
no Neotrópico e contava espécies pelo nome, não pela chave do GBIF; por isso dava 730 aves e
297 mamíferos com genoma, contra 727 e 293 na Tabela S1.
"""

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_DIR / "data" / "inputs"
OUT_DIR = REPO_DIR / "outputs" / "region_outputs" / "Neotropics"

# Mesma lista de 01_coverage_by_region.py
NEOTROPICAL_ISO2 = {
    # Mesoamérica
    "MX", "BZ", "CR", "GT", "HN", "NI", "PA", "SV",
    # Caribe
    "CU", "DO", "HT", "PR", "JM", "TT", "BB", "BS",
    "AG", "LC", "GD", "VC", "DM",
    # América do Sul
    "AR", "BO", "BR", "CL", "CO", "EC", "GF",
    "GY", "PE", "PY", "SR", "UY", "VE",
}

TETRAPOD_GROUPS = ["Amphibia", "Aves", "Mammalia", "Lepidosauria", "Testudines", "Crocodylia"]

LEVEL_RANK = {"Contig": 1, "Scaffold": 2, "Chromosome": 3}
RANK_LEVEL = {v: k for k, v in LEVEL_RANK.items()}


def simplify_assembly_level(raw):
    """'Complete Genome' entra em 'Chromosome'."""
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


def main():
    asm = pd.read_csv(INPUT_DIR / "famscan_all_assemblies_filtered.tsv", sep="\t")
    asm["species"] = asm["Organism Name"].astype(str).str.strip()
    asm["rank"] = asm["Assembly Level"].apply(simplify_assembly_level).map(LEVEL_RANK)
    best_by_name = asm.groupby("species")["rank"].max()

    sp = pd.read_csv(INPUT_DIR / "species_country_filtered_clean.tsv", sep="\t")
    sp = sp[sp["iso2"].isin(NEOTROPICAL_ISO2) & sp["parent_name"].isin(TETRAPOD_GROUPS)]
    sp = sp[["usageKey", "parent_name", "species"]].drop_duplicates()
    sp["rank"] = sp["species"].astype(str).str.strip().map(best_by_name)

    # Uma linha por espécie (usageKey): melhor nível entre os nomes ligados a ela
    per_species = sp.groupby(["usageKey", "parent_name"], as_index=False)["rank"].max()
    missing = per_species["rank"].isna().sum()
    if missing:
        print(f"⚠ {missing} espécies sem nível de montagem casado; ficam fora da contagem.")
    per_species = per_species.dropna(subset=["rank"])
    per_species["level"] = per_species["rank"].astype(int).map(RANK_LEVEL)

    table = (
        per_species.pivot_table(index="parent_name", columns="level", values="usageKey",
                                aggfunc="nunique", fill_value=0)
        .reindex(columns=["Chromosome", "Contig", "Scaffold"], fill_value=0)
    )
    table["n_species_with_genome"] = table.sum(axis=1)
    table = table.rename_axis("group").reset_index().rename_axis(None, axis=1).sort_values("group")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "assembly_levels_by_group_Neotropics.csv"
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    print(f"✅ Tabela S5 salva em: {out_csv}")


if __name__ == "__main__":
    main()
