"""
Cobertura genômica por grupo nas três regiões, com a composição por nível de montagem
(contig, scaffold, chromosome) dentro de cada barra.

Rode depois de 01_coverage_by_region.py. Saídas em outputs/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

REGIONS = ["Europe", "NorthAmerica", "Neotropics"]


# ---------------------------------------------------------------------
# Leitura dos dados de cobertura
# ---------------------------------------------------------------------
def load_region_group_sampling(region_name: str) -> pd.DataFrame:
    """
    Lê o CSV de cobertura por grupo para uma região.

    Espera arquivo:
        region_outputs/<Região>/sampling_by_group_<Região>_unique.csv

    com colunas:
        - group
        - n_species_total
        - n_species_with_genome
        - sampling_proportion

    Casos com n_species_total == 0 são tratados como não aplicáveis (NA),
    e não como 100% de cobertura.
    """
    region_dir = BASE_OUTDIR / region_name
    fname = f"sampling_by_group_{region_name}_unique.csv"
    path = region_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de cobertura por grupo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "group",
        "n_species_total",
        "n_species_with_genome",
        "sampling_proportion",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Arquivo {path} não tem todas as colunas esperadas. "
            f"Ausentes: {sorted(missing)}. Colunas encontradas: {list(df.columns)}"
        )

    df["n_species_total"] = pd.to_numeric(
        df["n_species_total"], errors="coerce"
    )
    df["n_species_with_genome"] = pd.to_numeric(
        df["n_species_with_genome"], errors="coerce"
    )
    df["sampling_proportion"] = pd.to_numeric(
        df["sampling_proportion"], errors="coerce"
    )

    # Recalcula a proporção a partir das contagens para evitar que 0/0
    # seja interpretado como 100% em arquivos gerados anteriormente.
    valid_total = df["n_species_total"] > 0
    df["coverage_pct"] = np.nan
    df.loc[valid_total, "coverage_pct"] = (
        100.0
        * df.loc[valid_total, "n_species_with_genome"]
        / df.loc[valid_total, "n_species_total"]
    ).clip(lower=0.0, upper=100.0)

    df["region"] = region_name

    return df[
        [
            "group",
            "region",
            "n_species_total",
            "n_species_with_genome",
            "coverage_pct",
        ]
    ]


def simplify_assembly_level(raw: str):
    """
    Mapeia o texto de 'Assembly Level' para:
      - Contig
      - Scaffold
      - Chromosome

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


# ---------------------------------------------------------------------
# Figura principal
# ---------------------------------------------------------------------
def plot_group_comparison_3regions_with_assemblies():
    """
    Compara a cobertura genômica por grupo de tetrápodes entre:
      - Europe
      - NorthAmerica
      - Neotropics

    Cada barra é empilhada por nível de montagem:
      - Contig
      - Scaffold
      - Chromosome

    Os rótulos mostram:
        XX.X% REGION
        (genomas/espécies totais)

    Grupos ausentes da região são mostrados como:
        REGION: absent
        (0/0)
    """
    print("🔄 Lendo coberturas por grupo/região...")

    dfs_cov = []
    for reg in REGIONS:
        try:
            dfs_cov.append(load_region_group_sampling(reg))
        except Exception as exc:
            print(f"⚠ Problema ao carregar cobertura para {reg}: {exc}")

    if not dfs_cov:
        print("❌ Nenhuma região pôde ser carregada. Abortando.")
        return

    df_cov_all = pd.concat(dfs_cov, ignore_index=True)

    groups = sorted(df_cov_all["group"].dropna().unique().tolist())

    pivot_cov = (
        df_cov_all.pivot(index="group", columns="region", values="coverage_pct")
        .reindex(index=groups, columns=REGIONS)
    )
    pivot_total = (
        df_cov_all.pivot(index="group", columns="region", values="n_species_total")
        .reindex(index=groups, columns=REGIONS)
    )
    pivot_genomes = (
        df_cov_all.pivot(
            index="group", columns="region", values="n_species_with_genome"
        )
        .reindex(index=groups, columns=REGIONS)
    )

    # ------------------------------------------------------------------
    # Composição dos assemblies
    # ------------------------------------------------------------------
    print("🔄 Lendo species_country_filtered_clean.tsv...")
    sp_path = INPUT_DIR / "species_country_filtered_clean.tsv"
    sp = pd.read_csv(sp_path, sep="\t")

    print("🔄 Lendo famscan_all_assemblies_filtered.tsv...")
    asm_path = INPUT_DIR / "famscan_all_assemblies_filtered.tsv"
    asm = pd.read_csv(asm_path, sep="\t")

    required_sp = {"species", "parent_name", "iso2"}
    missing_sp = required_sp.difference(sp.columns)
    if missing_sp:
        raise ValueError(
            f"{sp_path} não tem as colunas necessárias: {sorted(missing_sp)}"
        )

    required_asm = {"Assembly Level", "Organism Name"}
    missing_asm = required_asm.difference(asm.columns)
    if missing_asm:
        raise ValueError(
            f"{asm_path} não tem as colunas necessárias: {sorted(missing_asm)}"
        )

    asm["assembly_category"] = asm["Assembly Level"].apply(
        simplify_assembly_level
    )
    asm["species"] = asm["Organism Name"].astype(str).str.strip()

    rank_map = {"Contig": 1, "Scaffold": 2, "Chromosome": 3}
    asm["level_rank"] = asm["assembly_category"].map(rank_map).fillna(0)

    asm_best = (
        asm.sort_values("level_rank")
        .drop_duplicates(subset=["species"], keep="last")
        [["species", "assembly_category"]]
    )

    sp["species"] = sp["species"].astype(str).str.strip()
    sp["iso2"] = sp["iso2"].astype(str).str.strip().str.upper()

    sp_grp = (
        sp[["species", "parent_name", "iso2"]]
        .dropna(subset=["species", "parent_name", "iso2"])
        .rename(columns={"parent_name": "group"})
    )

    neotropical_iso2 = {
        "MX", "BZ", "CR", "GT", "HN", "NI", "PA", "SV",
        "CU", "DO", "HT", "PR", "JM", "TT", "BB", "BS",
        "AG", "LC", "GD", "VC", "DM",
        "AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF",
        "GY", "PE", "PY", "SR", "UY", "VE",
    }

    europe_iso2 = {
        "AL", "AD", "AM", "AT", "AZ", "BY", "BE", "BA", "BG", "HR",
        "CY", "CZ", "DK", "EE", "FI", "FR", "GE", "DE", "GR", "HU",
        "IS", "IE", "IT", "KZ", "XK", "LV", "LI", "LT", "LU", "MT",
        "MD", "MC", "ME", "NL", "MK", "NO", "PL", "PT", "RO", "RU",
        "SM", "RS", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "VA",
        "GB", "UK",
    }

    north_america_iso2 = {"US", "CA", "MX", "GL", "BM"}

    region_iso2 = {
        "Neotropics": neotropical_iso2,
        "Europe": europe_iso2,
        "NorthAmerica": north_america_iso2,
    }

    rows = []

    for reg in REGIONS:
        iso_set = region_iso2.get(reg, set())
        if not iso_set:
            continue

        sp_reg = sp_grp[sp_grp["iso2"].isin(iso_set)].copy()
        if sp_reg.empty:
            print(f"⚠ Nenhuma espécie mapeada para {reg}.")
            continue

        sp_reg_unique = sp_reg[["species", "group"]].drop_duplicates()

        merged = sp_reg_unique.merge(asm_best, on="species", how="inner")
        merged = merged.dropna(subset=["assembly_category"])

        if merged.empty:
            print(f"⚠ Nenhuma espécie com assembly mapeada para {reg}.")
            continue

        counts = (
            merged.groupby(["group", "assembly_category"])["species"]
            .nunique()
            .rename("n_species")
            .reset_index()
        )

        pivot_asm = counts.pivot_table(
            index="group",
            columns="assembly_category",
            values="n_species",
            fill_value=0,
        )

        for col in ["Contig", "Scaffold", "Chromosome"]:
            if col not in pivot_asm.columns:
                pivot_asm[col] = 0

        pivot_asm = pivot_asm[["Contig", "Scaffold", "Chromosome"]]
        pivot_asm["n_total_genomes"] = pivot_asm.sum(axis=1)

        frac = pivot_asm.copy().astype(float)
        nonzero = frac["n_total_genomes"] > 0

        frac.loc[nonzero, ["Contig", "Scaffold", "Chromosome"]] = (
            frac.loc[nonzero, ["Contig", "Scaffold", "Chromosome"]]
            .div(frac.loc[nonzero, "n_total_genomes"], axis=0)
        )
        frac.loc[~nonzero, ["Contig", "Scaffold", "Chromosome"]] = 0.0

        for grp in frac.index:
            rows.append(
                {
                    "region": reg,
                    "group": grp,
                    "frac_contig": float(frac.loc[grp, "Contig"]),
                    "frac_scaffold": float(frac.loc[grp, "Scaffold"]),
                    "frac_chromosome": float(frac.loc[grp, "Chromosome"]),
                }
            )

    df_frac = pd.DataFrame(rows)

    if df_frac.empty:
        print(
            "⚠ Não foi possível calcular as frações de assembly. "
            "Gerando gráfico apenas com cobertura total."
        )
        plot_group_comparison_3regions_only(df_cov_all)
        return

    # Matrizes dos segmentos empilhados. Valores NA viram zero somente
    # para o desenho das barras; continuam NA nos rótulos e na interpretação.
    contig_pct = pd.DataFrame(0.0, index=groups, columns=REGIONS)
    scaffold_pct = pd.DataFrame(0.0, index=groups, columns=REGIONS)
    chrom_pct = pd.DataFrame(0.0, index=groups, columns=REGIONS)

    for grp in groups:
        for reg in REGIONS:
            cov = pivot_cov.loc[grp, reg]
            if pd.isna(cov):
                continue

            row = df_frac[
                (df_frac["group"] == grp) & (df_frac["region"] == reg)
            ]

            if row.empty:
                # Não inventa composição de assembly quando ela não está disponível.
                # A cobertura é mostrada como uma barra de contorno cinza.
                continue

            r = row.iloc[0]
            contig_pct.loc[grp, reg] = cov * r["frac_contig"]
            scaffold_pct.loc[grp, reg] = cov * r["frac_scaffold"]
            chrom_pct.loc[grp, reg] = cov * r["frac_chromosome"]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("🎨 Gerando figura comparativa com composição de montagem...")

    assembly_colors = {
        "Contig": "#1f77b4",
        "Scaffold": "#ff7f0e",
        "Chromosome": "#2ca02c",
    }

    region_short = {
        "Europe": "E",
        "NorthAmerica": "NA",
        "Neotropics": "NEO",
    }

    y = np.arange(len(groups))
    n_regions = len(REGIONS)
    height = 0.8 / n_regions

    # Mais espaço horizontal para os novos rótulos.
    fig_height = max(5.0, 0.9 * len(groups) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    for i, reg in enumerate(REGIONS):
        vals_contig = contig_pct[reg].to_numpy(dtype=float)
        vals_scaf = scaffold_pct[reg].to_numpy(dtype=float)
        vals_chrom = chrom_pct[reg].to_numpy(dtype=float)

        offset = (i - (n_regions - 1) / 2) * height
        ys = y + offset

        ax.barh(
            ys,
            vals_contig,
            height=height,
            color=assembly_colors["Contig"],
            edgecolor="none",
            label="Contig" if i == 0 else None,
        )
        ax.barh(
            ys,
            vals_scaf,
            height=height,
            left=vals_contig,
            color=assembly_colors["Scaffold"],
            edgecolor="none",
            label="Scaffold" if i == 0 else None,
        )
        ax.barh(
            ys,
            vals_chrom,
            height=height,
            left=vals_contig + vals_scaf,
            color=assembly_colors["Chromosome"],
            edgecolor="none",
            label="Chromosome" if i == 0 else None,
        )

        short = region_short.get(reg, reg)

        for grp, yi, xi_cont, xi_scaf, xi_chrom in zip(
            groups, ys, vals_contig, vals_scaf, vals_chrom
        ):
            cov_val = pivot_cov.loc[grp, reg]
            n_total = pivot_total.loc[grp, reg]
            n_genomes = pivot_genomes.loc[grp, reg]

            # Combinação grupo × região ausente do arquivo.
            if pd.isna(n_total):
                continue

            n_total_int = int(round(n_total))
            n_genomes_int = (
                0 if pd.isna(n_genomes) else int(round(n_genomes))
            )

            # Grupo biologicamente ausente da região: 0/0 = NA.
            if n_total_int == 0:
                label = f"{short}: absent\n(0/0)"
                label_x = 0.7
            else:
                total_len = xi_cont + xi_scaf + xi_chrom

                # Caso a cobertura exista, mas não haja informação de composição
                # de assembly, desenha apenas o contorno da barra.
                if total_len == 0 and cov_val > 0:
                    ax.barh(
                        yi,
                        cov_val,
                        height=height,
                        facecolor="none",
                        edgecolor="0.45",
                        linewidth=0.8,
                    )
                    total_len = cov_val

                label = (
                    f"{cov_val:.1f}% {short}\n"
                    f"({n_genomes_int}/{n_total_int})"
                )
                label_x = total_len + 0.7

            ax.text(
                label_x,
                yi,
                label,
                va="center",
                ha="left",
                fontsize=7.5,
                linespacing=0.9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()

    ax.set_xlim(0, 118)
    ax.set_xlabel("Species with available genomes (%)")
    ax.set_ylabel("Tetrapod group")
    ax.set_title(
        "Genomic coverage and assembly composition by group across regions"
    )

    ax.legend(
        title="Best assembly level",
        fontsize=8,
        title_fontsize=9,
        loc="lower right",
    )

    ax.grid(axis="x", linewidth=0.4, alpha=0.3)

    plt.tight_layout()

    out_png = (
        PATH_BASE
        / "plot_sampling_by_group_3regions_with_assemblies_counts.png"
    )
    out_pdf = (
        PATH_BASE
        / "plot_sampling_by_group_3regions_with_assemblies_counts.pdf"
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"📊 PNG salvo em: {out_png}")
    print(f"📄 PDF salvo em: {out_pdf}")


# ---------------------------------------------------------------------
# Fallback sem composição dos assemblies
# ---------------------------------------------------------------------
def plot_group_comparison_3regions_only(df_cov_all: pd.DataFrame):
    """Gera uma versão simples quando os níveis de montagem não estão disponíveis."""
    groups = sorted(df_cov_all["group"].dropna().unique().tolist())

    pivot_cov = (
        df_cov_all.pivot(index="group", columns="region", values="coverage_pct")
        .reindex(index=groups, columns=REGIONS)
    )
    pivot_total = (
        df_cov_all.pivot(index="group", columns="region", values="n_species_total")
        .reindex(index=groups, columns=REGIONS)
    )
    pivot_genomes = (
        df_cov_all.pivot(
            index="group", columns="region", values="n_species_with_genome"
        )
        .reindex(index=groups, columns=REGIONS)
    )

    region_short = {
        "Europe": "E",
        "NorthAmerica": "NA",
        "Neotropics": "NEO",
    }

    y = np.arange(len(groups))
    n_regions = len(REGIONS)
    height = 0.8 / n_regions

    fig_height = max(5.0, 0.9 * len(groups) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    for i, reg in enumerate(REGIONS):
        vals = pivot_cov[reg].fillna(0).to_numpy(dtype=float)
        offset = (i - (n_regions - 1) / 2) * height
        ys = y + offset

        ax.barh(ys, vals, height=height, label=reg)

        short = region_short.get(reg, reg)

        for grp, yi, xi in zip(groups, ys, vals):
            n_total = pivot_total.loc[grp, reg]
            n_genomes = pivot_genomes.loc[grp, reg]
            cov_val = pivot_cov.loc[grp, reg]

            if pd.isna(n_total):
                continue

            n_total_int = int(round(n_total))
            n_genomes_int = (
                0 if pd.isna(n_genomes) else int(round(n_genomes))
            )

            if n_total_int == 0:
                label = f"{short}: absent\n(0/0)"
                label_x = 0.7
            else:
                label = (
                    f"{cov_val:.1f}% {short}\n"
                    f"({n_genomes_int}/{n_total_int})"
                )
                label_x = xi + 0.7

            ax.text(
                label_x,
                yi,
                label,
                va="center",
                ha="left",
                fontsize=7.5,
                linespacing=0.9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Species with available genomes (%)")
    ax.set_ylabel("Tetrapod group")
    ax.set_title("Genomic coverage by group across regions")
    ax.legend(title="Region", fontsize=8, title_fontsize=9)
    ax.grid(axis="x", linewidth=0.4, alpha=0.3)

    plt.tight_layout()

    out_png = PATH_BASE / "plot_sampling_by_group_3regions_counts_simple.png"
    out_pdf = PATH_BASE / "plot_sampling_by_group_3regions_counts_simple.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"📊 PNG simples salvo em: {out_png}")
    print(f"📄 PDF simples salvo em: {out_pdf}")


if __name__ == "__main__":
    plot_group_comparison_3regions_with_assemblies()
