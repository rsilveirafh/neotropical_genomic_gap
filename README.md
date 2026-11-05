# Genomics in the Neotropics: genomic coverage of tetrapods

Code and data for the bioinformatic case study in:

> Pirani RM, Rainha RN, Magalhães FM, da Silveira Filho RR, Werneck FP, Domingos FMCB and
> Vasconcellos MM (2026) **Genomics in the Neotropics: Advances, biases, and future directions for
> evolutionary research in Tetrapoda.** *Genetics and Molecular Biology*.
> doi:[10.1590/1678-4685-GMB-2025-0276](https://doi.org/10.1590/1678-4685-GMB-2025-0276)

## What this is

Genome sequencing has mostly benefited species from high-income regions, which leaves the
Neotropics, the most biodiverse region on Earth, underrepresented. To measure this gap, we
combined genome assembly metadata from **NCBI** with species occurrence data from **GBIF** for
the six tetrapod groups (Amphibia, Aves, Crocodylia, Lepidosauria, Mammalia and Testudines). The
pipeline asks, for each group, family and country: of all the species that occur there, how many
have at least one genome assembly?

It estimates:

1. the number of species in the Neotropical region, per group and family;
2. the number of those species with at least one genome assembly in NCBI;
3. genomic coverage (species with a genome ÷ all species), per group, per family and per country;
4. the assembly level of the genomes (contig, scaffold or chromosome);
5. the accumulation of species with genomes over time (2002–2025);
6. the same coverage for **Europe** and **North America**, for comparison.

Main results: Aves and Mammalia have many high-quality genomes, while the more diverse
Amphibia and Lepidosauria remain neglected (about 2% and 3% of Neotropical species have a genome),
and the Neotropics lag behind Europe and North America in every group except Crocodylia (which
has a single species in Europe).

## How the data were built

The analysis starts from curated tables in `data/inputs/`, assembled beforehand from public
databases (full details in the article's Supplementary Methods):

- **Genomes (NCBI Datasets).** For each group, all assemblies were retrieved with the NCBI Datasets
  CLI (`datasets summary genome taxon <taxid> --assembly-version latest`) and converted to a table
  with `dataformat tsv genome`. Only the best assembly per species was kept
  (chromosome/complete > scaffold > contig; ties broken by the most recent release).
- **Occurrences (GBIF).** Species–country lists from the GBIF API (`/occurrence/search`, filtered by
  `establishmentMeans = NATIVE` and `occurrenceStatus = PRESENT`). Species were treated as extant
  when GBIF marks them as non-extinct or when they have at least one record after 1950.
- **Taxonomy.** GBIF and NCBI family names were reconciled through the GBIF Backbone Taxonomy
  (normalised names, exact matching, then conservative fuzzy matching checked by hand).
- **Neotropical region.** A fixed list of ISO 3166-1 alpha-2 country codes covering Mesoamerica,
  the Caribbean and South America (see `NEOTROPICAL_ISO2` in `scripts/01_coverage_by_region.py`). A
  species counts as Neotropical if it occurs in at least one of these countries.

## Repository structure

```
data/inputs/       input tables (see below)
scripts/
  01_coverage_by_region.py              coverage by group, family and country, per region (Tables S1–S4, Figures S1–S5, Figure 3)
  02_compare_regions.py                 coverage by group across the three regions (Figure 1)
  03_compare_regions_with_assemblies.py same comparison, with the assembly-level composition of each bar
  04_assembly_levels_by_group.py        best assembly level per species, by group (Table S5)
  05_map_coverage_by_country.py         country maps of coverage for the six groups (Figure 4)
supplementary/
  tables/csv/      Tables S1–S5 (csv)
  tables/xlsx/     Tables S1–S5 (formatted for the journal)
  figures/         Figures S1–S5
tools/
  remove_content_control.py  helper that strips content controls from .docx files (manuscript editing)
outputs/           created when the scripts run (not versioned)
```

### Inputs

| File | Content |
|---|---|
| `families_seen_live.csv` | extant species per country and group (`iso2`, `speciesKey`, `group`) |
| `species_country_filtered_clean.tsv` | species–country matrix from GBIF (`usageKey`, `parent_name`, `iso2`, `species`) |
| `species_family_cache.csv` | GBIF species key to family, after reconciliation with NCBI (`speciesKey`, `family`) |
| `famscan_all_assemblies_filtered.tsv` | NCBI assemblies (`Organism Name`, `Assembly Level`, `Assembly Release Date`, …) |
| `representativity_by_group.csv` | worldwide coverage by group (`group`, `n_species_with_genome`, `n_gbif_species`) |

## How to run

Requires Python 3.10 or later.

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/01_coverage_by_region.py   # about 2 minutes; run it first
python scripts/02_compare_regions.py
python scripts/03_compare_regions_with_assemblies.py
python scripts/04_assembly_levels_by_group.py
python scripts/05_map_coverage_by_country.py   # downloads Natural Earth country borders
```

The scripts can be run from any folder. Results go to `outputs/`:

| Output | In the article |
|---|---|
| `outputs/plot_sampling_by_group_3regions_horizontal.png` | Figure 1 |
| `outputs/region_outputs/Neotropics/cumulative_genomes_over_time_Neotropics.png` | Figure 3 |
| `outputs/region_outputs/Neotropics/sampling_by_group_Neotropics_unique.csv` | Table S1 |
| `outputs/region_outputs/Neotropics/sampling_by_family_Neotropics_unique.csv` | Table S2 |
| `outputs/region_outputs/Neotropics/sampling_country_by_group_Neotropics.csv` | Table S3 |
| `outputs/region_outputs/Neotropics/sampling_country_by_family_Neotropics.csv` | Table S4 |
| `outputs/assembly_levels_by_group_neotropics.csv` | Table S5 |
| `outputs/region_outputs/Neotropics/map_country_group_3x2_Neotropics_quantiles.tif` | Figure 4 |
| `outputs/region_outputs/Neotropics/plot_sampling_by_family_<group>.png` | Figures S1–S5 |

Each region folder (`Neotropics`, `Europe`, `NorthAmerica`) also holds the family rankings,
assembly-level composition, country × group heatmap and richness × coverage plots.

In the coverage tables, `n_species_total` is the number of species, `n_species_with_genome` the
number with at least one assembly, and `sampling_proportion` the ratio between them (0–1).

## License

Code under the MIT License (see `LICENSE`). The input tables derive from NCBI and GBIF; please
cite those sources and the article when reusing them.

## Contact

Ricardo R. da Silveira Filho · [github.com/rsilveirafh](https://github.com/rsilveirafh)
