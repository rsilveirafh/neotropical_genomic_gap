# neotropical_genomic_gap

Code and data for **"Genomics in the Neotropics: Advances, Biases, and Future Directions for
Evolutionary Research in Tetrapods"** (accepted in *Genetics and Molecular Biology*).

The pipeline measures how many tetrapod species have a genome assembly, by group, family and
country, and compares the Neotropics with Europe and North America.

## Structure

```
scripts/
  01_pipeline_regions.py         genomic coverage by region (offline version)
  02_compare_regions_by_group.py coverage by group across the three regions
data/inputs/                     input tables (GBIF species lists, assemblies, caches)
supplementary/
  tables/csv, tables/xlsx        Tables S1–S5
  figures/                       Figures S1–S5
tools/
  remove_content_control.py      helper to strip content controls from .docx files
```

## Inputs

| File | Content |
|---|---|
| `families_seen_live.csv` | species seen per country and group (`iso2`, `speciesKey`, `group`) |
| `species_country_filtered_clean.tsv` | species with a genome assembly, by country (`usageKey`, `parent_name`, `iso2`) |
| `species_family_cache.csv` | species-to-family lookup (`speciesKey`, `family`) |
| `famscan_all_assemblies_filtered.tsv` | assemblies (`Organism Name`, `Assembly Level`, `Assembly Release Date`) |
| `representativity_by_group.csv` | global coverage by group (`n_species_with_genome`, `n_gbif_species`) |

## Running

Requires Python 3 with the packages in `requirements.txt`.

```sh
pip install -r requirements.txt
python scripts/01_pipeline_regions.py
python scripts/02_compare_regions_by_group.py
```

Outputs are written to `region_outputs/<Region>/`.

## License

MIT (see `LICENSE`).
