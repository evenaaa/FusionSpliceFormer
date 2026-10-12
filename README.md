# FusionSpliceFormer

FusionSpliceFormer predicts splice pathogenicity by combining reference/variant sequence representations with numerical features. Pretrained weights are included, with prediction supported on CPU and NVIDIA GPU.

[Install](#install) · [References](#prepare-references) · [Run](#quick-start) · [Input](#input) · [Output](#output) · [Options](#options)

## Install

We recommend **Python 3.10 and PyTorch 2.7.1**, based on our development and validation environment. Open a terminal in the extracted project folder containing `predict_cpu.py` and `requirements.txt`.

Choose **one** installation:

**CPU**

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**NVIDIA GPU** — requires a CUDA-capable GPU and compatible driver.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Prepare references

Download these five files into `downloads/`, keeping their filenames and annotation versions. FASTA/GTF files may remain compressed. Allow **at least 50 GB free space** for setup; prepared references occupy approximately **20 GB**.

| Resource | Download |
| --- | --- |
| GRCh38 genome | [GRCh38.primary_assembly.genome.fa.gz](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.primary_assembly.genome.fa.gz) |
| MANE v1.4 | [MANE.GRCh38.v1.4.ensembl_genomic.gtf.gz](https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.4/MANE.GRCh38.v1.4.ensembl_genomic.gtf.gz) |
| GENCODE v47 Basic | [gencode.v47.basic.annotation.gtf.gz](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz) |
| phyloP | [hg38.phyloP100way.bw](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw) |
| phastCons | [hg38.phastCons100way.bw](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw) |

Prepare once:

```bash
python prepare_references.py -I downloads
```

This prepares your downloaded files in `refs/`; it does not download them.

<details>
<summary>Custom reference location or interrupted setup</summary>

Use `-R other_refs` for both preparation and prediction to change the destination. Existing files are checked rather than overwritten. If reference construction is interrupted, retain that directory and prepare into a new destination. Changing annotation versions can change predictions.

</details>

## Quick start

Run from the project folder, choosing either CPU or GPU:

```bash
# CPU
python predict_cpu.py -I toy/input.vcf -O toy/output.tsv

# NVIDIA GPU
python predict_gpu.py -I toy/input.vcf -O toy/output.tsv
```

Interrupted runs can be resumed by rerunning the same command with unchanged inputs and settings. Existing output files are not overwritten.

## Input

Use an **hg38/GRCh38 VCF** (`.vcf` or `.vcf.gz`), as in [toy/input.vcf](toy/input.vcf):

```text
##fileformat=VCFv4.2
##reference=GRCh38
#CHROM  POS       ID  REF  ALT  QUAL  FILTER  INFO
chr1    51791663  .   T    C    .     PASS    .
chr1    51816311  .   C    T    .     PASS    .
```

Only **SNVs in hg38/GRCh38** are supported. Use tab-separated columns and 1-based positions; `1` and `chr1` are both accepted. All ALT alleles are processed, so filter unwanted records or alleles beforehand.

<details>
<summary>Optional: specify target sites in a TSV</summary>

To specify targets yourself, use a TSV with one variant–site pair per row:

```text
chrom  pos       ref  alt  site_pos  strand  site_type
chr1   51791663  T    C    51791661  -       acceptor
chr1   51816311  C    T    51816312  -       donor
```

`site_pos` is the 1-based exon boundary base adjacent to the intron, not an AG/GT base. Use genomic-forward REF/ALT bases and targets present in the prepared annotations. Pass this file with `-I`; each pair produces one result row.

</details>

## Output

VCF output contains **one row per unique CHROM/POS/REF/ALT**, reporting the highest-scoring annotated target within 5 kb:

```text
CHROM  POS  REF  ALT  GENE  TRANSCRIPTS  SITE_POS  STRAND  SITE_TYPE  SCORE  PREDICTION
```

- `SCORE`: average prediction from the ten models for the reported target.
- `PREDICTION`: **Pathogenic** if the score ≥0.5, otherwise **Benign**; **Not_scored** if unavailable.
- Gene, transcript and site columns describe the reported target. `SITE_POS` is its 1-based exon boundary; multiple transcript IDs are separated by `;`.

Predictions are intended for research and variant prioritization, not clinical diagnosis.

To output **all target-site results** instead:

```bash
python predict_cpu.py -I toy/input.vcf -O toy/output_details.tsv --details
```

## Options

| Option | Purpose | Default |
| --- | --- | --- |
| `-B` | Batch size; reduce if memory is limited | 4 |
| `-R` | Prepared reference directory | refs |
| `--details` | Report all target sites | Off |
| `--chunk-size` | VCF event chunk size | 1024 |

These options apply to CPU and GPU. Run `python predict_cpu.py -h` or `python predict_gpu.py -h` for all options.

## License

FusionSpliceFormer is released under the [MIT License](LICENSE). Third-party resource terms apply separately.
