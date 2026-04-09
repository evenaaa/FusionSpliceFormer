# 1. Installation
Follow these steps to get the model up and running on your local machine.

## 1.1 Clone Repository

```shell
git clone https://github.com/evenaaa/FusionSpliceFormer.git
cd FusionSpliceFormer
```

## 1.2 Requirement
### 1.2.1 OS Requirements
Our packages have been tested and are confirmed to be compatible with the following systems:

- Ubuntu 22.04.3 LTS
- RHEL 7.9

It should be compatible with other common Linux systems.

### 1.2.2 Software Dependencies

We recommend that you use the package management tool **Anaconda** to configure the environment and manage dependencies to avoid conflicts with the system environment or the dependencies of other projects.

```shell
# create environment
conda create -n env python=3.10.0 -y

# activate environment
conda activate env
```

The Python dependencies required to run the project are listed in the requirements.txt file. You can install them using the following command:

```shell
# install required dependencies
pip install -r requirements.txt
```
### 1.2.3 Dataset Requirements

Our model requires an assembly file and annotation databases to assist in locating genes and their strands. Therefore, we kindly ask you to follow the instructions carefully and download the listed files.

#### step 1: Download the Ensemble genome assembly file

```shell
# download
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.p14.genome.fa.gz

# unzip
gunzip GRCh38.p14.genome.fa.gz
```

After getting the fa files, we need to use samtools to build the index. Please make sure that you have installed samtools.

```shell
# samtools index 
samtools faidx GRCh38.p14.genome.fa
```

#### step 2: Download PhyloP and PhastCons annotation data

```shell
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw
```

#### step 3: Unzip gencode gtf annotation data

```shell
cd dataset/annodata/gtf
unzip gencode.v47.anno.zip
```

#### File Check
make sure that the files listed in the table are present in the path of the "dataset" directory:

- dataset/annodata/fa/GRCh38.p14.genome.fa
- dataset/annodata/fa/GRCh38.p14.genome.fa.fai
- dataset/annodata/conservation/hg38.phyloP100way.bw
- dataset/annodata/conservation/hg38.phastCons100way.bw
- dataset/annodata/gtf/gencode.v47.anno.gtf


## 2. usage

### 2.1 Preparing input data

To run the data annotation file, you need to prepare a tab-delimited annotation input site file. 
This file is the target file that you want to annotate. The format and content of the input file are as follows:

| Data Field  |                  Description                 |
|:------|:---------------------------------------------|
| chrom | Chromosome identifier in the reference genome |
| pos   |Start position of the variant on the chromosome|
| ref   | Reference base(s) at the genomic position     |
| alt   | Alternative base(s) observed in the sample    |

Also, if you only want to use it for testing purposes, you can use the file located in the local path: ``dataset/toy.vcf``

### 2.2 Annotate variants

Run **FusionSpliceFormer.py** to predict mutation effects. 

```shell
python FusionSpliceFormer.py -vcf dataset/toy.vcf -seq -out dataset/toy.predict.csv
```
