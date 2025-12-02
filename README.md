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
conda create -n env python=3.9.10 -y

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

#### step 2: Download PhyloP annotation data

```shell
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw
```

#### step 3: Download annotation data and trained model files

Due to GitHub's data limitations, we have placed some annotation files and pre-trained model files in a cloud storage drive to provide users with the ability to download them and facilitate the reproduction of the project.
```commandline
https://pan.baidu.com/s/1Y5Y5FGSJm7FXnAUVfXdKhQ?pwd=cj3u
```


## 2. usage

### 2.1 features Annotation

To run the data annotation file, you need to prepare a tab-delimited annotation input site file. 
This file is the target file that you want to annotate. The format and content of the input file are as follows:

| Data Field  |                  Description                 |
|:------|:---------------------------------------------|
| chrom | Chromosome identifier in the reference genome |
| pos   |Start position of the variant on the chromosome|
| ref   | Reference base(s) at the genomic position     |
| alt   | Alternative base(s) observed in the sample    |
| nm_id | RefSeq transcript accession number            |

Also, if you only want to use it for testing purposes, you can first use the file located in the local path: ``dataset/toy.vcf``

The feature annotations perform tokenization of numerical and sequence features on the target input file. The resulting feature data will be saved in ``toy.features.vcf`` and ``toy.seqIds.json``

```shell
python features/data_annotation.py -i toy.vcf -vcf toy.features.tsv -seq toy.seqIds.json
```

### 2.2 Annotate variants
Run **predict.py** to predict mutation effects. 

```shell
python predict/predict.py -vcf toy.features.tsv -seq toy.seqIds.json -o toy.predict.tsv
```
