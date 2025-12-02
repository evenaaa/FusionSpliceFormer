#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/14 12:09
# @Author  : even
# @File    : data_annotation.py
import argparse
from utills.exon_annotation import *
from utills.features_annotation import *
from utills.seq_tokenize import *

def print_prompt(step,prompt):
    print('\033[1;36m┌────────────────────────────────────────────┐\033[0m')
    print(f'\033[1;36m│ ► Step {step}: {prompt} │\033[0m')
    print('\033[1;36m└────────────────────────────────────────────┘\033[0m')

def data_annotation():
    parser = argparse.ArgumentParser(description='This is a script for features annotation.')
    parser.add_argument('-i', help='It must have input data', default=r'dataset/toy.vcf')
    parser.add_argument('-vcf', help='VCF output file.', default=r'dataset/toy.features.vcf')
    parser.add_argument('-seq', help='Seq Output file.', default=r'dataset/toy.seqIds.json')

    # Parse command-line parameters
    args = parser.parse_args()

    # Obtain the value of the -i parameter
    variant_path = args.i
    feature_path=args.vcf
    seq_path=args.seq

    # start exon annotation
    print_prompt(step=1,prompt='BedTools Input File')
    gtf_df=exon_annotation(variant_path=variant_path)

    # start feature annotation
    print_prompt(step=2,prompt='Feature Annotation')
    feature_df=features_annotation(gtf_df,feature_path=feature_path)

    # seq tokenize
    print_prompt(step=3,prompt='Seq Tokenize')
    seq_tokenize(
            feature_df=feature_df,
            token_path=seq_path,
            tokenizers_path='dataset/modeldata',
    )

if __name__ == '__main__':
    data_annotation()