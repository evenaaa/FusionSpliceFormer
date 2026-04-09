#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/9 21:22
# @Author  : even
# @File    : FusionSpliceFormer.py
from features.features import features
from predict.predict import prediction_pipeline
import argparse
def print_prompt(pipename):
    print('\033[1;36m┌────────────────────────────────────────────┐\033[0m')
    print(f'\033[1;36m│           ►  {pipename.upper()} │\033[0m')
    print('\033[1;36m└────────────────────────────────────────────┘\033[0m')

def pipeline():


    parser = argparse.ArgumentParser(description='This is a script for prediction.')
    parser.add_argument('-vcf', help='It must have input vcf file', default=r'dataset/toy.vcf')
    parser.add_argument('-out', help='Prediction output file.', default=r'dataset/toy.predict.csv')

    # Parse command-line parameters
    args = parser.parse_args()

    # Get the value of the -i parameter
    variant_path=args.vcf
    outfile=args.out
    print_prompt('features pipeline')
    acceptor_df,donor_df=features(variant_path=variant_path)
    print_prompt('prediction pipeline')
    prediction_pipeline(acceptor_df, donor_df, outfile=outfile)
if __name__ == '__main__':
    pipeline()