#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/9 18:04
# @Author  : even
# @File    : features.py
from features.utills.exon_annotation import get_bed_annotation
from features.utills.features_annotation import features_annotation

def features(variant_path='../dataset/toy.gtf.txt'):

    variant_df=get_bed_annotation(
        variant_path=variant_path,
    )

    feature_acceptor_df=features_annotation(
        variant_df=variant_df,
        region='acceptor')

    feature_donor_df=features_annotation(
        variant_df=variant_df,
        region='donor')
    return feature_acceptor_df,feature_donor_df
