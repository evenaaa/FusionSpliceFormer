"""Predict all prepared events using trained weights on an NVIDIA GPU."""
from fusionspliceformer.predict import prediction_main

if __name__ == '__main__':
    prediction_main('cuda')
