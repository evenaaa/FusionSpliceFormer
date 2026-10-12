"""Predict all prepared events using trained weights on the CPU."""
from fusionspliceformer.predict import prediction_main

if __name__ == '__main__':
    prediction_main('cpu')
