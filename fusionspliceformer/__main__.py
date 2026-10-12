"""The same inference interface as the CPU/GPU scripts."""
import argparse
from .predict import prediction_main


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args,remaining=parser.parse_known_args()
    prediction_main(args.device,remaining)


if __name__=='__main__': main()
