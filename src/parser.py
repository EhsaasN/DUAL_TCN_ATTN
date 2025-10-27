import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='LSTM_NDT',
                    help="model name")
parser.add_argument('--Device',
                    metavar='-D',
                    type=str,
                    required=False,
                    default='cpu',
                    help="cuda or cpu")
parser.add_argument('--test',
                    action='store_true',
                    help="test the model")
parser.add_argument('--retrain',
                    action='store_true',
                    help="retrain the model")
parser.add_argument('--less',
                    action='store_true',
                    help="train using less data")
parser.add_argument('--scheduler',
                    type=str,
                    required=False,
                    default='step',
                    choices=['step', 'cosine', 'reduce_on_plateau', 'exponential'],
                    help="learning rate scheduler type: 'step', 'cosine', 'reduce_on_plateau', or 'exponential'")
parser.add_argument('--early_stopping_patience',
                    type=int,
                    required=False,
                    default=7,
                    help="number of epochs to wait before early stopping")
args = parser.parse_args()