import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--o', default='dvmcar.pt', type=str, help='output file')
args = parser.parse_args()

print(args.o.split('.'))