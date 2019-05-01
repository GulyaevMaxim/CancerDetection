import pandas
import argparse
import os
import sys

path = os.path.abspath(__file__).rsplit(os.path.sep, 2)[0]
path = os.path.join(path, 'common')
sys.path.append(path)
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='paths out')
    parser.add_argument('-i', help='path in')
    args = parser.parse_args()
    list_dir = os.listdir(args.i)
    first_file = pandas.read_csv(args.i + '/' + list_dir[0])

    submission_names = first_file.values[:, 0]
    labels = first_file.values[:, 1]

    for id_file in range(len(list_dir) - 1):
        buf_csv = pandas.read_csv(args.i + '/' + list_dir[id_file])
        labels += buf_csv.values[:, 1]

    labels /= len(list_dir)

    utils.generate_submission(submission_names, labels, args.p)


if __name__ == '__main__':
    main()
