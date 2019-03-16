import pandas
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='paths out')
    parser.add_argument('-i', help='path in')
    args = parser.parse_args()
    list_dir = os.listdir(args.i)
    first_file = pandas.read_csv(args.i + '/' + list_dir[0])

    submission_names = first_file.to_numpy()[:, 0]
    labels = first_file.to_numpy()[:, 1]

    for id_file in range(len(list_dir) - 1):
        buf_csv = pandas.read_csv(args.i + '/' + list_dir[id_file])
        labels += buf_csv.to_numpy()[:, 1]

    labels /= len(list_dir)
    #labels[labels > 0.75] = 1

    df = pandas.DataFrame({'id': submission_names, 'label': labels})
    df.to_csv('{}.csv.gz'.format(args.p), index=False,
              compression='gzip')


if __name__ == '__main__':
    main()
