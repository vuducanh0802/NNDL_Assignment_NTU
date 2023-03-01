#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import csv

from tqdm import tqdm



def load_csvfile(filename, column=None, delimiter='\t'):
    """opens text file and return sentences in a list. Specify a column header if required."""
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(
                    f,
                    delimiter=delimiter,
                    quotechar='"', #None,
                    doublequote=False,
                    lineterminator="\n",
                    #quoting=csv.QUOTE_NONE,
        )
        if column is not None:
            lines = [row[column] for row in reader]
        else:
            lines = [row for row in reader]
    return lines

def write_lines_to_file(lines,outdir,filename,outname):
    filter_out_empty_lines = lines #[l for l in lines if len(l)>0]
    print(f"outlines:{len(filter_out_empty_lines)}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = f"{outdir}/{filename}{outname}"
    print(f"saving: {outfile}")
    open(outfile, 'w', encoding='utf-8').writelines(l.strip()+'\n' for l in filter_out_empty_lines)

def get_max(alist: list):
    alist = [int(n) for n in alist]
    return max([(v,i) for i,v in enumerate(alist)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="input tsv file")

    args = parser.parse_args()

    lines = load_csvfile(args.input, "n_frames")
    print(get_max(lines))


if __name__ == "__main__":
    main()
