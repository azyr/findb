import findb.manipulator
import azlib.azlogging
import logging
import argparse
import os
import signal
import multiprocessing
import os.path
import pandas as pd


g_shutting_down = False


def sigint_handler(signal, frame):
    global g_shutting_down
    print("\nCtrl-C pressed, saving data to continue later...")
    g_shutting_down = True


def convert_csv(csvfile, outdir, col, sep):
    if g_shutting_down:
        return
    d_filename = os.path.join(outdir, os.path.basename(csvfile)[:-4] + "_D.csv")
    ds_filename = os.path.join(outdir, os.path.basename(csvfile)[:-4] + "_DS.csv")
    w_filename = os.path.join(outdir, os.path.basename(csvfile)[:-4] + "_W.csv")
    if os.path.isfile(d_filename) and os.path.isfile(w_filename) and os.path.isfile(ds_filename):
        logging.debug("Skipped {}".format(csvfile))
        return
    numerical_col = True
    try:
        int(col)
    except ValueError:
        numerical_col = False
    df = pd.DataFrame.from_csv(csvfile, sep=sep)
    if numerical_col:
        series = df[df.columns[col]]
    else:
        series = df[col]
    try:
        d, w, ds = findb.manipulator.deltaconvert(series)
    except findb.manipulator.DeltaConversionException as err:
        logging.info("Error converting {}: {}".format(csvfile, err))
        return
    d.to_csv(d_filename, sep=sep)
    ds.to_csv(ds_filename, sep=sep)
    w.to_csv(w_filename, sep=sep)
    logging.info("Succesfully converted {}".format(csvfile))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="extract the financial data from xls file")

    parser.add_argument("csvfiles", nargs='+', help="files to extract")
    parser.add_argument("col", help="column to convert")
    parser.add_argument("-mt", action="store_true", help="process multithreaded")
    parser.add_argument("--sep", default=",", help="separator for csvfile")
    parser.add_argument("--outdir", default="", help="output directory")

    vgroup = parser.add_mutually_exclusive_group()
    vgroup.add_argument("-v", action="count", default=0, help="verbosity")
    vgroup.add_argument("--quiet", action="store_true", help="disable stdout-output")

    args = parser.parse_args()

    azlib.azlogging.quick_config(args.v, args.quiet, fmt="")

    csvfiles = sorted(args.csvfiles)
    g_shutting_down = False
    signal.signal(signal.SIGINT, sigint_handler)
    print(args.sep)

    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    if not args.mt:
        for csvfile in csvfiles:
            convert_csv(csvfile, args.outdir, args.col, args.sep)
    else:
        def process_fun(csvfile):
            convert_csv(csvfile, args.outdir, args.col, args.sep)
        with multiprocessing.Pool() as pool:
            logging.info("Started pool with {} processes".format(pool._processes))
            pool.map(process_fun, csvfiles)
