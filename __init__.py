import argparse
import logging
import yaml
import os
import concurrent
import yahoodl
import sys
import re
import deltaconv
from datetime import datetime, timedelta, date
import azlib as az
import azlib.azlogging


def read_groups(groupfile):
    sym_groups = {}
    try:
        with open(groupfile, mode='r') as myfile:
            data = yaml.load(myfile)
            for k, v in data.items():
                sym_groups[k] = v
    except FileNotFoundError:
        logging.warning("{} file not found, no symbol groups defined ...".format(groupfile))
    return sym_groups


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="tool for constructing financial database")

    parser.add_argument("selection", nargs="?", help="symbols to check correlation for")

    vgroup = parser.add_mutually_exclusive_group()
    vgroup.add_argument("-v", action="count", default=0, help="verbosity")
    vgroup.add_argument("--quiet", action="store_true", help="disable stdout-output")

    parser.add_argument("--provider", default="yahoo", choices=["yahoo"], help="data provider")
    parser.add_argument("--db", default="db", help="database folder for yahoo data")
    parser.add_argument("--groups", default="yahoogroups.yaml", help="yaml-file containing symbol groups")
    parser.add_argument("--modifygroups", action="store_true",
                        help="allow script to remove non-available entries from groups-file")
    parser.add_argument("--dlthreads", type=int, default=5, help="number of threads to use to download data")
    parser.add_argument("-dl", action="store_true", help="download data")
    parser.add_argument("-usd", action="store_true", help="convert data to USD")
    parser.add_argument("-delta", action="store_true", help="calculate changes")

    args = parser.parse_args()

    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    if args.quiet:
        az.azlogging.basic_config(level=logging.WARNING, format=fmt, datefmt='%j-%H:%M:%S')
    else:
        if args.v == 0:
            az.azlogging.basic_config(level=logging.INFO, format=fmt, datefmt='%j-%H:%M:%S')
        elif args.v == 1:
            az.azlogging.basic_config(level=logging.DEBUG, format=fmt, datefmt='%j-%H:%M:%S')
        elif args.v >= 2:
            az.azlogging.basic_config(level=logging.NOTSET, format=fmt, datefmt='%j-%H:%M:%S')

    db_directory = args.db
    if not os.path.exists(db_directory):
        os.makedirs(db_directory)

    symbols_to_fetch = []

    if args.dl and args.provider == "yahoo":

        if not args.selection:
            logging.critical("Cannot download data because no symbols were selected.")
            sys.exit(2)
        for sym in args.selection:
            if ".csv" in sym.lower():
                logging.critical("When downloading, select symbol/group names instead of filenames.")
                sys.exit(2)

        logging.debug("Reading symbol group definitions ...")
        sym_groups = read_groups(args.groups)

        symbols_to_fetch = []
        groups_selected = []
        for entry in args.symbols:
            expanded = az.globber(entry, sym_groups)
            if not expanded:
                symbols_to_fetch.append(entry)
            else:
                groups_selected += expanded
                for group in expanded:
                    symbols_to_fetch += sym_groups[group]
        logging.debug("Groups selected: {}".format(set(groups_selected)))

        symbols_to_fetch = set(symbols_to_fetch)

        symbols_to_remove_from_groups = []

        yahoodl_results = {}

        # grab the data from Yahoo with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.dlthreads) as executor:
            future_to_symbol = {executor.submit(yahoodl.dl, sym, db_directory): sym for sym in symbols_to_fetch}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    res = future.result()
                except Exception as err:
                    logging.error("Error when processing {}: {}".format(symbol, err))
                    yahoodl_results[symbol] = err
                if type(res) is str:
                    if res == "Not Found":
                        symbols_to_remove_from_groups.append(symbol)
                    yahoodl_results[symbol] = res
                    logging.debug("Result when downloading {}: {}".format(symbol, res))

        # check for possible unexpected errors
        errors = ""
        for sym, res in yahoodl_results.items():
            if type(res) is Exception:
                errors += "Exception occurred when downloading {}: \n\n{}".format(sym, res)
        if errors:
            logging.critical(errors)
            sys.exit(2)

        for symbol in symbols_to_remove_from_groups:
            symbols_to_fetch.remove(symbol)

        if args.modifygroups and os.path.isfile(args.groups) and symbols_to_remove_from_groups:
            logging.info("Removing invalid symbols from {} ...".format(args.groups))
            with open(args.groups, mode='r') as myfile:
                lines = myfile.readlines()
            for symbol in symbols_to_remove_from_groups:
                for i in range(len(lines) - 1, -1, -1):
                    matches = re.search('[\'"]{}[\'"]'.format(symbol), lines[i])
                    if matches:
                        del lines[i]
                        logging.info("Removed {} from line {}.".format(symbol, i))
            with open(args.groups, mode='w') as myfile:
                myfile.writelines(lines)

    if symbols_to_fetch:
        filenames = []
        for symbol in symbols_to_fetch:
            filenames.append(os.path.join(db_directory, symbol + ".csv"))
    else:
        filenames = args.selection

    if args.usd:
        yahoodl.convert_to_usd(filenames)

    if args.delta:
        for filename in filenames:
            # check how many columns and generate changes based on the last column
            with open(filename, 'r') as csvfile:
                header = csvfile.readline()
                splitted = header.split(',')
            if deltaconv.convert(filename, len(splitted) - 1):
                logging.debug("Delta-conversion/spike fixes for {} succesfully completed.".format(filename))
            else:
                logging.info("Delta-conversion/spike fixes for {} failed.".format(filename))
