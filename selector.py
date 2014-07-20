import yaml
import logging
import os.path
import pickle
import hashlib
import azlib as az
import pandas as pd
from datetime import datetime, date
# from .settings import *

FILTER_DICT = {
    "date": 0,
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "adjclose": 6
}


def read_groups(groupsfile):

    try:
        with open(groupsfile, mode='rb') as myfile:
            sha = hashlib.sha1(myfile.read())
            groupshash = sha.digest()
    except FileNotFoundError:
        pass

    splitted = groupsfile.split(".")
    cachefile = splitted[0] + ".p"
    if os.path.isfile(cachefile):
        fromcache = pickle.load(open(cachefile, 'rb'))
        if groupshash and groupshash == fromcache["__SHA1__"]:
            del fromcache["__SHA1__"]
            return fromcache
        elif not groupshash:
            logging.warning("{} not found, using cachefile {}".format(groupsfile, cachefile))
    # if ".yaml" in groupsfile:
    # logging.debug("Reading symbol group definitions from {}...".format(groupsfile))
    sym_groups = {}
    if not groupshash:
        logging.warning("{} file not found, no symbol groups defined ...".format(groupsfile))
        return sym_groups

    with open(groupsfile, mode='r') as myfile:
        logging.debug("Reading groups from non-cached {}...".format(groupsfile))
        data = yaml.load(myfile)
        if data:
            for k, v in data.items():
                sym_groups[k] = v

    sym_groups["__SHA1__"] = groupshash
    pickle.dump(sym_groups, open(cachefile, 'wb'))
    del sym_groups["__SHA1__"]
    return sym_groups
    # else:
    #     return pickle.load(open(groupsfile, 'rb'))


def selections_to_symbols(selections, groupsfile):

    sym_groups = read_groups(groupsfile)

    symbols = []
    groups_selected = []
    for entry in selections:
        expanded = az.globber(entry, sym_groups)
        if not expanded:
            symbols.append(entry)
        else:
            groups_selected += expanded
            for group in expanded:
                symbols += sym_groups[group]
    logging.debug("Groups selected: {}".format(set(groups_selected)))

    symbols = list(set(symbols))
    symbols.sort()
    return symbols


def get_yahoo_bars(selections, bartype=""):

    home = os.path.expanduser("~")
    findbdir = os.path.join(home, 'findb')
    dbfile = os.path.join(findbdir, 'db.h5')
    groupfile = os.path.join(findbdir, 'yahoogroups.yaml')
    symbols = selections_to_symbols(selections, groupfile)

    store = pd.io.pytables.HDFStore(dbfile, 'r')

    res = {}

    for sym in symbols:

        if not bartype:
            symloc = "/yahoo/{}".format(sym)
        else:
            symloc = "/yahoo/{}{}_".format(sym, bartype)
        res[sym] = store[symloc]

    store.close()
    return res

#
# def get_weekly_delta(selections):
#
#     home = os.path.expanduser("~")
#     findbdir = os.path.join(home, 'findb')
#     dbfile = os.path.join(findbdir, 'db.h5')
#     groupfile = os.path.join(findbdir, 'yahoogroups.yaml')
#     symbols = selections_to_symbols(selections, groupfile)
#
#     store = pd.io.pytables.HDFStore(dbfile, 'r')
#
#     res = {}
#
#     for sym in symbols:
#
#         symloc = "/yahoo/{}_W".format(sym)
#         res[sym] = store[symloc]
#
#     store.close()
#     return res




# def get_daily_bars(selections, **kwargs):
#
#     use_dateord = False
#     if "dateord" in kwargs:
#         use_dateord = kwargs["dateord"]
#
#     if "filter" in kwargs:
#         txtfilter = kwargs["filter"]
#         txtfilter = txtfilter.split(",")
#         for i in len(txtfilter):
#             txtfilter[i] = txtfilter[i].strip().lower()
#         filter = {FILTER_DICT[s]: i for i, s in enumerate(txtfilter)}
#     else:
#         filter = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
#
#
#     symbols = selections_to_symbols(selections, YAHOO_GROUPS)
#     logging.info("Fetching data for {} symbols...".format(len(symbols)))
#     results = {}
#
#     # TODO: possibly this could be faster?
#     for symbol in symbols:
#         filename = os.path.join(PROGRAM_DIR, "db", "daily", "ohlcvc", symbol + ".csv")
#         # filenames.append(filename)
#         with open(filename, 'r') as csvfile:
#             data = [[]] * len(filter)
#             csvfile.readline()  # skip header
#             lines = csvfile.readlines()
#             lines.reverse()
#             for line in lines:
#                 splitted = line.split(",")
#                 if 0 in filter:
#                     date = splitted[0]
#                     if use_dateord:
#                         date = datetime.strptime(date, "%Y-%m-%d").date().toordinal()
#                     data[filter[0]].append(date)
#                 if 1 in filter:
#                     data[filter[1]].append(float(splitted[1]))
#                 if 2 in filter:
#                     data[filter[2]].append(float(splitted[2]))
#                 if 3 in filter:
#                     data[filter[3]].append(float(splitted[3]))
#                 if 4 in filter:
#                     data[filter[4]].append(float(splitted[4]))
#                 if 5 in filter:
#                     data[filter[5]].append(int(splitted[5]))
#                 if 6 in filter:
#                     # always use the USD column
#                     data[filter[6]].append(float(splitted[-1]))
#             results[symbol] = data
#
#     return results