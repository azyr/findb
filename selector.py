import yaml
import logging
import os.path
import pickle
import hashlib
import azlib as az
import pandas as pd
from datetime import datetime, date
import findb.manipulator

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


def selections_to_symbols(selections, groupsfile=None):

    if not groupsfile:
        home = os.path.expanduser("~")
        findbdir = os.path.join(home, 'findb')
        groupsfile = os.path.join(findbdir, 'yahoogroups.yaml')

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
    if groups_selected:
        logging.debug("Groups selected: {}".format(set(groups_selected)))

    symbols = list(set(symbols))
    symbols.sort()
    return symbols


def get_yahoo_bars(selections, bartype="", **kwargs):

    kwargs.setdefault("fetch_missing", True)
    kwargs.setdefault("dlthreads", 5)
    kwargs.setdefault("batchsize", 100)
    kwargs.setdefault("dl_conv_usd", True)
    kwargs.setdefault("modifygroups", True)

    home = os.path.expanduser("~")
    findbdir = os.path.join(home, 'findb')
    dbfile = os.path.join(findbdir, 'db.h5')
    groupfile = os.path.join(findbdir, 'yahoogroups.yaml')
    symbols = selections_to_symbols(selections, groupfile)



    if kwargs["fetch_missing"]:

        missing_symbols = []
        missing_deltas = []
        store = pd.io.pytables.HDFStore(dbfile, 'r')

        for sym in symbols:
            basesymloc = "/yahoo/{}".format(sym)
            symloc = "/yahoo/{}_{}".format(sym, bartype)
            if basesymloc not in store:
                missing_symbols.append(sym)
                missing_deltas.append(sym)
            elif symloc not in store:
                missing_deltas.append(sym)

        store.close()

        if missing_symbols:
            logging.info("Downloading {} missing symbols...".format(len(missing_symbols)))
            findb.manipulator.download_yahoo(missing_symbols, kwargs["dlthreads"], findbdir, kwargs["batchsize"],
                           kwargs["dl_conv_usd"], kwargs["modifygroups"])

        if bartype and missing_deltas:
            logging.info("Calculating deltas for {} symbols...".format(len(missing_deltas)))
            findb.manipulator.fetch_deltas(missing_deltas, findbdir)

    res = {}
    store = pd.io.pytables.HDFStore(dbfile, 'r')

    for sym in symbols:
        basesymloc = "/yahoo/{}".format(sym)
        symloc = "/yahoo/{}_{}".format(sym, bartype)
        if not bartype:
            if basesymloc not in store:
                logging.warning("No data found for {}".format(sym))
            else:
                res[sym] = store[basesymloc]
        else:
            if symloc not in store:
                logging.warning("No delta-data found for {}".format(sym))
            else:
                res[sym] = store[symloc]

    store.close()
    return res
