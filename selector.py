import logging
import os.path
import pickle
import hashlib
import azlib as az
import pandas as pd
import findb.manipulator
import yaml


def read_groups(groupsfile):
    """Read groupfile. Return a dictionary.

    Return a dictionary where key is the group name and value
    is a list of symbols for the given group. If cache-file
    exist (same name as 'groupsfile' argument but .p extension)
    with right SHA1 hash it will be used instead.

    Arguments:
    groupsfile -- name of the groupsfile to read (yaml-file)
    """

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
    """Use globber patterns to select symbols/groups.

    Use globber pattern(s) to match against group definitions
    that are fetched from groupsfile. If there is a match (or many
    matches) return the matching symbols as a sorted list.

    Arguments:
    selections   -- list of globber patterns
    groupsfile   -- groups will be read from this (yaml) file
    """

    if type(selections) is str:
        selections = [selections]

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
    """Return a dictionary of yahoo bars or delta-returns given selections.

    Returning dictionary will have key as a name of the symbol and value
    is the data that was fetched for that symbol.

    Arguments:
    selections   -- yahoo symbols or symbol groups to fetch
    bartype      -- what kind of data to fetch
                    "" = bars, "D" = daily delta,
                    "W" = weekly delta, "DS" = daily delta-score

    Keyword arguments:
    fetch_missing     -- fetch missing data (default: True)
    dlthreads         -- # of threads to use for download (default: 5)
    batchsize         -- batchsize for download (default: 100)
    dl_conv_usd       -- convert to usd after download (default: True)
    modifygroups      -- remove non-existing symbols from groupsfile 
                         (default: True)
    update_freq       -- update frequency (business days) when fetching data
                         (default: 1)
    """

    kwargs.setdefault("fetch_missing", True)
    kwargs.setdefault("dlthreads", 5)
    kwargs.setdefault("batchsize", 100)
    kwargs.setdefault("dl_conv_usd", True)
    kwargs.setdefault("modifygroups", True)
    kwargs.setdefault("update_freq", 1)

    home = os.path.expanduser("~")
    findbdir = os.path.join(home, 'findb')
    dbfile = os.path.join(findbdir, 'db.h5')
    groupfile = os.path.join(findbdir, 'yahoogroups.yaml')
    symbols = selections_to_symbols(selections, groupfile)

    if not os.path.isfile(dbfile):
        findb.manipulator.create_empty_db()

    if kwargs["fetch_missing"]:
        findb.manipulator.download_yahoo(symbols, kwargs["dlthreads"], findbdir, kwargs["batchsize"],
                           kwargs["dl_conv_usd"], kwargs["modifygroups"], kwargs["update_freq"])
        if bartype:
            findb.manipulator.fetch_deltas(symbols, findbdir)

#        missing_symbols = []
#        missing_deltas = []
#        db = tables.open_file('db.h5', 'r')
#
#        for sym in symbols:
#            basesymloc = "/yahoo/{}".format(sym)
#            if basesymloc not in db:
#                missing_symbols.append(sym)
#                missing_deltas.append(sym)
#            else:
#                attrs = db.get_node(basesymloc)._v_attrs
#                if not "last_update" in attrs:
#                    missing_symbols.append(sym)
#                    missing_deltas.append(sym)
#                
#        
#            symloc = "/yahoo/{}_{}".format(sym, bartype)
#            if basesymloc not in store:
#                missing_symbols.append(sym)
#                missing_deltas.append(sym)
#            elif symloc not in store:
#                missing_deltas.append(sym)
#
#
#        if missing_symbols:
#            logging.info("Downloading {} missing symbols...".format(len(missing_symbols)))
#            findb.manipulator.download_yahoo(missing_symbols, kwargs["dlthreads"], findbdir, kwargs["batchsize"],
#                           kwargs["dl_conv_usd"], kwargs["modifygroups"], kwargs["update_freq"])
#
#        if bartype and missing_deltas:
#            logging.info("Calculating deltas for {} symbols...".format(len(missing_deltas)))
#            findb.manipulator.fetch_deltas(missing_deltas, findbdir)
#
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
