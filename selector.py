import logging
import os.path
import pickle
# import hashlib
import azlib as az
import findb.manipulator
# import yaml


class ParseException(Exception):
    pass


def read_groups(shortcuts_file):
    """Read groupfile. Return a dictionary.

    Return a dictionary where key is the group name and value
    is a list of symbols for the given group. If cache-file
    exist (same name as 'groupsfile' argument but .p extension)
    with right SHA1 hash it will be used instead.

    Arguments:
    groupsfile -- name of the groupsfile to read (yaml-file)
    """
    with open(shortcuts_file, mode='r') as scfile:
        lines = scfile.readlines()
    res = {}
    groupnow = None
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if line[0] == "#":  # comment line
            continue
        if line[0] == "/":
            line = line[1:]
        if len(line) == 0:
            continue
        if line[-1] == ":":  # group definition
            groupnow = line[:-1]
            res[groupnow] = []
            continue
        if '/' not in line:
            raise ParseException("{}:{}: Data Provider not defined"
                                 .format(shortcuts_file, i + 1))
        res[groupnow].append(line)
    return res
    # try:
    #     with open(groupsfile, mode='rb') as myfile:
    #         sha = hashlib.sha1(myfile.read())
    #         groupshash = sha.digest()
    # except FileNotFoundError:
    #     pass

    # splitted = groupsfile.split(".")
    # cachefile = splitted[0] + ".p"
    # if os.path.isfile(cachefile):
    #     fromcache = pickle.load(open(cachefile, 'rb'))
    #     if groupshash and groupshash == fromcache["__SHA1__"]:
    #         del fromcache["__SHA1__"]
    #         return fromcache
    #     elif not groupshash:
    #         logging.warning("{} not found, using cachefile {}".format(groupsfile, cachefile))
    # # if ".yaml" in groupsfile:
    # # logging.debug("Reading symbol group definitions from {}...".format(groupsfile))
    # sym_groups = {}
    # if not groupshash:
    #     logging.warning("{} file not found, no symbol groups defined ...".format(groupsfile))
    #     return sym_groups

    # with open(groupsfile, mode='r') as myfile:
    #     logging.debug("Reading groups from non-cached {}...".format(groupsfile))
    #     data = yaml.load(myfile)
    #     if data:
    #         for k, v in data.items():
    #             sym_groups[k] = v

    # sym_groups["__SHA1__"] = groupshash
    # pickle.dump(sym_groups, open(cachefile, 'wb'))
    # del sym_groups["__SHA1__"]
    # return sym_groups
    # # else:
    # #     return pickle.load(open(groupsfile, 'rb'))


def selections_to_symbols(selections, shortcuts_file=None):
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

    if not shortcuts_file:
        home = os.path.expanduser("~")
        findbdir = os.path.join(home, 'findb')
        shortcuts_file = os.path.join(findbdir, 'shortcuts.conf')

    sym_groups = read_groups(shortcuts_file)

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
    dl_threads         -- # of threads to use for download (default: 5)
    update_freq       -- update frequency (business days) when fetching data
                         (default: 1)
    """

    fetch_missing = kwargs.pop("fetch_missing", True)
    dl_threads = kwargs.pop("dl_threads", 5)
    update_freq = kwargs.pop("update_freq", 1)
    findb_dir = kwargs.pop("findb_dir", findb.manipulator.default_findb_dir())
    groups_file = kwargs.pop('groups_file', os.path.join(findb_dir, 'yahoo_groups.yaml'))
    for kwarg in kwargs:
        raise Exception("Keyword argument '{}' not supported.".format(kwarg))

    if bartype == "":
        suffix = ""
    elif bartype == "D":
        suffix = "~D"
    elif bartype == "W":
        suffix = "~W"
    elif bartype == "DS":
        suffix = "~DS"
    else:
        raise Exception('Bartype "{}" not supported.'.format(bartype))

    if fetch_missing:
        selections = findb.manipulator.download_yahoo(selections, findb_dir=findb_dir,
                                                      update_freq=update_freq, 
                                                      dl_threads=dl_threads)
        if bartype:
            selections = findb.manipulator.fetch_deltas(selections, findb_dir=findb_dir)
    else:
        selections = selections_to_symbols(selections, groups_file)

    res = {}
    yahoo_dir = os.path.join(findb_dir, 'db', 'Yahoo')
    for sym in selections:
        fpath = os.path.join(yahoo_dir, sym + suffix)
        data = pickle.load(open(fpath, 'rb'))['data']
        if data is not None:
            res[sym] = data

    return res


def get_data(selections, datatype="C", **kwargs):
    """Generic fetcher"""
    fetch_missing = kwargs.pop("fetch_missing", True)
    dl_threads = kwargs.pop("dl_threads", 5)
    update_freq = kwargs.pop("update_freq", 1)
    findb_dir = kwargs.pop("findb_dir", findb.manipulator.default_findb_dir())
    shortcuts_file = kwargs.pop('shortcuts_file', os.path.join(findb_dir, 'shortcuts.conf'))
    for kwarg in kwargs:
        raise Exception("Keyword argument '{}' not supported.".format(kwarg))
    db_dir = os.path.join(findb_dir, 'db')
    delta_conv = False
    if datatype == "C":
        suffix = ""
    elif datatype == "A":
        suffix = ""
    elif datatype == "D":
        suffix = "~D"
        delta_conv = True
    elif datatype == "W":
        suffix = "~W"
        delta_conv = True
    elif datatype == "DS":
        suffix = "~DS"
        delta_conv = True
    else:
        raise Exception('Datatype "{}" is not supported.'.format(datatype))
    if fetch_missing:
        symbols = findb.manipulator.download_data(selections,
                                                  delta_convert=delta_conv,
                                                  dl_threads=dl_threads,
                                                  update_freq=update_freq,
                                                  findb_dir=findb_dir,
                                                  shortcuts_file=shortcuts_file)[1]
    else:
        symbols = selections_to_symbols(selections, shortcuts_file)
    res = {}
    for sym in symbols:
        if not findb.manipulator.symbol_in_db(sym, db_dir):
            continue
        fpath = os.path.join(db_dir, sym + suffix)
        if not os.path.isfile(fpath):
            continue
        data = pickle.load(open(fpath, 'rb'))['data']
        if data is not None:
            splitted = sym.split('/')
            if datatype == "C":
                data_provider = splitted[0]
                if data_provider == "Yahoo":
                    col = "AdjClose(USD)"
                elif data_provider == "Quandl":
                    assert len(splitted) >= 3
                    qdl_db = splitted[1]
                    if qdl_db == "WIKI":
                        col = "Adj. Close"
                    elif qdl_db == "BAVERAGE":
                        col = "24h Average"
                    else:
                        raise Exception("Quantdl database '{}' => unknown close column"
                                        .format(qdl_db))
                else:
                    raise Exception("Unknown data provider: {}".format(data_provider))
                res[sym] = data[col]
            else:
                res[sym] = data
    return res
