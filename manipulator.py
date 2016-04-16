import argparse
import json
import logging
import os
import concurrent.futures
import yahoodl
import Quandl
import hashlib
import pickle
import threading
import os.path
import findb.selector
import azlib as az
import azlib.azlogging
import pandas as pd
import pandas.tseries.offsets
# import pandas.io.data
import bottleneck as bn
import numpy as np
from datetime import datetime
from pprint import pformat


# """Lookup table for FRED symbols.
# 
# Fred is missing some currencies that appear in Yahoo Finance:
#     ILS = Israeli New Sheqel
#     RUB = Russian Ruble
#     IDR = Indonesian Rupiah
#     ARS = Argentine Peso
# """
# fred_currencies = {
#     # USD/XYZ
#     "EUR": "DEXUSEU",
#     "GBP": "DEXUSUK",
#     "CAD": "DEXCAUS",
#     "AUD": "DEXUSAL",
#     "NZD": "DEXUSNZ",
#     # XYZ/USD
#     "CHF": "DEXSZUS",
#     "JPY": "DEXJPUS",
#     "MXN": "DEXMXUS",
#     "HKD": "DEXHKUS",
#     "ZAR": "DEXSFUS",
#     "SEK": "DEXSDUS",
#     "SGD": "DEXSIUS",
#     "NOK": "DEXNOUS",
#     "DKK": "DEXDNUS",
#     "BRL": "DEXBZUS",
#     "CNY": "DEXCHUS",
#     "INR": "DEXINUS",
#     "KRW": "DEXKOUS",
#     "MYR": "DEXMAUS",
#     "TWD": "DEXTAUS"
# }


class FXDataNotFoundException(Exception):
    pass


class DeltaConversionException(Exception):
    pass


def default_findb_dir():
    home = os.path.expanduser('~')
    return os.path.join(home, 'findb')


def check_pfile_uptodate(fpath, update_freq):
    if not os.path.isfile(fpath):
        return False
    try:
        fdict = pickle.load(open(fpath, 'rb'))
    except EOFError as err:
        os.remove(fpath)
        logging.info("Removed corrupt file from db ({}): {!r}".format(fpath, err))
        return False
    if 'do_not_update' in fdict and fdict['do_not_update']:
        return True
    if 'last_update' not in fdict:
        return False
    return az.utcnow() - pd.tseries.offsets.BDay(update_freq) < fdict['last_update']


def save_pfile(data, fpath, save_hash=False):
    fdict = {'data': data, 'last_update': az.utcnow()}
    if save_hash:
        fdict['sha1'] = hashlib.sha1(pickle.dumps(data)).digest()
    pickle.dump(fdict, open(fpath, 'wb'))


def load_pfile(fpath):
    return pickle.load(open(fpath, 'rb'))['data']


def load_yahoo_sym_data(findb_dir=None):
    if not findb_dir:
        findb_dir = default_findb_dir()
    fpath = os.path.join(findb_dir, 'yahoo_symbols.csv')
    return pd.DataFrame.from_csv(fpath, sep='|')


def check_yahoo_uptodate(sym, update_freq=1, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    filepath = os.path.join(db_dir, 'Yahoo', sym)
    return check_pfile_uptodate(filepath, update_freq)


def check_quandl_uptodate(sym, update_freq=1, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    filepath = os.path.join(db_dir, 'Quandl', sym)
    return check_pfile_uptodate(filepath, update_freq)


def symbol_in_db(sym, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    filepath = os.path.join(db_dir, sym)
    return os.path.isfile(filepath)


def dl_and_process_yahoo_symbol(sym, currency=None, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    yahoo_dir = os.path.join(db_dir, 'Yahoo')
    fpath = os.path.join(yahoo_dir, sym)
    df = yahoodl.dl(sym, currency)
    df = df.copy()
    # df = convert_to_usd(df, db_dir)
    save_pfile(df, fpath, save_hash=True)


def dl_and_process_quandl_symbol(sym, db_dir=None, authtoken=""):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    quandl_dir = os.path.join(db_dir, 'Quandl')
    fpath = os.path.join(quandl_dir, sym)
    df = Quandl.get(sym, authtoken=authtoken)
    dirname = os.path.dirname(fpath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_pfile(df, fpath, save_hash=True)


# def update_fred_fxdata(update_freq=1, db_dir=None):
#     if not db_dir:
#         db_dir = os.path.join(default_findb_dir(), 'db')
#     sym_to_update = []
#     for sym in fred_currencies.values():
#         fpath = os.path.join(db_dir, 'FRED', sym)
#         if not check_pfile_uptodate(fpath, update_freq):
#             sym_to_update.append(sym)
#     if not sym_to_update:
#         return
#     start = datetime(1900, 1, 1)
#     end = datetime(2020, 1, 1)
#     logging.debug("{} FRED currencies up-to-date, downloading {} pairs..."
#                   .format(len(fred_currencies) - len(sym_to_update), len(sym_to_update)))
#     fxdata = pandas.io.data.DataReader(sym_to_update, 'fred', start, end)
#     for sym in fxdata:
#         fpath = os.path.join(db_dir, 'FRED', sym)
#         save_pfile(fxdata[sym], fpath)


# This is required to serialize access to certain pandas operations
mylock = threading.Lock()


def convert_to_usd(x, db_dir=None):
    """
    x = series
    TODO: implement this! FRED doesnt provide fxdata anymore!
    """
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    return x
    # lastcol = df.columns[-1]
    # currency = lastcol[-4:-1]
    # if currency == "USD":  # conversion not required
    #     return df
    # try:
    #     fxname = fred_currencies[currency]
    # except KeyError:
    #     raise FXDataNotFoundException(currency)
    # fxfileloc = os.path.join(db_dir, 'FRED', fxname)
    # fxdata = load_pfile(fxfileloc)
    # # fill missing dates (i.e. US holidays) using the last available value
    # fxdata = fxdata.fillna(method='ffill')
    # # Seems like division and multiply operators are not thread safe even for read operations!
    # with mylock:
    #     # take inverse of usdxxx pairs (ending "XXUS")
    #     if fxname[-2:] == "US":
    #         fxdata = 1 / fxdata
    # # import threading
    # # tident = hex(threading.current_thread().ident)
    # # pickle.dump([df, fxdata], open('/home/seb/temp/dfdump/{}'.format(tident), 'wb'))
    #     df["AdjClose(USD)"] = df[lastcol] * fxdata
    #     return df

def strip_data_provider(sym, provider):
    try:
        return sym[sym.index(provider + '/') + len(provider) + 1:]
    except ValueError:
        return sym

def freeze_files(selections, **kwargs):
    return set_flags(selections, {'do_not_update': True}, **kwargs)


def defreeze_files(selections, **kwargs):
    return set_flags(selections, {'do_not_update': False}, **kwargs)


def set_flags(selections, flags, **kwargs):
    """Set flags for pfiles (internal use only)

    Return list of failed symbols.
    """
    if 'data' in flags:
        raise Exception("data is a reserved flag, please check flags")
    if 'last_update' in flags:
        raise Exception("last_update is a reserved flag, please check flags")
    findb_dir = kwargs.pop("findb_dir", findb.manipulator.default_findb_dir())
    shortcuts_file = kwargs.pop('shortcuts_file', os.path.join(findb_dir, 'shortcuts.conf'))
    for kwarg in kwargs:
        raise Exception("Keyword argument '{}' not supported.".format(kwarg))
    symbols = findb.selector.selections_to_symbols(selections, shortcuts_file)
    failed = []
    for sym in symbols:
        fpath = os.path.join(findb_dir, 'db', sym)
        if not os.path.isfile(fpath):
            logging.warning("Cannot set flags for {}: file not found".format(fpath))
            failed.append(sym)
            continue
        try:
            fdict = pickle.load(open(fpath, 'rb'))
        except EOFError as err:
            logging.warning("Cannot set flags for {}: corrupt file".format(fpath))
            failed.append(sym)
            continue
        file_changed = False
        for flag in flags:
            if not flag in fdict or flags[flag] != fdict[flag]:
                fdict[flag] = flags[flag]
                file_changed = True
        if file_changed:
            pickle.dump(fdict, open(fpath, 'wb'))
    return failed


def download_data(selections, **kwargs):
    """Generic fetcher"""
    delta_convert = kwargs.pop("delta_convert", True)
    dl_threads = kwargs.pop("dl_threads", 5)
    update_freq = kwargs.pop("update_freq", 1)
    retry = kwargs.pop("retry", False)
    findb_dir = kwargs.pop("findb_dir", findb.manipulator.default_findb_dir())
    shortcuts_file = kwargs.pop('shortcuts_file', os.path.join(findb_dir, 'shortcuts.conf'))
    for kwarg in kwargs:
        raise Exception("Keyword argument '{}' not supported.".format(kwarg))

    symbols = findb.selector.selections_to_symbols(selections, shortcuts_file)
    yahoo_symbols = [s for s in symbols if s.find('Yahoo/') == 0]
    quandl_symbols = [s for s in symbols if s.find('Quandl/') == 0]
    while True:
        downloaded = set()
        if yahoo_symbols:
            yres = findb.manipulator.download_yahoo(yahoo_symbols, findb_dir=findb_dir,
                                                    update_freq=update_freq,
                                                    dl_threads=dl_threads)
            yres = ["Yahoo/" + s for s in yres]
            downloaded = downloaded.union(set(yres))
        if quandl_symbols:
            qres = findb.manipulator.download_quandl(quandl_symbols, findb_dir=findb_dir,
                                                    update_freq=update_freq,
                                                    dl_threads=dl_threads)
            qres = ["Quandl/" + s for s in qres]
            downloaded = downloaded.union(set(qres))
        converted = downloaded
        if delta_convert:
            converted = fetch_deltas(downloaded, findb_dir=findb_dir)
        if retry and len(downloaded) != len(symbols):
            logging.debug("Downloaded {} / {}, retrying..."
                          .format(len(downloaded), len(symbols)))
            continue
        return downloaded, converted


def download_yahoo(selections, **kwargs):

    """Download daily data from Yahoo Finance.

    Returns a list of succesfully processed symbols.

    Arguments:
    selections       -- Symbols or symbol groups to download
    dl_threads       -- # of threads to use when downloading (default: 25)
    findbdir         -- Database directory (default: $HOME/findb)
    modify_groups    -- Remove non-existent symbols from yahoogroups.yaml.
                        This is not thread-safe so use carefully when other
                        processes might access the file simultaneously.
                        (default: True)
    update_freq      -- How many business days to wait until updating data
                        (default: 1)
    """

    dl_threads = kwargs.pop('dl_threads', 25)
    findb_dir = kwargs.pop('findb_dir', default_findb_dir())
    db_dir = kwargs.pop('db_dir', os.path.join(findb_dir, 'db'))
    update_freq = kwargs.pop('update_freq', 1)
    for kwarg in kwargs:
        raise Exception("Keyword argument '{}' not supported.".format(kwarg))
    if not selections:
        raise Exception("No symbols selected")
    if type(selections) is str:
        selections = [selections]

    yahoo_dir = os.path.join(db_dir, 'Yahoo')
    if not os.path.exists(yahoo_dir):
        os.makedirs(yahoo_dir)
    fred_dir = os.path.join(db_dir, 'FRED')
    if not os.path.exists(fred_dir):
        os.makedirs(fred_dir)

    all_symbols = [strip_data_provider(x, "Yahoo") for x in selections]
    ysd = load_yahoo_sym_data(findb_dir)
    # filter out bad data (everything should be in sym_data file)
    good_symbols = list(filter(lambda x: x in ysd.index, all_symbols))
    diff = set(all_symbols).difference(set(good_symbols))
    if diff:
        logging.warning("Symbols not present in yahoo_symbols.csv:\n{}".format(pformat(diff)))
    if not good_symbols:
        logging.debug("No symbols to download.")
        return
    l = lambda x: not check_yahoo_uptodate(x, update_freq=update_freq, db_dir=db_dir)
    symbols_to_dl = list(filter(l, good_symbols))
    uptodate = set(good_symbols).difference(set(symbols_to_dl))
    logging.info("{} Yahoo symbols up-to-date, {} to update."
                 .format(len(uptodate), len(symbols_to_dl)))
    if not symbols_to_dl:
        logging.debug("No symbols to download.")
        return sorted(list(uptodate))

    succesful_symbols = []
    symbols_without_data = []
    symbols_without_currency = []
    symbols_without_fxdata = []
    symbols_with_invalid_data = []

    # only need the warning/error messages
    if 'urllib3.connectionpool' in logging.Logger.manager.loggerDict:
        lg = logging.Logger.manager.loggerDict['urllib3.connectionpool']
        lg.setLevel(logging.WARNING)

    # no more threads than symbols to download to prevent deadlocks
    num_threads = min(len(symbols_to_dl), dl_threads)
    yahoodl.configure_downloader(num_threads)

    logging.debug("Starting to download with {} thread(s)...".format(num_threads))
    start_time = az.utcnow()
    sym_processed = 0
    total_processed = 0
    dlps = dl_and_process_yahoo_symbol  # we need an abbreviation
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        fut_to_sym = {executor.submit(dlps, sym, ysd.loc[sym, 'Currency'], db_dir):
                      sym for sym in symbols_to_dl}
        for fut in concurrent.futures.as_completed(fut_to_sym):
            total_processed += 1
            sym = fut_to_sym[fut]
            ex = fut.exception()
            if ex:
                logging.debug(ex.__repr__())
            if not ex:
                sym_processed += 1
                succesful_symbols.append(sym)
            elif type(ex) is yahoodl.YahooDataNotFoundException:
                symbols_without_data.append(sym)
            elif type(ex) is yahoodl.CurrencyNotFoundException:
                symbols_without_currency.append(sym)
            elif type(ex) is yahoodl.InvalidDataException:
                symbols_with_invalid_data.append(sym)
            elif type(ex) is FXDataNotFoundException:
                symbols_without_fxdata.append(sym)
            time_elapsed = az.utcnow() - start_time
            sym_per_sec = total_processed / time_elapsed.total_seconds()
            progress_str = "{}/{} (suc/total) symbols processed. Speed: {:.2f} symbols/second."\
                .format(sym_processed, total_processed, sym_per_sec)
            if total_processed % 500 == 0:
                logging.info(progress_str)
            else:
                logging.debug(progress_str)

    if symbols_without_data:
        logging.warning("Symbols without data: \n{}".format(pformat(symbols_without_data)))
    if symbols_without_currency:
        logging.warning("Symbols without currency: \n{}".format(pformat(symbols_without_currency)))
    if symbols_without_currency:
        logging.warning("Symbols without fxdata: \n{}".format(pformat(symbols_without_fxdata)))
    if symbols_with_invalid_data:
        logging.warning("Symbols with invalid data: \n{}"
                        .format(pformat(symbols_with_invalid_data)))

    progress_str = "{}/{} (suc/total) symbols processed. Speed: {:.2f} symbols/second."\
        .format(sym_processed, total_processed, sym_per_sec)
    logging.info(progress_str)
    if symbols_without_data:
        logging.warning("{} symbols without data.".format(len(symbols_without_data)))
    if symbols_without_currency:
        logging.warning("{} symbols without currency.".format(len(symbols_without_currency)))
    if symbols_without_fxdata:
        logging.warning("{} symbols without fxdata.".format(len(symbols_without_fxdata)))
    if symbols_with_invalid_data:
        logging.warning("{} symbols with invalid data.".format(len(symbols_with_invalid_data)))

    # if modify_groups and os.path.isfile(groups_file) and symbols_without_data:
    #     logging.warning("Removing {} symbols without data from {} ..."
    #                     .format(len(symbols_without_data), groups_file))
    #     with open(groups_file, mode='r') as myfile:
    #         lines = myfile.readlines()
    #     for symbol in symbols_without_data:
    #         lam = lambda x: not re.search('[\'"]{}[\'"]'.format(symbol), x)
    #         lines = list(filter(lam, lines))
    #         # for i in range(len(lines) - 1, -1, -1):
    #         #     if re.match('[\'"]{}[\'"]'.format(symbol), lines[i]
    #         #         del lines[i]
    #         #         # logging.info("Removed {} from line {}.".format(symbol, i))
    #     with open(groups_file, mode='w') as myfile:
    #         myfile.writelines(lines)

    return sorted(list(set(succesful_symbols).union(uptodate)))


def download_quandl(selections, **kwargs):

    """Download daily data from Quandl.

    Returns a list of succesfully processed symbols.

    Arguments:
    selections       -- Symbols or symbol groups to download
    dl_threads       -- # of threads to use when downloading (default: 25)
    findbdir         -- Database directory (default: $HOME/findb)
    update_freq      -- How many business days to wait until updating data
                        (default: 1)
    auth_token       -- Authentication token for Quandl
    """

    dl_threads = kwargs.pop('dl_threads', 25)
    findb_dir = kwargs.pop('findb_dir', default_findb_dir())
    db_dir = kwargs.pop('db_dir', os.path.join(findb_dir, 'db'))
    update_freq = kwargs.pop('update_freq', 1)
    auth_token = kwargs.pop('auth_token', '')
    if not auth_token:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        settings = json.load(open(os.path.join(script_dir, "settings.json")))
        auth_token = settings["quandl_token"]
    for kwarg in kwargs:
        raise Exception("Keyword argument '{}' not supported.".format(kwarg))
    if not selections:
        raise Exception("No symbols selected")
    if type(selections) is str:
        selections = [selections]

    quandl_dir = os.path.join(db_dir, 'Quandl')
    if not os.path.exists(quandl_dir):
        os.makedirs(quandl_dir)

    all_symbols = [strip_data_provider(x, "Quandl") for x in selections]
    l = lambda x: not check_quandl_uptodate(x, update_freq=update_freq, db_dir=db_dir)
    symbols_to_dl = list(filter(l, all_symbols))
    uptodate = set(all_symbols).difference(set(symbols_to_dl))
    logging.info("{} Quandl symbols up-to-date, {} to update."
                 .format(len(uptodate), len(symbols_to_dl)))
    if not symbols_to_dl:
        logging.debug("No symbols to download.")
        return sorted(list(uptodate))

    succesful_symbols = []
    symbols_without_data = []

    # no more threads than symbols to download to prevent deadlocks
    num_threads = min(len(symbols_to_dl), dl_threads)

    logging.debug("Starting to download with {} thread(s)...".format(num_threads))
    start_time = az.utcnow()
    sym_processed = 0
    total_processed = 0
    dlps = dl_and_process_quandl_symbol  # we need an abbreviation
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        fut_to_sym = {executor.submit(dlps, sym, db_dir, auth_token):
                      sym for sym in symbols_to_dl}
        for fut in concurrent.futures.as_completed(fut_to_sym):
            total_processed += 1
            sym = fut_to_sym[fut]
            ex = fut.exception()
            if ex:
                logging.debug("{}: {}".format(sym, ex.__repr__()))
            if not ex:
                sym_processed += 1
                succesful_symbols.append(sym)
            elif type(ex) is Quandl.Quandl.DatasetNotFound:
                symbols_without_data.append(sym)
            elif type(ex) is Quandl.Quandl.CallLimitExceeded:
                # sometimes this exception is thrown when data just
                # simple doesn't exist! don't take it seriously ...
                symbols_without_data.append(sym)
                # logging.warning("Cancelling download because" + 
                #                 " daily limit has been exceeded.")
                # for fut in fut_to_sym:
                #     try:
                #         fut.cancel()
                #     except concurrent.futures.CancelledError:
                #         logging.debug("{} cancelled already".format(fut_to_sym[fut]))
                # break
            time_elapsed = az.utcnow() - start_time
            sym_per_sec = total_processed / time_elapsed.total_seconds()
            progress_str = "{}/{} (suc/total) symbols processed. Speed: {:.2f} symbols/second."\
                .format(sym_processed, total_processed, sym_per_sec)
            if total_processed % 50 == 0:
                logging.info(progress_str)
            else:
                logging.debug(progress_str)

    if symbols_without_data:
        logging.warning("Symbols without data: \n{}".format(pformat(symbols_without_data)))

    progress_str = "{}/{} (suc/total) symbols processed. Speed: {:.2f} symbols/second."\
        .format(sym_processed, total_processed, sym_per_sec)
    logging.info(progress_str)
    if symbols_without_data:
        logging.warning("{} symbols without data.".format(len(symbols_without_data)))

    return sorted(list(set(succesful_symbols).union(uptodate)))


def change_to_score(change):
    """Create a change-score out of a return 'change'

    Used internally. Changescore shows proper magnitude of loss/gain.
    Simple change percent will make gains always look bigger than losses
    when standardized.

    Arguments:
    change -- return to calculate the change-score for
    """
    if change >= 1:
        changescore = change - 1
    elif change < 1:
        changescore = -1 / change
        changescore += 1
    return changescore


def deltaconvert(series, visualize=False, max_adj_outliers=10):

    """Perform delta-conversion to given pd.Series.

    Delta-conversion returns 3 series as a tuple and possibly error message.

    First series (D) contains daily returns where all the data has been removed
    that could make comparison difficult with other assets.

    Second series (W) contains weekly returns where all the data has been removed
    that could make comparison difficult with other assets.

    Third series (DS) contains daily returns where all the outliers and erroneus
    data points have been removed but holes in data are not taken into account.
    This is more suitable for calculating performance scores etc.

    Arguments:
    series           -- series to use
    visualize        -- visualize results
    max_adj_outliers -- maximum number of adjancent outliers, if there are
                        actually more adjancent outliers than this then they will
                        not be considered outliers anymore. default value: 10
    """
    # MEDIAN_LEN = 50
    ZSCORE_CUT_RATIO = 2

    series = series.dropna()

    if len(series) < 50:
        raise DeltaConversionException("Not enough data")

    lines_taken = 0

    if series.index[0] > series.index[-1]:
        raise DeltaConversionException("Wrong cronological order")

    # closes = []
    # dates = []
    # datesord = []

    # for line in lines:
        # splitted = line.split(",")
        # closes.append(float(splitted[column]))
        # dt = datetime.strptime(splitted[0], "%Y-%m-%d").date()
        # dates.append(dt)
        # datesord.append(dt.toordinal())

    # if datesord[-1] < datesord[0]:
    #     closes.reverse()
    #     dates.reverse()
    #     datesord.reverse()
    #     lines.reverse()

    closes = series
    dates = series.index

    num_invalid_prices = 0
    deltapct = [np.nan]
    changescores = [np.nan]
    invalid_price_indices = []
    for i in range(1, len(series)):
        if closes[i - 1] > 0 and closes[i] > 0:
            change = closes[i] / closes[i - 1]
            deltapct.append(change - 1)
            changescore = change_to_score(change)
            changescores.append(changescore)
        else:
            deltapct.append(np.nan)
            changescores.append(np.nan)
            num_invalid_prices += 1
            invalid_price_indices.append(i)
            logging.debug("Cannot determine changescore at {} ({} / {})".format(dates[i], closes[i], closes[i - 1]))

    # # remove zeroes (data may only end with price zero if stock goes bankrupt...)
    # first_nonzero_idx = [i for i, val in enumerate(closes[:-1]) if val == 0]
    # del closes[:first_nonzero_idx]
    # del dates[:first_nonzero_idx]
    # lines_taken += first_nonzero_idx
    # if first_nonzero_idx > 0:
    #     logging.debug("{}: removed {} zero-lines from the beginning.".format(filename, first_nonzero_idx))

    num_gaps = 0
    num_invalid_chrono_orders = 0
    gap_indices = []
    for i in range(len(dates) - 1, 0, -1):
        d = (dates[i] - dates[i - 1]).days
        # standard weekends are only allowed
        if d == 3:
            if dates[i].weekday() != 0:  # not monday
                # deltapct[i] = np.nan
                # changescores[i] = np.nan
                num_gaps += 1
                gap_indices.append(i)
                logging.log(5, "Non-weekend gap of 2 day(s) at {}".format(dates[i]))
        elif d > 1:
            # deltapct[i] = np.nan
            # changescores[i] = np.nan
            num_gaps += 1
            gap_indices.append(i)
            logging.log(5, "Non-weekend gap of {} day(s) at {}".format(d, dates[i]))
        elif d <= 0:
            del deltapct[i], dates[i], closes[closes.index[i]], changescores[i]
            logging.warning(5, "Invalid chronological order ({} day(s)) at {}"
                            .format(d - 1, dates[i]))
            num_invalid_chrono_orders += 1

    deltapct = np.asarray(deltapct)
    changescores = np.asarray(changescores)
    std_score = bn.nanstd(changescores)
    zscores = np.abs(changescores) / std_score
    mean_z = bn.nanmean(zscores)
    zscores_set = list(set(zscores[(~np.isnan(zscores)) & (zscores > 0)]))
    zscores_set.sort()

    outlier_z = None
    maxpctdiff = 0
    for i in range(int(len(zscores_set) * .95), len(zscores_set)):
        pctdiff = zscores_set[i] / zscores_set[i - 1]
        maxpctdiff = pctdiff
        # logging.info("{}: {}".format(i / len(zscores_set), pctdiff))
        if pctdiff >= 2:
            outlier_z = zscores_set[i]
            second_highest_z = zscores_set[i - 1]
            break

    possible_outliers = []
    confirmed_outliers = []
    localmean_factors = []

    if outlier_z:
        logging.log(5, "Outlier z-score: {:.2f}, earlier z-score: {:.2f}, mean z-score: {:.5f}"
                    .format(outlier_z, second_highest_z, mean_z))

        for i in range(len(zscores)):
            if zscores[i] >= outlier_z:
                localmean = bn.nanmean(zscores[max(0, i - 50):min(len(zscores) + 1, i + 50)])
                localmean_factor = np.sqrt(mean_z / localmean)
                score = (zscores[i] / second_highest_z) * localmean_factor
                logging.log(5, "Possible outlier at {}: localmean_factor: {:.2f}, zscore: {:.2f}, score: {:.2f}"
                            .format(dates[i], localmean_factor, zscores[i], score))
                if score >= ZSCORE_CUT_RATIO:
                    logging.debug("Possible outlier at {} (z-score={:.2f}, deltapct={:.2%})"
                                  .format(dates[i], zscores[i], deltapct[i]))
                    # deltapct[i] = np.nan
                    possible_outliers.append(i)
                    localmean_factors.append(localmean_factor)

        if len(possible_outliers) == 1:
            confirmed_outliers = possible_outliers

        for i in range(1, len(possible_outliers)):

            firstidx = possible_outliers[i - 1]
            secondidx = possible_outliers[i]
            # opposite signs and not too far from each other
            if deltapct[firstidx] * deltapct[secondidx] < 0 \
                    and secondidx - firstidx + 1 <= max_adj_outliers:
                firstnonan = None
                for i2 in range(firstidx, -1, -1):
                    if not np.isnan(deltapct[i2]):
                        firstnonan = i2
                        break
                confirmed = False
                if not firstnonan:
                    confirmed = True
                if firstnonan:
                    if i == 1:
                        left_mean = bn.nanmedian(closes[max(0, firstnonan - (max_adj_outliers - 1)):firstnonan + 1])
                    else:
                        left_mean = bn.nanmedian(closes[max(0, possible_outliers[i - 2], \
                                                            firstnonan - (max_adj_outliers - 1)):firstnonan + 1])
                    right_mean = bn.nanmedian(closes[firstidx:secondidx])
                    changescore = change_to_score(right_mean / left_mean)
                    zscore = abs(changescore) / std_score
                    score_left_vs_mid = (zscore / second_highest_z) * localmean_factors[i - 1]
                    left_mean = right_mean
                    right_mean = bn.nanmedian(closes[secondidx:min(secondidx + max_adj_outliers, len(closes))])
                    changescore = change_to_score(right_mean / left_mean)
                    zscore = abs(changescore) / std_score
                    score_mid_vs_right = (zscore / second_highest_z) * localmean_factors[i]
                    if score_left_vs_mid > ZSCORE_CUT_RATIO * .75 and score_mid_vs_right > ZSCORE_CUT_RATIO * .75:
                        confirmed = True
                if confirmed:
                    indices = [i2 for i2 in range(firstidx, secondidx + 1)]
                    deltapct[indices] = np.nan
                    confirmed_outliers += indices

    else:
        logging.debug("No possible outliers found based on initial z-score analysis (maxpctdiff: {})"
                      .format(maxpctdiff))

    if visualize:
        # TODO: make this work with DataFrame
        pass
        # closes_arr = np.asarray(closes.get_values())
        # datesord = np.asarray(datesord)
        # plt.subplot(2, 1, 1)
        # plt.plot(datesord - datesord[0], closes_arr, 'b*')
        # plt.plot(datesord[gap_indices] - datesord[0], closes_arr[gap_indices], 'ob')
        # plt.plot(datesord[confirmed_outliers] - datesord[0], closes_arr[confirmed_outliers], 'or')
        # plt.plot(datesord[invalid_price_indices] - datesord[0], closes_arr[invalid_price_indices], 'om')
        # plt.subplot(2, 1, 2)
        # plt.plot(datesord - datesord[0], zscores, 'o')
        # plt.show()

    logging.debug("Conversion result: lines = {}, invalid closes = {}, gaps = {}, invalid dates = {}, outliers = {}"
                  .format(len(series) - lines_taken, num_invalid_prices, num_gaps,
                          num_invalid_chrono_orders, len(confirmed_outliers)))

    indices_to_rem = list(set(gap_indices + confirmed_outliers + invalid_price_indices))
    # datesordmod = np.delete(datesord, indices_to_rem)
    datesmod = dates.copy()
    datesmod = datesmod.delete(indices_to_rem)
    deltapctmod = np.delete(deltapct, indices_to_rem)
    closesmod = closes.drop(closes.index[indices_to_rem])
    assert(not np.any(np.isnan(deltapctmod[1:])))

    weeklydeltapct = []
    weeklydatesmod = []
    lastidx = -1
    # resample to W-FRI (could be done with pandas)
    for i in range(len(closesmod)):
        if datesmod[i].weekday() == 4:
            dd = (datesmod[i] - datesmod[lastidx]).days
            if lastidx >= 0 or dd == 7:
                if closesmod[lastidx] >= 0:
                    weeklydeltapct.append(closesmod[i] / closesmod[lastidx] - 1)
                    weeklydatesmod.append(datesmod[i])
            else:
                logging.log(5, "Weekly bar at {} skipped (delta: {} days)".format(datesmod[i], i, dd))
            lastidx = i

    res_daily = pd.Series(deltapctmod, datesmod)
    res_weekly = pd.Series(weeklydeltapct, weeklydatesmod)


    indices_to_rem = list(set(confirmed_outliers + invalid_price_indices))
    datesmod = dates.copy()
    datesmod = datesmod.delete(indices_to_rem)
    deltapctmod = np.delete(deltapct, indices_to_rem)
    assert(not np.any(np.isnan(deltapctmod[1:])))

    res_dailyscore = pd.Series(deltapctmod, datesmod)

    return res_daily, res_weekly, res_dailyscore


def check_delta_uptodate(sym, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    symloc = os.path.join(db_dir, sym)
    dloc = symloc + '~D'
    wloc = symloc + '~W'
    dsloc = symloc + '~DS'
    if not os.path.isfile(dloc) or not os.path.isfile(wloc) or not os.path.isfile(dsloc):
        return False
    try:
        source_sha1 = pickle.load(open(symloc, 'rb'))['sha1']
    except EOFError as err:
        os.remove(symloc)
        logging.info("Removed corrupted file from db ({}): {!r}".format(symloc, err))
        return False
    try:
        if pickle.load(open(dloc, 'rb'))['source_sha1'] != source_sha1:
            return False
        if pickle.load(open(wloc, 'rb'))['source_sha1'] != source_sha1:
            return False
        if pickle.load(open(dsloc, 'rb'))['source_sha1'] != source_sha1:
            return False
    except EOFError as err:
        if os.path.isfile(dloc):
            os.remove(dloc)
        if os.path.isfile(wloc):
            os.remove(wloc)
        if os.path.isfile(dsloc):
            os.remove(dsloc)
        logging.info("Removed corrupted delta files for symbol {}: {!r}".format(sym, err))
        return False
    return True


def save_deltas(data, sym, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    symloc = os.path.join(db_dir, sym)
    dloc = symloc + '~D'
    wloc = symloc + '~W'
    dsloc = symloc + '~DS'
    source_sha1 = pickle.load(open(symloc, 'rb'))['sha1']
    d_data = {'source_sha1': source_sha1, 'data': data[0]}
    w_data = {'source_sha1': source_sha1, 'data': data[1]}
    ds_data = {'source_sha1': source_sha1, 'data': data[2]}
    pickle.dump(d_data, open(dloc, 'wb'))
    pickle.dump(w_data, open(wloc, 'wb'))
    pickle.dump(ds_data, open(dsloc, 'wb'))


def save_invalid_deltas(sym, err, db_dir=None):
    if not db_dir:
        db_dir = os.path.join(default_findb_dir(), 'db')
    symloc = os.path.join(db_dir, sym)
    dloc = symloc + '~D'
    wloc = symloc + '~W'
    dsloc = symloc + '~DS'
    source_sha1 = pickle.load(open(symloc, 'rb'))['sha1']
    d_data = {'source_sha1': source_sha1, 'data': None, 'error': err}
    w_data = {'source_sha1': source_sha1, 'data': None, 'error': err}
    ds_data = {'source_sha1': source_sha1, 'data': None, 'error': err}
    pickle.dump(d_data, open(dloc, 'wb'))
    pickle.dump(w_data, open(wloc, 'wb'))
    pickle.dump(ds_data, open(dsloc, 'wb'))


def fetch_deltas(selections, findb_dir=None, visualize=False):

    """Fetch deltas for the given selections and save to database.

    Arguments:
    selections  -- yahoo symbols or symbol groups to calculate the deltas for
    findbdir    -- database directory
    visualize   -- visualize results
    """

    if type(selections) is str:
        selections = [selections]

    if not findb_dir:
        findb_dir = default_findb_dir()

    all_symbols = selections

    db_dir = os.path.join(findb_dir, 'db')
    existing_symbols = list(filter(lambda x: symbol_in_db(x, db_dir), all_symbols))
    diff = set(all_symbols).difference(set(existing_symbols))
    if diff:
        logging.info("{} of the requested symbols do not exist in the db.".format(len(diff)))
    deltas_to_calc = list(filter(lambda x: not check_delta_uptodate(x, db_dir),
                                 existing_symbols))
    calculated_deltas = set(existing_symbols).difference(set(deltas_to_calc))
    if not deltas_to_calc:
        logging.info("All the deltas have already been calculated.")
        return sorted(list(calculated_deltas))
    logging.info("{} symbols already have deltas, {} to new deltas to calculate."
                 .format(len(existing_symbols) - len(deltas_to_calc), len(deltas_to_calc)))

    start_time = az.utcnow()
    total_processed = 0
    succesful_conversions = []
    for sym in deltas_to_calc:
        symloc = os.path.join(db_dir, sym)
        df = load_pfile(symloc)
        logging.debug("Delta-converting {}...".format(sym))
        try:
            splitted = sym.split('/')
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
                elif qdl_db == "GOOG":
                    col = "Close"
                elif qdl_db == "YAHOO":
                    col = "Adjusted Close"
                elif qdl_db == "ECB":
                    col = "Value"
                elif qdl_db == "CURRFX":
                    col = "Rate"
                elif qdl_db == "FRED":
                    col = "Value"
                elif qdl_db == "BOE":
                    col = "Value"
                elif qdl_db == "CME":
                    col = "Settle"
                elif qdl_db == "ICE":
                    col = "Settle"
                elif qdl_db == "NASDAQOMX":
                    col = "Index Value"
                else:
                    raise Exception("Quandl database '{}' is not supported for delta conversion"
                                    .format(qdl_db))
            else:
                raise Exception("Provider {} is not supported for delta conversion"
                                .format(data_provider))
            res = deltaconvert(df[col])
            succesful_conversions.append(sym)
            save_deltas(res, sym, db_dir=db_dir)
        except DeltaConversionException as err:
            logging.debug("{}: {!r}".format(sym, err))
            save_invalid_deltas(sym, err.args[0], db_dir=db_dir)
        total_processed += 1
        time_elapsed = az.utcnow() - start_time
        sym_per_sec = total_processed / time_elapsed.total_seconds()
        progress_str = "{}/{} (suc/total) symbols processed. Speed: {:.2f} symbols/second."\
            .format(len(succesful_conversions), total_processed, sym_per_sec)
        if total_processed % 500 == 0:
            logging.info(progress_str)
        else:
            logging.debug(progress_str)

    progress_str = "{}/{} (suc/total) symbols processed. Speed: {:.2f} symbols/second."\
        .format(len(succesful_conversions), total_processed, sym_per_sec)
    logging.info(progress_str)

    return sorted(list(set(succesful_conversions).union(calculated_deltas)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="tool for constructing financial database")

    parser.add_argument("selections", nargs="*", help="symbols to select")

    vgroup = parser.add_mutually_exclusive_group()
    vgroup.add_argument("-v", action="count", default=0, help="verbosity")
    vgroup.add_argument("--quiet", action="store_true", help="disable stdout-output")

    parser.add_argument("--provider", default="yahoo", choices=["yahoo", "eia"], help="data provider")
    parser.add_argument("--modifygroups", action="store_true",
                        help="allow script to remove non-available entries from groups-file")
    parser.add_argument("--dl_threads", type=int, default=5, help="number of threads to use to download data")
    parser.add_argument("-dl", action="store_true", help="download data")
    parser.add_argument("-usd", action="store_true", help="convert data to USD")
    parser.add_argument("-delta", action="store_true", help="calculate changes")
    parser.add_argument("--visualize", action="store_true", help="visualize delta-conversion")

    args = parser.parse_args()

    azlib.azlogging.quick_config(args.v, args.quiet)

    # this old code is not up-to-date

    # home = os.path.expanduser("~")
    # findbdir = os.path.join(home, 'findb')
    # if not os.path.exists(findbdir):
    #     os.makedirs(findbdir)

    # symbols = []
    # if args.dl and args.provider == "yahoo":
    #     # symbols not assigned on purpose
    #     download_yahoo(args.selections, args.dl_threads, findbdir, args.batchsize, args.usd, args.modifygroups)

    # if args.delta:
    #     fetch_deltas(args.selections, findbdir, args.visualize)
