import argparse
import logging
import os
import concurrent.futures
import yahoodl
import hashlib
import re
import io
import pickle
import tables
import os.path
import azlib as az
import azlib.azlogging
import pandas as pd
import pandas.tseries.offsets
import pandas.io.data
import bottleneck as bn
import numpy as np
import findb.selector
from datetime import datetime
from pandas import DataFrame, Series
from yahoodl import fred_currencies


def convert_to_usd(symbol, dbfile):
    """Convert yahoo symbol to USD in the HDF-file.

    Arguments:
    symbol  -- symbol to convert
    dbfile  -- database file
    """
    store = pd.io.pytables.HDFStore(dbfile, 'a')
    symloc = "/yahoo/{}".format(symbol)
    ydata = store[symloc]
    lastcol = ydata.columns[-1]
    currency = lastcol[-4:-1]
    if currency == "USD":  # conversion not required
        store.close()
        return False
    fxfile = fred_currencies[currency]
    fxfileloc = "/fred/{}".format(fxfile)
    xxxusd = store[fxfileloc]
    # fill missing dates (i.e. US holidays) using the last available value
    xxxusd.fillna(method='ffill')
    # take inverse of usdxxx pairs (ending "XXUS")
    if fxfile[-2:] == "US":
        xxxusd = 1 / xxxusd
    ydata["AdjClose(USD)"] = ydata[lastcol] * xxxusd
    store[symloc] = ydata
    store.close()


def create_empty_db():
    """Create empty database directory."""
    home = os.path.expanduser("~")
    findbdir = os.path.join(home, 'findb')
    dbfile = os.path.join(findbdir, 'db.h5')
    if os.path.isfile(dbfile):
        raise Exception("{} already exists.".format(dbfile))
    db = tables.open_file(dbfile, 'w')
    db.create_group('/', 'yahoo')
    db.create_group('/', 'fred')
    db.close()

def download_yahoo(selections, dl_threads, findbdir, batchsize, conv_to_usd=True, modify_groups=False, update_freq=1):

    """Download daily data from Yahoo Finance.

    Arguments:
    selections       -- symbols or symbol groups to download
    dl_threads       -- # of threads to use when downloading
    findbdir         -- database directory
    batchsize        -- how many symbols to download in one iteration
                        (bigger number = faster / more memory used)
    conv_to_usd      -- convert to USD after finished downloading the data
    modify_groups    -- remove non-existent symbols from yahoogroups.yaml
    update_freq      -- how many business days to wait until updating data
    """

    if not selections:
        raise Exception("No symbols selected")
    for sym in selections:
        if ".csv" in sym:
            raise Exception("When downloading select symbol/group names instead of filenames")

    # if os.path.isfile(os.path.join(findbdir, 'yahoogroups.p')):
    #     groupfile = os.path.join(findbdir, 'yahoogroups.p')
    # else:
    groupfile = os.path.join(findbdir, 'yahoogroups.yaml')
        # if not os.path.isfile(groupfile):
        #     with open(groupfile, 'w') as yamlfile:
        #         yamlfile.write("# empty groupsfile, remove .p cachefile after each write\n")

    symbols = findb.selector.selections_to_symbols(selections, groupfile)

    symbols_to_remove_from_groups = []

    results = {}

    dbfile = os.path.join(findbdir, 'db.h5')
    db = tables.open_file(dbfile, 'a')

    if not "/yahoo" in db:
        db.create_group('/', 'yahoo')

    symbols_to_fetch = []
    for sym in symbols:
        sympath = '/yahoo/{}'.format(sym)
        if not sympath in db:
            symbols_to_fetch.append(sym)
            continue
        attrs = db.get_node(sympath)._v_attrs
        if not 'last_update' in attrs:
            symbols_to_fetch.append(sym)
            continue
        # more than one businessday ago
        if az.utcnow().to_datetime() - pd.tseries.offsets.BDay(update_freq) > attrs['last_update']:
            symbols_to_fetch.append(sym)
            continue
        logging.debug("{} up-to-date, skipping download...".format(sym))

    db.close()

    # pd_est_now = pd.Timestamp(datetime.utcnow(), tz='UTC').tz_convert('US/Eastern')
    # est_now = pd_est_now.to_datetime()

    fred_symbols_to_dl = []
    if conv_to_usd:
        db = tables.open_file(dbfile, 'a')
        if not '/fred' in db:
            db.create_group('/', 'fred')

        for sym in fred_currencies.values():
            sympath = '/fred/{}'.format(sym)
            if not sympath in db:
                fred_symbols_to_dl.append(sym)
                continue
            attrs = db.get_node(sympath)._v_attrs
            if not 'last_update' in attrs:
                fred_symbols_to_dl.append(sym)
                continue
            # more than one businessday ago
            if az.utcnow().to_datetime() - pd.tseries.offsets.BDay(1) > attrs['last_update']:
                fred_symbols_to_dl.append(sym)
                continue
            logging.debug("{} up-to-date, skipping download...".format(sym))

        db.close()

        if fred_symbols_to_dl:
            # download fx-rates from fred
            start = datetime(1900, 1, 1)
            end = datetime(2020, 1, 1)
            logging.info("Downloading fx-data from FRED...")
            fxdata = pandas.io.data.DataReader(fred_symbols_to_dl, 'fred', start, end)
            logging.debug("Saving fx-data to HDF file...")
            for sym in fred_symbols_to_dl:
                fxdata[sym].to_hdf(dbfile, '/fred/{}'.format(sym))
            logging.debug("Updating timestamps in HDF file...")

            db = tables.open_file(dbfile, 'a')
            for sym in fred_symbols_to_dl:
                attrs = db.get_node('/fred/{}'.format(sym))._v_attrs
                attrs['last_update'] = az.utcnow().to_datetime()
            db.close()
            logging.debug("fx-data from FRED saved to HDF-file.")

    downloaded_symbols = []

    num_batches = len(list(az.chunks(symbols_to_fetch, batchsize)))
    batch_no = 1
    for batch in az.chunks(symbols_to_fetch, batchsize):

        results = {}
        # grab the data from Yahoo with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=dl_threads) as executor:
            future_to_symbol = {executor.submit(yahoodl.dl, sym): sym for sym in batch}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                # try:
                res = future.result()
                # except Exception as err:
                #     logging.error("Error when processing {}: {}".format(symbol, err))
                #     results[symbol] = err
                if res:
                    # dates should be in ascending order, faster than reindexing DataFrame later
                    lines = res.splitlines()
                    header = lines[0]
                    del lines[0]
                    lines.reverse()
                    lines.insert(0, header)
                    res = "\n".join(lines)

                    results[symbol] = res
                    downloaded_symbols.append(symbol)
                    logging.debug("Downloaded {} from Yahoo succesfully.".format(symbol))
                else:
                    # logging.debug("{} was not found.".format(symbol))
                    symbols_to_remove_from_groups.append(symbol)

        logging.debug("Saving batch data to the HDF-file...")
        for sym, res in results.items():
            df = DataFrame.from_csv(io.StringIO(res))
            # very rarely this will fail for some reason! (for example on CANCDA.SW)
            # have to report this bug (or maybe it was corrupted db file)
            df.to_hdf(dbfile, '/yahoo/{}'.format(sym))

        logging.debug("Updating timestamps on the HDF-file...")
        db = tables.open_file(dbfile, 'a')
        for sym, res in results.items():
            targetloc = '/yahoo/{}'.format(sym)
            if targetloc in db:
                attrs = db.get_node(targetloc)._v_attrs
                attrs['last_update'] = az.utcnow().to_datetime()
        db.close()
        logging.debug("Data succesfully saved to the HDF-file.")

        if modify_groups and os.path.isfile(groupfile) and symbols_to_remove_from_groups:
            logging.info("Removing invalid symbols from {} ...".format(groupfile))
            with open(groupfile, mode='r') as myfile:
                lines = myfile.readlines()
            for symbol in symbols_to_remove_from_groups:
                for i in range(len(lines) - 1, -1, -1):
                    matches = re.search('[\'"]{}[\'"]'.format(symbol), lines[i])
                    if matches:
                        del lines[i]
                        logging.info("Removed {} from line {}.".format(symbol, i))
            with open(groupfile, mode='w') as myfile:
                myfile.writelines(lines)

        if conv_to_usd:
            logging.debug("Converting to USD...")
            for sym in results:
                convert_to_usd(sym, dbfile)
            logging.debug("Succesfully converted to USD.")

        if num_batches > 1:
            logging.info("Finished downloading batch {}/{}.".format(batch_no, num_batches))
        batch_no += 1

    return downloaded_symbols

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

def deltaconvert(df, column="AdjClose(USD)", visualize=False, max_adj_outliers=10):

    """Perform delta-conversion based on the given column in a pandas.DataFrame.

    Delta-conversion returns 3 series as a tuple and possibly error message.
    
    First series (D) contains daily returns where all the data has been removed
    that could make comparison difficult with other assets.

    Second series (W) contains weekly returns where all the data has been removed
    that could make comparison difficult with other assets.

    Third series (DS) contains daily returns where all the outliers and erroneus
    data points have been removed but holes in data are not taken into account.
    This is more suitable for calculating performance scores etc.

    Arguments:
    df               -- pd.DataFrame to use
    column           -- column header to use
    visualize        -- visualize results
    max_adj_outliers -- maximum number of adjancent outliers, if there are
                        actually more adjancent outliers than this then they will
                        not be considered outliers anymore. default value: 10
    """

    # MEDIAN_LEN = 50
    ZSCORE_CUT_RATIO = 2

    df = df.dropna()

    if len(df) < 50:
        return None, "Not enough data"

    lines_taken = 0

    if df.index[0] > df.index[-1]:
        return None, "Wrong cronological order"

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

    closes = df["AdjClose(USD)"]
    dates = df.index

    num_invalid_prices = 0
    deltapct = [np.nan]
    changescores = [np.nan]
    invalid_price_indices = []
    for i in range(1, len(df)):
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
                  .format(len(df) - lines_taken, num_invalid_prices, num_gaps,
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

    res_daily = Series(deltapctmod, datesmod)
    res_weekly = Series(weeklydeltapct, weeklydatesmod)


    indices_to_rem = list(set(confirmed_outliers + invalid_price_indices))
    datesmod = dates.copy()
    datesmod = datesmod.delete(indices_to_rem)
    deltapctmod = np.delete(deltapct, indices_to_rem)
    assert(not np.any(np.isnan(deltapctmod[1:])))

    res_dailyscore = Series(deltapctmod, datesmod)

    return (res_daily, res_weekly, res_dailyscore), None


def fetch_deltas(selections, findbdir=None, visualize=False):

    """Fetch deltas for the given selections (yahoo) and save to HDF-database.

    Arguments:
    selections  -- yahoo symbols or symbol groups to calculate the deltas for
    findbdir    -- database directory
    visualize   -- visualize results
    """

    if not findbdir:
        home = os.path.expanduser("~")
        findbdir = os.path.join(home, 'findb')

    groupfile = os.path.join(findbdir, 'yahoogroups.yaml')
    symbols = findb.selector.selections_to_symbols(selections, groupfile)

    dbfile = os.path.join(findbdir, 'db.h5')
    for sym in symbols:
        store = pd.io.pytables.HDFStore(dbfile, 'r')
        symloc = "/yahoo/{}".format(sym)
        if symloc not in store:
            store.close()
            logging.debug("{} was not found in HDS-file, skipping...".format(sym))
            continue
        dloc = '/yahoo/{}_D'.format(sym)
        wloc = '/yahoo/{}_W'.format(sym)
        dsloc = '/yahoo/{}_DS'.format(sym)
        underlying_sha1 = hashlib.sha1(pickle.dumps(store[symloc])).digest()
        db = tables.open_file(dbfile, 'r')
        delta_calculation_needed = False
        if not dloc in db or "underlying_sha1" not in db.get_node(dloc)._v_attrs or \
                        underlying_sha1 != db.get_node(dloc)._v_attrs["underlying_sha1"]:
            delta_calculation_needed = True
        if not wloc in db or "underlying_sha1" not in db.get_node(wloc)._v_attrs or \
                        underlying_sha1 != db.get_node(wloc)._v_attrs["underlying_sha1"]:
            delta_calculation_needed = True
        if not dsloc in db or "underlying_sha1" not in db.get_node(dsloc)._v_attrs or \
                        underlying_sha1 != db.get_node(dsloc)._v_attrs["underlying_sha1"]:
            delta_calculation_needed = True
        db.close()
        store.close()
        if delta_calculation_needed:
            store = pd.io.pytables.HDFStore(dbfile, 'a')
            logging.debug("Performing delta-conversion to {}...".format(sym))
            res, err = deltaconvert(store[symloc], -1, visualize)
            if res:
                store[dloc] = res[0]
                store[wloc] = res[1]
                store[dsloc] = res[2]
                store.close()
                db = tables.open_file(dbfile, 'a')
                db.get_node(dloc)._v_attrs["underlying_sha1"] = underlying_sha1
                db.get_node(wloc)._v_attrs["underlying_sha1"] = underlying_sha1
                db.get_node(dsloc)._v_attrs["underlying_sha1"] = underlying_sha1
                db.close()
                logging.info("Delta-conversion for {} finished.".format(sym))
            else:
                store.close()
                logging.warning("Error when delta-converting {}: {}".format(sym, err))
        else:
            logging.debug("Delta-conversion not needed for {}.".format(sym))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="tool for constructing financial database")

    parser.add_argument("selections", nargs="*", help="symbols to select")

    vgroup = parser.add_mutually_exclusive_group()
    vgroup.add_argument("-v", action="count", default=0, help="verbosity")
    vgroup.add_argument("--quiet", action="store_true", help="disable stdout-output")

    parser.add_argument("--provider", default="yahoo", choices=["yahoo", "eia"], help="data provider")
    parser.add_argument("--modifygroups", action="store_true",
                        help="allow script to remove non-available entries from groups-file")
    parser.add_argument("--dlthreads", type=int, default=5, help="number of threads to use to download data")
    parser.add_argument("-dl", action="store_true", help="download data")
    parser.add_argument("-usd", action="store_true", help="convert data to USD")
    parser.add_argument("-delta", action="store_true", help="calculate changes")
    parser.add_argument("--visualize", action="store_true", help="visualize delta-conversion")
    parser.add_argument("--batchsize", type=int, default=100, help="number of downloads per batch")

    args = parser.parse_args()

    az.azlogging.quick_config(args.v, args.quiet)

    home = os.path.expanduser("~")
    findbdir = os.path.join(home, 'findb')
    if not os.path.exists(findbdir):
        os.makedirs(findbdir)

    symbols = []
    if args.dl and args.provider == "yahoo":
        # symbols not assigned on purpose
        download_yahoo(args.selections, args.dlthreads, findbdir, args.batchsize, args.usd, args.modifygroups)

    if args.delta:
        fetch_deltas(args.selections, findbdir, args.visualize)
