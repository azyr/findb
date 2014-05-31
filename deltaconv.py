import logging
import argparse
# import os.path
import os
import azlib.azlogging
import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
from datetime import datetime


def change_to_score(change):
    if change >= 1:
        changescore = change - 1
    elif change < 1:
        changescore = -1 / change
        changescore += 1
    return changescore


def convert(filename, column=6, visualize=False, max_adj_outliers=10, modfolder="mod"):

    mod_dir = os.path.join(os.path.dirname(filename), modfolder)
    basename = os.path.basename(filename)[:-4]
    dailyfile = os.path.join(mod_dir, basename + "_D.csv")
    weeklyfile = os.path.join(mod_dir, basename + "_W.csv")

    if not os.path.exists(mod_dir):
        os.makedirs(mod_dir)

    if os.path.isfile(dailyfile):
        os.remove(dailyfile)
    if os.path.isfile(weeklyfile):
        os.remove(weeklyfile)

    logging.debug("Extracting spike-corrected changes from {}...".format(filename))

    with open(filename, 'r') as csvfile:
        lines = csvfile.readlines()

    # MEDIAN_LEN = 50
    ZSCORE_CUT_RATIO = 2

    if not lines:
        logging.debug("{} is empty.".format(filename))
        return False

    # remove header line if existent
    lines_taken = 0
    headerline = None
    try:
        splitted = lines[0].split(",")
        float(splitted[column])
    except ValueError:
        lines_taken = 1
        headerline = lines[0]
        del lines[0]

    if len(lines) < 50:
        logging.debug("Not enough data for outlier analysis ({} lines)".format(len(lines)))
        return False

    closes = []
    dates = []
    datesord = []

    for line in lines:
        splitted = line.split(",")
        closes.append(float(splitted[column]))
        dt = datetime.strptime(splitted[0], "%Y-%m-%d").date()
        dates.append(dt)
        datesord.append(dt.toordinal())

    if datesord[-1] < datesord[0]:
        closes.reverse()
        dates.reverse()
        datesord.reverse()
        lines.reverse()

    num_invalid_prices = 0
    deltapct = [np.nan]
    changescores = [np.nan]
    invalid_price_indices = []
    for i in range(1, len(closes)):
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
        d = datesord[i] - datesord[i - 1]
        # standard weekends are only allowed
        if d == 3:
            if dates[i].weekday() != 0:  # not monday
                # deltapct[i] = np.nan
                # changescores[i] = np.nan
                num_gaps += 1
                gap_indices.append(i)
                logging.log(5, "Non-weekend gap of {} day(s) at {}".format(2, dates[i]))
        elif d > 1:
            # deltapct[i] = np.nan
            # changescores[i] = np.nan
            num_gaps += 1
            gap_indices.append(i)
            logging.log(5, "Non-weekend gap of {} day(s) at {}".format(2, dates[i]))
        elif d <= 0:
            del deltapct[i], dates[i], closes[i], changescores[i]
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
    #
    # n, bins, _ = plt.hist(zscores[~np.isnan(zscores)], 50, normed=False, facecolor='g')
    # logging.debug("Starting to find outliers with standard deviation of {:.4%}".format(std_deltapct))
    # for i in range(len(n)):
    #     logging.info("{}: {}".format(bins[i], n[i]))
    # # plt.show()


    # num_outliers = 0
    # outlier_indices = []
    # start_stddeltapct = 1e20  # just make sure first iteration will be run

    # while std_deltapct / start_stddeltapct < 0.5:
    #
    #     no_check = False
    #     start_stddeltapct = std_deltapct
    #     logging.debug("Starting to find outliers with standard deviation of {:.4%}".format(std_deltapct))
    #
    #     for i in range(len(deltapct)):
    #         absdelta = abs(deltapct[i])
    #         zscore = abs(deltapct[i]) / std_deltapct
    #         # sometimes there is this strange error where prices are spiked by multiplier of 10,
    #         # this will totally screw up stddev so we need this extra check
    #         if no_check or absdelta > 9.89 or zscore > numstd:
    #             median_before = np.median(closes[max(i-MEDIAN_LEN, 0):i])
    #             median_after = np.median(closes[i:min(i+MEDIAN_LEN, len(closes) + 1)])
    #             distance_to_after = abs(closes[i] - median_after)
    #             distance_to_after_pct = abs((closes[i] / median_after) - 1)
    #             distance_to_after_zscore = distance_to_after_pct / std_deltapct
    #
    #             distance_between_medians = abs(median_after - median_before)
    #
    #             logging.log(5, "Line: {}, z-score: {}, close: {}, median_before: {}, median_after: {}, mzscore={}"
    #                         .format(i + lines_taken + 1, zscore, closes[i], median_before, median_after,
    #                                 distance_to_after_zscore))
    #
    #             if distance_to_after_pct > 9.89 * .9 or \
    #                     (distance_to_after > distance_between_medians and distance_to_after_zscore > numstd * .8):
    #
    #                 logging.debug("Outlier at line {} (z-score={}, deltapct={:.2%})"
    #                               .format(i + lines_taken + 1, zscore, absdelta))
    #                 deltapct[i] = np.nan
    #                 outlier_indices.append(i)
    #                 num_outliers += 1
    #                 no_check = True
    #                 std_deltapct = bn.nanstd(np.abs(deltapct))
    #                 logging.debug("Standard deviation of absolute deltas changed to: {:.4%}".format(std_deltapct))
    #
    #             # after an outlier the next data point will be corrupted as well, so we will remove it
    #             elif no_check:
    #                 deltapct[i] = np.nan
    #                 no_check = False

    if visualize:
        closes = np.asarray(closes)
        datesord = np.asarray(datesord)
        plt.subplot(2, 1, 1)
        plt.plot(datesord - datesord[0], closes, 'b*')
        plt.plot(datesord[gap_indices] - datesord[0], closes[gap_indices], 'ob')
        plt.plot(datesord[confirmed_outliers] - datesord[0], closes[confirmed_outliers], 'or')
        plt.plot(datesord[invalid_price_indices] - datesord[0], closes[invalid_price_indices], 'om')
        plt.subplot(2, 1, 2)
        plt.plot(datesord - datesord[0], zscores, 'o')
        plt.show()

    res = []
    for i in range(len(deltapct)):
        if not np.isnan(deltapct[i]):
            res.append("{},{}".format(datesord[i], deltapct[i]))

    logging.debug("Conversion result: lines = {}, invalid closes = {}, gaps = {}, invalid dates = {}, outliers = {}"
                  .format(len(lines) - lines_taken, num_invalid_prices, num_gaps,
                          num_invalid_chrono_orders, len(confirmed_outliers)))

    indices_to_rem = list(set(gap_indices + confirmed_outliers + invalid_price_indices))
    datesord = np.delete(datesord, indices_to_rem)
    dates = np.asarray(dates)
    dates = np.delete(dates, indices_to_rem)
    deltapct = np.delete(deltapct, indices_to_rem)
    closes = np.delete(closes, indices_to_rem)
    assert(not np.any(np.isnan(deltapct[1:])))

    weeklydeltapct = []
    weeklydatesord = []
    lastidx = -1
    for i in range(len(closes)):
        if dates[i].weekday() == 4:
            dd = datesord[i] - datesord[lastidx]
            if lastidx >= 0 or dd == 7:
                if closes[lastidx] >= 0:
                    weeklydeltapct.append(closes[i] / closes[lastidx] - 1)
                    weeklydatesord.append(datesord[i])
            else:
                logging.log(5, "Weekly bar at {} (idx: {}) skipped (delta: {} days)".format(dates[i], i, dd))
            lastidx = i

    with open(dailyfile, 'w') as myfile:
        for i in range(1, len(deltapct)):
            myfile.write("{},{}\n".format(datesord[i], deltapct[i]))

    with open(weeklyfile, 'w') as myfile:
        for i in range(len(weeklydatesord)):
            myfile.write("{},{}\n".format(weeklydatesord[i], weeklydeltapct[i]))

    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="fixes the yahoo market data file and converts it to a delta file")

    parser.add_argument("files", nargs="*", help="files to convert")
    parser.add_argument("column", nargs="?", type=int, default=6, help="datacolumn")

    parser.add_argument("-v", action="count", default=0, help="verbosity")
    parser.add_argument("--visualize", action="store_true", help="visualize data")

    args = parser.parse_args()

    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    if args.v == 0:
        azlib.azlogging.basic_config(level=logging.INFO, format=fmt, datefmt='%j-%H:%M:%S')
    elif args.v == 1:
        azlib.azlogging.basic_config(level=logging.DEBUG, format=fmt, datefmt='%j-%H:%M:%S')
    elif args.v >= 2:
        azlib.azlogging.basic_config(level=logging.NOTSET, format=fmt, datefmt='%j-%H:%M:%S')

    for filename in args.files:

        logging.info("Converting {} ...".format(filename))

        convert(filename, args.column, args.visualize)