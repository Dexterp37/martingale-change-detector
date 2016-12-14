import argparse
import json
import martingale as mg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from datetime import date, datetime
from plotting import plot_changes

def filter_range(histograms, min_date, max_date):
    filtered = []

    for h in histograms:
        d = datetime.strptime(h['date'][:10], "%Y-%m-%d").date()
        if d >= min_date and d <= max_date:
            filtered.append(h)

    return filtered


def merge_build_histograms(histograms):
    # Histogram data might contain entries for multiple Firefox builds.
    # We want to use all the available data and merge the histograms if
    # we have multiple entries for the same day.

    hist_by_date = {}
    for h in histograms:
        hist_date = h['date'][:10]

        # Ignore histograms with few submissions
        if h["count"] < 1000:
            continue

        # Ignore negative values in histograms
        values = np.array(h["values"])
        values[values < 0] = 0
        h["values"] = values

        if hist_date in hist_by_date:
            # If we already have some data for that day, merge the values.
            stored_values = hist_by_date[hist_date]['values']
            merged_values = np.sum(np.vstack((stored_values, np.array(h['values']))), axis=0)
            hist_by_date[hist_date]['values'] = merged_values
        else:
            hist_by_date[hist_date] = h

    # Make sure the histograms are returned sorted by date.
    return [h for d, h in sorted(hist_by_date.iteritems(), key=lambda t: t[0])]


def parse_histograms(path, min_date, max_date):
    histograms = {}
    num_days = (max_date - min_date).days + 1

    # Assume histograms have been exported with
    # https://github.com/mozilla/cerberus/blob/master/exporter/export.js
    for file_name in os.listdir(path):
        # Skip non-JSON files.
        if not file_name.endswith(".json"):
            continue

        # Load the data for this measurement.
        file_path = os.path.join(path, file_name)
        with open(file_path) as json_file:
            dict_name = os.path.splitext(file_name)[0]
            series = json.load(json_file)
            merged_series = merge_build_histograms(filter_range(series, min_date, max_date))

            if len(merged_series) == num_days:
                # Remove histograms with missing days
                histograms[dict_name] = merged_series

    return histograms


def parse_arguments():
  parser = argparse.ArgumentParser(
    description="Analyse Firefox telemetry histograms to detect regressions using martingales.")
  parser.add_argument("--fromdate", required=True,
                      type=lambda d: datetime.strptime(d, "%Y%m%d").date(),
                      help="The beginning of the dates range to consider for analysis.")
  parser.add_argument("--todate", required=True,
                      type=lambda d: datetime.strptime(d, "%Y%m%d").date(),
                      help="The end of the dates range to consider for analysis.")
  parser.add_argument("--datadir", action="store", required=True,
                      help="The directory that contains the histogram data")
  parser.add_argument("--outdir", action="store", required=False, default='detections',
                      help="The directory that will contain the detections data. Defaults to 'detections'")
  parser.add_argument("--strangeness", action="store", required=False, default='cosine',
                      type=lambda s: parser.error("Invalid strangeness") if s not in mg.SUPPORTED_STRANGENESS.keys() else s,
                      help="The strangeness measure to use for detection. Supported values are {}. Defaults to 'cosine'"\
                          .format(", ".join(mg.SUPPORTED_STRANGENESS.keys())))
  parser.add_argument("--enable-plots", dest="enable_plots", action="store_true", required=False, default=False,
                      help="Enable the plotting of the discovered anomalies to the output directory.")
  parser.add_argument("--threshold", action="store", required=False, default=20, type=float,
                      help="The threshold to use for the detection. Defaults to 20")

  return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Fix a random seed to get reproducible results.
    np.random.seed(42)

    # Parse the histograms from the 'histograms' directory.
    histograms = parse_histograms(args.datadir, args.fromdate, args.todate)
    print "{} histograms parsed.".format(len(histograms))

    # Flush the directory that we will use to hold our output.
    shutil.rmtree(args.outdir, ignore_errors=True)
    os.makedirs(args.outdir)

    # Run the detection algorithm on all the parsed histograms.
    anomalies = {}
    detected = 0
    total = 0

    print "Running the martingales detection using the {} strangeness with a threshold of {}"\
        .format(args.strangeness, args.threshold)

    chosen_strangeness_func = mg.SUPPORTED_STRANGENESS[args.strangeness]

    for name in histograms:
        raw_changes, martingale, pvalues, strangeness, augmented_data =\
            mg.detect_changes(histograms, name, chosen_strangeness_func, threshold=args.threshold)

        # Update the stats.
        if len(raw_changes) > 0:
          detected += 1
          anomalies[name] = raw_changes

        # Only draw the plots if requested.
        if args.enable_plots:
            coroutine = plot_changes(histograms, name, raw_changes, martingale, pvalues,
                                     strangeness, augmented_data)
            changes = next(coroutine)

            if len(changes) > 0:
                fig = plt.figure(figsize=(17, 16))
                coroutine.send(fig)
                fig.savefig(os.path.join(args.outdir, '{}.png'.format(name)))
                plt.close(fig)

        total += 1

    # Save the detected changes to a JSON file for further processing.
    print "Histograms with changes: {} of {}".format(detected, total)
    with open('detections/anomalies.json', 'w') as outfile:
        json.dump(anomalies, outfile)
