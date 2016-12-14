# Anomaly detection for Telemetry histograms
This repository implements a martingale approach to detect changes in Telemetry histograms.
Some background on using martingales for testing exchangeability online can be found [here](http://www.vovk.net/cp/05.pdf).

## How to use
There are two ways to execute the tool. The first is running it from an [Anaconda](https://www.continuum.io/downloads) terminal. This is the easiest way, as the Anaconda environment comes with all the required dependencies pre-installed.

    python detector\main.py --from 20160920 --to 20161019 --datadir histograms --enable-plots

If running from a different environment, dependencies can be installed using `setuptools` by using the following command from the root project directory:

    pip install .

And then the script can be executed as previously reported.

The following is the list of supported command line options for the current version:

* `--fromdate`, the beginning of the dates range to consider for analysis, in `YYYYMMDD` format.
* `--todate`,  the end of the dates range to consider for analysis, in `YYYYMMDD` format.
* `--datadir`, the directory that contains the histogram data.
* `--outdir`, the directory that will contain the detections data. Defaults to `detections`.
* `--strangeness`, the strangeness measure to use for detection. Supported values are `cluster`, `hellinger`, `bhattacharyya` and `cosine` (the default).
* `--enable-plots`, enable plotting the discovered anomalies to the output directory.
* `--threshold`, the threshold to use for the detection. Defaults to 20.

## Fetching sample data
Fetching Telemetry histogram data can be done by using the export script from [cerberus](https://github.com/mozilla/cerberus/blob/master/exporter/export.js), the system Mozilla currently uses to perform anomaly detection on time-series of histogram data.

In order to do that, we start by fetching the histogram/measurement definition from the Firefox repository using `wget`:

    wget https://raw.githubusercontent.com/mozilla/gecko-dev/master/toolkit/components/telemetry/Histograms.json -O Histograms.json

Then we execute the script that fetches the time-series of histograms for the 3 most recent Firefox builds. Please note that the script must be run from the same directory as the `Histogram.json` file.

    nodejs export.js

This will take a bit to run and will output the histogram data as JSON files under the `histograms` directory.

A sample archive for the histogram data aggregated in the period from the 20th of September 2016 to the 19th October 2016 can he downloaded from [here](https://drive.google.com/open?id=0B-tN21rUReH2S2NySlozTXQ5YXc). This is the same date we have used to test the system.
