# Hamradio_copilot

Developed on the base of idea by Rudy, N2WQ - TNX

See live demo for ITU zone 28: https://azure.s53m.com/matrigs-desktop/copilot/

## DX Cluster Data Analysis Scripts: Readme

This repository contains two Python scripts for working with DX Cluster data:

  * **connect.py** - Connects to a DX Cluster server, collects spotted callsigns, and stores them in an SQLite database.
  * **analyseSNR.py** - Analyzes SNR data from the SQLite database and generates an HTML report.

### connect.py

This script connects to a DX Cluster server, retrieves spotted callsign information, and stores it in a local SQLite database (`callsigns.db`). It filters the data based on the specified ITU Zone and enriches it with additional information like band and CQZone.

**Features:**

* Connects to the DX Cluster server and handles login.
* Parses received data to extract callsign, frequency, mode, SNR, and spotter information using a regular expression.
* Uses a cache to avoid redundant lookups for callsign details (CQZone and ITU Zone).
* Calculates the band category based on the frequency.
* Handles reconnection attempts with exponential backoff in case of connection issues.
* Stores callsign information in an SQLite database with timestamps and spotter information.

**Running the script:**

```
python connect.py -l <your_callsign> -i <itu_zone> [--address <dx_cluster_host>] [--port <dx_cluster_port>] [-d]
```

* `-l <your_callsign>`: Specify your callsign for login.
* `-i <itu_zone>`: Specify the ITU Zone to track (integer value).
* `--address <dx_cluster_host>` (optional): Override the default DX Cluster host (telnet.reversebeacon.net).
* `--port <dx_cluster_port>` (optional): Override the default DX Cluster port (7001).
* `-d` (optional): Enable debug output.

**Dependencies:**

* Python 3
* `sqlite3` library
* `re` library
* `logging` library
* `argparse` library (optional, for command-line arguments)

**Note:**

* The script expects a file named `cty.plist` to be present in the same directory. This file contains callsign information including ITU Zone data.

cty.plist available at: https://github.com/dh1tw/DX-Cluster-Parser/blob/master/cty.plist

### analyseSNR.py

This script analyzes SNR data stored in the SQLite database (`callsigns.db`) generated by `connect.py`. It generates an HTML report summarizing the data with visualizations and statistics.

**Features:**

* Connects to the SQLite database and retrieves callsign data.
* Filters data based on a specified time range.
* Calculates statistics for SNR data per band and ITU Zone.
* Creates a heatmap visualization of SNR distribution.
* Generates an HTML report with the analysis results and visualizations.
* Optionally fetches and integrates solar data from an external source (requires internet access).

**Running the script:**

```
python analyseSNR.py [-f <frequency>] [-l <lower_threshold>] [-u <upper_threshold>] [-r <time_range>] [-o <output_folder>] [--use-s3] [-d] [--include-solar-data]
```

* `-f <frequency>` (optional): Specify the frequency of data collection in minutes (default: 1 minute).
* `-l <lower_threshold>` (optional): Define the lower threshold for data count visualization (empty square, default: 5).
* `-u <upper_threshold>` (optional): Define the upper threshold for data count visualization (filled square, default: 10).
* `-r <time_range>` (optional): Specify the number of hours of data to analyze from the current time (default: 0.25 hours).
* `-o <output_folder>` (optional):  Specify the folder to save the generated `index.html` report (default: "local_html").
* `--use-s3` (optional): Enable uploading the generated report to an S3 bucket (requires configuration).
* `-d` (optional): Enable debug output.
* `--include-solar-data` (optional): Include fetching and presenting solar data in the report.

**Dependencies:**

* Python 3
* `pandas` library
* `argparse` library
* `boto3` library (optional, for S3 uploads)
* `requests` library (optional, for external data fetching)
* `xml.etree.ElementTree` library (optional, for XML parsing)
* `os` library
* `html` library
* `sqlite3` library
* `numpy` library (for numeric operations)
* `matplotlib.colors` library (for custom colormap)

**Note:**

* None at the moment :)
