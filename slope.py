# Slope calculator and visualizer for debuging purposes
import argparse
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
import sys

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Calculate SNR slope for a given zone and band.')
    parser.add_argument('-z', '--zone', type=int, required=True, help='Zone number (integer).')
    parser.add_argument('-b', '--band', type=str, required=True, help='Band (e.g., "20").')
    parser.add_argument('-t', '--time-range', type=float, default=24.0, help='Time range in hours to consider. Default is 24 hours.')
    args = parser.parse_args()
    return args.zone, args.band, args.time_range

def fetch_snr_data(zone, band, time_range):
    """
    Fetches SNR data for the specified zone and band from the SQLite database.
    Filters data within the specified time range.
    """
    try:
        conn = sqlite3.connect('callsigns.db')
        query = """
        SELECT snr, timestamp
        FROM callsigns
        WHERE zone = ? AND band = ?
        """
        df = pd.read_sql_query(query, conn, params=(zone, band))
        conn.close()
    except Exception as e:
        print(f"Error accessing the database: {e}")
        sys.exit(1)

    if df.empty:
        print(f"No data found for zone {zone} and band {band}.")
        sys.exit(1)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    # Filter data within the specified time range
    time_threshold = pd.Timestamp.utcnow() - pd.Timedelta(hours=time_range)
    df = df[df['timestamp'] >= time_threshold]

    if df.empty:
        print(f"No data found for zone {zone}, band {band}, in the last {time_range} hours.")
        sys.exit(1)

    return df

def calculate_average_per_minute(df):
    """
    Calculates the average SNR for each minute.
    Returns a DataFrame with averaged SNR and corresponding minute timestamps.
    """
    # Round timestamps to the nearest minute
    df['minute'] = df['timestamp'].dt.floor('T')  # 'T' is for minute frequency

    # Group by the 'minute' column and calculate the average SNR
    df_avg = df.groupby('minute').agg({'snr': 'mean'}).reset_index()

    # Rename columns for clarity
    df_avg.rename(columns={'minute': 'timestamp', 'snr': 'avg_snr'}, inplace=True)

    return df_avg

def calculate_slope(df_avg):
    """
    Calculates the slope of averaged SNR over time using linear regression.
    """
    # Ensure there are at least two data points
    if len(df_avg) < 2:
        print("Not enough data points to calculate slope.")
        return np.nan

    # Prepare data for regression
    # Convert timestamps to numeric values (e.g., Unix timestamp in seconds)
    time_values = df_avg['timestamp'].astype(np.int64) // 1e9 / 60  # Convert to minutes
    snr_values = df_avg['avg_snr']

    # Check if all time values are identical
    if time_values.nunique() == 1:
        print("All timestamps are identical. Cannot calculate slope.")
        return np.nan

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(time_values, snr_values)

    return slope

def main():
    zone, band, time_range = parse_arguments()

    print(f"Fetching data for Zone: {zone}, Band: {band}, Time Range: {time_range} hours")

    df = fetch_snr_data(zone, band, time_range)

    # Calculate average SNR per minute
    df_avg = calculate_average_per_minute(df)

    # Sort data by timestamp
    df_avg_sorted = df_avg.sort_values(by='timestamp')

    # Print averaged SNR values along with their minute timestamps
    print("\nAveraged SNR Values per Minute:")
    for idx, row in df_avg_sorted.iterrows():
        timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
        avg_snr = row['avg_snr']
        print(f"Time: {timestamp}, Average SNR: {avg_snr:.2f}")

    # Calculate the slope using the averaged data
    slope = calculate_slope(df_avg_sorted)

    if not np.isnan(slope):
        print(f"\nCalculated Slope of Averaged SNR over Time: {slope:.6f} dB per minute")
    else:
        print("\nCould not calculate slope due to insufficient data.")

    # Optional: Plot the averaged SNR over time
    # Make sure to import matplotlib at the top of your script
    import matplotlib.pyplot as plt

    # Plot averaged SNR over time
    plt.figure(figsize=(10, 6))
    plt.plot(df_avg_sorted['timestamp'], df_avg_sorted['avg_snr'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Average SNR (dB)')
    plt.title(f'Average SNR over Time for Zone {zone}, Band {band}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
