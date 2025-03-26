import time
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests
import xml.etree.ElementTree as ET
import os
import htmlmin
import sqlite3
import numpy as np
import html
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

parser = argparse.ArgumentParser(description='Analyze SNR and generate HTML report.')
parser.add_argument("-f", "--frequency", help="Specify how often data is collected (in minutes). Default = 1",
                    type=float, default=1)
parser.add_argument("-l", "--lower",
                    help="Specify the lower end of the data count threshold (empty square). Default = 5",
                    type=int, default=5)
parser.add_argument("-u", "--upper",
                    help="Specify the upper end of the data count threshold (filled square). Default = 10",
                    type=int, default=10)
parser.add_argument("-r", "--range", type=float, default=0.25,
                    help="Specify # of hours of data from current time to analyze. Default = 0.25")
parser.add_argument("-o", "--output-folder", help="Specify the local folder to save the index.html file.",
                    default="local_html")
parser.add_argument("--use-s3", action="store_true", help="Enable uploading to S3")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging output")
parser.add_argument("--include-solar-data", action="store_true", help="Include fetching and presenting solar data")
args = parser.parse_args()
frequency = args.frequency
sparse = args.lower
busy = args.upper
span = args.range
output_folder = args.output_folder
use_s3 = args.use_s3
debug = args.debug
include_solar_data = args.include_solar_data

#band_order = ['160', '80', '40', '30', '20', '17', '15', '12', '10', '6']
band_order = ['160', '80', '40', '20', '15', '10']

# Mapping zone numbers to descriptions...
zone_name_map = {
    1: 'Northwestern Zone of North America: KL (Alaska), VY1/VE8 Yukon, the Northwest and Nunavut Territories west of 102 degrees (Includes the islands of Victoria, Banks, Melville, and Prince Patrick).',
    2: 'Northeastern Zone of North America: VO2 Labrador, the portion of VE2 Quebec north of the 50th parallel, the VE8 Northwest and Nunavut Territories east of 102 degrees (Includes the islands of King Christian, King William, Prince of Wales, Somerset, Bathurst, Devon, Ellesmere, Baffin and the Melville and Boothia Peninsulas, excluding Akimiski Island).',
    3: 'Western Zone of North America: VE7, W6, and the W7 states of Arizona, Idaho, Nevada, Oregon, Utah, and Washington.',
    4: 'Central Zone of North America: VE3, VE4, VE5, VE6, VE8 Akimiski Island, and W7 states of Montana and Wyoming. W0, W9, W8 (except West Virginia), W5, and the W4 states of Alabama, Tennessee, and Kentucky.',
    5: 'Eastern Zone of North America: 4U1UN, CY9, CY0, FP, VE1, VE9, VY2, VO1 and the portion of VE2 Quebec south of the **th parallel. VP9, W1, W2, W3 and the W4 states of Florida, Georgia, South Carolina, North Carolina, Virginia and the W8 state of West Virginia.',
    6: 'Southern Zone of North America: XE/XF, XF4 (Revilla Gigedo).',
    7: 'Central American Zone: FO (Clipperton), HK0 (San Andres and Providencia), HP, HR, TG, TI, TI9, V3, YN and YS.',
    8: 'West Indies Zone: C6, CO, FG, FJ, FM, FS, HH, HI, J3, J6, J7, J8, KG4 (Guantanamo), KP1, KP2, KP4, KP5, PJ (Saba, St. Maarten, St. Eustatius), V2, V4, VP2, VP5, YV0 (Aves Is.), ZF, and 8P.',
    9: 'Northern Zone of South America: FY, HK, HK0 (Malpelo), P4, PJ (Bonaire, Curacao), PZ, YV, 8R, and 9Y.',
    10: 'Western Zone of South America: CP, HC, HC8, and OA.',
    11: 'Central Zone of South America: PY, PY0, and ZP.',
    12: 'Southwest Zone of South America: 3Y (Peter I), CE, CE0 (Easter Is., Juan Fernandez Is.), and some Antarctic stations.',
    13: 'Southeast Zone of South America: CX, LU, VP8 Islands, and some Antarctic stations.',
    14: 'Western Zone of Europe: C3, CT, CU, DL, EA, EA6, El, F, G, GD, GI, GJ, GM. GU, GW, HB, HB0, LA, LX, ON, OY, OZ, PA, SM, ZB, 3A and 4U1ITU.',
    15: 'Central European Zone: ES, HA, HV, I, IS0, LY, OE, OH, OH0, OJ0, OK, OM, S5, SP, T7, T9, TK, UA2, YL, YU, ZA, 1A0, Z3, 9A, 9H and 4U1VIC.',
    16: 'Eastern Zone of Europe: UR-UZ, EU-EW, ER, UA1, UA3, UA4, UA6, UA9 (S, T, W), and R1MV (Malyj Vysotskij).',
    17: 'Western Zone of Siberia: EZ, EY, EX, UA9 (A, C, F, G, J, K, L, M, Q, X) UK, UN-UQ, UH, UI and UJ-UM.',
    18: 'Central Siberian Zone: UA8 (T, V), UA9 (H, O, U, Y, Z), and UA0 (A, B, H, O, S, U, W).',
    19: 'Eastern Siberian Zone: UA0 (C, D, E, I, J, K, L, Q, X, Z).',
    20: 'Balkan Zone: E4, JY, LZ, OD, SV, SV5, SV9, SV/A, TA, YK, YO, ZC4, 4X and 5B.',
    21: 'Southwestern Zone of Asia: 4J, 4K, 4L, A4, A6, A7, A9, AP, EK, EP, HZ, YA, YI, 7O and 9K.',
    22: 'Southern Zone of Asia: A5, S2, VU, VU (Lakshadweep Is.), 4S, 8Q, and 9N.',
    23: 'Central Zone of Asia: JT, UA0Y, BY3G-L (Nei Mongol), BY9, BY0.',
    24: 'Eastern Zone of Asia: BQ9 (Pratas), BV, BY1, BY2, BY3A-F (Tian Jin), BY3M-R (He Bei), BY3S-X (Shan Xi), BY4, BY5, BY6, BY7, BY8, VR and XX.',
    25: 'Japanese Zone: HL, JA and P5.',
    26: 'Southeastern Zone of Asia: HS, VU (Andaman and Nicobar Islands), XV(3W), XU, XW, XZ and 1S (Spratly Islands).',
    27: 'Philippine Zone: DU (Philippines), JD1 (Minami Torishima), JD1 (Ogasawara), T8(KC6) (Palau), KH2 (Guam), KH0 (Marianas Is.), V6 (Fed. States of Micronesia) and BS7 (Scarborough Reef).',
    28: 'Indonesian Zone: H4, P2, V8, YB, 4W (East Timor), 9M and 9V.',
    29: 'Western Zone of Australia: VK6, VK8, VK9X (Christmas Is.), VK9C (Cocos-Keeling Is.) and some Antarctic stations.',
    30: 'Eastern Zone of Australia: FK/C (Chesterfield), VK1-5, VK7, VK9L (Lord Howe Is.), VK9W (Willis Is.), VK9M (Mellish Reef), VK0 (Macquarie Is.) and some Antarctic stations.',
    31: 'Central Pacific Zone: C2, FO (Marquesas), KH1, KH3, KH4, KH5, KH5K, KH6, KH7, KH7K, KH9, T2, T3, V7 and ZK3.',
    32: 'New Zealand Zone: A3, FK (except Chesterfield), FO (except Marquesas and Clipperton), FW, H40(Temotu), KH8, VK9N (Norfolk Is.) VP6 (Pitcairn and Ducie), YJ, ZK1, ZK2, ZL, ZL7, ZL8, 3D2, 5W and some Antarctic stations.',
    33: 'Northwestern Zone of Africa: CN, CT3, EA8, EA9, IG9, IH9 (Pantelleria Is.), S0, 3V and 7X.',
    34: 'Northeastern Zone of Africa: ST, SU and 5A.',
    35: 'Central Zone of Africa: C5, D4, EL J5, TU, TY, TZ, XT, 3X, 5N, 5T, 5U, 5V, 6W, 9G and 9L.',
    36: 'Equatorial Zone of Africa: D2, TJ, TL, TN, S9, TR, TT, ZD7, ZD8, 3C, 3C0, **, 9Q, 9U and 9X.',
    37: 'Eastern Zone of Africa: C9, ET, E3, J2, T5, 5H, 5X, 5Z, 7O and 7Q.',
    38: 'South African Zone: A2, V5, ZD9, Z2, ZS1-ZS8, 3DA, 3Y (Bouvet Is.), 7P, and some Antarctic stations.',
    39: 'Madagascar Zone: D6, FT-W, FT-X, FT-Z, FH, FR, S7, VK0 (Heard Is.) VQ9, 3B6/7, 3B8, 3B9, 5R8 and some Antarctic stations.',
    40: 'North Atlantic Zone: JW, JX, OX, R1FJ (Franz Josef Land), and TF.'
}

def get_aws_credentials():
    access_key = input("Enter your AWS Access Key ID: ")
    secret_key = input("Enter your AWS Secret Access Key: ")
    bucket = input("Enter the name of the S3 Bucket you'd like to write to: ")
    return {
        'aws_access_key_id': access_key,
        'aws_secret_access_key': secret_key,
        's3_bucket': bucket
    }

def upload_file_to_s3(file_name, bucket_name, acc_key, sec_key):
    """
    Uploads the html file to the AWS S3 bucket.
    """ 
    creds = {
        'aws_access_key_id': acc_key,
        'aws_secret_access_key': sec_key
    }
    s3_client = boto3.client('s3', **creds)
    obj_name = 'index.html'

    try:
        s3_client.upload_file(file_name, bucket_name, obj_name, ExtraArgs={'ContentType':'text/html; charset=utf-8'})
        print(f"File {file_name} uploaded successfully to {bucket_name}/{obj_name}")
        return True
    except FileNotFoundError:
        print(f"The file {file_name} was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {e}")
    quit(1)

def reformat_table(table):
    """
    Reformats the pivot table with improved zone tooltips.
    Ensures the returned DataFrame has a standard 0-based index.
    """
    try:
        # Create a base DataFrame with all zones (1-40)
        all_zones_df = pd.DataFrame({'zone': range(1, 41)})
        
        if table is None or table.empty:
            # Create zone_display with improved tooltip structure
            all_zones_df['zone_display'] = all_zones_df['zone'].apply(
                lambda x: f'<div class="zone-tooltip" style="display: inline-block; width: 100%; text-align: center; cursor: help;" title="{zone_name_map.get(int(x), "Unknown Zone")}">{str(int(x)).zfill(2)}</div>'
            )
            for band in band_order:
                all_zones_df[band] = ''
            # Reset index here to ensure 0-39 index
            return all_zones_df.reset_index(drop=True) 

        # Flatten the pivot table (which might have zone index 1-40)
        flattened = table.reset_index()
        
        # Merge with all_zones_df to ensure all zones are present
        flattened = pd.merge(all_zones_df, flattened, on='zone', how='left')
        
        # Create zone_display with improved tooltip structure
        flattened['zone_display'] = flattened['zone'].apply(
            lambda x: f'<div class="zone-tooltip" style="display: inline-block; width: 100%; text-align: center; cursor: help;" title="{zone_name_map.get(int(x), "Unknown Zone")}">{str(int(x)).zfill(2)}</div>'
        )
        
        # Sort by zone and reset index to ensure 0-39 index
        flattened = flattened.sort_values(by='zone').reset_index(drop=True) 
        
        # Ensure all band columns exist
        for band in band_order:
            if band not in flattened.columns:
                flattened[band] = ''
        
        # Select and order columns
        columns = ['zone', 'zone_display'] + band_order
        flattened = flattened[columns]
        
        # Fill NaN values with empty strings
        flattened = flattened.fillna('')
        
        return flattened # Returns DataFrame with index 0-39
    except Exception as e:
        print(f"Error in reformat_table: {str(e)}")
        # Return a default structure in case of error, ensuring 0-39 index
        all_zones_df = pd.DataFrame({'zone': range(1, 41)})
        all_zones_df['zone_display'] = all_zones_df['zone'].apply(
            lambda x: f'<div class="zone-tooltip" style="display: inline-block; width: 100%; text-align: center; cursor: help;" title="Error">{str(int(x)).zfill(2)}</div>'
        )
        for band in band_order:
             all_zones_df[band] = ''
        return all_zones_df.reset_index(drop=True) # Ensure 0-39 index on error return
      
def delete_old(df, time_hours):
    """
    Deletes entries older than the specified time range from the dataframe.
    """
    day_ago = datetime.now(dt.timezone.utc) - timedelta(hours=time_hours)
    df = df[df['timestamp'] >= day_ago].reset_index(drop=True)
    return df

def slope_to_unicode(slope):
    """
    Converts a slope value to a corresponding Unicode character.
    """
    if pd.isna(slope): # Handle NaN slope explicitly
        return ''
    elif -0.1 <= slope <= 0.1:
        return '\u21D4'  # ⇔
    elif 0.1 < slope <= 0.3:
        return '\u21D7'  # ⇗
    elif slope > 0.3:
        return '\u21D1'  # ⇑
    elif -0.3 <= slope < -0.1:
        return '\u21D8'  # ⇘
    elif slope < -0.3:
        return '\u21D3'  # ⇓
    else:
        return '' # Should not happen if NaN is handled

def compute_slope(df, zone, band):
    """
    Computes the slope of SNR over time for a given zone and band.
    """
    relevant_spots = df[(df['zone'] == zone) & (df['band'] == band)].copy()

    if len(relevant_spots) < 2:
        return np.nan

    relevant_spots.loc[:, 'minute'] = relevant_spots['timestamp'].dt.floor('min')
    avg_per_minute = relevant_spots.groupby('minute')['snr'].mean().reset_index()

    if len(avg_per_minute) < 2:
        return np.nan

    time_values = avg_per_minute['minute'].astype(np.int64) // 1e9 / 60
    snr_values = avg_per_minute['snr']

    try:
        slope, _, _, _, _ = linregress(time_values, snr_values)
        return slope
    except Exception as e:
        print(f"Error calculating slope: {e}")
        return np.nan

def compute_ema_slope(df, zone, band, span_minutes=5):
    """
    Computes the slope of the Exponential Moving Average (EMA) of SNR over time.
    Returns slope and the EMA series for sparkline.
    """
    relevant_spots = df[(df['zone'] == zone) & (df['band'] == band)].copy()

    if len(relevant_spots) < 2:
        return np.nan, pd.Series(dtype=float) # Return NaN slope and empty series for sparkline

    relevant_spots.loc[:, 'minute'] = relevant_spots['timestamp'].dt.floor('min')
    avg_per_minute = relevant_spots.groupby('minute')['snr'].mean().reset_index()

    if len(avg_per_minute) < 2:
        # Still return the (short) series for potential sparkline if needed
        return np.nan, avg_per_minute.set_index('minute')['snr'].sort_index() 

    # Ensure the index is datetime for EMA calculation
    avg_per_minute = avg_per_minute.set_index('minute').sort_index()

    # Calculate EMA
    # Adjust span based on expected data frequency (e.g., 1 minute)
    ema_snr = avg_per_minute['snr'].ewm(span=span_minutes, adjust=False).mean()

    # Prepare data for regression on EMA
    time_values = ema_snr.index.astype(np.int64) // 1e9 / 60 # Convert to minutes
    snr_values_ema = ema_snr.values

    if len(time_values) < 2 or time_values.nunique() == 1:
        return np.nan, ema_snr # Return NaN slope but still return EMA data for sparkline

    try:
        slope, _, _, _, _ = linregress(time_values, snr_values_ema)
        return slope, ema_snr # Return slope and EMA data
    except Exception as e:
        print(f"Error calculating EMA slope: {e}")
        return np.nan, ema_snr # Return NaN slope but still return EMA data

def generate_sparkline_svg(data, trend_slope, width=50, height=15, stroke_width=1, default_color='black'):
    """
    Generates an SVG sparkline from a pandas Series of data, coloring the last segment based on trend.
    """
    if data.empty or len(data) < 2:
        return ""

    y = data.dropna().values # Drop NaN values before plotting
    if len(y) < 2:
        return ""
        
    x = np.arange(len(y))

    # Normalize data
    min_y, max_y = np.min(y), np.max(y)
    range_y = max_y - min_y if max_y > min_y else 1
    # Add small epsilon to prevent division by zero if range_y is 0
    range_y = range_y if range_y > 1e-9 else 1 
    norm_y = (y - min_y) / range_y * (height - stroke_width * 2) + stroke_width

    min_x, max_x = np.min(x), np.max(x)
    range_x = max_x - min_x if max_x > min_x else 1
    # Add small epsilon to prevent division by zero if range_x is 0
    range_x = range_x if range_x > 1e-9 else 1
    norm_x = (x - min_x) / range_x * (width - stroke_width * 2) + stroke_width

    # Determine trend color
    if pd.isna(trend_slope) or (-0.1 <= trend_slope <= 0.1):
        trend_color = default_color  # black or grey for stable/NaN
    elif trend_slope > 0.1:
        trend_color = '#28a745'  # green
    else: # slope < -0.1
        trend_color = '#dc3545'  # red

    # Create points strings
    all_points = " ".join([f"{px:.2f},{height - py:.2f}" for px, py in zip(norm_x, norm_y)])
    main_points = " ".join([f"{px:.2f},{height - py:.2f}" for px, py in zip(norm_x[:-1], norm_y[:-1])]) # All except last point
    last_segment_points = " ".join([f"{px:.2f},{height - py:.2f}" for px, py in zip(norm_x[-2:], norm_y[-2:])]) # Last two points

    # Generate SVG with two polylines
    svg_content = f'<polyline points="{main_points}" fill="none" stroke="{default_color}" stroke-width="{stroke_width}"/>'
    svg_content += f'<polyline points="{last_segment_points}" fill="none" stroke="{trend_color}" stroke-width="{stroke_width}"/>'
    
    return f'<svg width="{width}" height="{height}" style="vertical-align: middle; margin-right: 3px;">{svg_content}</svg>'


def get_intensity(count, max_count=1000):
    """
    Calculate color intensity using exponential scaling.
    """
    min_intensity = 0.2
    max_additional = 0.8
    a = 5.0 / max_count
    intensity = min_intensity + max_additional * (1 - np.exp(-a * count))
    return intensity

def hsl_to_rgb(h, s, l):
    """
    Convert HSL to RGB values.
    """
    if s == 0:
        return (l, l, l)
    
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    return (r, g, b)

def snr_to_color(val, count):
    """
    Convert SNR value and station count to color.
    """
    if pd.isna(val) or val == ' ':
        return 'background-color: #ffffff; padding: 1px 2px;'
    
    try:
        val = float(val)
        # Base color selection based on SNR
        if val >= 0:
            hue = 120  # green
        elif val >= -10:
            hue = 90   # yellow-green
        elif val >= -15:
            hue = 200  # blue
        else:
            hue = 220  # dark blue
                
        # Get intensity based on station count
        intensity = get_intensity(count)
            
        # Adjust lightness to avoid too dark colors
        sat = 0.8  # 80% saturation
        min_lightness = 0.3  # Minimum lightness (30%)
        max_lightness = 0.7  # Maximum lightness (70%)
        lightness = min_lightness + intensity * (max_lightness - min_lightness)
            
        # Convert HSL to RGB
        rgb_color = hsl_to_rgb(hue/360, sat, lightness)
        hex_color = '#{:02x}{:02x}{:02x}'.format(*[int(x * 255) for x in rgb_color])
            
        return f'background-color: {hex_color}; padding: 1px 2px; text-align: center; font-size: 0.85rem;'
    except ValueError:
        return 'background-color: #ffffff; padding: 1px 2px;'


def combine_snr_count(zone, band, median_snr, q1_snr, q3_snr, count, ema_slope, sparkline_svg, df, row_index):
    """
    Combines SNR stats (Median, IQR), count, and sparkline (with embedded trend) for table cell display.
    """
    try:
        # Format Median and IQR
        if pd.isna(median_snr):
            snr_display = "N/A"
        else:
            median_snr_int = int(round(median_snr))
            if pd.isna(q1_snr) or pd.isna(q3_snr):
                snr_display = f"{median_snr_int}"
            else:
                q1_int = int(round(q1_snr))
                q3_int = int(round(q3_snr))
                snr_display = f"{median_snr_int} <span class='iqr-text'>[{q1_int}/{q3_int}]</span>"

        # Format count
        count_display = f'<span class="count-text">({count})</span>'

        # Combine SNR and count display
        display_text = f"{snr_display} {count_display}"

        # Combine sparkline and text
        if count > 0:
             cell_content = f'{sparkline_svg}{display_text}'
        else:
             cell_content = "" # Empty if no count

        # Generate tooltip content (spotted stations) - only if count > 0
        tooltip_data = None
        if count > 0:
            relevant_spots = df[(df['zone'] == zone) & (df['band'] == band)].copy()
            unique_stations = sorted(set(relevant_spots['spotted_station']))
            
            tooltip_id = f"tooltip_{row_index}_{band}" # Use row_index (0-39) for unique tooltip ID
            tooltip_content_html = '<div class="station-list">'
            for station in unique_stations:
                display_station = station.replace('.', '/') # Keep existing logic
                tooltip_content_html += f'<div>{html.escape(display_station)}</div>'
            tooltip_content_html += '</div>'
            tooltip_data = (tooltip_id, tooltip_content_html)

            # Wrap cell content in tooltip span if there's tooltip data
            cell_content = f'<span class="tooltip" data-tooltip-content="#{tooltip_id}">{cell_content}</span>'

        return cell_content, tooltip_data

    except Exception as e:
        print(f"Error processing zone {zone} and band {band}: {str(e)}")
        return "", None

def generate_empty_cell_style(total_zones=40):
    """Generate CSS for empty cells for all zones"""
    empty_cell_styles = []
    for i in range(total_zones):
        for band in band_order:
            empty_cell_styles.append(f"""
            #T_table_row{i}_col{band} {{
                background-color: #ffffff;
                text-align: center;
            }}
            """)
    return "\n".join(empty_cell_styles)

def generate_html_template(snr_table_html, tooltip_content_html, caption_string):
    """
    Generates HTML template with improved tooltip styles.
    """
    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="60">
        <title>SNR Report</title>
        <style>
            body {{
                margin: 0;
                padding: 4px;
                font-family: 'Roboto', monospace;
                font-size: 0.85rem;
                background: #ffffff;
            }}
    
            table {{
                border-collapse: collapse;
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                table-layout: fixed;
            }}
    
            th {{
                position: sticky;
                top: 0;
                background-color: rgba(255, 255, 255, 0.95);
                z-index: 10;
                padding: 2px;
                font-size: 0.85rem;
                border: 1px solid #ddd;
                font-weight: bold;
            }}
    
            td {{
                padding: 1px 2px;
                border: 1px solid #ddd;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                /* Ensure SVG and text align */
                vertical-align: middle; 
            }}
    
            tr:nth-child(even) {{
                background-color: rgba(0, 0, 0, 0.02);
            }}
    
            td:first-child {{
                width: 35px;
                font-weight: bold;
                cursor: help;
                padding: 0;
            }}

            .zone-tooltip {{
                padding: 1px 2px;
                background-color: rgba(0, 0, 0, 0.02);
                transition: background-color 0.2s;
            }}

            .zone-tooltip:hover {{
                background-color: rgba(0, 0, 0, 0.05);
            }}
               
            .tippy-box[data-theme~='zone'] {{
                background-color: #333;
                color: white;
                font-size: 0.8rem;
                line-height: 1.3;
                max-width: none !important;
                width: auto !important;
            }}

            .tippy-box[data-theme~='zone'] .tippy-content {{
                padding: 8px 12px;
            }}
    
            .tippy-box[data-theme~='zone'] .tippy-arrow {{
                color: #333;
            }}
    
            .tippy-content {{
                padding: 0 !important;
                font-size: 0.8rem;
                max-width: none !important;
                width: auto !important;
                background: white;
            }}

            .tooltip {{
                cursor: pointer;
                /* Allow tooltip span to contain block/inline-block elements */
                display: inline-block; 
                width: 100%;
            }}

            .station-list {{
                display: inline-grid;
                grid-template-columns: repeat(4, minmax(70px, max-content));
                gap: 2px;
                padding: 4px;
                background: #e4f0f3;
                color: #333333;
                width: fit-content;
                max-width: 100%;
            }}

            .station-list div {{
                padding: 2px 4px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
    
            .tooltip_templates {{
                display: none;
            }}
    
            caption {{
                padding: 2px;
                font-size: 0.85rem;
                font-weight: bold;
            }}

            .count-text {{
                font-size: 0.5rem;
                color: #666;
            }}

            .iqr-text {{
                font-size: 0.5rem; /* Slightly smaller for IQR */
                color: #888; /* Lighter color for IQR */
            }}

            td svg {{ /* Style for sparkline SVG */
                vertical-align: middle;
                margin-right: 3px; /* Space between sparkline and text */
            }}

            .legend {{
                position: fixed;
                bottom: 20px;
                left: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                z-index: 1000;
                text-align: center;
            }}
        </style>
        <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css">
    </head>
    <body>
        {snr_table_html}
        {tooltip_content_html}
        <div style="text-align: center; margin-top: 20px;">
            <small>Make your own SNR overview: <a href="https://github.com/s53zo/Hamradio_copilot">https://github.com/s53zo/Hamradio_copilot</a> <a href="https://azure.s53m.com/copilot/index.html">ALL RX</a> <a href="https://azure.s53m.com/copilot/index_s53m.html">S53M RX only</a></small>
        </div>

        <script src="https://unpkg.com/@popperjs/core@2/dist/umd/popper.min.js"></script>
        <script src="https://unpkg.com/tippy.js@6/dist/tippy-bundle.umd.min.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                tippy('.tooltip', {{
                    content(reference) {{
                        const id = reference.getAttribute('data-tooltip-content');
                        const template = document.querySelector(id);
                        return template.innerHTML;
                    }},
                    allowHTML: true,
                    maxWidth: 'none',
                    interactive: true,
                    animation: 'scale',
                    placement: 'top',
                    theme: 'light',
                }});
            }});
        </script>
    </body>
    </html>
    """
    return template

def run(access_key=None, secret_key=None, s3_buck=None, include_solar_data=False):
    # Connect to the SQLite database
    conn = sqlite3.connect('callsigns.db')
  
    # Read data from the SQLite table `callsigns` into a pandas DataFrame
    query = """
    SELECT zone, band, CAST(snr AS FLOAT) as snr, timestamp, spotter, spotted_station
    FROM callsigns
    """
    
    try:
        df = pd.read_sql_query(
            query,
            conn,
            dtype={
                'zone': 'Int64',
                'band': 'str',
                'snr': 'float',
                'spotter': 'str',
                'spotted_station': 'str'
            }
        )
        
        # Convert 'timestamp' from UNIX timestamp (seconds) to datetime with UTC timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce')
    except Exception as e:
        print(f"Error: Unable to read data from the SQLite database. {e}")
        return
    finally:
        conn.close()

    if debug:
        print("Initial DataFrame:")
        print(df.head())

    df = delete_old(df, span)

    if debug:
        print(f"DataFrame after deleting entries older than {span} hours:")
        print(df.head())

    # Set 'zone' as a categorical variable with categories from 1 to 40
    df['zone'] = df['zone'].astype(int)
    df['zone'] = df['zone'].astype('category')
    df['zone'] = df['zone'].cat.set_categories(range(1, 41))

    # Set 'band' as a categorical variable with desired order
    df['band'] = df['band'].astype(str)
    df['band'] = df['band'].astype('category')
    df['band'] = df['band'].cat.set_categories(band_order)

    all_zones = pd.Index(range(1, 41), name='zone') # Index 1-40 for accessing data

    # --- Calculate Aggregated Stats (Median, IQR, Count) ---
    agg_funcs = {
        'snr': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        # Use a unique name for the count aggregation to avoid potential conflicts
        'spotter': [('unique_spots', lambda x: len(set(zip(x, df.loc[x.index, 'spotted_station']))))] 
    }
    
    # Group, aggregate, and unstack by band
    stats_table = df.groupby(['zone', 'band'], observed=True).agg(agg_funcs).unstack(level='band')
    
    # Select stats directly using MultiIndex columns and reindex to ensure all zones/bands
    # These tables will have index 1-40
    median_table = stats_table.get(('snr', 'median'), pd.DataFrame()).reindex(index=all_zones, columns=band_order)
    q1_table = stats_table.get(('snr', '<lambda_0>'), pd.DataFrame()).reindex(index=all_zones, columns=band_order)
    q3_table = stats_table.get(('snr', '<lambda_1>'), pd.DataFrame()).reindex(index=all_zones, columns=band_order)
    count_table = stats_table.get(('spotter', 'unique_spots'), pd.DataFrame()).reindex(index=all_zones, columns=band_order).fillna(0).astype(int)

    # --- Prepare data for EMA slope and Sparklines ---
    # Calculate per-minute averages for the entire dataset (within the time span)
    df_minute_avg = df.copy()
    df_minute_avg['minute'] = df_minute_avg['timestamp'].dt.floor('min')
    per_minute_snr = df_minute_avg.groupby(['zone', 'band', 'minute'], observed=True)['snr'].mean().reset_index()
        
    # --- Reformat base table structure ---
    # Reformat the base structure for the final table (using median table as template for zones/bands layout)
    # display_table_base will have index 0-39
    display_table_base = reformat_table(median_table.copy()) # Use copy to avoid modifying original

    if debug:
        print("Count Table:")
        print(count_table.head())
        print("Median Table:")
        print(median_table.head())
        print("Q1 Table:")
        print(q1_table.head())
        print("Q3 Table:")
        print(q3_table.head())

    now = dt.datetime.now(dt.timezone.utc).strftime("%b %d, %Y %H:%M")
    caption_string = f"Last {int(span*60)} minutes SNR (Median [Q1,Q3]) of spots in S5 and around - refresh at {now} GMT"

    # Create the final display table structure from the base (index 0-39)
    combined_table = display_table_base.copy()
    
    # Tooltip content list
    tooltip_contents = []
    
    # --- Pre-calculate styles ---
    # Create styles DataFrame matching combined_table structure (index 0-39, columns including 'zone', 'zone_display')
    styles_df = pd.DataFrame('', index=combined_table.index, columns=combined_table.columns)
    # Set default white background for all cells initially
    for col in styles_df.columns:
         styles_df[col] = 'background-color: #ffffff; padding: 1px 2px;'

    # --- Populate combined_table and styles_df ---
    for band in band_order:
        if band in median_table.columns: # Check if band exists in the data
            combined_results = []
            # Iterate using combined_table's index (0-39)
            for idx in combined_table.index: 
                zone = idx + 1 # Zone number is index + 1 (1-40)
                
                # Safely get median, q1, q3, count using .loc with the zone number (1-40) on the original stat tables
                # Use .get() for safer access in case zone doesn't exist (though reindex should prevent this)
                median_snr = median_table.get(band, pd.Series(dtype=float)).get(zone, np.nan)
                q1_snr = q1_table.get(band, pd.Series(dtype=float)).get(zone, np.nan)
                q3_snr = q3_table.get(band, pd.Series(dtype=float)).get(zone, np.nan)
                count = count_table.get(band, pd.Series(dtype=int)).get(zone, 0)

                # Get recent per-minute data for EMA slope and sparkline
                # Filter the pre-calculated per_minute_snr DataFrame using zone (1-40)
                recent_snr_data = per_minute_snr[
                    (per_minute_snr['zone'] == zone) & (per_minute_snr['band'] == band)
                ].set_index('minute')['snr'].sort_index()

                # Calculate EMA slope and get EMA series
                # Pass the main df for calculation, as it contains all necessary raw data
                ema_slope, ema_series = compute_ema_slope(df, zone, band) 

                # Generate sparkline from EMA series, passing slope for trend coloring
                sparkline_svg = generate_sparkline_svg(ema_series, trend_slope=ema_slope) 

                # Combine display elements
                result = combine_snr_count(
                    zone, band, median_snr, q1_snr, q3_snr, count, ema_slope, sparkline_svg, df, idx # Pass idx (0-39) for tooltip ID
                )
                combined_results.append(result)

                # Calculate and store style string in styles_df using index 'idx' (0-39)
                if not pd.isna(median_snr) and count > 0:
                    styles_df.loc[idx, band] = snr_to_color(median_snr, count)
                # else: style remains default white
            
            combined_table[band] = [result[0] for result in combined_results]
            tooltip_contents.extend([content for _, content in combined_results if content])
        else:
            combined_table[band] = ' ' # Ensure column exists even if no data
            styles_df[band] = 'background-color: #ffffff; padding: 1px 2px;' # Ensure style column exists

    # Handle 'zone' column separately (already done in display_table_base)
    combined_table['zone_display'] = display_table_base['zone_display']

    # Ensure the columns are ordered as per band_order
    combined_table = combined_table[['zone_display'] + band_order]
    combined_table = combined_table.rename(columns={'zone_display': 'zone'})
    
    # Ensure styles_df has the correct columns ('zone' + band_order) matching combined_table
    styles_df = styles_df.reindex(columns=combined_table.columns, fill_value='background-color: #ffffff; padding: 1px 2px;')
    # Set default style for the 'zone' column (no background color)
    styles_df['zone'] = '' 

    # Apply the styles DataFrame directly using axis=None (elementwise)
    styled_table = combined_table.style.apply(lambda x: styles_df, axis=None).set_caption(caption_string)

    # Apply other non-background properties separately
    styled_table.set_properties(subset=['zone'], **{'font-weight': 'bold'})
    styled_table.set_properties(**{'text-align': 'center'})

    # Set table styles
    styled_table.set_table_styles([
        {'selector': 'caption', 'props': [
            ('font-size', '0.85rem'),
            ('font-weight', 'bold'),
            ('padding', '2px')
        ]},
        {'selector': 'th', 'props': [
            ('font-size', '0.85rem'),
            ('padding', '2px'),
            ('position', 'sticky'),
            ('top', '0'),
            ('background-color', 'rgba(255, 255, 255, 0.95)'),
            ('z-index', '1'),
            ('font-weight', 'bold'),
            ('white-space', 'nowrap')
        ]},
        {'selector': 'td', 'props': [
            ('padding', '1px 2px'),
            ('font-size', '0.85rem'),
            ('white-space', 'nowrap')
        ]},
        {'selector': 'td:first-child', 'props': [
            ('font-weight', 'bold'),
            ('width', '35px'),
            ('min-width', '35px')
        ]}
    ])

    # Convert to HTML
    html_table = styled_table.hide(axis="index").to_html()
    html_table = html_table.replace(
        '<table ',
        '<table style="width: 100%; max-width: 800px; margin: 0 auto; table-layout: fixed;" '
    )

    # Build tooltip content HTML
    tooltip_content_html = ''
    for tooltip_id, content_html in tooltip_contents:
        tooltip_content_html += f'''
        <div class="tooltip_templates">
            <div id="{tooltip_id}">
                {content_html}
            </div>
        </div>
        '''

    # Generate final HTML
    final_html = generate_html_template(html_table, tooltip_content_html, caption_string)

    # Minify the HTML
    minified_html = htmlmin.minify(final_html, remove_empty_space=True, remove_comments=True)

    # Save files
    with open("index.html", "w", encoding="utf-8") as text_file:
        text_file.write(minified_html)

    os.makedirs(output_folder, exist_ok=True)
    local_file_path = os.path.join(output_folder, "index.html")
    with open(local_file_path, "w", encoding="utf-8") as local_file:
        local_file.write(minified_html)

    print(f"Table updated in index.html and saved to '{output_folder}' at {now}")
    
    if use_s3:
        if not (access_key and secret_key and s3_buck):
            access_key = input("Enter your AWS Access Key ID: ")
            secret_key = input("Enter your AWS Secret Access Key: ")
            s3_buck = input("Enter the name of the S3 Bucket you'd like to write to: ")
        upload_file_to_s3("index.html", s3_buck, access_key, secret_key)

if __name__ == '__main__':
    time_to_wait = frequency * 60  # Time to wait in between re-running program

    if use_s3:
        credentials = get_aws_credentials()
        aws_access_key = credentials['aws_access_key_id']
        secret_access_key = credentials['aws_secret_access_key']
        s3_bucket = credentials['s3_bucket']
    else:
        aws_access_key = None
        secret_access_key = None
        s3_bucket = None

    while True:
        run(aws_access_key, secret_access_key, s3_bucket, include_solar_data)
        time.sleep(time_to_wait)
