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
    """
    try:
        # Create a base DataFrame with all zones
        all_zones = pd.DataFrame({'zone': range(1, 41)})
        
        if table is None or table.empty:
            # Create zone_display with improved tooltip structure
            all_zones['zone_display'] = all_zones['zone'].apply(
                lambda x: f'<div class="zone-tooltip" style="display: inline-block; width: 100%; cursor: help;" title="{zone_name_map.get(int(x), "Unknown Zone")}">{str(int(x)).zfill(2)}</div>'
            )
            for band in band_order:
                all_zones[band] = ''
            return all_zones

        # Flatten the pivot table
        flattened = table.reset_index()
        
        # Merge with all_zones to ensure all zones are present
        flattened = pd.merge(all_zones, flattened, on='zone', how='left')
        
        # Create zone_display with improved tooltip structure
        flattened['zone_display'] = flattened['zone'].apply(
            lambda x: f'<div class="zone-tooltip" style="display: inline-block; width: 100%; cursor: help;" title="{zone_name_map.get(int(x), "Unknown Zone")}">{str(int(x)).zfill(2)}</div>'
        )
        
        # Sort by zone and reset index
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
        
        return flattened
    except Exception as e:
        print(f"Error in reformat_table: {str(e)}")
        return all_zones
      
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
    if -0.1 <= slope <= 0.1:
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
        return ''

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
            
        return f'background-color: {hex_color}; padding: 1px 2px; font-size: 0.85rem;'
    except ValueError:
        return 'background-color: #ffffff; padding: 1px 2px;'


def combine_snr_count(mean_table_all_row, mean_table_s53m_row, count_table_all, count_table_s53m, band, df, df_s53m, row_index):
    """
    Combines SNR and count data for both all spots and S53M spots with proper empty value handling.
    """
    try:
        zone = mean_table_all_row['zone']
        
        # Process all spots data
        snr_all = mean_table_all_row[band] if band in mean_table_all_row else None
        if isinstance(snr_all, str) and snr_all.strip() == '':
            snr_all = None
        elif snr_all is not None:
            try:
                snr_all = float(snr_all)
            except (ValueError, TypeError):
                snr_all = None
        
        count_all = count_table_all.at[row_index, band] if band in count_table_all.columns else 0
        if pd.isna(count_all) or (isinstance(count_all, str) and count_all.strip() == ''):
            count_all = 0
        else:
            try:
                count_all = int(count_all)
            except (ValueError, TypeError):
                count_all = 0

        # Process S53M spots data
        snr_s53m = mean_table_s53m_row[band] if band in mean_table_s53m_row else None
        if isinstance(snr_s53m, str) and snr_s53m.strip() == '':
            snr_s53m = None
        elif snr_s53m is not None:
            try:
                snr_s53m = float(snr_s53m)
            except (ValueError, TypeError):
                snr_s53m = None
        
        count_s53m = count_table_s53m.at[row_index, band] if band in count_table_s53m.columns else 0
        if pd.isna(count_s53m) or (isinstance(count_s53m, str) and count_s53m.strip() == ''):
            count_s53m = 0
        else:
            try:
                count_s53m = int(count_s53m)
            except (ValueError, TypeError):
                count_s53m = 0

        # Generate display text
        if snr_all is None and count_all == 0:
            display_text_all = "--"
        elif snr_all is None:
            display_text_all = f'N/A ({count_all})'
        else:
            display_text_all = f'{int(round(snr_all))} ({count_all})'

        if snr_s53m is None and count_s53m == 0:
            display_text_s53m = "--"
        elif snr_s53m is None:
            display_text_s53m = f'N/A ({count_s53m})'
        else:
            display_text_s53m = f'{int(round(snr_s53m))} ({count_s53m})'

        # Add trend indicators
        all_slope = None
        s53m_slope = None

        if snr_all is not None and count_all > 0:
            all_slope = compute_slope(df, zone, band)
            all_arrow = slope_to_unicode(all_slope)
            # Create styled arrow span
            arrow_color = 'black'
            if not pd.isna(all_slope):
                if all_slope > 0.3:
                    arrow_color = '#28a745'  # strong increase
                elif 0.1 < all_slope <= 0.3:
                    arrow_color = '#28a745'  # slight increase
                elif all_slope < -0.3:
                    arrow_color = '#dc3545'  # strong decrease
                elif -0.3 <= all_slope < -0.1:
                    arrow_color = '#dc3545'  # slight decrease
            
            display_text_all = f'{display_text_all} <span style="color: {arrow_color};">{all_arrow}</span>'

        if snr_s53m is not None and count_s53m > 0:
            s53m_slope = compute_slope(df_s53m, zone, band)
            s53m_arrow = slope_to_unicode(s53m_slope)
            # Create styled arrow span
            arrow_color = 'black'
            if not pd.isna(s53m_slope):
                if s53m_slope > 0.3:
                    arrow_color = '#28a745'  # strong increase
                elif 0.1 < s53m_slope <= 0.3:
                    arrow_color = '#28a745'  # slight increase
                elif s53m_slope < -0.3:
                    arrow_color = '#dc3545'  # strong decrease
                elif -0.3 <= s53m_slope < -0.1:
                    arrow_color = '#dc3545'  # slight decrease
            
            display_text_s53m = f'{display_text_s53m} <span style="color: {arrow_color};">{s53m_arrow}</span>'

        # Create tooltip content
        tooltip_content = None
        if count_all > 0 or count_s53m > 0:
            tooltip_id = f"tooltip_{row_index}_{band}"
            tooltip_content_html = '<div class="station-list">'
            
            if count_all > 0:
                tooltip_content_html += '<div style="font-weight: bold; margin-bottom: 4px;">All Spots:</div>'
                relevant_spots = df[(df['zone'] == zone) & (df['band'] == band)].copy()
                unique_stations = sorted(set(relevant_spots['spotted_station']))
                for station in unique_stations:
                    tooltip_content_html += f'<div>{html.escape(station)}</div>'
            
            if count_s53m > 0:
                tooltip_content_html += '<div style="font-weight: bold; margin-top: 8px; margin-bottom: 4px;">S53M Spots:</div>'
                relevant_spots = df_s53m[(df_s53m['zone'] == zone) & (df_s53m['band'] == band)].copy()
                unique_stations = sorted(set(relevant_spots['spotted_station']))
                for station in unique_stations:
                    tooltip_content_html += f'<div>{html.escape(station)}</div>'
            
            tooltip_content_html += '</div>'
            tooltip_content = (tooltip_id, tooltip_content_html)

        # Create the cell HTML with updated layout
        cell_html = f'''
        <div class="tooltip" data-tooltip-content="#{tooltip_id if tooltip_content else ''}">
            <div class="grid">
                <div>{display_text_all}</div>
                <div>{display_text_s53m}</div>
            </div>
        </div>
        '''

        return cell_html.strip(), tooltip_content

    except Exception as e:
        print(f"Error processing zone {zone if 'zone' in locals() else 'unknown'} and band {band}: {str(e)}")
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
    css_styles = """
        /* Base styles */
        body {
            margin: 0;
            padding: 4px;
            font-family: 'Roboto', monospace;
            font-size: 0.85rem;
            background: #ffffff;
            overflow-x: auto;
        }
        
        /* Table styles */
        table {
            border-collapse: collapse;
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            table-layout: auto;
        }
        
        th {
            position: sticky;
            top: 0;
            background-color: rgba(255, 255, 255, 0.95);
            z-index: 10;
            padding: 4px 8px;
            font-size: 0.85rem;
            border: 1px solid #ddd;
            font-weight: bold;
            text-align: center;
            white-space: nowrap;
        }
        
        td {
            padding: 0;
            border: 1px solid #ddd;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Header subtext */
        th span.subtext {
            display: block;
            font-size: 0.7rem;
            color: #666;
            font-weight: normal;
            white-space: pre-line;
        }
        
        /* Zone column */
        td:first-child {
            width: 45px;
            min-width: 45px;
            font-weight: bold;
            cursor: help;
            padding: 2px;
        }
        
        /* Band columns */
        th:not(:first-child), 
        td:not(:first-child) {
            min-width: 140px;
        }
        
        /* Grid layout for cells */
        td > div.grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            min-height: 28px;  /* Ensure consistent height */
        }
        
        td > div.grid > div {
            padding: 4px 2px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 60px;  /* Minimum width for each half */
            position: relative;  /* For border positioning */
        }
        
        /* Left cell */
        td > div.grid > div:first-child {
            border-right: 1px solid #ddd;
            margin-right: -1px;  /* Prevent double border */
        }
        
        /* Right cell (S53M) */
        td > div.grid > div:last-child {
            color: #2563eb;  /* Blue color for S53M values */
        }
        
        /* Band columns - wider to accommodate split */
        th:not(:first-child), 
        td:not(:first-child) {
            min-width: 160px;  /* Increased from 140px */
            width: 160px;      /* Fixed width */
        }
        
        /* Responsive adjustments */
        @media (max-width: 1200px) {
            th:not(:first-child), 
            td:not(:first-child) {
                min-width: 140px;
                width: 140px;
            }
            
            td > div.grid > div {
                min-width: 50px;
            }
        }
        
        /* Zone tooltip styles */
        .zone-tooltip {
            padding: 1px 2px;
            background-color: rgba(0, 0, 0, 0.02);
            transition: background-color 0.2s;
        }
        
        .zone-tooltip:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        /* Tooltip styles */
        .tooltip {
            cursor: pointer;
            display: block;
            width: 100%;
            height: 100%;
            white-space: nowrap;
        }
        
        .tippy-box[data-theme~='zone'] {
            background-color: #333;
            color: white;
            font-size: 0.8rem;
            line-height: 1.3;
            max-width: none !important;
            width: auto !important;
        }
        
        .tippy-box[data-theme~='zone'] .tippy-content {
            padding: 8px 12px;
        }
        
        .tippy-box[data-theme~='zone'] .tippy-arrow {
            color: #333;
        }
        
        .tippy-content {
            padding: 0 !important;
            font-size: 0.8rem;
            max-width: none !important;
            width: auto !important;
            background: white;
        }
        
        /* Station list in tooltip */
        .station-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 4px;
            padding: 8px;
            background: #e4f0f3;
            color: #333333;
            max-width: 600px;
        }
        
        .station-list div {
            padding: 2px 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Hide tooltip templates */
        .tooltip_templates {
            display: none;
        }
        
        /* Caption styles */
        caption {
            padding: 2px;
            font-size: 0.85rem;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        /* Utility classes */
        .text-blue-600 {
            color: #2563eb !important;
        }
        
        .text-center {
            text-align: center;
        }
        
        /* Trend indicators */
        .trend-up {
            color: #28a745;
        }
        
        .trend-down {
            color: #dc3545;
        }
        
        .trend-stable {
            color: #000000;
        }
        
        /* Footer styles */
        .footer-text {
            text-align: center;
            margin-top: 20px;
            font-size: 0.8rem;
        }
        
        .footer-text a {
            color: #2563eb;
            text-decoration: none;
        }
        
        .footer-text a:hover {
            text-decoration: underline;
        }
        
        /* Solar data panel */
        .solar-panel {
            position: fixed;
            left: 5%;
            padding: 10px;
            z-index: 1000;
            font-family: 'Roboto', monospace;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Legend panel */
        .legend {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 80%;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            z-index: 1000;
            font-size: 0.85rem;
        }
        
        /* Alternating row colors */
        tr:nth-child(even) {
            background-color: rgba(0, 0, 0, 0.02);
        }
        
        /* Responsive design */
        @media (min-width: 1400px) {
            table {
                max-width: 1400px;
            }
            
            th:not(:first-child), 
            td:not(:first-child) {
                /* min-width: 160px; */
                min-width: 200px;       /* or larger if needed */
                white-space: nowrap;    /* ensures text does not wrap */
            }
        }
        
        @media (max-width: 1200px) {
            table {
                max-width: 1000px;
            }
            
            th:not(:first-child), 
            td:not(:first-child) {
                min-width: 120px;
            }
        }
        
        @media (max-width: 768px) {
            td > div.grid > div {
                padding: 2px 4px;
                font-size: 0.75rem;
            }
            
            .station-list {
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            }
        }
    """

    template = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="60">
        <title>SNR Report</title>
        <style>
        {css_styles}
        </style>
        <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css">
    </head>
    <body>
        {snr_table_html}
        {tooltip_content_html}
        <div style="text-align: center; margin-top: 20px;">
            <small>Make your own SNR overview: <a href="https://github.com/s53zo/Hamradio_copilot">https://github.com/s53zo/Hamradio_copilot</a></small>
        </div>

        <script src="https://unpkg.com/@popperjs/core@2/dist/umd/popper.min.js"></script>
        <script src="https://unpkg.com/tippy.js@6/dist/tippy-bundle.umd.min.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                tippy('.tooltip', {{
                    content(reference) {{
                        const id = reference.getAttribute('data-tooltip-content');
                        const template = document.querySelector(id);
                        return template ? template.innerHTML : '';
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
    </html>"""
    return template

def run(access_key=None, secret_key=None, s3_buck=None, include_solar_data=False):
    # Connect to the SQLite database
    conn = sqlite3.connect('callsigns.db')
    
    # Modified query to get both all spots and S53M spots
    query = """
    SELECT 
        zone, 
        band, 
        CAST(snr AS FLOAT) as snr, 
        timestamp, 
        spotter, 
        spotted_station,
        CASE WHEN spotter = 'S53M' THEN 1 ELSE 0 END as is_s53m
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
                'spotted_station': 'str',
                'is_s53m': 'int'
            }
        )
        
        # Convert 'timestamp' from UNIX timestamp to datetime with UTC timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce')
    except Exception as e:
        print(f"Error: Unable to read data from the SQLite database. {e}")
        return
    finally:
        conn.close()

    df = delete_old(df, span)

    # Set categorical variables
    df['zone'] = df['zone'].astype(int)
    df['zone'] = df['zone'].astype('category')
    df['zone'] = df['zone'].cat.set_categories(range(1, 41))

    df['band'] = df['band'].astype(str)
    df['band'] = df['band'].astype('category')
    df['band'] = df['band'].cat.set_categories(band_order)

    all_zones = pd.Index(range(1, 41), name='zone')

    # Create separate tables for all spots and S53M spots
    # All spots count table
    count_table_all = df.groupby(['zone', 'band'], observed=True).agg(
        count=('spotter', lambda x: len(set(zip(x, df.loc[x.index, 'spotted_station']))))
    ).reset_index().pivot(
        index='zone',
        columns='band',
        values='count'
    ).reindex(all_zones, fill_value=0)

    # S53M spots count table
    s53m_df = df[df['is_s53m'] == 1]
    count_table_s53m = s53m_df.groupby(['zone', 'band'], observed=True).agg(
        count=('spotter', lambda x: len(set(zip(x, s53m_df.loc[x.index, 'spotted_station']))))
    ).reset_index().pivot(
        index='zone',
        columns='band',
        values='count'
    ).reindex(all_zones, fill_value=0)

    # All spots mean table
    mean_table_all = df.pivot_table(
        values='snr',
        index='zone',
        columns='band',
        aggfunc='median',
        observed=True,
        dropna=False
    ).reindex(all_zones)

    # S53M spots mean table
    mean_table_s53m = s53m_df.pivot_table(
        values='snr',
        index='zone',
        columns='band',
        aggfunc='median',
        observed=True,
        dropna=False
    ).reindex(all_zones)

    # Convert numeric columns to float
    for table in [mean_table_all, mean_table_s53m]:
        numeric_columns = table.select_dtypes(include=['number']).columns
        table[numeric_columns] = table[numeric_columns].astype('float')

    # Ensure all tables have the same index
    all_zones = sorted(set(count_table_all.index) | set(mean_table_all.index))
    count_table_all = count_table_all.reindex(all_zones, fill_value=0)
    count_table_s53m = count_table_s53m.reindex(all_zones, fill_value=0)
    mean_table_all = mean_table_all.reindex(all_zones)
    mean_table_s53m = mean_table_s53m.reindex(all_zones)

    # Reformat the tables
    count_table_all = reformat_table(count_table_all)
    count_table_s53m = reformat_table(count_table_s53m)
    mean_table_all = reformat_table(mean_table_all)
    mean_table_s53m = reformat_table(mean_table_s53m)

    if debug:
        print("Count Table:")
        print(count_table.head())
        print("Mean Table:")
        print(mean_table.head())

    now = dt.datetime.now(dt.timezone.utc).strftime("%b %d, %Y %H:%M")
    caption_string = f"Last {int(span*60)} minutes SNR of spots in S5 and around - refresh at {now} GMT"

    # Initialize combined table from mean_table_all
    combined_table = mean_table_all.copy()
    
    # Tooltip content list
    tooltip_contents = []

    # Create color mapping for each cell
    color_styles = pd.DataFrame('', index=mean_table_all.index, columns=mean_table_all.columns)
    
    # Iterate through bands and create combined display and colors
    for band in band_order:
        if band in mean_table_all.columns:
            combined_results = []
            for idx, row in mean_table_all.iterrows():
                # Get corresponding S53M row
                s53m_row = mean_table_s53m.iloc[idx]
                
                # Combine SNR and count display for both all spots and S53M spots
                result = combine_snr_count(
                    row,
                    s53m_row,
                    count_table_all,
                    count_table_s53m,
                    band,
                    df,
                    s53m_df,
                    idx
                )
                combined_results.append(result)

                # Get counts for color styling
                count = count_table_all.at[idx, band] if band in count_table_all.columns else 0
                try:
                    count = int(count) if count != '' else 0
                    snr = row[band] if band in row else None
                    if not pd.isna(snr) and count > 0:
                        color_styles.at[idx, band] = snr_to_color(snr, count)
                except (ValueError, TypeError):
                    pass
            
            combined_table[band] = [result[0] for result in combined_results]
            tooltip_contents.extend([content for _, content in combined_results if content])
        else:
            combined_table[band] = ' '

    # Handle 'zone' column separately
    if 'zone' in combined_table.columns and 'zone' in mean_table_all.columns:
        combined_table['zone'] = mean_table_all['zone']
        combined_table['zone_display'] = mean_table_all['zone_display']

    # Ensure the columns are ordered as per band_order
    combined_table = combined_table[['zone_display'] + band_order]
    combined_table = combined_table.rename(columns={'zone_display': 'zone'})
    color_styles = color_styles[band_order]  # Match columns with combined_table

    # Apply the styles to the combined table
    styled_table = combined_table.style.apply(lambda x: color_styles, axis=None).set_caption(caption_string)

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
            ('white-space', 'nowrap'),
            ('text-align', 'center')  # Center align the headers
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

    # Create the band headers with sub-text
    band_headers = {
        'zone': 'Zone',
        '160': '160m\nAll / S53M',
        '80': '80m\nAll / S53M',
        '40': '40m\nAll / S53M',
        '20': '20m\nAll / S53M',
        '15': '15m\nAll / S53M',
        '10': '10m\nAll / S53M'
    }
    
    styled_table.columns = band_headers

    # Convert to HTML
    html_table = styled_table.hide(axis="index").to_html()
    html_table = html_table.replace(
        '<table ',
        '<table style="width: 100%; max-width: 1300px; margin: 0 auto; table-layout: auto;" '
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
