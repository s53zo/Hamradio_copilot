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
import numpy as np  # Needed for numeric operations
import html  # For escaping HTML content
from matplotlib.colors import LinearSegmentedColormap  # For custom colormap
from scipy.stats import linregress  # For linear regression

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
include_solar_data = args.include_solar_data  # Get the include_solar_data flag

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
    """
    Prompts the user to input their AWS credentials and the bucket they'd like to upload to.

    :return: A dictionary containing 'aws_access_key_id', 'aws_secret_access_key', and 's3_bucket'.
    """
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

    :param file_name: The name of the html file being uploaded.
    :param bucket_name: The name of the bucket being uploaded to.
    :param acc_key: The AWS access key for S3 access.
    :param sec_key: The secret access key for the AWS access key.
    :return: Boolean True if the file was uploaded successfully. False if not uploaded successfully.
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
    Reformats the pivot table into a DataFrame convenient for HTML display.
    """
    # Flatten the pivot table
    flattened = table.reset_index()
    # Ensure 'zone' is numeric
    flattened['zone'] = pd.to_numeric(flattened['zone'], errors='coerce')
    # Sort by 'zone'
    flattened = flattened.sort_values(by='zone')
    # Create 'zone_display' with HTML formatting
    flattened['zone_display'] = flattened['zone'].apply(lambda x: f'<span title="{zone_name_map.get(x, "")}">{str(int(x)).zfill(2)}</span>')
    flattened.reset_index(drop=True, inplace=True)
    # Rearrange columns
    columns_order = ['zone', 'zone_display'] + band_order
    existing_columns = [col for col in columns_order if col in flattened.columns]
    flattened = flattened[existing_columns]
    # Fill NaN with appropriate values
    flattened = flattened.fillna({' ': ' '})
    return flattened

def delete_old(df, time_hours):
    """
    Deletes entries older than the specified time range from the dataframe.

    :param df: The dataframe being modified.
    :param time_hours: The range of age before deletion in hours.
    :return: The dataframe without the older entries.
    """
    # 'timestamp' is already converted to datetime with UTC timezone
    day_ago = datetime.now(dt.timezone.utc) - timedelta(hours=time_hours)

    # Filter out old entries
    df = df[df['timestamp'] >= day_ago].reset_index(drop=True)
    return df

def slope_to_unicode(slope):
    """
    Converts a slope value to a corresponding Unicode character based on the specified ranges.

    :param slope: The slope value (float) to convert.
    :return: A Unicode character representing the direction of the slope.
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
        return ''  # In case of NaN or other unexpected values

def compute_slope(df, zone, band):
    """
    Computes the slope of SNR over time for a given zone and band.
    :param df: The dataframe containing SNR data.
    :param zone: The zone to filter on.
    :param band: The band to filter on.
    :return: The slope of SNR over time.
    """
    # Filter DataFrame for the given zone and band
    relevant_spots = df[(df['zone'] == zone) & (df['band'] == band)]

    if len(relevant_spots) < 2:
        return np.nan  # Not enough data points to compute a slope

    # Convert timestamps to numerical values (e.g., Unix timestamp in minutes)
    relevant_spots['minute'] = relevant_spots['timestamp'].dt.floor('min')  # Round to nearest minute
    avg_per_minute = relevant_spots.groupby('minute')['snr'].mean().reset_index()

    if len(avg_per_minute) < 2:
        return np.nan

    time_values = avg_per_minute['minute'].astype(np.int64) // 1e9 / 60  # Convert to minutes
    snr_values = avg_per_minute['snr']

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(time_values, snr_values)

    return slope
        
def combine_snr_count(snr, count, band, zone, df, row_index):
    if pd.isna(snr) and count == 0:
        return "", None
    elif pd.isna(snr):
        display_text = f'N/A ({count})'
    else:
        display_text = f'{int(round(snr))} ({count})'

    # Compute the slope for this zone and band
    slope = compute_slope(df, zone, band)
    # Convert the slope to a Unicode arrow
    slope_arrow = slope_to_unicode(slope)

    # Determine the arrow color based on the same thresholds as in slope_to_unicode
    if pd.isna(slope) or (-0.1 <= slope <= 0.1):
        arrow_color = 'black'  # Stable or undefined slope
    elif 0.1 < slope <= 0.3:
        arrow_color = 'green'  # Slight positive slope
    elif slope > 0.3:
        arrow_color = 'green'  # Strong positive slope
    elif -0.3 <= slope < -0.1:
        arrow_color = 'red'    # Slight negative slope
    elif slope < -0.3:
        arrow_color = 'red'    # Strong negative slope
    else:
        arrow_color = 'black'  # Default color


    # Style the arrow: increase font size and apply color
    styled_arrow = f'<span style="font-size: 16pt; color: {arrow_color};">{slope_arrow}</span>'

    # Append the styled arrow to the display text
    display_text_with_arrow = f'{display_text} {styled_arrow}'

    # Filter DataFrame for the given zone and band
    relevant_spots = df[(df['zone'] == zone) & (df['band'] == band)]

    # Get unique spotted stations
    unique_stations = sorted(set(relevant_spots['spotted_station']))

    # Generate a unique ID for the tooltip content
    tooltip_id = f"tooltip_content_{row_index}_{band}"

    # Split the list into chunks of 10
    chunk_size = 10
    chunks = [unique_stations[i:i + chunk_size] for i in range(0, len(unique_stations), chunk_size)]

    # Create tooltip content HTML with multiple columns
    tooltip_content_html = "<div style='display: flex;'>"
    for chunk in chunks:
        tooltip_content_html += "<ul style='margin: 0 10px; padding: 0; list-style: none;'>"
        for station in chunk:
            escaped_station = html.escape(station)
            tooltip_content_html += f"<li>{escaped_station}</li>"
        tooltip_content_html += "</ul>"
    tooltip_content_html += "</div>"

    # HTML with data-tooltip-content attribute
    cell_html = f'''
        <span class="tooltip" data-tooltip-content="#{tooltip_id}">{display_text_with_arrow}</span>
    '''

    # Return the cell HTML and the tooltip content
    return cell_html, (tooltip_id, tooltip_content_html)
    
def create_custom_colormap():
    # Define the colors and positions according to the percentage mapping
    colors = [
        (0.00, "#ffffff"),  # 0%
        (0.10, "#bff0ff"),  # 10% 
        (0.20, "#6ed6fd"),  # 20% 
        (0.30, "#02bfff"),  # 30% 
        (0.40, "#009aff"),  # 40% 
        (0.50, "#1ebe3e"),  # 50% 
        (0.60, "#bfff02"),  # 60% 
        (0.70, "#fdff00"),  # 70% 
        (0.80, "#ffcd2e"),  # 80%
        (0.90, "#ff7603"),  # 90% 
        (1.00, "#ff0002"),  # 100%
    ]

    positions = [pos for pos, color in colors]
    color_codes = [color for pos, color in colors]

    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, color_codes)))
    return cmap

def run(access_key=None, secret_key=None, s3_buck=None, include_solar_data=False):
    # Connect to the SQLite database
    conn = sqlite3.connect('callsigns.db')
    
    # Read data from the SQLite table `callsigns` into a pandas DataFrame
    query = """
    SELECT zone, band, snr, timestamp, spotter, spotted_station
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
        conn.close()  # Close the database connection after reading the data

    if debug:
        print("Initial DataFrame:")
        print(df.head())

    df = delete_old(df, span)  # Ignore any data older than the specified range from the database.

    if debug:
        print(f"DataFrame after deleting entries older than {span} hours:")
        print(df.head())

    # Set 'zone' as a categorical variable with categories from 1 to 40
    df['zone'] = df['zone'].astype(int)
    df['zone'] = df['zone'].astype('category')
    df['zone'] = df['zone'].cat.set_categories(range(1, 41))

    # Set 'band' as a categorical variable with desired order
    df['band'] = df['band'].astype(str)  # Convert 'band' to string if necessary
    df['band'] = df['band'].astype('category')
    df['band'] = df['band'].cat.set_categories(band_order)

    # Generate the pivot tables for the aggregated data
    count_table = df.pivot_table(
        values='snr',
        index='zone',
        columns='band',
        aggfunc='count',
        fill_value=0,
        dropna=False
    )
    mean_table = df.pivot_table(
        values='snr',
        index='zone',
        columns='band',
        aggfunc='median',  # Median calculation
        dropna=False
    )

    # Reformat the tables
    count_table = reformat_table(count_table)
    mean_table = reformat_table(mean_table)

    if debug:
        print("Count Table:")
        print(count_table.head())
        print("Mean Table:")
        print(mean_table.head())

    now = dt.datetime.now(dt.timezone.utc).strftime("%b %d, %Y %H:%M")
    caption_string = f"Last {int(span*60)} minutes spots in ITU Zone 28 overview - refresh at {now} GMT"

    # Compute snr_min and snr_max using 10th and 90th percentiles
    if not df['snr'].empty:
        snr_min = df['snr'].quantile(0.1)
        snr_max = df['snr'].quantile(0.9)
    else:
        snr_min = -20  # Default values if 'snr' is empty
        snr_max = 0

    if snr_min == snr_max:
        snr_min -= 10  # Adjust to avoid division by zero
        snr_max += 10

    # Create custom colormap
    custom_cmap = create_custom_colormap()

    # Define the function to map SNR values to colors
    def snr_to_color(val):
        if pd.isna(val) or val == ' ':
            return 'background-color: #ffffff'  # Set empty cells to white
        else:
            try:
                val = float(val)
            except ValueError:
                return 'background-color: #ffffff'  # Also set invalid values to white
            # Clip the SNR value between snr_min and snr_max
            val_clipped = max(min(val, snr_max), snr_min)
            # Normalize between 0 and 1
            norm_val = (val_clipped - snr_min) / (snr_max - snr_min)
            # Get color from custom_cmap
            rgba_color = custom_cmap(norm_val)
            # Convert RGBA to hex
            rgb_color = tuple(int(255 * x) for x in rgba_color[:3])
            hex_color = '#%02x%02x%02x' % rgb_color
            return f'background-color: {hex_color}'

    # Apply color map to mean_table (excluding 'zone' and 'zone_display' columns)
    means_no_zone = mean_table.drop(columns=['zone', 'zone_display'], errors='ignore')

    # Convert all values to float
    means_no_zone = means_no_zone.apply(pd.to_numeric, errors='coerce')

    color_table1 = means_no_zone.applymap(snr_to_color)

    # Combine mean_table and count_table into a single table with desired cell content
    combined_table = mean_table.copy()

    # Tooltip content list
    tooltip_contents = []

    # Iterate over each band to combine mean and count using the helper function
    for band in band_order:
        if band in mean_table.columns and band in count_table.columns:
            combined_results = mean_table.apply(
                lambda row: combine_snr_count(
                    row[band],
                    count_table.at[row.name, band],
                    band,
                    row['zone'],  # 'zone' is numeric
                    df,
                    row.name  # Pass the row index to generate unique IDs
                ),
                axis=1
            )
            combined_table[band] = combined_results.apply(lambda x: x[0])
            # Collect tooltip contents
            for result in combined_results:
                content = result[1]
                if content:
                    tooltip_contents.append(content)
        else:
            combined_table[band] = ' '

    # Handle 'zone' column separately (keep as is)
    if 'zone' in combined_table.columns and 'zone' in mean_table.columns:
        combined_table['zone'] = mean_table['zone']
        combined_table['zone_display'] = mean_table['zone_display']

    # Ensure the columns are ordered as per band_order
    combined_table = combined_table[['zone_display'] + band_order]
    # Rename 'zone_display' to 'zone' for display
    combined_table = combined_table.rename(columns={'zone_display': 'zone'})

    # Apply the styles to the combined table.
    styled_table1 = combined_table.style.apply(lambda x: color_table1, axis=None).set_caption(caption_string)

    styled_table1.set_properties(subset=['zone'], **{'font-weight': 'bold'})
    styled_table1.set_properties(**{'text-align': 'center'})

    # Set table styles to different parts of the table.
    styled_table1.set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '13pt'), ('font-weight', 'bold')]},
        {'selector': 'th',
         'props': [('font-size', '12pt'), ('word-wrap', 'break-word'), ('position', 'sticky'),
                   ('top', '0'), ('background-color', 'rgba(255, 255, 255, 0.75)'), ('z-index', '1')]},
        {'selector': 'td:first-child', 'props': [('font-weight', 'bold')]}  # First column
    ])

    # Convert the styled table to HTML.
    html1 = styled_table1.hide(axis="index").to_html()

    html1 = html1.replace('<table ',
                          '<table style="width: 60vw; table-layout: fixed; margin-left: auto; margin-right: auto;" ')

    if debug:
        print("Generated HTML Table:")
        print(html1[:500])  # Print the first 500 characters of the HTML

    # Build the tooltip content HTML
    tooltip_content_html = ''
    for tooltip_id, content_html in tooltip_contents:
        tooltip_content_html += f'''
        <div class="tooltip_templates">
            <div id="{tooltip_id}">
                {content_html}
            </div>
        </div>
        '''

    # Include Tooltip.js from CDN
    tooltip_js_cdn = "https://unpkg.com/@popperjs/core@2/dist/umd/popper.min.js"
    tooltip_js_library = "https://unpkg.com/tippy.js@6/dist/tippy-bundle.umd.min.js"
    tooltip_css_cdn = "https://unpkg.com/tippy.js@6/animations/scale.css"

    # Prepare solar data if included
    if include_solar_data:
        # Fetch solar widget XML data.
        try:
            solar_response = requests.get("https://www.hamqsl.com/solarxml.php")
            xml_data = solar_response.content
            root = ET.fromstring(xml_data)
        except Exception as e:
            print(f"Error fetching solar data: {e}")
            solar_table_html = ""
        else:
            # Extract solar and band condition data
            solar_data = {
                "SFI": root.findtext("solardata/solarflux"),
                "Sunspots": root.findtext("solardata/sunspots"),
                "A-Index": root.findtext("solardata/aindex"),
                "K-Index": root.findtext("solardata/kindex"),
                "X-Ray": root.findtext("solardata/xray"),
                "Signal_Noise": root.findtext("solardata/signalnoise"),
                "Aurora": root.findtext("solardata/aurora"),
                "Lat.": root.findtext("solardata/latdegree"),
            }

            conditions = {
                "80m-40m": {"Day": "", "Night": ""},
                "30m-20m": {"Day": "", "Night": ""},
                "17m-15m": {"Day": "", "Night": ""},
                "12m-10m": {"Day": "", "Night": ""},
            }

            for band in root.findall("solardata/calculatedconditions/band"):
                band_name = band.get("name")
                time_of_day = band.get("time")
                condition = band.text
                if band_name in conditions:
                    conditions[band_name][time_of_day.capitalize()] = condition

            # HTML content for solar data and band conditions
            solar_table_html = f"""
            <div style="width: 100%; text-align: center; font-weight: bold; margin-bottom: 5px;">Solar Data by N0NBH</div>
            <hr>
            <div style="display: flex; justify-content: center; margin-bottom: 5px;">
                <div style="margin-right: 20px;">
                    <span style="font-weight: bold;">SFI:</span> <span style="font-weight: bold;">{solar_data['SFI']}</span>
                </div>
                <div>
                    <span style="font-weight: bold;">SSN:</span> <span style="font-weight: bold;">{solar_data['Sunspots']}</span>
                </div>
            </div>
            <div style="display: inline-flex; justify-content: center; align-items: center; width: 100%; text-align: center; white-space: nowrap;">
                <div style="margin-right: 10px;">
                    <span style="font-weight: bold;">A:</span> <span style="font-weight: bold;">{solar_data['A-Index']}</span>
                </div>
                <div style="margin-right: 10px;">
                    <span style="font-weight: bold;">K:</span> <span style="font-weight: bold;">{solar_data['K-Index']}</span>
                </div>
                <div>
                    <span style="font-weight: bold;">X:</span> <span style="font-weight: bold;">{solar_data['X-Ray']}</span>
                </div>
            </div>

            <div style="display: flex; justify-content: center; align-items: center; white-space: nowrap;">
                <div style="margin-right: 20px;">
                    <span style="font-weight: bold;">Aurora:</span> <span style="font-weight: bold;">{solar_data['Aurora']}</span>
                </div>
                <div>
                    <span style="font-weight: bold;">Lat.:</span> <span style="font-weight: bold;">{solar_data['Lat.']}</span>
                </div>
            </div>

            <hr>
            <div style="width: 100%; text-align: center; font-weight: bold; margin-top: 10px;">Band Conditions</div>
            <table style="width: 60%; margin: 0 auto; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="padding: 5px; text-align: center; font-weight: bold;">Band</th>
                        <th style="padding: 5px; text-align: center; font-weight: bold;">Day</th>
                        <th style="padding: 5px; text-align: center; font-weight: bold;">Night</th>
                    </tr>
                </thead>
                <tbody>
            """

            # Add the conditions for each band with color coding.
            for band, condition in conditions.items():
                day_condition = condition.get("Day", "N/A")
                night_condition = condition.get("Night", "N/A")
                if day_condition == "Good":
                    day_color = "green"
                elif day_condition == "Fair":
                    day_color = "orange"
                else:
                    day_color = "red"
                if night_condition == "Good":
                    night_color = "green"
                elif night_condition == "Fair":
                    night_color = "orange"
                else:
                    night_color = "red"

                solar_table_html += f"""
                <tr>
                    <td style="padding: 5px; text-align: center; font-weight: bold; white-space: nowrap;">{band}</td>
                    <td style="padding: 5px; text-align: center; font-weight: bold; color: {day_color}; white-space: nowrap;">{day_condition}</td>
                    <td style="padding: 5px; text-align: center; font-weight: bold; color: {night_color}; white-space: nowrap;">{night_condition}</td>
                </tr>
                """

            solar_table_html += """
                </tbody>
            </table>
            <div style="margin-top: 5px; text-align: center">
                <span style="font-weight: bold;">Signal Noise:</span> <span style="font-weight: bold;">{Signal_Noise}</span>
            </div>
            """

            # Formatting the placeholders directly in `solar_table_html`
            solar_table_html = solar_table_html.format(
                SFI=solar_data['SFI'],
                Sunspots=solar_data['Sunspots'],
                A_Index=solar_data['A-Index'],
                K_Index=solar_data['K-Index'],
                X_Ray=solar_data['X-Ray'],
                Signal_Noise=solar_data['Signal_Noise'],
                Aurora=solar_data['Aurora'],
                Lat=solar_data['Lat.']
            )

            # Wrap the table in a fixed div.
            solar_table_html = f"""
            <div style="position: fixed; left: 5%; padding-top: 0.75%; padding: 10px; z-index: 1000; font-family: 'Roboto', monospace;">
                {solar_table_html}
            </div>
            """
    else:
        solar_table_html = ""

    # Legend HTML block
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 20px; 
        left: 50%; 
        transform: translateX(-50%); 
        width: 80%; 
        background-color: rgba(255, 255, 255, 0.95); 
        font-weight: bold; 
        padding: 15px; 
        border: 1px solid #ccc; 
        border-radius: 8px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        z-index: 1000;
    ">
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <!-- SNR Levels Section -->
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 14pt; margin-right: 20px;">SNR Levels:</div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <div style="width: 20px; height: 20px; background-color: #a3cce9; margin-right: 5px;"></div>
                    <span style="font-size: 12pt;">Very Weak (≤ -15 dB)</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <div style="width: 20px; height: 20px; background-color: #b6e3b5; margin-right: 5px;"></div>
                    <span style="font-size: 12pt;">Weak (-15 to -10 dB)</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <div style="width: 20px; height: 20px; background-color: #f7c896; margin-right: 5px;"></div>
                    <span style="font-size: 12pt;">Moderate (-10 to -3 dB)</span>
                </div>
                
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #e57373; margin-right: 5px;"></div>
                    <span style="font-size: 12pt;">Strong (≥ -3 dB)</span>
                </div>
            </div>
            <!-- Slope Symbols Section -->
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 14pt; margin-right: 20px;">SNR Trend:</div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <span style="font-size: 16pt;">⇑</span>
                    <span style="font-size: 12pt; margin-left: 5px;">Strong Increase</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <span style="font-size: 16pt;">⇗</span>
                    <span style="font-size: 12pt; margin-left: 5px;">Slight Increase</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <span style="font-size: 16pt;">⇔</span>
                    <span style="font-size: 12pt; margin-left: 5px;">Stable</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-right: 15px;">
                    <span style="font-size: 16pt;">⇘</span>
                    <span style="font-size: 12pt; margin-left: 5px;">Slight Decrease</span>
                </div>
                
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 16pt;">⇓</span>
                    <span style="font-size: 12pt; margin-left: 5px;">Strong Decrease</span>
                </div>
            </div>
        </div>
    </div>
    """

    # Adjust the left padding depending on whether solar data is included
    left_padding = "120px" if include_solar_data else "20px"

    # Simplified CSS styles for the tooltip
    final_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="{int(frequency * 60)}">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow-y: hidden; /* Prevent vertical scrolling */
            }}
            html {{
                height: 100%;
            }}
            .tooltip {{
                cursor: pointer;
            }}
            .tippy-content {{
                font-size: 12pt;
            }}
            /* Hide the tooltip content divs */
            .tooltip_templates {{
                display: none;
            }}
        </style>
        <!-- Include Tooltip.js CSS -->
        <link rel="stylesheet" href="{tooltip_css_cdn}">
    </head>
    <body>
        <div style="display: flex; width: 100%; height: 100%;">
            {solar_table_html}
            <div style="position: relative; flex-grow: 1; padding-left: {left_padding}; overflow-y: auto; font-family: 'Roboto', monospace;">
                <div style="max-height: 80vh; overflow-y: auto; padding-top: 0.75%; padding-bottom: 10%;">
                    {html1}
                    <!-- Tooltip content is included here and will be hidden -->
                    {tooltip_content_html}
                </div>
                <div>{legend_html}</div>
            </div>
        </div>
        <!-- Include Tooltip.js JS -->
        <script src="{tooltip_js_cdn}"></script>
        <script src="{tooltip_js_library}"></script>
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

    # Minify the final_html before writing it to the file
    minified_html = htmlmin.minify(final_html, remove_empty_space=True, remove_comments=True)

    # Write the minified HTML to index.html
    with open("index.html", "w", encoding="utf-8") as text_file:  # Write minified HTML data to index.html file.
        text_file.write(minified_html)

    # Save a copy to the specified local folder
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
        upload_file_to_s3("index.html", s3_buck, access_key, secret_key)  # Upload index.html to S3 bucket

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
        
