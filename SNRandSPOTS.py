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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
import html

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
parser.add_argument("--nodered-url", default="http://10.0.10.9:2880/ui/#!/3?socketid=wW6AIvSxAn8ySJyyAAAx",
                    help="URL of the Node-RED dashboard")
args = parser.parse_args()
frequency = args.frequency
sparse = args.lower
busy = args.upper
span = args.range
output_folder = args.output_folder
use_s3 = args.use_s3
debug = args.debug
include_solar_data = args.include_solar_data

band_order = ['160', '80', '40', '20', '15', '10']

# Your existing zone_name_map dictionary remains unchanged
zone_name_map = {
    # ... [Keep your existing zone_name_map dictionary exactly as it is] ...
}

def get_aws_credentials():
    """Existing get_aws_credentials function"""
    access_key = input("Enter your AWS Access Key ID: ")
    secret_key = input("Enter your AWS Secret Access Key: ")
    bucket = input("Enter the name of the S3 Bucket you'd like to write to: ")
    return {
        'aws_access_key_id': access_key,
        'aws_secret_access_key': secret_key,
        's3_bucket': bucket
    }

def upload_file_to_s3(file_name, bucket_name, acc_key, sec_key):
    """Existing upload_file_to_s3 function"""
    # ... [Keep your existing upload_file_to_s3 function exactly as it is] ...

# Keep all your existing helper functions unchanged:
def reformat_table(table):
    """Existing reformat_table function"""
    # ... [Keep your existing reformat_table function exactly as it is] ...

def delete_old(df, time_hours):
    """Existing delete_old function"""
    # ... [Keep your existing delete_old function exactly as it is] ...

def slope_to_unicode(slope):
    """Existing slope_to_unicode function"""
    # ... [Keep your existing slope_to_unicode function exactly as it is] ...

def compute_slope(df, zone, band):
    """Existing compute_slope function"""
    # ... [Keep your existing compute_slope function exactly as it is] ...

def combine_snr_count(snr, count, band, zone, df, row_index):
    """Existing combine_snr_count function"""
    # ... [Keep your existing combine_snr_count function exactly as it is] ...

def create_custom_colormap():
    """Existing create_custom_colormap function"""
    # ... [Keep your existing create_custom_colormap function exactly as it is] ...

def generate_html_template(snr_table_html, solar_table_html, tooltip_content_html, caption_string, nodered_url):
    """
    Generates the combined HTML template with SNR data and Node-RED dashboard.
    """
    template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="60">
    <title>Radio Propagation and Contest Dashboard</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
            background: #f5f5f5;
            overflow-x: hidden;
        }}

        .main-container {{
            display: flex;
            padding: 20px;
            gap: 20px;
            min-height: calc(100vh - 40px);
        }}

        .data-panel {{
            flex: 3;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            position: relative;
            min-width: 800px;
            overflow-x: auto;
        }}

        .dashboard-panel {{
            flex: 2;
            background: #1a202c;
            border-radius: 8px;
            overflow: hidden;
            min-width: 400px;
            height: calc(100vh - 40px);
        }}

        .snr-table {{
            margin: 20px auto;
            max-width: 100%;
            overflow-x: auto;
        }}

        .solar-data {{
            position: fixed;
            left: 20px;
            top: 20px;
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

        #nodered-frame {{
            width: 100%;
            height: 100%;
            border: none;
        }}

        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}

        @media (max-width: 1400px) {{
            .main-container {{
                flex-direction: column;
            }}
            .data-panel, .dashboard-panel {{
                min-width: 100%;
            }}
            .dashboard-panel {{
                height: 800px;
            }}
            .solar-data {{
                position: relative;
                left: auto;
                top: auto;
                margin-bottom: 20px;
            }}
            .legend {{
                position: relative;
                bottom: auto;
                left: auto;
                right: auto;
                margin-top: 20px;
            }}
        }}
    </style>
    <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css">
</head>
<body>
    <div class="main-container">
        <div class="data-panel">
            {solar_table_html if solar_table_html else ''}
            <div class="snr-table">
                {snr_table_html}
            </div>
            <div class="legend">
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 14pt; margin-right: 20px;">SNR Trend:</span>
                        <span style="font-size: 16pt; margin: 0 10px;">⇑</span><span>Strong Increase</span>
                        <span style="font-size: 16pt; margin: 0 10px;">⇗</span><span>Slight Increase</span>
                        <span style="font-size: 16pt; margin: 0 10px;">⇔</span><span>Stable</span>
                        <span style="font-size: 16pt; margin: 0 10px;">⇘</span><span>Slight Decrease</span>
                        <span style="font-size: 16pt; margin: 0 10px;">⇓</span><span>Strong Decrease</span>
                    </div>
                </div>
                <small>Make your own SNR overview: <a href="https://github.com/s53zo/Hamradio_copilot">https://github.com/s53zo/Hamradio_copilot</a></small>
            </div>
        </div>
        <div class="dashboard-panel">
            <iframe id="nodered-frame" src="{nodered_url}"></iframe>
        </div>
    </div>
    {tooltip_content_html}
    <script src="https://unpkg.com/@popperjs/core@2/dist/umd/popper.min.js"></script>
    <script src="https://unpkg.com/tippy.js@6/dist/tippy-bundle.umd.min.js"></script>
    <script>
        function resizeIframe() {{
            const iframe = document.getElementById('nodered-frame');
            const container = document.querySelector('.dashboard-panel');
            iframe.style.height = container.clientHeight + 'px';
        }}

        window.addEventListener('load', resizeIframe);
        window.addEventListener('resize', resizeIframe);

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
    """Your existing run function with modifications for the new template"""
    # ... [Keep all the existing code up until the HTML generation] ...
    
    # When you reach the HTML generation part, replace it with:
    final_html = generate_html_template(
        snr_table_html=html1,
        solar_table_html=solar_table_html if include_solar_data else "",
        tooltip_content_html=tooltip_content_html,
        caption_string=caption_string,
        nodered_url=args.nodered_url
    )

    # Minify the final HTML
    minified_html = htmlmin.minify(final_html, remove_empty_space=True, remove_comments=True)

    # Write the minified HTML to index.html
    with open("index.html", "w", encoding="utf-8") as text_file:
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
        upload_file_to_s3("index.html", s3_buck, access_key, secret_key)

if __name__ == '__main__':
    time_to_wait = frequency * 60

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
