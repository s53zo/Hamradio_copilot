#!/usr/bin/env python3
# connect.py

import re
import socket
import plistlib
from datetime import datetime, timedelta
import sqlite3
import argparse
import os
import select
import time
import logging
import paho.mqtt.client as mqtt
import json

# Regular expression pattern for telnet data
telnet_pattern = re.compile(
    r'DX de (\S+):\s+(\d+\.\d+)\s+(\S+)\s+(FT8|FT4|CW)\s+([+-]?\d+)\s*dB'
)

# SQLite database file name
db_file = 'callsigns.db'

# Initialize the callsign cache
callsign_cache = {}

def setup_database():
    """
    Sets up the SQLite database and creates the necessary table if it doesn't exist.
    Includes the 'spotted_station' field.
    """
    logging.debug("Setting up the database.")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create table to store callsign information if it doesn't already exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS callsigns (
            zone INTEGER,
            band INTEGER,
            snr INTEGER,
            timestamp REAL,
            spotter TEXT,
            spotted_station TEXT
        )
    ''')

    conn.commit()
    logging.debug("Database setup complete.")
    return conn, cursor

def insert_batch(cursor, buffer_list):
    """
    Inserts a batch of data into the SQLite database using executemany.
    Now includes 'spotted_station'.
    """
    logging.debug(f"Inserting batch of {len(buffer_list)} entries into the database.")
    cursor.executemany('''
        INSERT INTO callsigns (zone, band, snr, timestamp, spotter, spotted_station)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', buffer_list)

def delete_old_entries(cursor):
    """
    Deletes entries older than 15 minutes from the SQLite database to keep the data current.
    """
    time_ago = datetime.now().timestamp() - timedelta(minutes=15).total_seconds()
    logging.debug(f"Deleting entries older than timestamp {time_ago}.")
    cursor.execute('DELETE FROM callsigns WHERE timestamp <= ?', (time_ago,))

def search_list(call_sign, cty_list):
    """
    Search through the cty.plist file to find CQZone and CountryCode information for the provided callsign.
    Uses a cache to avoid redundant lookups.
    """
    original_call_sign = call_sign  # Keep the original callsign for reference
    logging.debug(f"Searching for callsign '{call_sign}' in cty_list.")

    # Check if the callsign is in the cache
    if original_call_sign in callsign_cache:
        logging.debug(f"Found '{original_call_sign}' in cache.")
        return callsign_cache[original_call_sign]

    # If not in cache, perform the lookup
    while len(call_sign) >= 1 and call_sign not in cty_list:
        logging.debug(f"Callsign '{call_sign}' not found. Truncating last character.")
        call_sign = call_sign[:-1]
    if len(call_sign) == 0:
        logging.debug(f"No matching callsign found for '{original_call_sign}'.")
        callsign_cache[original_call_sign] = None  # Cache the negative result
        return None
    else:
        info = cty_list[call_sign]
        cq_zone = info.get("CQZone")
        country_code = info.get("ADIF")
        logging.debug(f"Found CQZone '{cq_zone}' and CountryCode '{country_code}' for callsign '{call_sign}'.")
        # Cache the result
        callsign_cache[original_call_sign] = {"CQZone": cq_zone, "CountryCode": country_code}
        return callsign_cache[original_call_sign]

def calculate_band(freq, is_telnet=True):
    """
    Calculate the radio band category based on the frequency.
    For telnet data, the frequency is in kHz.
    For MQTT data, the frequency is in Hz.
    """
    logging.debug(f"Calculating band for frequency '{freq}' with is_telnet={is_telnet}.")

    if is_telnet:
        freq_khz = float(freq)  # Frequency is in kHz
    else:
        freq_hz = float(freq)
        freq_khz = freq_hz / 1000.0  # Convert Hz to kHz

    if 1800 <= freq_khz <= 2000:
        return 160
    elif 3500 <= freq_khz <= 4000:
        return 80
    elif 7000 <= freq_khz <= 7300:
        return 40
    elif 10100 <= freq_khz <= 10150:
        return 30
    elif 14000 <= freq_khz <= 14350:
        return 20
    elif 18068 <= freq_khz <= 18168:
        return 17
    elif 21000 <= freq_khz <= 21450:
        return 15
    elif 24890 <= freq_khz <= 24990:
        return 12
    elif 28000 <= freq_khz <= 29700:
        return 10
    elif 50000 <= freq_khz <= 54000:
        return 6
    else:
        logging.debug(f"Frequency '{freq_khz}' does not match any known band.")
        return None

def login_to_server(s, login_callsign):
    """
    Handles the login process with the DX Cluster server.
    """
    logging.debug("Starting login process.")
    data = ''
    s.setblocking(1)  # Temporarily set socket to blocking mode
    while True:
        try:
            chunk = s.recv(1024).decode('utf-8', errors='ignore')
            data += chunk
            logging.debug(f"Received during login: {chunk}")
            if any(prompt in data.lower() for prompt in ['login:', 'call:', 'please enter your call:']):
                break
        except socket.error as e:
            logging.error(f"Socket error during login: {e}")
            time.sleep(0.1)
            continue
    # Send the callsign
    s.sendall((login_callsign + '\n').encode('utf-8'))
    logging.debug(f"Sent login callsign: {login_callsign}")
    s.setblocking(0)  # Set socket back to non-blocking mode

def reconnect(host, port, login_callsign, max_retries=10):
    """
    Attempt to reconnect to the DX Cluster server with an exponential backoff strategy.
    """
    logging.debug(f"Attempting to reconnect to {host}:{port} with max retries {max_retries}.")
    retries = 0
    backoff_time = 5  # Start with 5 seconds of wait time, then double for each retry

    while retries < max_retries:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))  # Try reconnecting
            logging.debug(f"Successfully connected to {host}:{port}")
            login_to_server(s, login_callsign)  # Handle the login process
            s.setblocking(0)  # Set socket to non-blocking mode
            logging.debug(f"Successfully logged in as {login_callsign}")
            print(f"Connected and logged in to {host}:{port} as {login_callsign}")
            return s
        except (socket.error, socket.timeout) as e:
            retries += 1
            logging.debug(f"Reconnection attempt {retries} failed. Error: {e}")
            print(f"Reconnection attempt {retries} failed. Error: {e}")

            if retries >= max_retries:
                logging.error("Max retries reached. Exiting.")
                print("Max retries reached. Exiting.")
                raise Exception("Unable to reconnect after multiple attempts.")

            logging.debug(f"Retrying in {backoff_time} seconds...")
            print(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff: double the wait time

    return None  # In case the loop exits without success

def run(connection_type, receiver_countries, mqtt_host, mqtt_port, host, port, login_callsign):
    """
    Main function to connect to the DX Cluster or MQTT broker, receive and process data, and store it in the SQLite database.
    Handles data processing and resource cleanup.
    """
    last_update_time = datetime.now().timestamp()
    logging.debug(f"Starting main run function with connection_type={connection_type}, receiver_countries={receiver_countries}")

    # Load the cty.plist file with callsign information
    try:
        with open("cty.plist", 'rb') as infile:
            cty_list = plistlib.load(infile, dict_type=dict)
            logging.debug("Loaded cty.plist successfully.")
    except FileNotFoundError:
        logging.error("Error: cty.plist not found.")
        print(f"Error: cty.plist not found.")
        return

    # Set up the SQLite database
    conn, cursor = setup_database()

    buffer_entry_count = 0  # Track how many valid entries are processed between updates
    buffer_list = []  # List to hold data entries before inserting into the database

    if connection_type == 'telnet':
        # Establish the initial socket connection
        s = reconnect(host, port, login_callsign)

        buffer = ""  # Buffer to store incoming data

        try:
            while True:
                now = datetime.now()  # Cache the current time
                ready_to_read, _, _ = select.select([s], [], [], 1)  # Wait up to 1 second for data

                if ready_to_read:
                    try:
                        data = s.recv(1024).decode('utf-8', errors='ignore')  # Non-blocking read, only if data is available
                        logging.debug(f"Received data: {data}")

                        if not data:
                            logging.debug("Connection closed by server.")
                            print("Connection closed by server.")
                            s = reconnect(host, port, login_callsign)  # Reconnect if the connection is closed
                            continue

                        buffer += data  # Append the received data to the buffer

                    except UnicodeDecodeError as e:
                        logging.error(f"Decoding error: {e}")
                        print(f"Decoding error: {e}")
                        continue
                    except socket.error as e:
                        logging.error(f"Socket error: {e}. Reconnecting...")
                        print(f"Socket error: {e}. Reconnecting...")
                        s = reconnect(host, port, login_callsign)
                        continue

                    # Split the buffer by newlines; the last part may be incomplete
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Save the incomplete line back to the buffer
                    logging.debug(f"Processing {len(lines)-1} complete lines from buffer.")

                    for line in lines[:-1]:  # Process all complete lines
                        line = line.strip()
                        logging.debug(f"Processing line: {line}")

                        match = telnet_pattern.search(line)  # Use the telnet regex

                        if match:
                            # Mapping receiver call to spotter and sender call to spotted station
                            spotter_callsign = match.group(1).rstrip('-#')  # Spotter (receiver call)
                            frequency = match.group(2)
                            call_sign = match.group(3)  # Spotted station (sender call)
                            mode = match.group(4)
                            snr = match.group(5).replace(" ", "")
                            timestamp = now.timestamp()

                            logging.debug(f"Matched data - Spotter: {spotter_callsign}, Frequency: {frequency}, Spotted Station: {call_sign}, Mode: {mode}, SNR: {snr}")

                            # Check if the spotter's country code matches any of the specified receiver countries
                            spotter_result = search_list(spotter_callsign, cty_list)
                            if spotter_result and str(spotter_result["CountryCode"]) in receiver_countries:
                                # Proceed to process the spot
                                # Get the CQZone of the spotted callsign
                                spotted_result = search_list(call_sign, cty_list)
                                if spotted_result:
                                    cq_zone = spotted_result["CQZone"]
                                    spotted_station = call_sign  # The spotted station is the call_sign
                                else:
                                    cq_zone = None
                                    spotted_station = "Unknown"

                                band = calculate_band(frequency, is_telnet=True)
                                if band and cq_zone and snr:
                                    # Add the enhanced callsign info to the buffer list
                                    buffer_list.append((cq_zone, band, snr, timestamp, spotter_callsign, spotted_station))
                                    logging.debug(f"Added entry to buffer list: {(cq_zone, band, snr, timestamp, spotter_callsign, spotted_station)}")
                                    buffer_entry_count += 1  # Increment the count of valid entries processed
                                else:
                                    logging.debug("Invalid data encountered. Skipping entry.")
                            else:
                                logging.debug(f"Spotter '{spotter_callsign}' country code does not match receiver countries '{receiver_countries}'. Skipping entry.")
                        else:
                            logging.debug("No regex match found for line. Skipping entry.")

                # Every 500 lines or 30 seconds, update the database with the new info
                if ((buffer_entry_count >= 500) or (now.timestamp() - last_update_time > 30)) and buffer_list:
                    logging.debug(f"Updating database with {len(buffer_list)} entries.")
                    insert_batch(cursor, buffer_list)
                    delete_old_entries(cursor)  # Keep the database size manageable
                    conn.commit()  # Commit the changes

                    last_update_time = now.timestamp()

                    # Get the current time and print it along with the update message
                    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Database updated on {current_time}. Processed {buffer_entry_count} total entries from the buffer.")

                    # Reset buffer entry count and list after each update
                    buffer_entry_count = 0
                    buffer_list = []
        except KeyboardInterrupt:
            # Handle clean exit on Ctrl+C
            logging.info("KeyboardInterrupt received. Exiting gracefully...")
            print("Exiting... Please wait while resources are being cleaned up.")
        finally:
            # Clean up resources
            try:
                if s:
                    s.close()
                    logging.debug("Socket connection closed.")
            except Exception as e:
                logging.error(f"Error closing socket: {e}")

            try:
                if conn:
                    conn.commit()
                    conn.close()
                    logging.debug("Database connection closed.")
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")

            logging.info("Cleanup complete. Goodbye!")
    elif connection_type == 'mqtt':
        # MQTT Connection and processing
        logging.debug(f"Connecting to MQTT broker at {mqtt_host}:{mqtt_port} and subscribing to relevant topics.")

        #client = mqtt.Client()
        import uuid
        client_id = f"mqtt_client_{uuid.uuid4()}"
        client = mqtt.Client(client_id=client_id)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.debug("Connected to MQTT broker successfully.")
                print("Connected to MQTT broker successfully.")
                # Subscribe to topics for the specified receiver countries
                for country_code in receiver_countries:
                    topic = f"pskr/filter/v2/+/+/+/+/+/+/+/{country_code}"
                    logging.debug(f"Subscribing to topic '{topic}'")
                    client.subscribe(topic)
            else:
                logging.error(f"Failed to connect to MQTT broker. Return code {rc}")

        def on_message(client, userdata, msg):
            nonlocal buffer_entry_count, buffer_list, last_update_time
            data = msg.payload.decode('utf-8')
            logging.debug(f"Received MQTT message on topic '{msg.topic}': {data}")

            # Parse the JSON payload
            try:
                spot = json.loads(data)
                logging.debug(f"Parsed JSON data: {spot}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error: {e}")
                return

            # Extract relevant fields
            spotter_callsign = spot.get('rc')  # Receiver call is the spotter
            frequency = spot.get('f')
            call_sign = spot.get('sc')  # Sender call is the spotted station
            mode = spot.get('md')
            snr = spot.get('rp')
            timestamp = spot.get('t')
            receiver_country_code = spot.get('ra')  # Receiver country code

            if str(receiver_country_code) not in receiver_countries:
                logging.debug(f"Receiver country code '{receiver_country_code}' does not match the specified countries '{receiver_countries}'. Skipping entry.")
                return

            logging.debug(f"Processing spot: Spotter={spotter_callsign}, Frequency={frequency}, Spotted Station={call_sign}, Mode={mode}, SNR={snr}")

            # Check if the spotter exists in cty_list
            spotter_result = search_list(spotter_callsign, cty_list)
            if spotter_result:
                # Proceed to process the spot
                # Get the CQZone of the spotted callsign
                spotted_result = search_list(call_sign, cty_list)
                if spotted_result:
                    cq_zone = spotted_result["CQZone"]
                    spotted_station = call_sign  # The spotted station is the call_sign
                else:
                    cq_zone = None
                    spotted_station = "Unknown"

                band = calculate_band(frequency, is_telnet=False)
                if band and cq_zone and snr is not None:
                    # Add the enhanced callsign info to the buffer list
                    buffer_list.append((cq_zone, band, snr, timestamp, spotter_callsign, spotted_station))
                    logging.debug(f"Added entry to buffer list: {(cq_zone, band, snr, timestamp, spotter_callsign, spotted_station)}")
                    buffer_entry_count += 1  # Increment the count of valid entries processed
                else:
                    logging.debug("Invalid data encountered. Skipping entry.")
            else:
                logging.debug(f"Spotter '{spotter_callsign}' not found in cty_list. Skipping entry.")

            # Every 500 lines or 30 seconds, update the database with the new info
            now = datetime.now()
            if ((buffer_entry_count >= 500) or (now.timestamp() - last_update_time > 30)) and buffer_list:
                logging.debug(f"Updating database with {len(buffer_list)} entries.")
                insert_batch(cursor, buffer_list)
                delete_old_entries(cursor)  # Keep the database size manageable
                conn.commit()  # Commit the changes

                last_update_time = now.timestamp()

                # Get the current time and print it along with the update message
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Database updated on {current_time}. Processed {buffer_entry_count} total entries from the buffer.")

                # Reset buffer entry count and list after each update
                buffer_entry_count = 0
                buffer_list = []

        client.on_connect = on_connect
        client.on_message = on_message

        try:
            client.connect(mqtt_host, mqtt_port, 60)
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            print(f"Failed to connect to MQTT broker: {e}")
            return

        # Start the loop
        try:
            client.loop_forever()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Exiting gracefully...")
            print("Exiting... Please wait while resources are being cleaned up.")
        finally:
            # Clean up resources
            try:
                client.disconnect()
                logging.debug("Disconnected from MQTT broker.")
            except Exception as e:
                logging.error(f"Error disconnecting MQTT client: {e}")

            try:
                if conn:
                    conn.commit()
                    conn.close()
                    logging.debug("Database connection closed.")
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")

            logging.info("Cleanup complete. Goodbye!")
    else:
        logging.error(f"Unknown connection type '{connection_type}'.")
        print(f"Unknown connection type '{connection_type}'.")

if __name__ == '__main__':
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Connect to a DX Cluster or MQTT broker, collect spotted callsigns, and store them in an SQLite database.")
    parser.add_argument("-c", "--connection-type", choices=['telnet', 'mqtt'], default='telnet', help="Choose the connection type: telnet or mqtt")
    parser.add_argument("-r", "--receiver-countries", help="Specify the receiver country codes (ADIF codes) to track, comma-separated", required=True)
    parser.add_argument("-d", "--debug", help="Enable debug output", action='store_true')

    # Telnet-specific arguments
    parser.add_argument("-a", "--address", help="Specify hostname/address of the DX Cluster", default=os.getenv("DX_CLUSTER_HOST", "telnet.reversebeacon.net"))
    parser.add_argument("-p", "--port", help="Specify port for the DX Cluster", type=int, default=int(os.getenv("DX_CLUSTER_PORT", 7001)))
    parser.add_argument("-l", "--login-callsign", help="Specify the callsign to send during login (for telnet)", default="")

    # MQTT-specific arguments
    parser.add_argument("--mqtt-host", help="Specify hostname/address of the MQTT broker", default="mqtt.pskreporter.info")
    parser.add_argument("--mqtt-port", help="Specify port for the MQTT broker", type=int, default=1883)

    args = parser.parse_args()

    # Set up logging configuration based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Process receiver countries input
    receiver_countries = [code.strip() for code in args.receiver_countries.split(',')]
    logging.debug(f"Receiver countries to track: {receiver_countries}")

    # Run the main function with provided arguments
    run(args.connection_type, receiver_countries, args.mqtt_host, args.mqtt_port, args.address, args.port, args.login_callsign)
