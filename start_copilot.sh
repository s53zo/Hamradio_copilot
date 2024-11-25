#!/usr/bin/bash
echo Starting connect
python3 /opt/hamradio-copilot/connect.py -c mqtt -r 499,497,503,504,206,239 &

echo Starting analyzeSNR
python3 /opt/hamradio-copilot/analyzeSNR.py -o /opt/hamradio-copilot/html/
