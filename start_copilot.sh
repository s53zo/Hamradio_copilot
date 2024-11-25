#!/bin/bash
/usr/bin/python3 /opt/hamradio-copilot/connect.py -c mqtt -r 499,497,503,504,206,239 &
/usr/bin/python3 /opt/hamradio-copilot/analyseSNR.py -o /opt/hamradio-copilot/html/
