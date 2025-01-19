#!/bin/bash
# python3 /opt/hamradio-copilot/connect.py -c mqtt -r 499,497,503,504,206,239 & python3 /opt/hamradio-copilot/analyzeSNR.py -o /opt/hamradio-copilot/html/

# Start the first script in the background
/usr/bin/python3 /opt/hamradio-copilot/webSNR.py -o /opt/hamradio-copilot/html/ & pid1=$!

# Start the second script in the background
/usr/bin/python3 /opt/hamradio-copilot/webSNR_S53M.py -o /opt/hamradio-copilot/html/ & pid2=$!

# Wait until at least one of them exits
wait -n $pid1 $pid2

# If you prefer to exit only when *all* scripts exit,
# you can replace the wait command above with:
#
#     while kill -0 "$pid1" 2>/dev/null && kill -0 "$pid2" 2>/dev/null; do
#         sleep 1
#     done
#
# That loop continually checks if both processes are alive.
# It only exits when either one is dead. 
# Typically, you'll want to exit (and let systemd restart) if any script fails.
