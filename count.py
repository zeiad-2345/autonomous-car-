import re
with open("src/perception/sign_recognition/live_sign_detector.py", "r") as f:
    text = f.read()

labels = """
    # ── Custom GTSRB Numbers ──
    "12": "priority",         # priority road
    "13": "priority",         # yield
    "14": "stop",             # stop
    "15": "no_entry",         # vehicles prohibited (blank circle)
    "17": "no_entry",         # no entry (red bar)
    "27": "crosswalk",        # pedestrian crossing
    "28": "crosswalk",        # children crossing
    "33": "one_way",          # turn right ahead
    "35": "one_way",          # ahead only
    "38": "highway_entrance", # keep right
    "39": "highway_entrance", # keep left
    "40": "roundabout",       # roundabout mandatory
"""

print(text.find("12"))
