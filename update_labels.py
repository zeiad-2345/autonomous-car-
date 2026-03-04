import re

with open("src/perception/sign_recognition/live_sign_detector.py", "r") as f:
    content = f.read()

# The model outputs GTSRB class IDs. Based on the signs in the image and GTSRB standard:
# 12: priority
# 13: yield (not in BFMC, but let's add)
# 14: stop
# 15: no entry (blanked circle)
# 17: no entry (red bar)
# 25: crosswalk?
# wait, I should look up standard GTSRB class IDs

# Actually, the user's photo shows:
# 12 is Priority (89%)
# 27 is Crosswalk (pedestrian) (95%)
# 14 is Stop (51%)

# Let's map GTSRB numbers to our labels.
# GTSRB mapping:
# 14 = Stop
# 12 = Priority
# 27 = Crosswalk
# 17 = No Entry
# 38 = Highway entrance / Keep right?
# Let's add them to LABEL_MAP
