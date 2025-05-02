from collections import Counter
import numpy as np
import os

labels = []
for f in os.listdir("label"):
    y = np.load(f"label/{f}")
    labels.extend(y.tolist())
print(Counter(labels))
