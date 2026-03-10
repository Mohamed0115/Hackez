import glob
import os

labels = glob.glob('labels/train/*.txt')
class_counts = {0: 0, 1: 0, 'other': 0}

for lbl in labels:
    with open(lbl, 'r') as f:
        for line in f:
            c = int(line.split()[0])
            if c in class_counts:
                class_counts[c] += 1
            else:
                class_counts['other'] += 1
                
print(f"Class counts in training labels: {class_counts}")
