#!/usr/bin/python3

import sys
import time
import cv2

n = 100
total_time = 0
for i in range(n):
    file_id = 1 + (i % (len(sys.argv) - 1))
    filename = sys.argv[file_id]
    start = time.time()
    img = cv2.imread(filename)
    end = time.time()
    total_time += end - start

t = 1000 * total_time / n
print("{:0.3f}ms per image".format(t))
