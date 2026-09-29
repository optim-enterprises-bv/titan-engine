#!/usr/bin/env python3
"""Routing trace -> TITAN_TIERED_PROFILE (`layer expert count` lines, decode-step uses only).
usage: profile.py TRACE > profile.txt"""
import collections, sys
counts = collections.Counter()
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) >= 4 and f[1] == "1":
        for e in f[3:]:
            counts[(int(f[0]), int(e))] += 1
for (layer, e), c in sorted(counts.items()):
    print(layer, e, c)
