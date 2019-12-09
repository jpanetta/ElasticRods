# Converts the curve OBJ written by rhino (which must consist of degree 1 B-splines)
# into a line OBJ.
import re, sys

if (len(sys.argv) != 3) and ((len(sys.argv) != 4)):
    print("Usage: convert_rhino_profile_obj.py rhino.obj [hole_pts.txt] standard.obj")
    exit(-1)

inpath, holepath, outpath = None, None, None

if (len(sys.argv) == 3):
    inpath, outpath = sys.argv[1:]

if (len(sys.argv) == 4):
    inpath, holepath, outpath = sys.argv[1:]

out_file = open(outpath, 'w')

parsingCurve = False
entries = []
lineElements = []
for line in open(inpath, 'r'):
    if line[0:2] == 'v ':
        out_file.write(line)
    elif line[0:4] == 'deg ':
        if (line.split()[1] != '1'): raise Exception('Only degree 1 curves are supported')
    elif line[0:5] == 'curv ':
        entries = line.strip().split()[3:]
        if entries[-1] == '\\': entries.pop()
        parsingCurve = True
    elif line[0:5] == 'parm ': # Terminate curv index list
        parsingCurve = False
        # Create line elements from the polyline indices stored in `entries`
        lineElements += zip(entries, entries[1:])
    elif parsingCurve:
        entries += line.strip().split()
        if entries[-1] == '\\': entries.pop()

# Read holes from the passed ASCII file.
# This file should have one line for each hole giving the x y z coordinates.
# These holes are then printed as dangling vertices in the output OBJ
if (holepath is not None):
    for line in open(holepath, 'r'):
        pts = line.strip().split()
        if (len(pts) == 2): pts.append('0.0')
        if (len(pts) != 3): raise Exception('Incorrect line format in holes file.')
        print('v {} {} {}'.format(pts[0], pts[1], pts[2]), file=out_file)

for v0, v1 in lineElements:
    print('l {} {}'.format(v0, v1), file=out_file)
