import sys, os

in_directory = "/Users/jpanetta/Downloads/florin_new"
out_directory = "/Users/jpanetta/Research/elastic_rods/examples/florin"
for name in os.listdir(in_directory):
    print(name)
    inpath = in_directory + '/' + name
    outpath = out_directory + '/' + name
    outfile = open(outpath, 'w')
    edges = set()
    for l in open(inpath, 'r'):
        fields = l.strip().split(' ')
        if (fields[0] == 'v'): outfile.write(' '.join(fields) + '\n')
        if (fields[0] == 'f'):
            for i in range(len(fields) - 1):
                edges.add(tuple(sorted([fields[1 + i].split('/')[0], fields[1 + (i + 1) % 4].split('/')[0]])))
    for e in edges:
        outfile.write('l {} {}\n'.format(e[0], e[1]))
