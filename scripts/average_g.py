import sys

fn = sys.argv[1]

gs = []
with open(fn, 'r', encoding='utf-8') as f:
    for line in f:
        gs.append([float(x) for x in line.split()])

gsum = []
for l in zip(*gs):
    gsum.append(sum(l)/len(l))

print(gsum)
        
        
