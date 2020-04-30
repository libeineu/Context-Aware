import sys

infn = sys.argv[1]
num = int(sys.argv[2])

with open(infn, 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines = [float(x.strip()) for x in lines]

g_min = 0 #min(lines)
g_max = 0.15 #max(lines)+1e-5

interval = (g_max-g_min) / num

print('min:  %.5f'%g_min)
print('max:  %.5f'%g_max)
print('mean: %.5f'%(sum(lines)/len(lines)))

cnts = [0] * num

for g in lines:
    cnts[int((g-g_min)/interval)] += 1

for i, c in enumerate(cnts):
    print("%.5f\t%d\t%.3f"%(g_min+i*interval, c, c/len(lines)))

