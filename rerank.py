import sys

fr= open(sys.argv[1],'r',encoding="utf-8")
fw = open(sys.argv[2],'w',encoding="utf-8")
dict = {}
count = 0
for line in fr.readlines():
    line = line.strip().replace('\n','').split('\t')
    dict[int(line[0])]=line[1]
    count+=1
#print(count)
    
sorted_list = sorted(dict.items(),key=lambda x:x[0])

for item in sorted_list:
    fw.write(item[1]+'\n')
