import pandas as pd
#1004-1023 blank relation
#56 REC->JON


df = pd.read_excel("/Users/charles/Desktop/CZYFYP/sampleset.xlsx")
data = df.values
print("There are "+(str)(len(data))+" sentences in this set")

count = {"ACQ":0, "SUP":0, "SUB":0, "INV":0, "SUE":0, "CMP":0, "JON":0, "CTL":0, "APP":0, "WIN":0, "BNK":0, "NONE":0, "REC":0}
idx = 1
for line in data:
    print(idx)
    idx = idx + 1
    tag = line[5]
    count[tag] = count[tag]+1

print(count)
