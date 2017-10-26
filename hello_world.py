import pandas as pd
import csv
import matplotlib.pyplot as pl

f = open("SBER_101001_171001.txt")
reader = csv.reader(f)
sber = []
for line in reader:
    sber.append(line)
    


sber_df = pd.read_csv("SBER_101001_171001.txt")

fig, ax1 = pl.subplots()
ax2 = ax1.twinx()

ax1.plot(sber_df["<CLOSE>"], color="red")
ax2.plot(sber_df["<VOL>"])

pl.show()








