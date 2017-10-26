import pandas as pd
import csv
import pylab as pl

f = open("SBER_101001_171001.txt")
reader = csv.reader(f)
sber = []
for line in reader:
    sber.append(line)
    


sber_df = pd.read_csv("SBER_101001_171001.txt")

# pl.plot(sber_df["<CLOSE>"])
pl.plot(sber_df["<VOL>"])