import sys
from agent import Agent
import csv
import numpy as np

with open('mystery_stock_daily_train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    a = None
    for line in readCSV:#sys.stdin:
        if line[0]=='open':
            continue
        row = line#line.split(',')
        row = np.array([float(x.strip()) for x in row])
        if not a:
            a = Agent(len(row))

        res = a.step(row)
        print(f"{res[0].name} {res[1]}")
