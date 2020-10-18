import sys
#from agent import Agent
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

a = None
i = 0
model = keras.models.load_model('./mymodel5')
last_ten = np.zeros((10,7))
for line in sys.stdin:
    if i==0:
        i+=1
        continue
    row = line.split(',')
    row = np.array([float(x.strip()) for x in row])
    '''if not a:
        a = Agent(len(row))

    res = a.step(row)
    print(f"{res[0].name} {res[1]}")
    '''
    #seed(1)
    
    
    if i < 10:
        print(f"HOLD {0}")
        last_ten[i,:] = row[0:7] # this is 'close' stock price 
    else:
        last_ten = np.roll(last_ten, -1)
        last_ten[-1,] = row[0:7]
        normal = last_ten

        minx = np.min(normal,axis=0)
        maxx = np.max(normal,axis=0)
        for j in range(normal.shape[1]):
            normal[:,j] = (normal[:,j]-minx[j])/(maxx[j]-minx[j])

        X_predict = np.array(normal).reshape((1, 10, 7)) 
        
        predictedVal = float(model.predict(X_predict))
        diff = predictedVal-float(row[0])
        percent = diff/float(row[0])
        
        #print(predictedVal)
        print(float(predictedVal))
        print(float(row[0]))
        if percent<-0.1:
            print(f"BUY 0.5")
            #print(f"BUY 0.2")
        elif percent<-0.05:
            negative = -1*percent
            negative = str(negative)
            print(f"BUY 0.2")
            #print(f"BUY 0.2")
        elif percent<0.02:
            print(f"HOLD {0}")
            
        elif percent<0.05:
            print(f"SELL 0.2")
        else:
            print(f"SELL 0.5")

    i += 1
