import pandas as pd
import numpy as np
#import get_prices as hist
import backtest as bt
import tensorflow as tf
from tensorflow import keras
from preprocessing import DataProcessing
#import pandas_datareader.data as pdr
#import fix_yahoo_finance as fix
#fix.pdr_override()

# ignore this part
'''
file = pd.read_csv("mystery_stock_daily_train.csv")
i = int(0.9 * len(file))
stock_train = file[0: i]
input_train = []
output_train = []
seq_len = 10
for i in range((len(stock_train)//seq_len)*seq_len - seq_len - 1):
    x = np.array(stock_train.iloc[i: i + seq_len]).reshape(10,157,1)
    y = np.array([stock_train.iloc[i + seq_len + 1, 0]], np.float64)
    minx = np.min(x,axis=0)
    maxx = np.max(x,axis=0)
    for j in range(x.shape[1]):
        x[:,j] = (x[:,j]-minx[j])/(maxx[j]-minx[j])
    if i==0:
        input_train=x
    else:
        input_train = np.dstack((input_train,x))
    output_train.append(y)
X_train = np.array(input_train)
Y_train = np.array(output_train)


# train the model. GO TO preprocessing.py TO CHANGE THE FEATURES! THE FEATURES ARE DETERMINED BY x = np.array(stock_train.iloc[i: i + seq_len, 0]. THIS TAKES THE 0TH COLUMN, WHICH IN THIS CASE IS 0)
process = DataProcessing("mystery_stock_daily_train.csv", 0.9)
process.gen_test(10)
process.gen_train(10)

X_train = np.moveaxis(process.X_train,-1,0) #769,10,7

Y_train = process.Y_train

X_test = np.moveaxis(process.X_test,-1,0)
Y_test = process.Y_test

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(10, 7), return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu, kernel_regularizer='l1'))

model.compile(optimizer="adam", loss="mean_squared_error")

model = keras.models.load_model('./mymodel4')
model.fit(X_train, Y_train, epochs=20)

print(model.evaluate(X_test, Y_test))

model.save('./mymodel5')
'''

model = keras.models.load_model('./mymodel5')

#bt.back_test(model,10,(1, 10, 1))

#data = pdr.get_data_yahoo("AAPL", "2017-12-19", "2018-01-03")
data = pd.read_csv("mystery_stock_daily_train.csv")
data = data.dropna()
stock = data[:,0:7]
stock = stock.dropna()
X_predict = np.array(stock).reshape((25, 10, 7)) / 200

for i in range(np.size(X_predict,0)):
    print(model.predict(np.array(X_predict[i]).reshape(1,10,1))*200)

# If instead of a full backtest, you just want to see how accurate the model is for a particular prediction, run this:
# data = pdr.get_data_yahoo("AAPL", "2017-12-19", "2018-01-03")
# stock = data["Adj Close"]
# X_predict = np.array(stock).reshape((1, 10)) / 200
# print(model.predict(X_predict)*200)
