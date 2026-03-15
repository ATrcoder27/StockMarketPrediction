import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import itertools

ticker = yf.Ticker(input("Ticker: "))
ticker_df = ticker.history(period="5y")
ticker_df.drop(['High', 'Low', 'Volume'], axis=1, inplace=True)
open = np.empty(shape = (1259), dtype=float)
close = np.empty(shape=(1259), dtype=float)

open = ticker_df[['Open']].to_numpy()
close = ticker_df[['Close']].to_numpy()

X = np.zeros((1260, 6))
Y = [0]*1260

for i in range(1200):
    X[i] = [open[i][0], close[i][0], open[i+1][0], close[i+1][0], open[i+2][0], close[i+2][0]]
    Y[i] = open[i+3][0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

baseline_value = np.mean(Y_test)
baseline_pred = np.full(len(Y_test), baseline_value)
baselineMSE = mean_squared_error(Y_test, baseline_pred)
print("Baseline MSE :", baselineMSE)

lin = linear_model.LinearRegression()
lin.fit(X_train, Y_train)
linPred = lin.predict(X_test)
linMSE = mean_squared_error(Y_test, linPred)
linMAE = mean_absolute_error(Y_test, linPred)
print("Linear MSE: ", linMSE)
print("Linear MAE: ", linMAE)

dt = DecisionTreeRegressor(max_depth=10)
dt.fit(X_train, Y_train)
dtPred = dt.predict(X_test)
dtMSE = mean_squared_error(Y_test, dtPred)
dtMAE = mean_absolute_error(Y_test, dtPred)
print("Decision Tree MSE: ", dtMSE)
print("Decision Tree MAE: ", dtMAE)

rf = RandomForestRegressor(max_depth=10, n_estimators=50)
rf.fit(X_train, Y_train)
rfPred = rf.predict(X_test)
rfMSE = mean_squared_error(Y_test, rfPred)
rfMAE = mean_absolute_error(Y_test, rfPred)
print("Random Forest MSE: ", rfMSE)
print("Random Forest MAE: ", rfMAE)

mlp = MLPRegressor(max_iter=1000)
mlp.fit(X_train, Y_train)
mlpPred = mlp.predict(X_test)
mlpMSE = mean_squared_error(Y_test, mlpPred)
mlpMAE = mean_absolute_error(Y_test, mlpPred)
print("Multi-Layer Perception MSE: ", mlpMSE)
print("Multi-Layer Perception MAE: ", mlpMAE)

knn = neighbors.KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, Y_train)
knnPred = knn.predict(X_test)
knnMSE = mean_squared_error(Y_test, knnPred)
knnMAE = mean_absolute_error(Y_test, knnPred)
print("K Nearest Neighbors MSE: ", knnMSE)
print("K Nearest Neighbors MAE: ", knnMAE)

X_np = np.array(X)
Y_np = np.array(Y)

X_lstm = X_np.reshape((X_np.shape[0], 3, 2))

X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(
    X_lstm, Y_np, test_size=0.2, shuffle=False
)

print("Train samples:", len(X_train_lstm))
print("Test samples:", len(X_test_lstm))

lstm = Sequential()
lstm.add(LSTM(128, return_sequences=False, input_shape=(3, 2)))
lstm.add(Dropout(0.1))
lstm.add(Dense(32, activation='relu'))
lstm.add(Dense(1))

lstm.compile(optimizer='adam', loss='mse')

history = lstm.fit(
    X_train_lstm, Y_train_lstm,
    validation_data=(X_test_lstm, Y_test_lstm),
    epochs=20,
    batch_size=8,
    verbose=1
)

lstmp = lstm.predict(X_lstm).flatten()
lstmPred = lstm.predict(X_test_lstm).flatten()

lstmMSQ = mean_squared_error(Y_test_lstm, lstmPred)
lstmMAE = mean_absolute_error(Y_test_lstm, lstmPred)

print("\nLSTM RESULTS")
print("LSTM MSE:", lstmMSQ)
print("LSTM MAE:", lstmMAE)


print("")
print("")
print("")

avgPred = (linPred + dtPred + rfPred + mlpPred + knnPred)/5
avgMSE = mean_squared_error(Y_test, avgPred)
print("Average MSE: ", avgMSE)

linCoeff = 1/linMSE
dtCoeff = 1/dtMSE
rfCoeff = 1/rfMSE
mlpCoeff = 1/mlpMSE
knnCoeff = 1/knnMSE

totalCoeff = linCoeff + dtCoeff + rfCoeff + mlpCoeff + knnCoeff

linWeight = linCoeff/totalCoeff
print("Linear Weight: ", linWeight)

dtWeight = dtCoeff/totalCoeff
print("Decision Tree Weight: ", dtWeight)

rfWeight = rfCoeff/totalCoeff
print("Random Forest Weight: ", rfWeight)

mlpWeight = mlpCoeff/totalCoeff
print("Multi-Layer Perception Weight: ", mlpWeight)

knnWeight = knnCoeff/totalCoeff
print("K Nearest Neighbors Weight: ", knnWeight)

totalPred = ((linPred * linWeight) + (dtPred * dtWeight) + (rfPred * rfWeight) + (mlpPred * mlpWeight) + (knnPred * knnWeight))
totalMSE = mean_squared_error(Y_test, totalPred)
print("Total MSE: ", totalMSE)

predMSE = [0]*1260
for i in range(1250):
    predMSE[i] = ((lin.predict([X[i]]) * linWeight) + (dt.predict([X[i]]) * dtWeight) + (rf.predict([X[i]]) * rfWeight) + (mlp.predict([X[i]]) * mlpWeight) + (knn.predict([X[i]]) * knnWeight))

eps = 1e-4
n = len(X_test)
predRoll = np.zeros(n)

wlin = wdt = wrf = wmlp = wknn = wlstm = 1/6

for i in range(n):
    linError  = abs(linPred[i]  - Y_test[i])
    dtError   = abs(dtPred[i]   - Y_test[i])
    rfError   = abs(rfPred[i]   - Y_test[i])
    mlpError  = abs(mlpPred[i]  - Y_test[i])
    knnError  = abs(knnPred[i]  - Y_test[i])
    lstmError = abs(lstmPred[i] - Y_test[i])

    linInv  = wlin  / (linError  + eps)
    dtInv   = wdt   / (dtError   + eps)
    rfInv   = wrf   / (rfError   + eps)
    mlpInv  = wmlp  / (mlpError  + eps)
    knnInv  = wknn  / (knnError  + eps)
    lstmInv = wlstm / (lstmError + eps)

    totalInv = linInv + dtInv + rfInv + mlpInv + knnInv + lstmInv

    wlin  = linInv  / totalInv
    wdt   = dtInv   / totalInv
    wrf   = rfInv   / totalInv
    wmlp  = mlpInv  / totalInv
    wknn  = knnInv  / totalInv
    wlstm = lstmInv / totalInv

    x_tab = np.array(X_test[i]).reshape(1, -1)
    x_lstm = np.array(X_np[i], dtype=np.float32).reshape(1, 3, 2)

    lin_p  = float(lin.predict(x_tab))
    dt_p   = float(dt.predict(x_tab))
    rf_p   = float(rf.predict(x_tab))
    mlp_p  = float(mlp.predict(x_tab))
    knn_p  = float(knn.predict(x_tab))
    lstm_p = float(lstm.predict(x_lstm, verbose=0))

    predRoll[i] = (lin_p*wlin + dt_p*wdt + rf_p*wrf + mlp_p*wmlp + knn_p*wknn + lstm_p*wlstm)

print(wlin, wdt, wrf, wmlp, wknn, wlstm)

print("")
print("")
print("")

print("Initializing Game...")
print("Initializing Game...")
money = int(input(f"Starting Amount: {ticker.info['currency']}"))
time = int(input("Length of Investment (in days): "))
stocks = 0

for i in range(time):
    pred_next = float(np.ravel(predMSE[i+1])[0])
    price_now = float(np.ravel(Y[i])[0])

    k=round(int((pred_next/price_now)*6),0)

    if (pred_next - price_now) > 0 and money > k * price_now:
        money -= k * price_now
        stocks += k
    elif (pred_next - price_now) < 0 and stocks >= k:
        money += k * price_now
        stocks -= k

final_price = float(np.ravel(Y[time])[0])
money = money + stocks * final_price

print(f"Final Amount: {ticker.info['currency']}", money)


