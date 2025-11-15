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

lin = linear_model.LinearRegression()
lin.fit(X_train, Y_train)
linPred = lin.predict(X_test)
linMSQ = mean_squared_error(Y_test, linPred)
linMAE = mean_absolute_error(Y_test, linPred)
print("Linear MSQ: ", linMSQ)
print("Linear MAE: ", linMAE)

dt = DecisionTreeRegressor(max_depth=10)
dt.fit(X_train, Y_train)
dtPred = dt.predict(X_test)
dtMSQ = mean_squared_error(Y_test, dtPred)
dtMAE = mean_absolute_error(Y_test, dtPred)
print("Decision Tree MSQ: ", dtMSQ)
print("Decision Tree MAE: ", dtMAE)

rf = RandomForestRegressor(max_depth=10, n_estimators=50)
rf.fit(X_train, Y_train)
rfPred = rf.predict(X_test)
rfMSQ = mean_squared_error(Y_test, rfPred)
rfMAE = mean_absolute_error(Y_test, rfPred)
print("Random Forest MSQ: ", rfMSQ)
print("Random Forest MAE: ", rfMAE)

mlp = MLPRegressor(max_iter=1000)
mlp.fit(X_train, Y_train)
mlpPred = mlp.predict(X_test)
mlpMSQ = mean_squared_error(Y_test, mlpPred)
mlpMAE = mean_absolute_error(Y_test, mlpPred)
print("Multi-Layer Perception MSQ: ", mlpMSQ)
print("Multi-Layer Perception MAE: ", mlpMAE)

knn = neighbors.KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, Y_train)
knnPred = knn.predict(X_test)
knnMSQ = mean_squared_error(Y_test, knnPred)
knnMAE = mean_absolute_error(Y_test, knnPred)
print("K Nearest Neighbors MSQ: ", knnMSQ)
print("K Nearest Neighbors MAE: ", knnMAE)

print("")
print("")
print("")

avgPred = (linPred + dtPred + rfPred + mlpPred + knnPred)/5
avgMSQ = mean_squared_error(Y_test, avgPred)
print("Average MSQ: ", avgMSQ)

linCoeff = 1/linMSQ
dtCoeff = 1/dtMSQ
rfCoeff = 1/rfMSQ
mlpCoeff = 1/mlpMSQ
knnCoeff = 1/knnMSQ

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
totalMSQ = mean_squared_error(Y_test, totalPred)
print("Total MSQ: ", totalMSQ)

predMSQ = [0]*1260
for i in range(1250):
    predMSQ[i] = ((lin.predict([X[i]]) * linWeight) + (dt.predict([X[i]]) * dtWeight) + (rf.predict([X[i]]) * rfWeight) + (mlp.predict([X[i]]) * mlpWeight) + (knn.predict([X[i]]) * knnWeight))
    

wlin = 1/5
wdt = 1/5
wrf = 1/5
wmlp = 1/5
wknn = 1/5

for i in range(len(X_test)):
  linError = abs(linPred[i] - Y_test[i])
  dtError = abs(dtPred[i] - Y_test[i])
  rfError = abs(rfPred[i] - Y_test[i])
  mlpError = abs(mlpPred[i] - Y_test[i])
  knnError = abs(knnPred[i] - Y_test[i])

  linInv  = wlin/(linError+0.0001)
  dtInv = wdt/(dtError+0.0001)
  rfInv = wrf/(rfError+0.0001)
  mlpInv = wmlp/(mlpError+0.0001)
  knnInv = wknn/(knnError+0.0001)
  totalInv = linInv + dtInv + rfInv + mlpInv + knnInv

  wlin = linInv/totalInv
  wdt = dtInv/totalInv
  wrf = rfInv/totalInv
  wmlp = mlpInv/totalInv
  wknn = knnInv/totalInv
print(wlin, wdt, wrf, wmlp, wknn)

predRoll = [0]*1260
for i in range(1250):
    predRoll[i] = ((lin.predict([X[i]]) * wlin) + (dt.predict([X[i]]) * wdt) + (rf.predict([X[i]]) * wrf) + (mlp.predict([X[i]]) * wmlp) + (knn.predict([X[i]]) * wknn))

print("")
print("")
print("")

print("Initializing Game...")
money = int(input(f"Starting Amount: {ticker.info['currency']}"))
time = int(input("Length of Investment (in days): "))
stocks = 0

for i in range(time):
    if predMSQ[i+1] - Y[i] > 0 and money > 1 * Y[i]:
        money = money - (1 * Y[i])
        stocks = stocks + 1
    elif predMSQ[i+1] - Y[i] < 0 and stocks > 1:
        money = money + (1 * Y[i])
        stocks = stocks - 1
money = (stocks * Y[time]) + money
print(f"Final Amount: {ticker.info['currency']}", money)