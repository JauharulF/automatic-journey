from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime(x, '%d-%b-%y')

series = read_csv('phosphor.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# check the autocorrelation to determine the lags.
print(series.head())
autocorrelation_plot(series)
pyplot.show()
# try to create the model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# check the bias
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
# retrain the model
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
next_val = model.fit(disp=0).forecast()
print('Forecast the next price is %.2f with error probablity of %.2f within range (%.2f and %.2f)' % (next_val[0], next_val[1], next_val[2][0][0], next_val[2][0][1]))
