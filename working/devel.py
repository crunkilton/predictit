## For RCP Trump-Biden averages

import numpy as np
import requests
import json
import pandas as pd
import altair as alt
#from plotnine import *
import datetime
import statsmodels
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.tsa.api import VAR
from sklearn.utils import resample
import matplotlib.pyplot as plt


def pull_rcp_biden_trump():
    j = requests.get('https://www.realclearpolitics.com/epolls/json/6247_historical.js?1453388629140&callback=return_json')

    jsonData = json.loads(j.text.split('(',1)[-1].rsplit(')',1)[0])

    results = pd.DataFrame()
    df = pd.DataFrame(jsonData['poll']['rcp_avg'])
    for idx, row in df.iterrows():
        temp_df = pd.DataFrame(row['candidate'])
        temp_df['date'] = row['date']
        results = results.append(temp_df, sort=True).reset_index(drop=True)
    results['ds'] = pd.to_datetime(results['date']).apply(lambda x: x.date())
    results['value'] = results['value'].astype(float)

    rcp = results.pivot(index = 'ds', values = 'value', columns = 'name')

    rcp['y'] = rcp['Biden'] - rcp['Trump']

    rcp = rcp[['y']].reset_index()
    rcp['ds'] = pd.to_datetime(rcp['ds'])

    return(rcp)

rcp = pull_rcp_biden_trump()

## does it look like it matches? 
# alt.Chart(rcp).mark_line().encode(x = 'ds:T', y = 'y:Q')


## RCP Series

## to do:
## check rcp series for patterns:
## ACF/PACF
## arima for looking
## hurst for mean reversion
## adf for unit root

## run some quick models arima, prophet, lag model - which is best? 
## join 538 approval ratings: do a viz, does including those change what this looks like? 
## vector autoregression: are those series related in any way? 
# see https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/ for how to do all of this in python

# depending on VAR - put in 538 AR term into model here, see if that helps
# (lower priority) do the converse: put trump v biden term into 538 AR model, see if that improves it

# decide on best model (or do a model and an avg one), do "create trump biden df" from predictit, gen preds and see how it looks
# concern: lower changes here - more dependent on poll dropping, which others might know better


########### PART 1 - EXPLORATION

## ACF/PACF
plot_pacf(rcp['y'])
plot_pacf(rcp['y'].diff().dropna())
# p = 1

plot_acf(rcp['y'].diff().dropna())

# d = 1

## ARIMA
## auto arima

ar_model = auto_arima(rcp['y'].diff().dropna())
ar_model ## R plays more nicely with this than Python

model = ARIMA(rcp['y'], order = (1, 1, 0))
fit = model.fit()
fit.summary()
# plotting errors
pd.DataFrame(fit.resid).plot()
pd.DataFrame(fit.resid).plot(kind='kde')
pd.DataFrame(fit.resid).describe()
## just looking at the differenced series
rcp['y'].diff().plot() # looks pretty stable after differencing

## HURST

def hurst(ts, lags):
    # calculate standard deviation of differenced series using various lags
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # # plot on log-log scale
    # plot(log(lags), log(tau)); show()
    # # calculate Hurst as slope of log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2.0
    return(hurst)

ts = rcp['y'].tolist()
hurst(ts = ts, lags = range(1, 50)) # doesn't look like it's mean reverting

## ADF
adfuller(rcp['y'].diff().dropna()) # can't reject the null of non-stationarity at zero, can at 1 difference


########## PART 2 - QUICK MODELS

arima_resids = fit.resid
lag_resids = rcp['y'].diff().dropna()

## raw
arima_resids.describe()
lag_resids.describe()

## abs error
np.abs(arima_resids).describe()
np.abs(lag_resids).describe()

# lag looks better. Let's stop here before spending more time on fancier looks. 

#### LOOKING AT 538 APPROVAL RATING IN COMBO WITH THIS
## (ar_538 from arbitrage file)
merged = pd.merge(rcp, ar_538, left_on = 'ds', right_on = 'ds', suffixes = ['_rcp', '_538'])

merged.corr() # this tracks: higher biden leads = lower trump AR

m = merged[['y_rcp', 'y_538']]


######### VAR ############

# GRANGER CAUSALITY

maxlag = 7
test = 'ssr_chi2test'
g_test = grangercausalitytests(m, maxlag = maxlag)

p_values = [round(g_test[i+1][0][test][1],4) for i in range(maxlag)]

p_values ## looks like yes causes 


## COINTEGRATION

cj = coint_johansen(m, det_order = -1, k_ar_diff = 7)
## tbh I have no idea how to interpret any of this output, so I'm going to use engle-granger 
coint_test = coint(m['y_538'], m['y_rcp'])
coint_test # can't reject the null of no cointegration

## THE ACTUAL VAR
n_back = 7
df_train, df_test = m[0:-n_back], m[-n_back:]

## difference for stationarity

df_train_diff = df_train.diff().dropna()

# test to make sure it's stationary

adfuller(df_train_diff['y_rcp'])
adfuller(df_train_diff['y_538']) ## looks good

# the model
model = VAR(df_train_diff)
## choosing best P (order) of VAR model
for i in [1,2,3,4,5,6,7]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

# 4 looks best here, lets try the automatic way

best = model.select_order(maxlags = 7)
best.summary() ## hmm two for 4, two for 7. Try both and see what is better maybe?
# start with 4
fitted = model.fit(4)
fitted.summary()

## durbin-watson for serial error correlation
# 2 is good, 0 = positive correlation, 4 = negative correlation
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(fitted.resid)
dw ## looks great!

## the forecast
lag_order = fitted.k_ar # to verify
for_forecast = df_train_diff.values[-lag_order:]
for_forecast

# Forecast
fc = fitted.forecast(y=for_forecast, steps=n_back)
df_fc = pd.DataFrame(data = fc, columns = ['y_rcp', 'y_538'])
most_recent_538 = df_train['y_538'].iloc[-1]
most_recent_rcp = df_train['y_rcp'].iloc[-1]

df_fc['rcp_forecast'] = df_train['y_rcp'].iloc[-1] + df_fc['y_rcp'].cumsum()

df_fc['538_forecast'] = df_train['y_538'].iloc[-1] + df_fc['y_538'].cumsum()

results = pd.concat([df_test.reset_index(), df_fc[['rcp_forecast', '538_forecast']]], axis = 1)

results['error_538_var'] = results['y_538'] - results['538_forecast'] 
results['error_rcp_var'] = results['y_rcp'] - results['rcp_forecast'] 
results['error_538_lag'] = results['y_538'] - most_recent_538
results['error_rcp_lag'] = results['y_rcp'] - most_recent_rcp
results

## now, putting it all together on a few test sets (skipping the assumptions checking because it's not looking like this is going to be useful)

n_back = 7
indexes_to_check_to = [1, 3, 6, 8, 11, 15, 21, 25, 33, 40, 43, 46] 
comparison_df = pd.DataFrame()

for idx in indexes_to_check_to:
    df_train, df_test = m[0:-(idx + n_back)], m[-(idx + n_back):-idx]

    df_train_diff = df_train.diff().dropna()
    model = VAR(df_train_diff)

    ## choosing best lag order of VAR model
    aic = []
    for i in [1,2,3,4,5,6,7]:
        result = model.fit(i)
        aic.append(result.aic)
        # print('Lag Order =', i)
        # print('AIC : ', result.aic)
        # print('BIC : ', result.bic)
        # print('FPE : ', result.fpe)
        # print('HQIC: ', result.hqic, '\n')

    min_aic = np.min(aic)
    lag_order = aic.index(min_aic) + 1

    fitted = model.fit(lag_order)
    ## the forecasts
    for_forecast = df_train_diff.values[-lag_order:]

    # Forecast
    fc = fitted.forecast(y=for_forecast, steps=n_back)
    df_fc = pd.DataFrame(data = fc, columns = ['y_rcp', 'y_538'])
    most_recent_538 = df_train['y_538'].iloc[-1]
    most_recent_rcp = df_train['y_rcp'].iloc[-1]

    df_fc['rcp_forecast'] = df_train['y_rcp'].iloc[-1] + df_fc['y_rcp'].cumsum()

    df_fc['538_forecast'] = df_train['y_538'].iloc[-1] + df_fc['y_538'].cumsum()

    results = pd.concat([df_test.reset_index(), df_fc[['rcp_forecast', '538_forecast']]], axis = 1)

    results['error_538_var'] = results['y_538'] - results['538_forecast'] 
    results['error_rcp_var'] = results['y_rcp'] - results['rcp_forecast'] 
    results['error_538_lag'] = results['y_538'] - most_recent_538
    results['error_rcp_lag'] = results['y_rcp'] - most_recent_rcp
    results['538_lag_over_var'] = np.abs(results['error_538_lag']) - np.abs(results['error_538_var'])
    results['rcp_lag_over_var'] = np.abs(results['error_rcp_lag']) - np.abs(results['error_rcp_var'])
    results['idx'] = idx
    tokeep = results[['idx', '538_lag_over_var', 'rcp_lag_over_var']]

    comparison_df = comparison_df.append(results) #for everything
    #comparison_df = comparison_df.append(tokeep) #for just seeing which is better

comparison_df
## breaking down by test start date
comparison_df.groupby('idx').mean() # average for all starts

## breaking down by prediction
comparison_df.reset_index().drop('idx', axis =1).groupby('level_0').mean() # average for all days out. change level_0 to index for smaller version

## overall average
comparison_df.mean()


## there are arguments for both - but for both 538 and RCP the lag is off by less. Might be able to fiddle with the 538 model to make a version using VAR which is better than what we have, especially if we use the "support trump" part of RCP. 

## (lower priority) ideas:

#### VAR #####
## 538 + RCP VAR models in production (downside for 538 is that we can't train for the whole series, just last 200 days, so maybe do VAR first).
## add in support trump/support biden features to the model so it's not just the difference. 
## test out for each iter: 
# ideal lag order, 
# whether assumptions met, etc - 
# make a big function I can use with any series
## see if another model can predict errors with external regressors (like series mean over time)

#### LM/RF ####
## do LM/RF model for 538 and RCP with lags of the other. Is that better than VAR?

## to add to arbitrage file: 1) lag model with historical errors for trump-biden, 2) any other ideas I have later

#### GENERAL
## get a way to bootstrap time series errors so I'm not just taking the average. 


####### MAKING THE LAG MODEL ###########

def make_lags(df, n_lags):
    for lag in range(1,n_lags):
        df[f'lag{lag}'] = df['y'].shift(lag)
    return(df)

def bootstrap_std(errors, n_bootstraps):
    st_devs = []

    for i in range(n_bootstraps):
        boot = resample(errors, replace = True, n_samples = len(errors))
        st_devs.append(np.std(boot))

    return(st_devs)

test_df = make_lags(rcp, 7)

test_errors = test_df['lag2'] - test_df['y']

boot = resample(test_errors, replace = True, n_samples = len(test_errors))

plt.hist(boot, density = True, bins = 20)

plt.hist(test_errors, density = True, bins = 20)

## this looks questionable. Maybe estimate one for the odds that it changes, then another for how far it will change? The problem there is that the odds of it staying at zero is not random but based on when polls will come out, which I don't want to have to track. Maybe draw an error from bootstrapped sample errors and try that?

lower_bound = -.55
upper_bound = .05
n_bootstraps = 50

## get a better name than this
def bootstrap_bounds(errors, n_bootstraps, lower_bound, upper_bound):
    prob_sample = []
    for sim in range(n_bootstraps):
        boot = resample(errors, replace = True, n_samples = len(errors))

        boot_prob = (len(boot[(boot >= lower_bound)]) - len(boot[boot > upper_bound]) ) / len(boot)

        prob_sample.append(boot_prob)

    bootstrapped_prob = np.mean(prob_sample)
    return(bootstrapped_prob)

##
prb = []
names = []
for lwr, upr in [(-400,-1.000001), (-1,-.0500001), (-.05, .04999), (.05, .99999), (1, 400)]:
    x = bootstrap_bounds(test_errors, 100, lwr, upr)
    prb.append(x)
    names.append(upr)

standardized_probabilities = [elem/np.sum(prb) for elem in prb]

## to add:
# convert the cost_df to difference from where we are at now
# take those differences and run them through the above function
# add those in as probability of each segment. 
# consider changing the simple model to do this too. 

# also: take 2016 averages, combine the next 30 days of those to the sample, then take the std to account for future variance.