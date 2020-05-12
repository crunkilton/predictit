## Mostly for 538 approval ratins here

from arbitrage import *

## steps:
## chose model
## choose hyperparameters

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

## a quick look
model = LinearRegression()#RandomForestRegressor()
df = df.dropna()
X_data = df.drop(['y'], axis = 1)
Y_data = df['y']
model.fit(X_data, Y_data)

preds = model.predict(X_data)
np.mean(np.abs(preds - Y_data))
## get data

## PROCESSING

def prep_model_df(ar_538, lag_order, shift = True):
    df = ar_538[['y', f'lag{lag_order}']]
    df = df.iloc[::-1].reset_index(drop = True)
    if shift:
        df['avg_30'] = (df['y'].rolling(window = 30).mean()).shift((lag_order))
        df['avg_90'] = (df['y'].rolling(window = 90).mean()).shift((lag_order))
        df['avg_lifetime'] = (np.cumsum(df['y'])/(df.index + 1)).shift(lag_order)
    else:
        df['avg_30'] = (df['y'].rolling(window = 30).mean())
        df['avg_90'] = (df['y'].rolling(window = 90).mean())
        df['avg_lifetime'] = (np.cumsum(df['y'])/(df.index + 1))

    df = df.dropna().reset_index(drop = True)
    return(df)

prep_df(ar_538, lag_order = 4, shift = False)

## WORKING ON MODEL PREDICTIONS HERE
ar_538
cost_df = predictIt(predictit_df).gen_538_cost_df()
most_recent_ar = ar_538['y'][0]
model_st_devs = pd.read_csv('model_errors_std.csv')
model_st_devs

df = ar_538
df['lag1'] = df['y'].shift(-1)
df['lag2'] = df['y'].shift(-2)
df['lag3'] = df['y'].shift(-3)
df['lag4'] = df['y'].shift(-4)
df['lag5'] = df['y'].shift(-5)
df['lag6'] = df['y'].shift(-6)
df['lag7'] = df['y'].shift(-7)

ar_data = ar_538

approval_538(ar_538).st_devs

## CHANGE THIS TO PREP DF CLASS

end_date = cost_df['dateEnd'].unique()
end_date = pd.to_datetime(end_date[0]).date()
current_date = ar_data['ds'].iloc[0].date()

days_off = end_date - current_date
datediff_index = days_off.days - 1

if datediff_index < 0:
    raise ValueError("""It is the day of the market close so don't use the model""")

## OR MAKE AN ARGUMENT HERE - STDEVS IS THE ARGUMENT

st_devs = model_st_devs
sd_to_use = st_devs.iloc[datediff_index]

### BUILD MODEL HERE
for_model = pd.read_csv('best_model.csv')

lag_order = for_model['lag'].iloc[datediff_index]

train_df = prep_df(self.df, lag_order = lag_order)

test_df = prep_df(self.df, lag_order = for_model['lag'].iloc[datediff_index], shift = False)
test_df = test_df.iloc[-1:]
test_df[f'lag{lag_order}'] = most_recent_ar

def fit_model(train_df, test_df, other_features, model_type):
    X_train = train_df.drop(['y'] + other_features, axis = 1)
    X_test = test_df.drop(['y'] + other_features, axis = 1)
    Y_train = train_df['y']

    ## the model
    if model_type == 'lm':
        model = LinearRegression(fit_intercept=False)
    elif model_type == 'rf':
        model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    return(preds)

estimate = fit_model(train_df, 
            test_df, 
            other_features = [for_model['additional_features'].iloc[datediff_index]],
            model_type = for_model['model'].iloc[datediff_index])
## CONSIDER ADDING ON THE BIAS HERE


## PUT RESULT AS MOST_RECENT_AR HERE


dist = scipy.stats.norm(estimate + .077, sd_to_use)

cost_df['actual_lower'] = cost_df['lower'] - .05
cost_df['actual_upper'] = cost_df['upper'] + .05

cost_df['prob'] = cdf_probs(dist, lower = cost_df['actual_lower'], upper = cost_df['actual_upper'])

#cost_df['prob'].apply(lambda x: round(x, 4))

cd = cost_df.drop(['actual_lower', 'actual_upper', 'dateEnd'], axis = 1)
cd['yes_margin'] = cd['prob'] - cd['real_yes_cost']
cd['no_margin'] = (1-cd['prob']) - cd['real_no_cost']
cd['prob'] = round(cd['prob'], 4)
cd = cd[['lower', 'upper', 'prob', 'bestBuyYesCost', 'bestBuyNoCost', 'yes_margin', 'no_margin']]

hours_since_update = round((datetime.datetime.now() - pd.to_datetime(ar_data['timestamp'][0])).seconds / (60*60), 1)

print(str(days_off.days) + ' days out, current rating is ' + str(round(most_recent_ar, 2)) + ', last update posted ' + str(hours_since_update) + ' hours ago at '+ ar_data['timestamp'][0])

return(cd)

raw_df = pd.read_csv('errors_raw.csv')
fordist = estimate - raw_df.groupby('lag').mean().iloc[datediff_index]
    

## 
predictIt(predictit_df).gen_538_cost_df()

df_for_model = prep_df(ar_538, lag_order = 1)

def test_model_errors(df, lag_order, other_features, model_type = 'lm'):

    X_data = df.drop(['y'] + other_features, axis = 1)
    Y_data = df['y']

    ## the model
    if model_type == 'lm':
        model = LinearRegression(fit_intercept=False)
    elif model_type == 'rf':
        model = RandomForestRegressor()
    model.fit(X_data, Y_data)

    preds = model.predict(X_data)

    error = preds - df['y']

    lag_error = np.mean(np.abs(df[f'lag{lag_order}'] - df['y']))

    model_error = np.mean(np.abs(preds - df['y']))
    model_std = np.std(preds - df['y'])


    print(f'{np.round(model_error, 4)} model error, {np.round(lag_error, 4)} lag error')

    return({'model_error':[model_error], 'model_std':[model_std]})

# test_model_errors(df = prep_df(ar_538, lag_order = 1), lag_order = 1, other_features = ['avg_90'], model_type='rf')

def find_best_model(ar_538):
    final_df = pd.DataFrame()

    for lag in [1,2,3,4,5,6,7]:
        for features in [['avg_lifetime', 'avg_30', 'avg_90'], ['avg_30', 'avg_90'], ['avg_30', 'avg_lifetime'], ['avg_90', 'avg_lifetime'], ['avg_30'], ['avg_90'], ['avg_lifetime']]:
            for model in ['lm', 'rf']:

                z = test_model_errors(df = prep_df(ar_538, lag_order = lag), lag_order = lag, other_features=features, model_type=model)

                toappend = pd.DataFrame({'lag': lag,
                'additional_features':features,
                #'additional_features': [str([e for e in all_features if e not in features])],
                'model': model
                }
                )

                toappend = pd.concat([toappend, pd.DataFrame(z)], axis = 1)

                final_df = final_df.append(toappend)

    final_df = final_df.sort_values(['lag', 'model_error']).reset_index(drop=True)

    answer = final_df.groupby(['lag']).first()

    for_model = answer.reset_index()[['lag', 'additional_features', 'model']]

    for_model.to_csv('best_model.csv', index = False)
    return(for_model)

for_model = find_best_model(ar_538)

## to do: make predictions, test them for sanity, 
# get sd for ranges of predictions, look to make sure they are normal
# do a 1x calculation for a set of ranges
# add to other function
for_model
##
2+2


def gen_model_stds(ar_538, for_model, n_back = 500):
    ans = []
    l = []
    for lag in range(len(for_model['lag'])):

        df_for_model = prep_df(ar_538, lag_order = lag + 1)
        max_index = max(df_for_model.index)
        min_index = max_index - n_back
        df = df_for_model

        for i in range(min_index, max_index):
            ## copy pasted data prep from test_model_errors:
                ## processing

            X_data = df.drop(['y'] + [for_model['additional_features'][lag]], axis = 1)
            Y_data = df['y']

            X = X_data.iloc[:i,:]
            Y = Y_data.iloc[:i]

            y_train, y_test  = Y[:i-2], Y[i-1]
            X_train, X_test = X[:i-2], X[i-1:]

            if for_model['model'][lag] == 'rf':
                model = RandomForestRegressor()
            elif for_model['model'][lag] == 'lm':
                model = LinearRegression()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            #avg_performance = np.mean(np.abs(preds - y_test))
            next_performance = preds - y_test
            #ans.append(avg_performance)
            ans.append(next_performance)
            l.append(lag)

            print(f'Lag {str(lag + 1)}, iter {str(i - min_index)} of {str(max_index - min_index)}')

    test_df = pd.DataFrame({'lag':l, 'error':ans})
    test_df['lag'] = test_df.lag + 1
    test_df['error'] = test_df['error'].astype(float)
    st_devs = test_df.groupby('lag')['error'].std()
    test_df.to_csv('errors_raw.csv', index = False)
    st_devs.to_csv('model_errors_std.csv', index = False)
    return(test_df)

## to do: rf predictions w/ standard errors - put into arbitrage.py function. 

## think about: why are RF predictions biased downward? see altair at bottom

##################################################
############## UNFINISHED/WORKING ################
##################################################

####### IF YOU WANT P VALUES

## finding the p values
import statsmodels.api as sm
mod = sm.OLS(Y_data,X_data) #sm.add_constant(X_data)
fii = mod.fit()

fii.summary2()

###### CROSS VALIDATION

## check cross validation - this is a quick first look, make into function later. 

X_data = df.drop(['y'], axis = 1)
Y_data = df['y']

X = X_data.copy()
y = Y_data.copy()

tscv = TimeSeriesSplit(n_splits=100)

ans = []
ans2 = []
for train_index, test_index in tscv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    avg_performance = np.mean(np.abs(preds - y_test))
    next_performance = preds[0] - y_test.iloc[0]
    ans.append(avg_performance)
    ans2.append(next_performance)

ans2


######################
######## JUNK/NOTES
######################

## misc altair


## a plot:
#alt.data_transformers.disable_max_rows()
## import altair as alt
## lines
alt.Chart(z).mark_line(size = .1).encode(
    alt.Y('simulated_ar',
        scale=alt.Scale(zero=False)
    ), 
    x = 'date', 
    color = 'sim_number')

## histogram of maxes

alt.Chart(pd.DataFrame.from_dict({'max':simulated_maxes})).mark_bar().encode(alt.X('max', bin = alt.Bin(extent=[45, 53], step=0.5)),
y = 'count()')



alt.Chart(test).mark_bar().encode(alt.X('diff', bin = alt.Bin(extent=[0, 5], step=0.1)),
y = 'count()')


forchart = test_df
forchart['iter'] = forchart.index
alt.Chart(test_df[test_df['lag'] == 2]).mark_line().encode(
    alt.Y('error'),
    alt.X('iter')
)

alt.Chart(test_df[test_df['lag'] == 2]).mark_bar().encode(
    alt.X('error', bin = True),
    y = 'count()'
)

t = rf_raw
t['iter'] = t.index - (700 * (t.lag - 1))
t['lag'] = t['lag'].astype(str)
t['abs_error'] = np.abs(t.error)
t['ma_10'] = t['abs_error'].rolling(window = 30).mean()

alt.Chart(t).mark_line().encode(
    x = 'iter',
    y = 'ma_10',
    color = 'lag')

alt.Chart(ar_538).mark_line().encode(
    x = 'ds',
    y = 'y')


raw_df = pd.read_csv('errors_raw.csv')

alt.Chart(raw_df).mark_bar().encode(
    alt.X('error:Q', bin=True),
    alt.Y('count()'),
    facet='lag'
)
