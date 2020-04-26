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


def prep_df(ar_538, lag_order):
    df = ar_538[['y', f'lag{lag_order}']]
    df = df.iloc[::-1].reset_index(drop = True)
    df['avg_30'] = (df['y'].rolling(window = 30).mean()).shift((lag_order))
    df['avg_90'] = (df['y'].rolling(window = 90).mean()).shift((lag_order))
    df['avg_lifetime'] = (np.cumsum(df['y'])/(df.index + 1)).shift(lag_order)
    df = df.dropna().reset_index(drop = True)
    return(df)

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

                z = test_model_errors(ar_538 = ar_538, lag_order = lag, other_features=features, model_type=model)

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
    return(for_model)

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

    test_df.to_csv('errors_raw.csv')

    test_df = pd.DataFrame({'lag':l + 1, 'error':ans})
    test_df['error'] = test_df['error'].astype(float)
    st_devs = test_df.groupby('lag')['error'].std()
    st_devs.to_csv('errors_std.csv')
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