
## ideas 4/3/2020:
## 1) betfair arbitrage
## 2) comment out everything so I'll understand it
## other stuff: make 538 AR predictions better, do tweet count model. 

########## THE FILE ##########

import pandas as pd 
import numpy as np 
import requests
import json
import datetime
import scipy.stats


class odds_calc():
    def __init__(self, odds, withdrawal_tax = False):
        self.odds = odds
        self.withdrawal_tax = withdrawal_tax

    def real_odds(self):
        profit = (1 - self.odds)
        profit_tax = profit * .1
        profit_post_tax = profit - profit_tax

        if self.withdrawal_tax:
            withdrawal_tax = (1 - profit_tax) * .05
            net_gains = profit_post_tax - withdrawal_tax
        else:
            net_gains = profit_post_tax

        new_total = self.odds + net_gains

        real_odds = self.odds/new_total

        return(real_odds)
    
    def ror_arbitrage(self):
        cost = np.sum(self.odds)
        revenue = len(self.odds) - 1

        profit_tax = []
        profit_tax.append((1 - np.mean(self.odds)) * (len(self.odds) - 1) * .1)
        profit_tax.append((1 - (np.sum(self.odds) - np.min(self.odds))/(len(self.odds) - 1)) * (len(self.odds) - 1) * .1)
        profit_tax.append((1 - (np.sum(self.odds) - np.max(self.odds))/(len(self.odds) - 1)) * (len(self.odds) - 1) * .1)
        
        profit_tax = np.array(profit_tax)

        if self.withdrawal_tax:
            withdrawal_tax = (revenue - profit_tax) * .05
        else:
            withdrawal_tax = 0
        
        ror = 100 * (((revenue - (withdrawal_tax + profit_tax))/cost) - 1)
        return(ror)

# ## examples
# odds_calc(odds = .93, withdrawal_tax=True).real_odds()
# odds_calc(odds = [.42, .50, .97], withdrawal_tax=False).ror_arbitrage()

### API HERE

def pull_predict_it(): 
    p = requests.get('https://www.predictit.org/api/marketdata/all/')

    z = p.json()
    full_df = pd.DataFrame()
    for i in range(len(z['markets'])):
        df = pd.DataFrame(z['markets'][i]['contracts'])
        df['contract_name'] = z['markets'][i]['name']
        df['url'] = z['markets'][i]['url']
        df['contract_id'] = z['markets'][i]['id']
        full_df = full_df.append(df)

    df = full_df
    return(df)

predictit_df = pull_predict_it()

class predictIt():

    def __init__(self, df):
        self.df = df

    def arbitrage_possibles(self, withdrawal_tax = True, all_profitable = False):
        
        df = self.df

        if withdrawal_tax:
            threshold = .95
        else:
            threshold = 1

        df['inv_buy_no'] = 1 - df['bestBuyNoCost']
        
        possibles = df[df['bestBuyNoCost'] < threshold].groupby(['contract_id','contract_name'])['inv_buy_no'].sum().sort_values(ascending = False)

        if all_profitable:
            profit_threshold = 1
        else:
            profit_threshold = 1.11

        targets = possibles[possibles > profit_threshold]

        return(targets)
    
    def arbitrage_roi(self):
        df = self.df
        arb_possibles = predictIt(df).arbitrage_possibles(withdrawal_tax = False, all_profitable = True)

        contracts = arb_possibles.index.get_level_values('contract_id').tolist()
        contracts

        ROIs = pd.DataFrame()

        for contract in contracts:
            odds = df[df['contract_id'] == contract]['bestBuyNoCost'].dropna().tolist()
            returns = odds_calc(odds).ror_arbitrage()
            ROIs = ROIs.append([returns])

        ROIs.columns = ['mean', 'best', 'worst']
        ROIs['contract_name'] = arb_possibles.index.get_level_values('contract_name').tolist()
        ROIs['contract_id'] = contracts

        ROIs = ROIs.sort_values('mean', ascending = False)

        ROIs = ROIs.merge(self.df[['contract_name', 'contract_id', 'dateEnd']].drop_duplicates()) # self

        ROIs['time_left'] = pd.to_datetime(ROIs['dateEnd'], errors = 'coerce') - datetime.datetime.now()

        ROIs['annualized_return'] = ROIs['mean']/(ROIs['time_left']/datetime.timedelta(365))

        ROIs['total_yes_value'] = arb_possibles.tolist()

        return(ROIs)

    def paired_outcome_arbitrage(self, paired_outcomes):
        paired_df = pd.DataFrame()
        holder = pd.DataFrame()

        df = self.df

        for market_name, market_id in paired_outcomes.items():

            paired = df[df['id'].isin(market_id)][['id','bestBuyYesCost', 'bestBuyNoCost']].reset_index()
            #print(paired)

            cost1 = paired['bestBuyYesCost'][0] + paired['bestBuyNoCost'][1]
            cost2 = paired['bestBuyYesCost'][1] + paired['bestBuyNoCost'][0]

            #print(cost1, cost2)
            #print('iter done')

            holder['markets'] = [market_name]
            holder['best_price'] = [np.min([cost1, cost2])]
            holder['avg_profit_tax'] = .1 * (1 - holder['best_price']/2)
            holder['raw_profit'] = 1 - holder['best_price']
            holder['post_tax_profit'] = 1 - (holder['best_price'] + holder['avg_profit_tax'])
            paired_df = paired_df.append(holder).sort_values(by = ['post_tax_profit'], ascending = False)

        return(paired_df)

    def gen_538_cost_df(self):
        df = self.df 
        
        ar = df[df['contract_name'].str.contains("Trump's 538 job")]
        if len(ar['contract_id'].unique()) == 1:
            r = ar['name'].str.replace('\%', '').str.replace(' to ', ',').str.replace(' or lower', '').str.replace(' or higher', '')
            r[0] = '00.0,' + r[0]
            r[len(r)-1] = r[len(r)-1] + ',100'

            low_bounds = r.apply(lambda x: x[:4]).astype(float)
            high_bounds = r.apply(lambda x: x[5:]).astype(float)

            cost_df = ar[['bestBuyYesCost', 'bestBuyNoCost', 'dateEnd']]
            cost_df['lower'] = low_bounds
            cost_df['upper'] = high_bounds

            cost_df['real_yes_cost'] = cost_df['bestBuyYesCost'].apply(lambda x: odds_calc(x).real_odds())

            cost_df['real_no_cost'] = cost_df['bestBuyNoCost'].apply(lambda x: odds_calc(x).real_odds())

            return(cost_df)
        else:
            print('multiple possible markets:')
            print(ar['contract_id'].unique())
            print(ar['contract_name'].unique())

    def gen_538_max_ar_df(self, contract_id = 6601):
        df = self.df 

        ar = df[df['contract_id'] == contract_id]

        r = ar['name'].str.replace('\%', '').str.replace(' to ', ',').str.replace(' or lower', '').str.replace(' or higher', '')
        r[0] = '00.0,' + r[0]
        r[len(r)-1] = r[len(r)-1] + ',100'

        low_bounds = r.apply(lambda x: x[:4]).astype(float)
        high_bounds = r.apply(lambda x: x[5:]).astype(float)

        cost_df = ar[['bestBuyYesCost', 'bestBuyNoCost', 'dateEnd']]
        cost_df['lower'] = low_bounds
        cost_df['upper'] = high_bounds

        cost_df['real_yes_cost'] = cost_df['bestBuyYesCost'].apply(lambda x: odds_calc(x).real_odds())

        cost_df['real_no_cost'] = cost_df['bestBuyNoCost'].apply(lambda x: odds_calc(x).real_odds())

        return(cost_df)


predictIt(predictit_df).arbitrage_roi()


def list_contests(df):
    uqs = df.groupby(['contract_name', 'contract_id']).size().reset_index().rename(columns={0:'count'})

    uqs = uqs.sort_values(by = ['contract_name'])

    for row in range(len(uqs)):
        print(uqs['contract_id'][row], uqs['contract_name'][row])

# list_contests(df[df['contract_name'].str.contains("Trump's 538 job")])
#4353
#4365
#4366


######## TRUMP 538 APPROVAL RATING HERE ##########
## calculate averages historically:

def pull_538_approval():
    ar_data = pd.read_csv("https://projects.fivethirtyeight.com/trump-approval-data/approval_topline.csv")
    ar_data = ar_data[ar_data['subgroup'] == 'All polls']
    ar_data['ds'] = pd.to_datetime(ar_data['modeldate'], format = "%m/%d/%Y")
    ar_data['y'] = ar_data['approve_estimate']
    ar_data = ar_data[['ds', 'y', 'timestamp']].reset_index(drop = True)

    return(ar_data)

ar_538 = pull_538_approval()



class approval_538():

    def __init__(self, df):

        df['lag1'] = df['y'].shift(-1)
        df['lag2'] = df['y'].shift(-2)
        df['lag3'] = df['y'].shift(-3)
        df['lag4'] = df['y'].shift(-4)
        df['lag5'] = df['y'].shift(-5)
        df['lag6'] = df['y'].shift(-6)
        df['lag7'] = df['y'].shift(-7)

        ndf = pd.melt(df, id_vars = ['ds', 'y', 'timestamp'], var_name = 'lag', value_name='pred')
        ndf['error'] = ndf['y'] - ndf['pred']


        st_devs = ndf.drop(['y', 'pred'], axis = 1).groupby('lag').std(ddof = 1)

        self.df = df
        self.ndf = ndf # for troubleshooting
        self.st_devs = st_devs
        self.most_recent_ar = df['y'][0]

    @staticmethod
    def cdf_probs(dist, lower, upper):
        prob = dist.cdf(upper) - dist.cdf(lower)
        return(prob)
    
    def cost_probs(self):
        ar_data = self.df
        st_devs = self.st_devs
        cost_df = predictIt(predictit_df).gen_538_cost_df()
        most_recent_ar = self.most_recent_ar

        end_date = cost_df['dateEnd'].unique()
        end_date = pd.to_datetime(end_date[0]).date()
        current_date = ar_data['ds'].iloc[0].date()

        days_off = end_date - current_date
        datediff_index = days_off.days - 1

        if datediff_index < 0:
            raise ValueError("""It is the day of the market close so don't use the model""")

        sd_to_use = st_devs.iloc[datediff_index]
        
        dist = scipy.stats.norm(most_recent_ar, sd_to_use)

        cost_df['actual_lower'] = cost_df['lower'] - .05
        cost_df['actual_upper'] = cost_df['upper'] + .05

        cost_df['prob'] = self.cdf_probs(dist, lower = cost_df['actual_lower'], upper = cost_df['actual_upper'])

        #cost_df['prob'].apply(lambda x: round(x, 4))

        cd = cost_df.drop(['actual_lower', 'actual_upper', 'dateEnd'], axis = 1)
        cd['yes_margin'] = cd['prob'] - cd['real_yes_cost']
        cd['no_margin'] = (1-cd['prob']) - cd['real_no_cost']
        cd['prob'] = round(cd['prob'], 4)
        cd = cd[['lower', 'upper', 'prob', 'bestBuyYesCost', 'bestBuyNoCost', 'yes_margin', 'no_margin']]

        hours_since_update = round((datetime.datetime.now() - pd.to_datetime(ar_data['timestamp'][0])).seconds / (60*60), 1)

        print(str(days_off.days) + ' days out, current rating is ' + str(round(most_recent_ar, 2)) + ', last update posted ' + str(hours_since_update) + ' hours ago at '+ ar_data['timestamp'][0])

        return(cd)
    
    @staticmethod
    def gen_shift(daily_std, days_until_end, num_sims):
        daily_change = np.random.normal(loc = 0, scale = daily_std, size = days_until_end * num_sims)

        return(daily_change)

    @staticmethod
    def hacky_sim_probs(sim_max, lower, upper):
        ans = list()
        for row in range(len(lower)):
            toappend = len([x for x in sim_max if x >= lower[row] and x < upper[row]])/len(sim_max)
            ans.append(toappend)
        
        return(ans)

        ## pandas series don't seem to play nice with this
        ##def draws(draw_from, lower, upper):

            # keeps = [item for item in draw_from if item < upper and item > lower]
            # ans = len(keeps)/len(draw_from)
            # return(ans)

        # max_df['aldskfj'] = draws(draw_from, max_df['lower'], max_df['upper']) - 'truth value of a Series is ambiguous'

    def max_ar_probs(self, method = 'simulation', num_sims = 1000):
        ## setup section
        df = self.df
        daily_std = self.st_devs['error'][0]
        today = self.df['ds'].iloc[0]
        current_ar = self.most_recent_ar
        days_until_end = (datetime.datetime(2020, 5, 31) - today).days + 1
        max_df = predictIt(predictit_df).gen_538_max_ar_df()
        most_recent_ar = self.most_recent_ar

        ## getting dates for the string at the end

        end_date = max_df['dateEnd'].unique()
        end_date = pd.to_datetime(end_date[0]).date()
        current_date = df['ds'].iloc[0].date()

        days_off = end_date - current_date
        datediff_index = days_off.days - 1

        ## version 1: simulate assuming random walk with sd = sd of series each day
        if method == 'simulation':
            ## generating daily change

            daily_change = self.gen_shift(daily_std, days_until_end, num_sims)

            ## list of sim numbers to append
            sims = list(range(1,num_sims + 1))
            sim_tracker = [item for item in sims for i in range(days_until_end)]

            ## list of dates to append
            dates = pd.date_range(today, periods = days_until_end).to_pydatetime().tolist()
            date_tracker = [date for i in range(len(sims)) for date in dates]

            ## beginning the df
            z = pd.DataFrame(daily_change, columns=['daily_change'])
            z['sim_number'] = sim_tracker
            z['net_shift'] = z.groupby('sim_number')['daily_change'].cumsum()
            z['simulated_ar'] = z['net_shift'] + current_ar
            z['date'] = date_tracker

            simulated_maxes = z.groupby('sim_number')['simulated_ar'].max().sort_values()

            highest_so_far = ar_538[ar_538['ds'] > datetime.datetime(2020, 3, 31)]['y'].max()

            ## finding the maxes
            sim_max = [elem if elem > highest_so_far else highest_so_far for elem in simulated_maxes]

            ## adding the probabilities to the df
            max_df['prob']  = self.hacky_sim_probs(sim_max, lower = max_df['lower'] - .05, upper = max_df['upper'] + .05)
        else:
            rolling_max = df['y'].rolling(window = days_until_end).max()

            diff = rolling_max - df['y']

            draw_from = diff.dropna().reset_index(drop = True) + most_recent_ar

            best_so_far = df[df['ds'] >= datetime.datetime(2020, 3, 31)]['y'].max()

            # for elem in draw_from:
            #     if elem < best_so_far:
            #         elem = best_so_far
            #     else:
            #         elem = elem
            ## CODY CHANGE ABOVE TO THIS

            draw_from = [elem if elem > best_so_far else best_so_far for elem in draw_from]

            max_df['prob'] = self.hacky_sim_probs(draw_from, max_df['lower'], max_df['upper'])

        ## cleaning
        cd = max_df.drop(['dateEnd'], axis = 1)
        cd['yes_margin'] = cd['prob'] - cd['real_yes_cost']
        cd['no_margin'] = (1-cd['prob']) - cd['real_no_cost']
        cd['prob'] = round(cd['prob'], 4)
        cd = cd[['lower', 'upper', 'prob', 'bestBuyYesCost', 'bestBuyNoCost', 'yes_margin', 'no_margin']]

        ## printing the string
        hours_since_update = round((datetime.datetime.now() - pd.to_datetime(df['timestamp'][0])).seconds / (60*60), 1)

        print(method + ', ' + str(days_off.days) + ' days out, current rating is ' + str(round(most_recent_ar, 2)) + ', last update posted ' + str(hours_since_update) + ' hours ago at '+ df['timestamp'][0])

        ## return
        return(cd)

# approval_538(ar_538).cost_probs()

# approval_538(ar_538).max_ar_probs(method = 'simulation')
# approval_538(ar_538).max_ar_probs(method = 'historical draws')

## as of 4/7/20, seems like the markets strongly expect a mean reversion, which these tests (obviously) do not account for. 

## INSERT THIS INTO FUNCTION - ADD ANOTHER OPTION, OR COMBINE THIS WITH THE ABOVE? 
## finds the highest value <days_until_end> out


# ## ADF test

# import statsmodels.tsa.stattools as ts

# for_adf = ar_538.iloc[::-1]
# for_adf

# ts.adfuller(for_adf['y'])  # can't reject the null - it's stationary.

## is it mean-reverting, trending, or random walk? Use hurst exponent - 0 = mean reverting, .5 = random walk, 1 = trending.

def hurst(ts, lags):
    # calculate standard deviation of differenced series using various lags
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # # plot on log-log scale
    # plot(log(lags), log(tau)); show()
    # # calculate Hurst as slope of log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2.0
    return(hurst)

# ## testing
# ts = for_adf['y'].tolist()

# hurst(ts = ts, lags = range(1, 500)) # over long term: mean reverting

# hurst(ts = ts, lags = range(1, 58)) # over medium term: a bit less mean reverting

# hurst(ts = ts, lags = range(1, 7)) # over short term: basically random walk

## idea for mean reversion: do a regression where there is a mean up to that point term + mean of last 90 days + lag1

# TO DO: COMPARE THE LAG PREDICTIONS TO A MODEL WITH VARIOUS LAGS + 30 DAY, 90 DAY, AND LIFETIME AVERAGES - BREAK OUT TESTING INTO A DIFFERENT FILE SO I CAN TEST OUT SCIKIT-LEARN random forest/lm




## max 538 AR by 5/31:
## arguments: end date, num_sims


## make the function: 1) change the below bits to self, 2) get the holder df from the first class, 3) add the probabilities in, 4) find out the margin yes/no vs the true probability, 5) do the same for the second version, (optional) 6) combine these two together to compare them. 




### another idea - find the average increase in max from current over rolling periods since the start of the series. Might be a better idea than using the random walk because there are ceilings/floors to support. 

## the problem: I don't think trump's approval follows a random walk over the long term - I expect it is pulled back and oscillates around a mean. 




paired_outcomes = {'ia_nh_pair':[14214, 19366],
                    'nv_bernie_nom':[18064, 7725],
                    'nv_bernie_pres':[18035, 7941],
                    'nh_bernie_nom':[18066, 7725],
                    'nh_bernie_pres':[18034, 7941],
                    'ca_bernie_nom':[18062, 7725],
                    'ca_bernie_pres':[18038, 7941],
                    'sc_biden_nom':[18063, 7729],
                    'sc_biden_pres':[18036, 7940],
                    'ma_biden_nom':[18061, 7729],
                    'ma_biden_pres':[18039, 7940],
                    'tx_biden_nom':[18065, 7729],
                    'tx_biden_pres':[18037, 7940],
}

# predictIt(predictit_df).paired_outcome_arbitrage(paired_outcomes)


 ####### BALANCE OF POWER IN GOV WORKING HERE #######

# dem house = 10461
# r house = 10463
# r senate = 10464
# d senate = 10462
# d house r senate = 10414
# r house r senate = 10415
# r house d senate = 10416
# d house d senate 10413


# ## this is a df that can optionally be fed into the other function
# d_house2 = df[df['id'].isin([10414, 10413])]
# dh_yes = d_house2['bestBuyYesCost'].sum()
# dh_no = d_house2['bestBuyNoCost'].sum()

# dh_yes2 = df[df['id']==10461]['bestBuyYesCost']
# dh_no2 = df[df['id']==10461]['bestBuyNoCost']


# df[df['id']==10460]

# d_senate2 = df[df['id'].isin([10416, 10413])]
# ds_yes = d_senate2['bestBuyYesCost'].sum()
# ds_no = d_senate2['bestBuyNoCost'].sum()

# r_senate2 = df[df['id'].isin([10414, 10415])]
# rh_yes = r_senate2['bestBuyYesCost'].sum()
# rh_no = r_senate2['bestBuyNoCost'].sum()

# r_house2 = df[df['id'].isin([10415, 10416])]
# rh_yes = r_house2['bestBuyYesCost'].sum()
# rh_no = r_house2['bestBuyNoCost'].sum()


# d_house
# dh_yes

## pick up later: take d house yes on one + no on another, created paired outcomes like in the other function, run through that calc, append to bottom of other one


# ca nom = 18062
# ca prez = 18038

## list every contract to help looking for paired outcomes

# ## helper thing for figuring out IDs
# intermed = df[df['contract_id'].isin([3633, 5883, 3698, 5973, 5989, 5967, 5963, 5972, 5958, 5959, 5970, 5969, 5960, 5971, 5961])].sort_values(['name'], ascending = False)[['id', 'shortName', 'contract_name']]

# def figure_out_ids(intermed):
#     for i in range(len(intermed['shortName'])):
#         print(intermed['shortName'][i], intermed['contract_name'][i], )

# for i in range(len(intermed['shortName'])):
#     print(intermed['shortName'][i], intermed['contract_name'][i], )

# intermed[intermed['shortName'].str.contains('Biden')]

# figure_out_ids(intermed)

def notes(): ## stuffed this here so I can find it in the outline

    ## possible duplicates:
    ## how many dems running for prez april 1 vs individuals drop out
    ## will nominee be woman vs nominee
    ## woman VP + VP
    ## balance of power, house senate
    ## dem nominee 70+
    ## gender comp of 2020 dem ticket
    ## which candidate will drop out
    ## second in x primary vs win primary? 
    ## iowa win NH vs iowa winner 6167 vs ___
    ## woman elected vp vs VP odds
    ## woman elected prez vs prez odds
    ## pledged delegates
    # compare brokerred conv to 538

    # ## 5973 Will the 2020 IA Democratic caucus winner win the nomination?
    # 5989 Will the 2020 IA Democratic caucus winner win the presidency?
    # 5967 Will the 2020 MA Democratic primary winner win the nomination?
    # 5963 Will the 2020 MA Democratic primary winner win the presidency?
    # 5972 Will the 2020 NH Democratic primary winner win the nomination?
    # 5958 Will the 2020 NH Democratic primary winner win the presidency?
    # 5970 Will the 2020 NV Democratic caucuses winner win the nomination?
    # 5959 Will the 2020 NV Democratic caucuses winner win the presidency?
    # 5969 Will the 2020 SC Democratic primary winner win the nomination?
    # 5960 Will the 2020 SC Democratic primary winner win the presidency?
    # 5971 Will the 2020 TX Democratic primary winner win the nomination?
    # 5961 Will the 2020 TX Democratic primary winner win the presidency?
    pass 
