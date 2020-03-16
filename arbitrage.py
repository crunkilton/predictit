######### NOTES ##########
## agenda
# 0: check the arbitrage finder - probably isn't accurate
# 1: tidy what is here
# 2: link to 538 on markets, see diff
# 3: (maybe) link to betfair prices
# 4: make tweet count models seaparately

## new projects: check arb for linked items (i.e. will winner of x primary win nom)
## make function accept list
## give range of outcomes for ror_arbitrage, check if it hits all
## return arb opportunties to pretty table
## check vs 538


## predictit and other arbitrage opportunities- notes
## 1) calculate what the actual odds/profit are for each given costs
## 2) check all yes/all no and compare to investing in stocks
## misc: if implied probability in one market is higher than another, how to get there
## pull prices via API


## look at different websites (i.e. betfair, another one, the iowa thing)

## look at non-political arbitrage

## net value


########## THE FILE ##########

import pandas as pd 
import numpy as np 
import requests
import json
import datetime

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

df = pull_predict_it()

predictIt(df).arbitrage_roi()


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

# sc prez = 18036
# sc nom = 18063
# biden prez = 7940
# biden nom = 7729

predictIt(df).paired_outcome_arbitrage(paired_outcomes)


def list_contests(df):
    uqs = df.groupby(['contract_name', 'contract_id']).size().reset_index().rename(columns={0:'count'})

    uqs = uqs.sort_values(by = ['contract_name'])

    for row in range(len(uqs)):
        print(uqs['contract_id'][row], uqs['contract_name'][row])

list_contests(df[df['contract_name'].str.contains('House')])
#4353
#4365
#4366


 ####### BALANCE OF POWER IN GOV WORKING HERE #######

# dem house = 10461
# r house = 10463
# r senate = 10464
# d senate = 10462
# d house r senate = 10414
# r house r senate = 10415
# r house d senate = 10416
# d house d senate 10413


## this is a df that can optionally be fed into the other function
d_house2 = df[df['id'].isin([10414, 10413])]
dh_yes = d_house2['bestBuyYesCost'].sum()
dh_no = d_house2['bestBuyNoCost'].sum()

dh_yes2 = df[df['id']==10461]['bestBuyYesCost']
dh_no2 = df[df['id']==10461]['bestBuyNoCost']


df[df['id']==10460]

d_senate2 = df[df['id'].isin([10416, 10413])]
ds_yes = d_senate2['bestBuyYesCost'].sum()
ds_no = d_senate2['bestBuyNoCost'].sum()

r_senate2 = df[df['id'].isin([10414, 10415])]
rh_yes = r_senate2['bestBuyYesCost'].sum()
rh_no = r_senate2['bestBuyNoCost'].sum()

r_house2 = df[df['id'].isin([10415, 10416])]
rh_yes = r_house2['bestBuyYesCost'].sum()
rh_no = r_house2['bestBuyNoCost'].sum()


d_house
dh_yes

## pick up later: take d house yes on one + no on another, created paired outcomes like in the other function, run through that calc, append to bottom of other one

df[df['contract_id'].isin([5967, 5963, 5971, 5961])]

# ca nom = 18062
# ca prez = 18038

## list every contract to help looking for paired outcomes

## helper thing for figuring out IDs
intermed = df[df['contract_id'].isin([3633, 5883, 3698, 5973, 5989, 5967, 5963, 5972, 5958, 5959, 5970, 5969, 5960, 5971, 5961])].sort_values(['name'], ascending = False)[['id', 'shortName', 'contract_name']]

def figure_out_ids(intermed):
    for i in range(len(intermed['shortName'])):
        print(intermed['shortName'][i], intermed['contract_name'][i], )

for i in range(len(intermed['shortName'])):
    print(intermed['shortName'][i], intermed['contract_name'][i], )

intermed[intermed['shortName'].str.contains('Biden')]

figure_out_ids(intermed)

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
