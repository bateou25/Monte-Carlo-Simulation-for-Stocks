# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 14:16:06 2015

@author: benj29
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')#set white background for visualization plots
%matplotlib inline
from __future__ import division

dframe = pd.read_csv("singleDataFile.csv",header=None)
dframe.head()
dframe2 = dframe.drop(7,axis=1)#drop last column
dframe2.head()
#rename columns
dframe2.columns = ['Stock','Date','Open','High','Low','Close','Volume']
#filter Stock column to select stocks
AC_df = dframe2[dframe2['Stock']=='AC']
GTCAP_df = dframe2[dframe2['Stock']=='GTCAP']
AEV_df = dframe2[dframe2['Stock']=='AEV']
#combine filtered columns to one dataframe
initial_comb = AC_df.combine_first(GTCAP_df)
myport_df = initial_comb.combine_first(AEV_df)
myport_df.info()
#convert date to datetime format
myport_df['Date'] = pd.to_datetime(myport_df['Date'])
myport_df.info()
#sort the Date column
myport_df.sort('Date',ascending=True,inplace=True)
myport_df.head()
myport_Close = myport_df.pivot('Date','Stock','Close')#pivot for Closing Price
myport_Close.head()
myport_rets = myport_Close.pct_change()#closing
myport_rets.head()
#historical view of the daily returns
myport_rets['AC'].plot(figsize=(10,4),legend=True)
myport_rets['GTCAP'].plot(figsize=(10,4),legend=True)
myport_rets['AEV'].plot(figsize=(10,4),legend=True)

########VOLUME PLOTS#############
#create plots for monthly volume
#AC
AC_vol = AC_df.drop(['Open','High','Low','Close'],axis=1)
AC_vol['Date'] = pd.to_datetime(AC_vol['Date'])
#aggregate to monthly volume
AC_vol_comb = AC_vol.set_index('Date'). \
                groupby('Stock').resample('M', how='sum')
#convert back to dataframe                
AC_vol_comb = AC_vol_comb.reset_index().reindex(columns=['Date','Volume'])
AC_vol_comb['Month'] = AC_vol_comb['Date'].dt.month#create col for Month                
sns.barplot("Month", y="Volume",data=AC_vol_comb,
            palette="BuGn_d")
#create plots for monthly volume
#GTCAP
GTCAP_vol = GTCAP_df.drop(['Open','High','Low','Close'],axis=1)
GTCAP_vol['Date'] = pd.to_datetime(GTCAP_vol['Date'])
#aggregate to monthly volume
GTCAP_vol_comb = GTCAP_vol.set_index('Date'). \
                groupby('Stock').resample('M', how='sum')
#convert back to dataframe                
GTCAP_vol_comb = GTCAP_vol_comb.reset_index().reindex(columns=['Date','Volume'])
GTCAP_vol_comb['Month'] = GTCAP_vol_comb['Date'].dt.month#create col for Month                
sns.barplot("Month", y="Volume",data=GTCAP_vol_comb,
            palette="BuGn_d")               
#create plots for monthly volume
#AEV
AEV_vol = AEV_df.drop(['Open','High','Low','Close'],axis=1)
AEV_vol['Date'] = pd.to_datetime(AEV_vol['Date'])
#aggregate to monthly volume
AEV_vol_comb = AEV_vol.set_index('Date'). \
                groupby('Stock').resample('M', how='sum')
#convert back to dataframe                
AEV_vol_comb = AEV_vol_comb.reset_index().reindex(columns=['Date','Volume'])
AEV_vol_comb['Month'] = AEV_vol_comb['Date'].dt.month#create col for Month                
sns.barplot("Month", y="Volume",data=AEV_vol_comb,
            palette="BuGn_d")    

#loop through different stocks to compare each other using seaborn
sns.pairplot(myport_rets.dropna())
#correlation bet. closing prices of all stock tickers
returns_fig = sns.PairGrid(myport_Close.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
#correlation plot bet. daily returns of all stock tickers
sns.corrplot(myport_rets.dropna(),annot=True)

#sns.jointplot(myport_rets['URC'],myport_rets['CEB'])#joint plot of both datasets
#sns.jointplot(myport_rets['URC'],myport_rets['CEB'],kind='hex')#plot using hex
#sns.jointplot(myport_rets['URC'],myport_rets['JGS'])#joint plot of both datasets
#sns.jointplot(myport_rets['URC'],myport_rets['JGS'],kind='hex')#plot using hex
#sns.jointplot(myport_rets['JGS'],myport_rets['CEB'])#joint plot of both datasets
#sns.jointplot(myport_rets['JGS'],myport_rets['CEB'],kind='hex')#plot using hex

#correlation plot bet. closing prices of all stock tickers
sns.corrplot(myport_Close.dropna(),annot=True)
#######################################
# RISK ANALYSIS
# (A) There are many ways we can quantify risk, one of the most basic ways
# using the info. we've gathered on daily percentage returns is by
# comparing the expected return with the standard deviation of the
# daily returns
mp_rets = myport_rets.dropna()
mp_rets.info()

area = np.pi*20 #define the area of the circle
plt.scatter(mp_rets.mean(),mp_rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
for label,x,y in zip(mp_rets.columns,mp_rets.mean(),mp_rets.std()):
    plt.annotate(
        label,
        xy = (x,y), xytext = (50,50),
        textcoords = 'offset points', ha='right', va='bottom',
        arrowprops = dict(arrowstyle='-',connectionstyle='arc3,rad=-0.3'))
# (B) VALUE AT RISK
# Define a value at risk parameter for our stocks.
# We can treat value at risk as the amount of money we could expect
# to lose (aka putting at risk) for a given confidence interval.

# (Ba) Using the "bootstrap"method
# for this method we will calculate the empirical quantiles from
# a histogram of daily returns.
#average daily return
sns.distplot(myport_rets['AEV'].dropna(),color='purple')

#test to get quantile of daily returns
#i.e the 0.05 empirical quantile of daily returns is at x %,
#where x=quantile(0.05). That means that with 95% confidence,
#(i.e. 95% of the times) our worst daily loss will not exceed x%.
#If we have a 1 milliondollar investment, our one-day % VaR
#is x*1,000,000 = $y
myport_rets['AEV'].quantile(0.01)

# (Bb) Using the Monte Carlo method
# using the method to run many trials with random market conditions,
# we will calculate portfolio losses for each trial.
# After this, we'll use the aggregation of all these simulations
# to establish how risky the stock is.
# We will use the Markov process.
# This means that the past info. on the price of a stock is
# independent of where the stock price will be in the future.
# Basically meaning, you can't perfectly predict the future solely
# based on the previous price of a stock.

days = 365
dt = 1/days
mu = mp_rets.mean()['AEV']#average of daily return
sigma = mp_rets.std()['AEV']#i.e. volatility of the stock
def stock_monte_carlo(start_price,days,mu,sigma):
    price = np.zeros(days)#define price array
    price[0] = start_price#first term will be the starting price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in xrange(1,days):
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))
    return price        
#test function
myport_df.pivot('Date','Stock','Open')    
start_price =  56.50#opening price as of 20150604
for run in xrange(100):#run monte carlo x times
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for AEV')

#return a histogram of all the final prices
runs = 10000
simulations= np.zeros(runs)
for run in xrange(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
#define q as the 1% empirical quantile
q = np.percentile(simulations,1)#99% of the values should fit w/in the output
plt.hist(simulations,bins=200)
plt.figtext(0.6,0.8,s='Start price: Php%.2f' %start_price)#starting price
#mean ending price
plt.figtext(0.6,0.7,'Mean final price: Php%.2f' % simulations.mean())
#variance of the price(within 99% confidence interval)
plt.figtext(0.6,0.6,'VaR(0.99): Php%.2f' % (start_price - q,))
plt.figtext(0.15,0.6,'q(0.99): Php%.2f' % q)#display 1% quantile
plt.axvline(x=q,linewidth=4,color='r')#plot a line at the 1% quantile result
#title
plt.title(u'Final price distribution for AEV Stock after %s days' %days,
          weight = 'bold') 