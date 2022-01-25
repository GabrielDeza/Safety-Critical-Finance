import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st

def greedy_gain(period_dates, pred, target,df_open,df_close,mu,std,freq_type ='day'):
    if freq_type =='day':
        freq_str = '%Y-%m-%d'
    if freq_type =='hour':
        freq_str = '%Y-%m-%d %H:%M:%S'
    returns = []
    amount = 10000

    for i in range(period_dates.shape[0]):
        c = df_close[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        o = df_open[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        if mu[i] > 0:
            new_amount = (amount/o) * c
            gain = ((new_amount - amount)/amount) * 100
            amount = new_amount
            returns.append(gain)

    init_investement = 10000
    net_gain_percent = ((amount - 10000)/10000) *100
    percent_days_traded = 100 * (len(returns) / len(period_dates))
    return net_gain_percent,init_investement, returns,percent_days_traded

def threshold_gain(period_dates, pred, target,df_open,df_close,mu,std,freq_type ='day'):
    if freq_type =='day':
        freq_str = '%Y-%m-%d'
    if freq_type =='hour':
        freq_str = '%Y-%m-%d %H:%M:%S'
    k =5
    returns = []
    amount = 10000
    for i in range(k,len(period_dates)):
        #get the past k true "diff" values (ie: Open - Close)
        #If my prediction is much stronger than the average of my predictions in the past
        # several days AND the confidence is less than the average of my prediction in the past several days, trade
        past_mu = mu[i-k:i-1]
        past_std = std[i - k:i - 1]
        std_dev = np.mean(past_std)
        mu_to_beat = np.mean(past_mu)

        c = df_close[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        o = df_open[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        if pred[i] >= mu_to_beat and pred[i] > 0 and std[i] < std_dev:
            new_amount = (amount / o) * c
            gain = ((new_amount - amount) / amount) * 100
            amount = new_amount
            returns.append(gain)
    net_gain_percent = ((amount - 10000) / 10000) * 100
    percent_days_traded = 100 * (len(returns) / len(period_dates))
    return net_gain_percent, returns,percent_days_traded

def avg_list(a):
    #return a.mean()
    return a[a > 0].mean()
def std(a):
    #return np.std(a)
    return np.std(a[a > 0])

def kelly_greedy_gain(period_dates, pred, target,df_open,df_close,mu,std,freq_type ='day'):
    f_list = []
    if freq_type =='day':
        freq_str = '%Y-%m-%d'
    if freq_type =='hour':
        freq_str = '%Y-%m-%d %H:%M:%S'
    returns = []
    amount = 10000
    for i in range(period_dates.shape[0]):
        c = df_close[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        o = df_open[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        if mu[i] > 0:
            f = mu[i] / (std[i] +0.00001)
            if f >= 1.0:
                f = 1.0
            if f <= 0.0:
                f = 0.001
            if math.isnan(f):
                f = 1.0
            new_amount = (((f*amount)/o) * c) + (1-f)*amount
            gain = ((new_amount - amount) / amount) * 100
            amount = new_amount
            returns.append(gain)
            f_list.append(f)
    init_investement = 10000
    net_gain_percent = ((amount - 10000)/10000) *100
    percent_days_traded = 100 * (len(returns) / len(period_dates))
    f = sum(f_list)/len(f_list)
    return net_gain_percent,init_investement, returns,percent_days_traded,f_list

def kelly_threshold_gain(period_dates, pred, target,df_open,df_close,mu,std,freq_type ='day'):
    if freq_type =='day':
        freq_str = '%Y-%m-%d'
    if freq_type =='hour':
        freq_str = '%Y-%m-%d %H:%M:%S'
    k =5
    f_list = []
    returns = []
    amount = 10000
    for i in range(k,len(period_dates)):
        #get the past k true "diff" values (ie: Open - Close)
        past_mu = mu[i - k:i - 1]
        past_std = std[i - k:i - 1]
        std_dev = np.mean(past_std)
        mu_to_beat = np.mean(past_mu)
        c = df_close[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        o = df_open[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        if mu[i] >= mu_to_beat and mu[i] > 0 and std[i] < std_dev:
            f = mu[i] / (std[i] +0.00001)
            if f >= 1.0:
                f = 1
            if f <= 0.0:
                f = 0.001
            if math.isnan(f):
                f = 1.0
            new_amount = (((f * amount) / o) * c) + ((1 - f) * amount)
            gain = ((new_amount - amount) / amount) * 100
            amount = new_amount
            returns.append(gain)
            f_list.append(f)
    net_gain_percent = ((amount - 10000) / 10000) * 100
    percent_days_traded = 100 * (len(returns) / len(period_dates))
    return net_gain_percent, returns,percent_days_traded



def plot_return_distributions(returns,PDT,f_dict,show_figs =True,save_figs= False,save_path = '.'):
    fig,ax = plt.subplots(nrows=4, ncols=2,figsize=(16, 10),sharex = 'col',sharey='col')
    for j,dataset in enumerate(['Training', 'Testing']):
        for i,strategy in enumerate(['Greedy','Kelly Greedy','Threshold','Kelly Threshold']):
            if dataset == 'Training':
                x = np.array(returns[dataset][strategy])
                if len(x)!=0:
                    bins = 30
                    ax[i,j].hist(x, density=True, bins=bins,zorder=2,label ='Returns')
                    ax[i, j].axvline(x=sum(x)/len(x),zorder=1,ymin = 0,ymax=1, label='Mean')
                ax[i, j].legend(loc="upper left")
                ax[i,j].set_ylabel('Percentage',fontsize=9)
                if i == 3:
                    ax[i,j].set_xlabel('Returns',fontsize = 9)
                med = np.median(x)
                kurt = st.kurtosis(x)
                skew = st.skew(x)
                ax[i,j].set_title(f"{strategy} Strategy on {dataset} \n "
                                  f"Days Traded = {round(PDT[dataset][strategy],1)} Mean ={round(sum(x)/len(x),2)} Median = {round(med,2)} Kurtosis = {round(kurt,2)} Skew = {round(skew,2)} %",fontsize=9.5)
            if dataset == 'Testing':
                x1 = np.array(returns[dataset][strategy])
                x2 = np.array(returns['Adversarial'][strategy])
                if len(x1)!=0:
                    bins = 30
                    ax[i,j].hist(x1, density=True, bins=bins,alpha =0.5,zorder=2,color='b',label='Predicted')
                    ax[i,j].hist(x2, density=True, bins=bins, alpha=0.5,zorder=2,color='r', label='Manipulated')
                    ax[i, j].axvline(x=sum(x1)/len(x1),color='b',ymin = 0,ymax=1,zorder=1, label='Mean')
                    ax[i, j].axvline(x=sum(x2)/len(x2),color='r',ymin = 0,ymax=1,zorder=1, label='Mean')
                ax[i,j].set_ylabel('Percentage',fontsize=9)
                ax[i, j].legend(loc="upper left",fontsize=7.5)
                if i == 3:
                    ax[i,j].set_xlabel('Returns',fontsize = 9)
                med1 = np.median(x1)
                med2 = np.median(x2)
                kurt1 = st.kurtosis(x1)
                kurt2 = st.kurtosis(x2)
                skew1 = st.skew(x1)
                skew2 = st.skew(x2)
                if 'Kelly' in strategy:
                    if len(x1)!=0 and len(x2) !=0:
                        ax[i,j].set_title(f"{strategy} Strategy on {dataset} \n  "
                                  f"Reg: Days Traded = {round(PDT[dataset][strategy],1)}% Mean ={round(sum(x1)/len(x1),2)} Median = {round(med1,2)} Kurtosis = {round(kurt1,2)} Skew = {round(skew1,2)} \n"
                                  f"Adv Days Traded = {round(PDT['Adversarial'][strategy],1)}% Mean ={round(sum(x2)/len(x2),2)} Median = {round(med2,2)} Kurtosis = {round(kurt2,2)} Skew = {round(skew2,2)}",fontsize=11)
                else:
                    if len(x1)!=0 and len(x2) !=0:
                        ax[i, j].set_title(f"{strategy} Strategy on {dataset} \n  "
                                       f"Reg: Days Traded = {round(PDT[dataset][strategy], 1)}% Mean ={round(sum(x1)/len(x1), 2)} Median = {round(med1, 2)} Kurtosis = {round(kurt1, 2)} Skew = {round(skew1, 2)} \n"
                                       f"Adv Days Traded = {round(PDT['Adversarial'][strategy], 1)}% Mean ={round(sum(x2)/len(x2), 2)} Median = {round(med2, 2)} Kurtosis = {round(kurt2, 2)} Skew = {round(skew2, 2)}",
                                        fontsize=11)
            ax[i,j].xaxis.set_tick_params(labelbottom=True)
    fig.subplots_adjust(hspace=0.7)
    fig.subplots_adjust(wspace=0.3)

    if save_figs:
        plt.savefig(save_path, dpi=100)
    if show_figs:
        plt.show()
    else:
        plt.show()
