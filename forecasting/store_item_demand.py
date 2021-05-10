# General imports
from IPython.display import display, HTML, Markdown
import math 
from datetime import datetime
from datetime import date
import pickle
import random 


# General data analysis imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Facebook Prophet
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

# for Hierarchical models
import hts
from hts.hierarchy import HierarchyTree


# My own code
import forecasting.bplot as bplot

###################################################################
#  _____                                            _             
# |  __ \                                          (_)            
# | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
# |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
# | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
# |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                | |                                         __/ |
#                |_|                                        |___/ 
####################################################################


def preprocess_data(file='data/train.csv'):
    df = pd.read_csv(file, parse_dates=['date'])
    df = df.astype({'store':str, 'item':str})
    
    summary = """
# no null entries
# 10 stores
# 50 items
# 50*10=500 separate time series, one for each store/item combination
# 1826 days = 5 years
# 1826*500 = 913'000 rows of sales data
    """
    print(summary)
    
    # ID for store_item combinations 
    df['store_item'] = df.apply(lambda x: f"{x['store']}_{x['item']}", axis=1)
    
    # bottom level
    df_bottom_level = df.pivot(index='date', columns='store_item', values='sales')
    #display(df_bottom_level.head(3))
    
    # middle level
    df_middle_level = df.groupby(['date', 'store']) \
                    .sum() \
                    .reset_index() \
                    .pivot(index='date', columns='store', values='sales')
    
    # total root level
    df_total = df.groupby('date')['sales'] \
                .sum() \
                .to_frame() \
                .rename(columns={"sales": "total"})
    
    
    # hierarchy df
    df_h = df_bottom_level.join(df_middle_level).join(df_total)

    # This is necessary because HTS expects index to have 'freq' set
    df_h = df_h.resample("D").sum()
    
    print(f"Number of time series at the bottom level: {df_bottom_level.shape[1]}")
    print(f"Number of time series at the middle level: {df_middle_level.shape[1]}")
    print(f"Number of time series at the top level   : {df_total.shape[1]}")

    # creating hierarchy
    stores = df['store'].unique()
    store_items = df['store_item'].unique()
    total = {'total':list(stores)}
    store = {k: [i for i in store_items if i.startswith(str(k)+'_')] for k in stores}
    hierarchy = {**total, **store}
    
    return df, df_h, hierarchy



#####################################################################################################
#  ______            _                 _                                          _           _     
# |  ____|          | |               | |                       /\               | |         (_)    
# | |__  __  ___ __ | | ___  _ __ __ _| |_ ___  _ __ _   _     /  \   _ __   __ _| |_   _ ___ _ ___ 
# |  __| \ \/ / '_ \| |/ _ \| '__/ _` | __/ _ \| '__| | | |   / /\ \ | '_ \ / _` | | | | / __| / __|
# | |____ >  <| |_) | | (_) | | | (_| | || (_) | |  | |_| |  / ____ \| | | | (_| | | |_| \__ \ \__ \
# |______/_/\_\ .__/|_|\___/|_|  \__,_|\__\___/|_|   \__, | /_/    \_\_| |_|\__,_|_|\__, |___/_|___/
#             | |                                     __/ |                          __/ |          
#             |_|                                    |___/                          |___/           
#####################################################################################################

def do_share_analysis(df_h, list_cols, n_sample=None):
    
    if(n_sample is not None):
        list_cols = random.sample(list_cols,n_sample)
    n_cols = len(list_cols)
    
    df_month = df_h.resample('MS', closed='left', label='left').sum()
    df_week  = df_h.resample('W-MON', closed='left', label='left').sum()
    
    df_share = df_month[list_cols].divide(df_month['total'], axis=0)*100
    ax = df_share.plot(title='Share of total sales - Monthly')
    ax.legend(bbox_to_anchor=(1.0, 1.0));

    df_share = df_week[list_cols].divide(df_week['total'], axis=0)*100
    ax = df_share.plot(title='Share of total sales - Weekly')
    ax.legend(bbox_to_anchor=(1.0, 1.0));

    df_share = df_h[list_cols].divide(df_h['total'], axis=0)*100
    ax = df_share.plot(title='Share of total sales - Daily')
    ax.legend(bbox_to_anchor=(1.0, 1.0));

    
    df_share = df_h[list_cols].divide(df_h['total'], axis=0)
    share_dict = df_share.mean(axis=0).to_dict()

    res_list = []
    for c in df_share.columns:
        res = df_share[c] - share_dict[c]
        res_list.append(res)
    df_res = pd.concat(res_list, axis=1)

    n_y_subplots = math.ceil(n_cols/2)
    
    fig, axs = plt.subplots(n_y_subplots, 2)
    fig.tight_layout()
    ix = 0
    for x in range(2):
        for y in range(n_y_subplots):
            ix_store_item = list_cols[ix]
            yvals, xvals, _ = axs[y,x].hist(df_res[ix_store_item], bins=30)
            ymax = yvals.max()*1.1
            xmean = df_res[ix_store_item].mean()
            axs[y,x].axvline(x=xmean, ymin=0, ymax=ymax, color='red')
            axs[y,x].title.set_text(f'store-item {ix_store_item}')
            ix +=1
   
    fig, axs = plt.subplots(1)
    ix_store_item = list_cols[0]
    plot_acf(df_res[ix_store_item], ax=axs, zero=False, lags=600, alpha=0.05,
             title=f'Autocorrelation {ix_store_item}')
    
    fig, axs = plt.subplots(n_cols)
    fig.tight_layout()
    ix = 0
    for y in range(n_cols):
        ix_store_item = list_cols[ix]
        plot_acf(df_res[ix_store_item], ax=axs[y], zero=False, lags=600, alpha=0.05,
                 title=f'Autocorrelation {ix_store_item}')
        ix +=1
        
    
    
    
## Verifying I can reproduce FbProphet implementation    
def mape(df_cv):
    df_cv['horizon'] = df_cv['ds']-df_cv['cutoff']
    df_cv['horizon'] = df_cv['horizon'].apply(lambda x: int(str(x).split()[0]))
    df_cv['mape_t'] = abs((df_cv['yhat'] - df_cv['y'])/df_cv['y'])
    df_for_mape = df_cv[['horizon', 'cutoff', 'mape_t']]

    # mean over the different cutoff for the same horizon
    df_for_mape2 = df_for_mape.groupby('horizon')['mape_t'].mean()

    # mean over rolling window for each horizon
    mapes_dict = {}
    n_days_window = 18
    for i in df_for_mape2.index:
        if((i-18)<0):
            continue
        slice_to_average = df_for_mape2[i-n_days_window:i]
        mapes_dict[i] = slice_to_average.mean()
    df_mape = pd.DataFrame.from_dict(mapes_dict, orient='index')
    df_mape.columns = ['mape']    
    return df_mape

    

        
#################################################
#  __  __           _      _ _             
# |  \/  |         | |    | (_)            
# | \  / | ___   __| | ___| |_ _ __   __ _ 
# | |\/| |/ _ \ / _` |/ _ \ | | '_ \ / _` |
# | |  | | (_) | (_| |  __/ | | | | | (_| |
# |_|  |_|\___/ \__,_|\___|_|_|_| |_|\__, |
#                                     __/ |
#                                    |___/ 
#################################################

### HIERARCHICAL TIME SERIES

## to drop ?? 
def calc_store_hier_preds(hierarchy, df_h, store_idx='1'):
    small_hier = {store_idx: hierarchy[store_idx]}
    small_df_cols = hierarchy[store_idx].copy()
    small_df_cols.append(store_idx)
    small_hdf = df_h[small_df_cols].copy()

    model_hts_prophet = hts.HTSRegressor(model='prophet', revision_method='OLS')
    model_hts_prophet = model_hts_prophet.fit(df=small_hdf, nodes=small_hier, 
                                              root=store_idx)
    pred_hts = model_hts_prophet.predict(steps_ahead=90)
    return pred_hts



def shf_split(df, n_years_min_training=3.5, horizon=90, n_max_splits=7):
    full_train_start = min(df.index)
    full_train_end   = max(df.index)
    min_train_end = full_train_start + pd.Timedelta(365*n_years_min_training, 'days')
    horizon_delta = pd.Timedelta(horizon, 'days')

    cutoffs = []
    cutoff = (full_train_end - horizon_delta)
    while(cutoff>min_train_end):
        cutoffs.append(cutoff)
        cutoff = cutoff - horizon_delta/2

    print (f"""
# full train start : '{full_train_start.date()}'
# full train end   : '{full_train_end.date()}'
# min train end    : '{min_train_end.date()}'
# horizon          :  {horizon} days
    """)
    
    if(n_max_splits is not None):
        #cutoffs = cutoffs[:n_max_splits]
        cutoffs = random.sample(cutoffs,n_max_splits)
    
    print(f'# number of cutoffs: {len(cutoffs)}')    
    splits = {}   
    for cutoff in cutoffs:
        print(str(cutoff.date()))
        df_shf_train = df[:cutoff]
        df_shf_val   = df[cutoff+pd.Timedelta(1, 'day'):cutoff+horizon_delta]
        splits[cutoff] = [df_shf_train, df_shf_val]
    
    return splits


## on_off_season_custom_seasonality
def is_on_season(ds):
    date = pd.to_datetime(ds)
    return ((date.month >= 3) & (date.month < 12))


def shf_update_modelX(modelX):
    
    def new_update_func(df_shf_train, df_shf_valid, store_items_list,
                        cutoff, H, deltas):
        m = prophet_model(modelX)
        shf_update_topdown_prophet(m,
                                   df_shf_train, df_shf_valid, store_items_list,
                                   cutoff, H, 
                                   deltas=deltas)
    return new_update_func
    

def do_all_shf_steps(splits, store_items_list, algo,
                     H=90, verbose=False):
    
    results = shf_loop(splits, store_items_list, algo=algo,
                       H=H, verbose=verbose)

    df_perf = eval_performance(results)

    mean_smape = df_perf.query(f'h_days==90')['smape'].mean()
    print(f'average SMAPE: {mean_smape:.3f}')
    ax = df_perf.query(f'h_days==90')['smape'].plot(kind='hist', bins=20, 
                                                    title=f'{algo} SMAPEs')  
    return results, df_perf


def shf_loop(shf_splits, store_items_list,
             algo='boris1', H=90, 
             verbose=False):
    
    start = datetime.now()
    cutoffs = list(shf_splits.keys())

    deltas = {}
    for store_item in store_items_list:
        deltas[store_item] = []
    
    for cutoff in cutoffs:
        if(verbose):
            print (f'# cutoff: {cutoff.date()}')
        df_shf_train = shf_splits[cutoff][0]
        df_shf_valid = shf_splits[cutoff][1]  

        # Use algo_mapping to call specific implementation of update method
        # for the specified algo
        algo_mapping[algo](df_shf_train, df_shf_valid, 
                           store_items_list,
                           cutoff, H,  
                           deltas)
       
    results = {}        
    for store_item in store_items_list:        
        results[store_item] = pd.concat(deltas[store_item], axis=0, ignore_index=True)
          
    end = datetime.now()
    time_delta = end - start
    if(verbose):
        print(f'# execution walltime: {str(time_delta)}')
    
    return results


## to keep
def shf_update_simple(df_shf_train, df_shf_valid, 
                      store_items_list,
                      cutoff, H, deltas, simple_type):
    ## This function updates dictionary deltas
    for store_item in store_items_list:    
        df_model_train = df_shf_train[store_item].to_frame().reset_index()
        df_model_valid = df_shf_valid[store_item].to_frame().reset_index()
        df_model_train.columns = ['ds', 'y']
        df_model_valid.columns = ['ds', 'y']

        if(simple_type=='naive'):
            pred_value = df_model_train['y'].iloc[-1]
        elif(simple_type=='average'):
            pred_value = df_model_train['y'].mean()
        else:
            raise Exception("simple_type should be eitehr 'naive' or 'average' ")
            
        
        df_item_valid = df_shf_valid[store_item].to_frame().reset_index()
        df_item_valid.columns = ['ds', 'y']
        
        df_comp = df_item_valid.copy().set_index('ds')
        df_comp['yhat'] = pred_value
        df_comp['cutoff'] = cutoff
        df_comp['h_days'] = (df_comp.index - cutoff).days
        df_comp['delta'] = df_comp['yhat'] - df_comp['y']
        deltas[store_item].append(df_comp.reset_index())

def shf_update_naive(df_shf_train, df_shf_valid, 
                      store_items_list, cutoff, H, deltas):
                            
    shf_update_simple(df_shf_train, df_shf_valid, 
                      store_items_list,cutoff, H, deltas, simple_type='naive')

def shf_update_average(df_shf_train, df_shf_valid, 
                      store_items_list, cutoff, H, deltas):
                            
    shf_update_simple(df_shf_train, df_shf_valid, 
                      store_items_list,cutoff, H, deltas, simple_type='average')

def shf_update_seasonal_naive(df_shf_train, df_shf_valid, 
                              store_items_list,
                              cutoff, H, deltas):
    ## This function updates dictionary deltas
    df_share = df_shf_train[store_items_list].divide(df_shf_train['total'], axis=0)
    share_dict = df_share.mean(axis=0).to_dict()

    df_model_train = df_shf_train['total'].to_frame().reset_index()
    df_model_valid = df_shf_valid['total'].to_frame().reset_index()
    df_model_train.columns = ['ds', 'y']
    df_model_valid.columns = ['ds', 'y']

    last_period_values  = df_model_train['y'].iloc[-7:].values
    forecast_start_date = df_model_train['ds'].iloc[-1]+pd.Timedelta(1, 'day')
    forecast_dates = pd.date_range(start=forecast_start_date, periods=H, freq='D')

    forecast_values = math.ceil(H/7)*list(last_period_values)
    forecast_values = forecast_values[:H] #to have exactly H values
    forecast_series = pd.Series(data=forecast_values, index=forecast_dates)
    df_forecast = forecast_series.reset_index()
    df_forecast.columns = ['ds', 'yhat']

    for store_item in store_items_list:
        store_item_deltas = []
        df_item_forecast = df_forecast.copy()
        df_item_forecast['yhat'] = df_item_forecast['yhat']*share_dict[store_item]
        df_item_valid = df_shf_valid[store_item].to_frame().reset_index()
        df_item_valid.columns = ['ds', 'y']

        # Here I do comparison plots and evaluate performance metrics
        df_comp = df_item_valid.set_index('ds').join(df_item_forecast.set_index('ds'))
        df_comp['cutoff'] = cutoff
        df_comp['h_days'] = (df_comp.index - cutoff).days
        df_comp['delta'] = df_comp['yhat'] - df_comp['y']
        deltas[store_item].append(df_comp.reset_index())


        
custom_events_nov = pd.DataFrame(
    {'holiday': 'unkown',
     'ds': pd.to_datetime(['2013-11-30',
                           '2014-11-30', 
                           '2015-11-30',
                           '2016-11-30',
                           '2017-11-30']),
     'lower_window': -5,
     'upper_window': 4,
    })


def prophet_model(model_name):
    if(model_name=='model1'):
        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True, 
                    daily_seasonality=False)
    elif(model_name=='model2'):
        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True, 
                    daily_seasonality=False,
                    seasonality_mode='multiplicative')
    elif(model_name=='model3'):
        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True, 
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    holidays=custom_events_nov)
    elif(model_name=='model4'):
        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=False, 
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    holidays=custom_events_nov)
        m.add_seasonality(name='weekly_on_season', period=7, 
                          fourier_order=3, condition_name='on_season')
        m.add_seasonality(name='weekly_off_season', period=7, 
                          fourier_order=3, condition_name='off_season')
    elif(model_name=='model5'):
        m = Prophet(yearly_seasonality=13,
                    weekly_seasonality=False, 
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    holidays=custom_events_nov)
        m.add_seasonality(name='weekly_on_season', period=7, 
                          fourier_order=3, condition_name='on_season')
        m.add_seasonality(name='weekly_off_season', period=7, 
                          fourier_order=3, condition_name='off_season')
    elif(model_name=='model6'):
        m = Prophet(yearly_seasonality=13,
                    weekly_seasonality=False, 
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    holidays=custom_events_nov)
        m.add_seasonality(name='weekly_peak_season', period=7, 
                          fourier_order=3, condition_name='peak_season')
        m.add_seasonality(name='weekly_norm_season', period=7, 
                          fourier_order=3, condition_name='norm_season')
        m.add_seasonality(name='weekly_off_season', period=7, 
                          fourier_order=3, condition_name='off_season')

    else:
        m=None
    
    return m
        
def model_first_check(model_name, df_train, H=90):
    m = prophet_model(model_name)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=H)
    future['off_season'] = ~future['ds'].apply(is_on_season)  #winter: Dec, Jan, Feb
    future['on_season']  = future['ds'].apply(is_on_season)    # on season: rest of year
    future['peak_season'] = future['ds'].apply(lambda x: x.month in (6,7))
    future['norm_season'] = future['ds'].apply(lambda x: x.month in (3,4,5,8,9,10,11))
    
    
    df_forecast = m.predict(future)
    
    # calculating residuals
    df_merged = df_train.merge(df_forecast, on='ds', how='left')
    df_merged['res'] = df_merged['yhat'] - df_merged['y']
    df_merged = df_merged.set_index('ds')
    ts_res = df_merged['res']

    # plotting residuals
    fig, axs = plt.subplots(1)
    yvals, xvals, _ = axs.hist(ts_res, bins=30)
    ymax = yvals.max()*1.1
    xmean = ts_res.mean()
    axs.axvline(x=xmean, ymin=0, ymax=ymax, color='red')
    print(f'residuals mean={xmean:.2f}')
    
    print('Top 10 residuals')
    ts_abs_res = abs(ts_res)
    display(ts_abs_res.sort_values(ascending=False).head(10))
    
    #print (f'max residual: {ts_res.max():.2f}', )
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    ax = ts_res.plot()
    #ax.set_xlim([date(2014, 1, 10), date(2016, 1, 1)])
    
    
    # checking correlation of residuals
    fig, axs = plt.subplots(1)
    axs = plot_acf(ts_res, ax=axs, zero=False, lags=300, alpha=0.05,
                   title=f'Autocorrelation')
    return m, df_forecast, ts_res



def shf_update_topdown_prophet(m,
                               df_shf_train, df_shf_valid, 
                               store_items_list,
                               cutoff, H, 
                               deltas):
    ## This function updates dictionary deltas
    df_share = df_shf_train[store_items_list].divide(df_shf_train['total'], axis=0)
    share_dict = df_share.mean(axis=0).to_dict()

    df_model_train = df_shf_train['total'].to_frame().reset_index()
    df_model_valid = df_shf_valid['total'].to_frame().reset_index()
    df_model_train.columns = ['ds', 'y']
    df_model_valid.columns = ['ds', 'y']
 
    ## Necessary IF using on_off_season_custom_seasonality
    df_model_train = df_model_train.copy()
    df_model_train['on_season']  =  df_model_train['ds'].apply(is_on_season)
    df_model_train['off_season'] = ~df_model_train['ds'].apply(is_on_season)
    df_model_train['peak_season'] = df_model_train['ds'].apply(lambda x: x.month in (6,7))
    df_model_train['norm_season'] = df_model_train['ds'].apply(lambda x: x.month in (3,4,5,8,9,10,11))
    
    m.fit(df_model_train)
    future = m.make_future_dataframe(periods=H)
    
    ## Necessary IF using on_off_season_custom_seasonality
    future['on_season']  =  future['ds'].apply(is_on_season)
    future['off_season'] = ~future['ds'].apply(is_on_season)
    future['peak_season'] = future['ds'].apply(lambda x: x.month in (6,7))
    future['norm_season'] = future['ds'].apply(lambda x: x.month in (3,4,5,8,9,10,11))
    
    df_forecast = m.predict(future)

    for store_item in store_items_list:
        store_item_deltas = []
        df_item_forecast = df_forecast.copy()
        df_item_forecast['yhat'] = df_item_forecast['yhat']*share_dict[store_item]
        df_item_valid = df_shf_valid[store_item].to_frame().reset_index()
        df_item_valid.columns = ['ds', 'y']

        # Here I do comparison plots and evaluate performance metrics
        df_comp = df_item_valid.set_index('ds').join(df_item_forecast.set_index('ds'))
        df_comp['cutoff'] = cutoff
        df_comp['h_days'] = (df_comp.index - cutoff).days
        df_comp['delta'] = df_comp['yhat'] - df_comp['y']
        deltas[store_item].append(df_comp.reset_index())

      
        
def shf_update_prophet_middle(df_shf_train, df_shf_valid, 
                              store_items_list,
                              cutoff, H, 
                              deltas):
    
    # to get sort list of stores from store_item_list
    stores_list = np.sort(list(set([int(x.split('_')[0]) for x in store_items_list])))
    stores_list = [str(x) for x in stores_list]
    
    ## This function updates dictionary deltas
    

    for store in stores_list:
        df_store_train = df_shf_train[str(store)].to_frame().reset_index()
        df_store_train.columns = ['ds', 'y']
 
        ## Necessary IF using on_off_season_custom_seasonality
        df_model_train = df_store_train.copy()
        df_model_train['on_season']  =  df_model_train['ds'].apply(is_on_season)
        df_model_train['off_season'] = ~df_model_train['ds'].apply(is_on_season)
        df_model_train['peak_season'] = df_model_train['ds'].apply(lambda x: x.month in (6,7))
        df_model_train['norm_season'] = df_model_train['ds'].apply(lambda x: x.month in (3,4,5,8,9,10,11))

        m = prophet_model('model6')
        m.fit(df_model_train)
        future = m.make_future_dataframe(periods=H)
        
        ## Necessary IF using on_off_season_custom_seasonality
        future['on_season']  =  future['ds'].apply(is_on_season)
        future['off_season'] = ~future['ds'].apply(is_on_season)
        future['peak_season'] = future['ds'].apply(lambda x: x.month in (6,7))
        future['norm_season'] = future['ds'].apply(lambda x: x.month in (3,4,5,8,9,10,11))


        df_store_forecast = m.predict(future)

        
        
        items_in_store_list = [x for x in store_items_list if x.split('_')[0]==store]
        df_share = df_shf_train[items_in_store_list].divide(df_shf_train[store], axis=0)
        share_dict = df_share.mean(axis=0).to_dict()     
        
        
        for store_item in items_in_store_list:
            store_item_deltas = []
            df_item_forecast = df_store_forecast.copy()
            df_item_forecast['yhat'] = df_item_forecast['yhat']*share_dict[store_item]
            df_item_valid = df_shf_valid[store_item].to_frame().reset_index()
            df_item_valid.columns = ['ds', 'y']

            # Here I do comparison plots and evaluate performance metrics
            df_comp = df_item_valid.set_index('ds').join(df_item_forecast.set_index('ds'))
            df_comp['cutoff'] = cutoff
            df_comp['h_days'] = (df_comp.index - cutoff).days
            df_comp['delta'] = df_comp['yhat'] - df_comp['y']
            deltas[store_item].append(df_comp.reset_index())


algo_mapping = {'model1':shf_update_modelX('model1'),
                'model2':shf_update_modelX('model2'),
                'model3':shf_update_modelX('model3'),
                'model4':shf_update_modelX('model4'),
                'model5':shf_update_modelX('model5'),
                'model6':shf_update_modelX('model6'),
                'naive':shf_update_naive,
                'seasonal_naive':shf_update_seasonal_naive,
                'average':shf_update_average,
                'middle-down':shf_update_prophet_middle
               }

def eval_performance(forecasts_dict):
    dfs_perf = []
    for k in forecasts_dict.keys():
        list_columns = ['ds', 'yhat', 'y', 'cutoff']
        df_cv = forecasts_dict[k][list_columns].copy()
        df_perf = performance_metrics(df_cv)
        df_perf['h_days'] = df_perf['horizon'].apply(lambda x: x.days)
        df_perf['store_item'] = k
        dfs_perf.append(df_perf)    
    merged_df_perf = pd.concat(dfs_perf, axis=0)
    return merged_df_perf


##########################################################################
#  _____                                  ____        _               _   
# |  __ \                                / __ \      | |             | |  
# | |__) | __ ___ _ __   __ _ _ __ ___  | |  | |_   _| |_ _ __  _   _| |_ 
# |  ___/ '__/ _ \ '_ \ / _` | '__/ _ \ | |  | | | | | __| '_ \| | | | __|
# | |   | | |  __/ |_) | (_| | | |  __/ | |__| | |_| | |_| |_) | |_| | |_ 
# |_|   |_|  \___| .__/ \__,_|_|  \___|  \____/ \__,_|\__| .__/ \__,_|\__|
#                | |                                     | |              
#                |_|                                     |_|              
###########################################################################

def prepare_output_file(df_forecast, df_h):
    df_forecast_total = df_forecast[['ds', 'yhat']].query(" ds>='2018-01-01' ")
    store_items_list = [f'{s}_{i}' for s in range(1,11) for i in range(1,51)]
    
    df_share = df_h[store_items_list].divide(df_h['total'], axis=0)
    share_dict = df_share.mean(axis=0).to_dict()

    results = []
    for item in range(1,51):
        for store in range(1,11):
            #print(store, ', ', item)
            df_tmp = df_forecast_total.copy()
            df_tmp['sales'] = df_tmp['yhat']*share_dict[f"{str(store)}_{str(item)}"]
            df_tmp['store'] = str(store)
            df_tmp['item'] = str(item)
            #results.append(df_tmp[['sales']])
            results.append(df_tmp[['ds', 'store', 'item', 'sales']])
            
    df_for_output = pd.concat(results).reset_index(drop=True)
    df_for_output.to_csv('final_predictions.csv')    
        
            
### Utilities ###


def write_pickle(filename, obj):
    outfile = open(filename,'wb')
    pickle.dump(obj, outfile)
    outfile.close()

def read_pickle(filename):
    infile = open(filename,'rb')
    object = pickle.load(infile)
    infile.close()
    return object




