# General imports
from IPython.display import display, HTML, Markdown
import math 
from datetime import datetime

# General data analysis imports
import pandas as pd

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



def cv_validation(df_prophet, progress_bar):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)
    df_cv = cross_validation(m, initial='1100 days', 
                             period='90 days', horizon = '180 days', 
                             disable_tqdm=(not progress_bar))
    df_p = performance_metrics(df_cv)
    
    return df_cv, df_p

    
def cv_loop(df_h, ts_list, progress_bar=True):
    cvs = {}
    perfs = {}
    for idx in ts_list:
        df_prophet = df_h[idx].to_frame().reset_index()
        df_prophet.columns = ['ds', 'y']
        df_cv, df_p = cv_validation(df_prophet, progress_bar)
        cvs[idx] = df_cv
        perfs[idx] = df_p    
    return cvs, perfs


def cv_plot(store_perfs, id_label, metric='smape',
            n_days_horizon=180, n_increments=10):
    increment_delta = math.floor(n_days_horizon/n_increments)  
    h_subset = list(range(increment_delta, n_days_horizon, increment_delta))
    
    ts_list = []
    ts_labels = []
    for key, df_p in store_perfs.items():   
        # convert delta into integers (days)
        df_plot = df_p[['horizon',metric]].copy()
        df_plot['horizon'] = df_plot['horizon'].apply(lambda x: x.days)

        # only plot subset of well-separated h since consecutive days are correlated
        df_to_plot = df_plot[df_plot['horizon'].isin(h_subset)]

        # convert to ts to reuse bplot function
        ts_to_plot = df_to_plot.set_index('horizon')[metric]
        ts_list.append(ts_to_plot)
        ts_labels.append(f'{id_label} {key}')
    
    
    bplot.plot_timeseries(ts_list, ts_labels, xlabel='horizon [days]', 
                          ylabel=metric.upper())

    
    
    
    
    
### HIERARCHICAL TIME SERIES

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


def plot_comparison(df_train, df_preds, idx='1_1',
                    xmin = None, xmax = None):
    df_y = df_train[[idx]].copy()
    df_y.columns =['y']

    df_yhat = df_preds[[idx]].copy()
    df_yhat.columns =['yhat']

    df_comp = df_yhat.join(df_y)

    ax = df_comp.plot(title="Sales - item level")
    ax.legend(bbox_to_anchor=(1.0, 1.0));
    x_range = [xmin, xmax]
    ax.set_xlim(x_range);
    
    
    
### Model evaluation

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
        cutoffs = cutoffs[:n_max_splits]
    
    print(f'# number of cutoffs: {len(cutoffs)}')    
    splits = {}   
    for cutoff in cutoffs:
        print(str(cutoff.date()))
        df_shf_train = df[:cutoff]
        df_shf_val   = df[cutoff+pd.Timedelta(1, 'day'):cutoff+horizon_delta]
        splits[cutoff] = [df_shf_train, df_shf_val]
    
    return splits


def shf_forecasts_loop(hierarchy, shf_splits, algo='naive',
                       n_store=None, n_store_items=None,
                       verbose=False):
    # Horizon
    H=90
    
    start = datetime.now()
    cutoffs = list(shf_splits.keys())

    # FIXME: take just 1 to speed up testing
    #cutoffs = cutoffs[1:4]

    forecasts_dict = {}    
    stores = hierarchy['total']

    # take just few to speed up testing
    if(n_store is not None):
        stores = stores[:n_store]
        
    for store in stores:
        print(f'# store: {store}')
        #store_items = [store_item for store in stores for store_item in hierarchy[store]]
        store_items = hierarchy[store]
        
        # take just few to speed up testing
        if(n_store_items is not None):
            store_items = store_items[:n_store_items]

        for store_item in store_items:
            print(f'# store_item: {store_item}')

            shf_forecasts = []
            for cutoff in cutoffs:
                #print (f'# cutoff: {cutoff.date()}')
                df_shf_train = shf_splits[cutoff][0]
                df_shf_valid = shf_splits[cutoff][1]  

                # Reformatting datasets (follwowing Prophet requirement)
                df_model_train = df_shf_train[store_item].to_frame().reset_index()
                df_model_valid = df_shf_valid[store_item].to_frame().reset_index()
                df_model_train.columns = ['ds', 'y']
                df_model_valid.columns = ['ds', 'y']

                ### Model specific part
                if(algo=='prophet'):                
                    m = Prophet(yearly_seasonality=True, 
                                weekly_seasonality=True, 
                                daily_seasonality=False)
                    m.fit(df_model_train)
                    future = m.make_future_dataframe(periods=H)
                    df_forecast = m.predict(future)
                elif(algo in ('naive','average')):
                    # simple models
                    last_train_tmstp = df_model_train['ds'].max()
                    forecast_naive  = \
                        df_model_train[df_model_train['ds']==last_train_tmstp]['y'].values[0]
                    forcast_average = df_model_train['y'].mean()
                    future_dates = pd.date_range(start=last_train_tmstp+pd.Timedelta(1, 'day'), 
                                                 freq='D', periods=H)
                    forecast = forecast_naive if algo=='naive' else forcast_average
                    df_forecast = pd.DataFrame({'ds':future_dates, 
                                                    'yhat':forecast})
                else:
                    raise ValueError("algo should be in ('naive', 'average', 'average')")
                ####
                
                # Here I do comparison plots and evaluate performance metrics
                df_comp = df_model_valid.set_index('ds').join(df_forecast.set_index('ds'))
                df_comp['cutoff'] = cutoff
                df_comp['h_days'] = (df_comp.index - cutoff).days
                df_comp['delta'] = df_comp['yhat'] - df_comp['y'] 
                #shf_forecasts.append(df_comp.reset_index().drop('ds', axis=1))
                shf_forecasts.append(df_comp.reset_index())

                plot = False
                if(plot):
                    xrange = [cutoff-pd.Timedelta(90, 'days'), cutoff+pd.Timedelta(100, 'days')]
                    fig, ax = plt.subplots()
                    ax.set_xlim(xrange)
                    fig1 = m.plot(df_forecast, ax)

                    xrange = [cutoff, cutoff+pd.Timedelta(100, 'days')]
                    ax = df_comp[['y','yhat']].plot()
                    ax.set_xlim(xrange)

            df_concat = pd.concat(shf_forecasts, axis=0, ignore_index=True)
            forecasts_dict[store_item] = df_concat
            
    end = datetime.now()
    delta = end - start
    if(verbose):
        print(f'# execution walltime: {str(delta)}')
    
    return forecasts_dict


def eval_performance(forecasts_dict):
    dfs_perf = []
    for k in forecasts_dict.keys():
        #list_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y', 'cutoff']
        list_columns = ['ds', 'yhat', 'y', 'cutoff']
        df_cv = forecasts_dict[k][list_columns].copy()
        df_perf = performance_metrics(df_cv)
        df_perf['h_days'] = df_perf['horizon'].apply(lambda x: x.days)
        df_perf['store_item'] = k
        dfs_perf.append(df_perf)    
    merged_df_perf = pd.concat(dfs_perf, axis=0)
    return merged_df_perf