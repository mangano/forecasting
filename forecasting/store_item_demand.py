# General imports
from IPython.display import display, HTML, Markdown
import math 

# General data analysis imports
import pandas as pd

# Facebook Prophet
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

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


def cv_plot(store_perfs, id_label,
            n_days_horizon=180, n_increments=10):
    increment_delta = math.floor(n_days_horizon/n_increments)  
    h_subset = list(range(increment_delta, n_days_horizon, increment_delta))
    
    ts_list = []
    ts_labels = []
    for key, df_p in store_perfs.items():   
        # convert delta into integers (days)
        df_plot = df_p[['horizon','mape']].copy()
        df_plot['horizon'] = df_plot['horizon'].apply(lambda x: x.days)

        # only plot subset of well-separated h since consecutive days are correlated
        df_to_plot = df_plot[df_plot['horizon'].isin(h_subset)]

        # convert to ts to reuse bplot function
        ts_to_plot = df_to_plot.set_index('horizon')['mape']
        ts_list.append(ts_to_plot)
        ts_labels.append(f'{id_label} {key}')
    
    
    bplot.plot_timeseries(ts_list, ts_labels, xlabel='horizon [days]', ylabel='MAPE')

