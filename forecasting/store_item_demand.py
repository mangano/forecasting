import pandas as pd
from IPython.display import display, HTML, Markdown


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