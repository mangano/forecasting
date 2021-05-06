from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
import pandas as pd

import plotly.io as pio
pio.templates.default = "none"

def plot_timeseries(data, labels, title='', xlabel='', ylabel='',
                    shapes=None, hoverformat = '.2f',
                    mode='lines+markers',
                    legendx=None, legendy=1.1,
                    x_range=None,y_range=None,width=None,height=None, visibility=None):

    if(type(data) == pd.DataFrame):
        traces = []
        for c in data.columns:
            traces.append(data[c])        
    else:
        traces = data
    
    if visibility==None:
        visible_traces=[True]*len(labels)
    else:
        visible_traces=visibility
    scatters = []
    for n, x in enumerate(traces):
        scatters.append(go.Scatter(x=x.index, y=x, name=labels[n], mode=mode,  visible=visible_traces[n]))
    data = scatters
    layout = go.Layout(height=height, width=width,
        title = title,
        legend=dict(orientation='h', x=legendx, y=legendy),
        yaxis=dict(title=ylabel,    
                   rangemode='tozero',
                   hoverformat = hoverformat,range=y_range),
        xaxis=dict(title=xlabel, range=x_range),
    )
    layout.update(dict(shapes = shapes))

    fig = go.Figure(data=data, layout=layout)
    fig.show()