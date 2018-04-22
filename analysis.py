from __future__ import print_function
import pickle
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
from hyperopt import STATUS_OK, STATUS_FAIL, Trials
from pprint import pprint
import ga_base as ga
import json
import os

def get_stream(id_index=0, num_traces=1, maxpoints=80):
    stream_ids = tls.get_credentials_file()['stream_ids']

    # Get stream id from stream id list
    stream_id = stream_ids[id_index]

    # Make instance of stream id object
    stream = go.Stream(
        token=stream_id,  # (!) link stream id to 'token' key
        maxpoints=maxpoints      # (!) keep a max of 80 pts on screen
    )

    traces = []

    for i in range(num_traces):

        # Initialize trace of streaming plot by embedding the unique stream_id
        traces.append(go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            stream=stream         # (!) embed stream id, 1 per trace
        ))

    data = go.Data(traces)
    unique_url = py.plot(data, filename='s7_first-stream')

    s = py.Stream(stream_id)

    return s

def stream_div_and_fit(maxpoints=100):

    stream_ids = tls.get_credentials_file()['stream_ids']

    fit_token = stream_ids[0]
    fit_stream = go.Stream(
        token=fit_token,  # (!) link stream id to 'token' key
        maxpoints=maxpoints      # (!) keep a max of 80 pts on screen
    )

    div_token = stream_ids[1]
    div_stream = go.Stream(
        token=div_token,  # (!) link stream id to 'token' key
        maxpoints=maxpoints      # (!) keep a max of 80 pts on screen
    )

    fit_trace = go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        stream=fit_stream,         # (!) embed stream id, 1 per trace
        yaxis='y',
        name='fitness'
    )

    div_trace = go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        stream=div_stream,         # (!) embed stream id, 1 per trace
        yaxis='y2',
        name='diversity'
    )

    layout = go.Layout(
        title='div fit stream',
        yaxis=go.YAxis(
            title='fitness'
        ),
        yaxis2=go.YAxis(
            title='diversity',
            side='right',
            overlaying='y'
        )
    )

    data = go.Data([div_trace, fit_trace])
    fig = go.Figure(data=data, layout=layout)
    unique_url = py.plot(fig, filename='div fit stream')

    div_s = py.Stream(div_token)
    fit_s = py.Stream(fit_token)

    div_s.open()
    fit_s.open()

    if 'pop' in os.listdir(os.getcwd()):
        with open('pop', 'r') as f:
            pop = pickle.load(f)
    else:
        pop = ga.get_pop(64)

    c = 0
    while True:
        c += 1
        pop.evolve()
        if c % 200 == 0:
             fit_s.write(dict(x=c, y=pop.average_fitness()))
             div_s.write(dict(x=c, y=pop.average_diversity()))
        if c % 200 == 0:
            pop.all_time_fittest_individual.show_imitation()
            pass
        if c % 100 == 0:
            pop.save('pop')
        if c % 1000 == 0:
            top_specs()

def top_specs():

    with open('pop', 'r') as f:
        pop = pickle.load(f)

    print(pop.all_time_fittest_individual.to_string())

if __name__ == "__main__":

    stream_div_and_fit(maxpoints=256)
    # top_specs()
    pass