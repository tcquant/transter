import psycopg2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from PIL.PyAccess import mode_map
# from fontTools.misc.py23 import tounicode

import plotly.express as px
from plotly.subplots import make_subplots
import os
from IPython.display import FileLink , display

def fetch_options(cursor, symbol, expiry):
    query = '''
        SELECT * 
        FROM ohlcv_options_per_minute oopm
        WHERE symbol = %s
        AND expiry_type = 'I'
        AND expiry = %s
        ORDER BY date_timestamp;
        '''
    cursor.execute(query, (symbol, expiry))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df

def FetchFuturesByDate(cursor, symbol, date):
    query = '''
            SELECT *
            FROM ohlcv_future_per_minute ofpm 
            WHERE ofpm.symbol = %s
            AND ofpm.expiry_type = 'I'
            AND DATE(date_timestamp) = %s
            ORDER BY date_timestamp ASC
        '''
    cursor.execute(query, (symbol,date))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    df['open'] = df['open'] // 100
    df['high'] = df['high'] // 100
    df['low'] = df['low'] // 100
    df['close'] = df['close'] // 100
    df.drop(columns = ['id'],inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def FetchOptionsByDate(cursor, symbol, date,expiry="IW1"):
    query = '''
            SELECT * 
            FROM ohlcv_options_per_minute oopm
            WHERE symbol = %s
            AND expiry_type = %s
            AND DATE(date_timestamp) = %s
            ORDER BY date_timestamp;
            '''
    cursor.execute(query, (symbol,expiry,date))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    df['open'] = df['open'] // 100
    df['high'] = df['high'] // 100
    df['low'] = df['low'] // 100
    df['close'] = df['close'] // 100
    df['strike'] = df['strike'] // 100
    return df









def FetchOptionsByDateCsv(symbol,date):
    df = pd.read_csv(f'C:\\Users\\Tanmay\\Desktop\\DB\\options_{symbol}_{date}.csv')
    df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
    df['expiry'] = pd.to_datetime(df['expiry'])
    return df


def fetch_futures(cursor, symbol, expiry):
    query = '''
        SELECT *
        FROM ohlcv_future_per_minute ofpm 
        WHERE ofpm.symbol = %s
        AND ofpm.expiry_type = 'I'
        AND DATE(ofpm.expiry) = %s
        ORDER BY date_timestamp ASC
    '''
    # Execute the query with parameters as a tuple
    cursor.execute(query, (symbol, expiry))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df


# Define the fetch_expiries function
def fetch_expiries(cursor, symbol):
    query = f'''
        SELECT DISTINCT ofpem.expiry 
        FROM ohlcv_options_per_minute ofpem 
        WHERE ofpem.symbol = '{symbol}'
        AND ofpem.expiry_type = 'I'
        GROUP BY ofpem.expiry 
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    df['expiry'] = pd.to_datetime(df['expiry']).dt.date
    return df

def fetch_expiries_():
    # Load expiry_future data and ensure 'expiry' column is datetime.date type
    expiry_future = pd.read_csv(r"C:\Users\Tanmay\Desktop\Files\expiry_future.csv")
    expiry_future['expiry'] = pd.to_datetime(expiry_future['expiry']).dt.date

    # Define error dates and convert to DataFrame
    error_dates = ['2020-03-26', '2020-05-28', '2020-06-25', '2020-07-30', '2020-11-26', '2020-12-31', '2021-01-28',
                   '2021-05-27', '2021-07-29', '2021-08-26', '2021-09-30', '2021-10-28', '2021-11-25', '2021-12-30']
    Error_dates = pd.DataFrame(error_dates, columns=['expiry'])
    Error_dates['expiry'] = pd.to_datetime(Error_dates['expiry']).dt.date

    # Convert Error_dates['expiry'] to a set for faster lookup
    error_dates_set = set(Error_dates['expiry'])

    # Filter expiry_future by excluding values present in Error_dates
    expiry_future_filtered = [i for i in expiry_future['expiry'] if i not in error_dates_set]

    # Convert filtered expiry_future to DataFrame
    expiry_future_filtered_df = pd.DataFrame(expiry_future_filtered, columns=['expiry'])

    for i, row in enumerate(expiry_future_filtered_df.itertuples()):
        if i < 10:
            print(f" {i} = {row.expiry}", end=' | ')
        else:
            print(f"{i} = {row.expiry}", end=' | ')
        if (i + 1) % 4 == 0:
            print()
    return expiry_future['expiry']

def fetch_futures_(filename):  # Replace 'expiry' with the correct column name
    futures = pd.read_csv(f"C:\\Users\\tanmay\\Documents\\db\\f\\{filename}.csv")
    if 'id' in futures.columns:   futures.drop(columns=['id'], inplace=True)
    if 'Unnamed: 0' in futures.columns : futures.drop(columns=['Unnamed: 0'], inplace=True)

    futures.drop_duplicates(inplace=True)
    futures['date_timestamp'] = pd.to_datetime(futures['date_timestamp'])
    futures['expiry'] = pd.to_datetime(futures['expiry'])

    futures['open'] = futures['open'].astype('int64') // 100
    futures['high'] = futures['high'].astype('int64') // 100
    futures['low'] = futures['low'].astype('int64') // 100
    futures['close'] = futures['close'].astype('int64') // 100
    futures['volume'] = futures['volume'].astype('int64')
    fill_missings(futures)
    return  futures

def fetch_options_(filename):
    options = futures = pd.read_csv(f"C:\\Users\\tanmay\\Documents\\db\\o_g\\{filename}_G.csv")
    if 'id' in options.columns: options.drop(columns=['id'],inplace=True)
    if 'Unnamed: 0' in options.columns : options.drop(columns=['Unnamed: 0'],inplace=True)
    options.drop_duplicates(inplace=True)
    options['date_timestamp'] = pd.to_datetime(options['date_timestamp'])
    options['expiry'] = pd.to_datetime(options['expiry'])
    options['expiry'] = options['expiry'] + pd.Timedelta(hours=15,minutes=30)
    options['strike'] = options['strike'].astype('int64')
    options['delta'] = round(options['delta'],3)
    # options['open'] = options['open'].astype('int64') // 100
    # options['high'] = options['high'].astype('int64') // 100
    # options['low'] = options['low'].astype('int64') // 100
    # options['close'] = options['close'].astype('int64') // 100
    options['volume'] = options['volume'].astype('int64')

    return options

def fill_missings(futures):
    # Generate the full range of timestamps for each day
    start_time = '09:15:00'
    end_time = '15:30:00'

    all_timestamps = pd.date_range(start=start_time, end=end_time, freq='min').time

    # Generate a DataFrame with all timestamps for each day in the data
    all_dates = futures['date_timestamp'].dt.date.unique()
    all_date_times = [pd.Timestamp.combine(date, time) for date in all_dates for time in all_timestamps]
    all_date_times_df = pd.DataFrame(all_date_times, columns=['date_timestamp'])

    # Merge with the original DataFrame to include all timestamps
    futures = pd.merge(all_date_times_df, futures, on='date_timestamp', how='left')

    # Forward and backword fill missing values
    futures = futures.ffill()

    # Sort by date_timestamp to maintain order
    futures = futures.sort_values('date_timestamp').reset_index(drop=True)


def genrate_signals(futures, long_window=26, short_window=9):
    futures['Short_EMA'] = futures['close'].ewm(span=short_window).mean()
    futures['Long_EMA'] = futures['close'].ewm(span=long_window).mean()

    futures['Signal'] = 0

    for i in range(26, len(futures)):
        if futures['Short_EMA'].iloc[i] > futures['Long_EMA'].iloc[i] and futures['Short_EMA'].iloc[i - 1] <= \
                futures['Long_EMA'].iloc[i - 1]:
            futures.at[futures.index[i], 'Signal'] = 1  # Buy signal
        elif futures['Short_EMA'].iloc[i] < futures['Long_EMA'].iloc[i] and futures['Short_EMA'].iloc[i - 1] >= \
                futures['Long_EMA'].iloc[i - 1]:
            futures.at[futures.index[i], 'Signal'] = -1  # Sell signal


def plot_signals(futures):
    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    candlestick = go.Candlestick(
        x=futures.index,
        open=futures['open'],
        high=futures['high'],
        low=futures['low'],
        close=futures['close'],
        name='Candlesticks'
    )
    fig.add_trace(candlestick)

    # Short EMA
    short_ema = go.Scatter(x=futures.index, y=futures['Short_EMA'], mode='lines', name='Short EMA')
    fig.add_trace(short_ema)

    # Long EMA
    long_ema = go.Scatter(x=futures.index, y=futures['Long_EMA'], mode='lines', name='Long EMA')
    fig.add_trace(long_ema)

    # Buy signals
    buy_signals = go.Scatter(
        x=futures[futures['Signal'] == 1].index,
        y=futures['Short_EMA'][futures['Signal'] == 1],
        mode='markers',
        marker=dict(symbol='triangle-up', color='yellow', size=12),
        name='Buy Signal'
    )
    fig.add_trace(buy_signals)

    # Sell signals
    sell_signals = go.Scatter(
        x=futures[futures['Signal'] == -1].index,
        y=futures['Short_EMA'][futures['Signal'] == -1],
        mode='markers',
        marker=dict(symbol='triangle-down', color='black', size=12),
        name='Sell Signal'
    )
    fig.add_trace(sell_signals)

    # Corrected title access
    symbol = futures.iloc[0]['Symbol'] if 'Symbol' in futures.columns else 'Unknown'
    fig.update_layout(
        title=f'{symbol} EMA Crossover Strategy',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        width=2000,
        height=1000
    )

    return fig

def _plot(futures):
    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    candlestick = go.Candlestick(
        x=futures['date_timestamp'],
        open=futures['open'],
        high=futures['high'],
        low=futures['low'],
        close=futures['close'],
        name='Candlesticks'
    )
    fig.add_trace(candlestick)

    # Short EMA
    short_ema = go.Scatter(x=futures['date_timestamp'], y=futures['Short_EMA'], mode='lines', name='Short EMA')
    fig.add_trace(short_ema)

    # Long EMA
    long_ema = go.Scatter(x=futures['date_timestamp'], y=futures['Long_EMA'], mode='lines', name='Long EMA')
    fig.add_trace(long_ema)

    # Buy signals
    buy_signals = go.Scatter(
        x=futures[futures['Signal'] == 1]['date_timestamp'],
        y=futures['Short_EMA'][futures['Signal'] == 1],
        mode='markers',
        marker=dict(symbol='triangle-up', color='yellow', size=12),
        name='Buy Signal'
    )
    fig.add_trace(buy_signals)

    # Sell signals
    sell_signals = go.Scatter(
        x=futures[futures['Signal'] == -1]['date_timestamp'],
        y=futures['Short_EMA'][futures['Signal'] == -1],
        mode='markers',
        marker=dict(symbol='triangle-down', color='black', size=12),
        name='Sell Signal'
    )
    fig.add_trace(sell_signals)

    fig.update_layout(
        title=f'{futures.iloc[0]["Symbol"]} EMA Crossover Strategy',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        width=2000,
        height=1000
    )

    return fig

import pandas as pd

def fetch_call_put(options):
    # Check and Drop 'id' Column
    if 'id' in options.columns:
        options.drop(columns=['id'], inplace=True)

    # Drop Duplicates
    options.drop_duplicates(inplace=True)

    # Convert 'date_timestamp' to datetime
    options['date_timestamp'] = pd.to_datetime(options['date_timestamp'])

    # Separate Call and Put Options
    copt = options[options['opt_type'] == 'CE'].copy()
    popt = options[options['opt_type'] == 'PE'].copy()

    # Drop Duplicates in Call and Put DataFrames
    copt.drop_duplicates(inplace=True)
    popt.drop_duplicates(inplace=True)

    # Define start and end times
    start_time = '09:15:00'
    end_time = '15:30:00'

    # Generate all timestamps for a single day
    all_timestamps = pd.date_range(start=start_time, end=end_time, freq='min').time

    # Extract unique dates
    all_dates = options['date_timestamp'].dt.date.unique()

    all_date_times = [pd.Timestamp.combine(date, time) for date in all_dates for time in all_timestamps]

    main_ce_df = pd.DataFrame()
    main_pe_df = pd.DataFrame()

    # Get unique strike values
    strike_values_call = options[options['opt_type']=='CE']['strike'].unique()
    strike_values_put = options[options['opt_type'] == 'PE']['strike'].unique()

    for strike in strike_values_call:
        # Create a DataFrame with all date-time combinations for the current strike
        d2c = [[dt, strike] for dt in all_date_times]
        d2c_df = pd.DataFrame(d2c, columns=['date_timestamp', 'strike'])

        # Filter call and put options for the current strike
        copt_strike = copt[copt['strike'] == strike]
        copt_strike['real'] = 1
        ce = pd.merge(d2c_df, copt_strike, on=['date_timestamp', 'strike'], how='left')
        ce.ffill(inplace=True) ; ce.bfill(inplace=True)
        ce['real'] = [1 if i == 1 else 0 for i in ce['real']]
        main_ce_df = pd.concat([main_ce_df, ce])

    for strike in strike_values_put:

        d2c = [[dt, strike] for dt in all_date_times]
        d2c_df = pd.DataFrame(d2c, columns=['date_timestamp', 'strike'])

        popt_strike = popt[popt['strike'] == strike]
        popt_strike['real'] = 1

        pe = pd.merge(d2c_df, popt_strike, on=['date_timestamp', 'strike'], how='left')
        pe.ffill(inplace=True) ; pe.bfill(inplace=True)
        pe['real'] = [1 if i == 1 else 0 for i in pe['real']]
        main_pe_df = pd.concat([main_pe_df, pe])

    # Reset index for the main DataFrames
    main_ce_df.reset_index(drop=True, inplace=True)
    main_pe_df.reset_index(drop=True, inplace=True)

    return main_ce_df, main_pe_df


def fetch_call_put_(options):
    # Check and Drop 'id' Column
    if 'id' in options.columns:
        options.drop(columns=['id'], inplace=True)

    # Drop Duplicates
    options.drop_duplicates(inplace=True)

    # Convert 'date_timestamp' to datetime
    options['date_timestamp'] = pd.to_datetime(options['date_timestamp'])

    # Separate Call and Put Options
    copt = options[options['opt_type'] == 'CE'].copy()
    popt = options[options['opt_type'] == 'PE'].copy()

    # Drop Duplicates in Call and Put DataFrames
    copt.drop_duplicates(inplace=True)
    popt.drop_duplicates(inplace=True)

    # Define start and end times
    start_time = '09:15:00'
    end_time = '15:30:00'

    # Generate all timestamps for a single day
    all_timestamps = pd.date_range(start=start_time, end=end_time, freq='min').time

    # Extract unique dates
    all_dates = options['date_timestamp'].dt.date.unique()

    # Create a list of all date-time combinations
    all_date_times = [pd.Timestamp.combine(date, time) for date in all_dates for time in all_timestamps]

    # Initialize main DataFrames for call and put options
    main_ce_df = pd.DataFrame()
    main_pe_df = pd.DataFrame()

    # Get unique strike values
    strike_values = options['strike'].unique()

    for strike in strike_values:
        # Create a DataFrame with all date-time combinations for the current strike
        d2c = [[dt, strike] for dt in all_date_times]
        d2c_df = pd.DataFrame(d2c, columns=['date_timestamp', 'strike'])
        d2c_df['date_timestamp'] = pd.to_datetime(d2c_df['date_timestamp'], errors='coerce')

        # Filter call and put options for the current strike
        copt_strike = copt[copt['strike'] == strike]
        popt_strike = popt[popt['strike'] == strike]
        copt_strike['date_timestamp'] = pd.to_datetime(copt_strike['date_timestamp'], errors='coerce')
        popt_strike['date_timestamp'] = pd.to_datetime(popt_strike['date_timestamp'], errors='coerce')

        # Merge with call and put data separately
        ce = pd.merge(d2c_df, copt_strike, on=['date_timestamp', 'strike'], how='left')
        pe = pd.merge(d2c_df, popt_strike, on=['date_timestamp', 'strike'], how='left')

        # Forward fill missing values and infer correct data types to avoid warnings
        ce.ffill(inplace=True)
        ce.bfill(inplace=True)
        ce.infer_objects(copy=False)  # Fix FutureWarning

        pe.ffill(inplace=True)
        pe.bfill(inplace=True)
        pe.infer_objects(copy=False)  # Fix FutureWarning

        # Append to the main DataFrames
        main_ce_df = pd.concat([main_ce_df, ce])
        main_pe_df = pd.concat([main_pe_df, pe])

    # Reset index for the main DataFrames
    main_ce_df.reset_index(drop=True, inplace=True)
    main_pe_df.reset_index(drop=True, inplace=True)

    return main_ce_df, main_pe_df


def stacked_graphs(data_streams, labels=None, colors=None, height = 300 , title="plots", xaxis_title="Index",output_file="plots.html"):
    """
    Plots stacked graphs with shared x-axis.

    Parameters:
    - data_streams: List of lists or arrays, where each inner list/array represents a data stream to plot.
    - labels: List of labels for each data stream. If None, default labels will be used.
    - colors: List of colors for each data stream. If None, default colors will be assigned.
    - title: Title of the plot.
    - xaxis_title: Label for the x-axis.
    - output_file: Name of the HTML file to save the plot to.
    """
    num_streams = len(data_streams)

    # Set default labels if not provided
    if labels is None:
        labels = [f"Data {i + 1}" for i in range(num_streams)]

    # Set default colors if not provided
    if colors is None:
        colors = ['#FF6347','#4682B4','#32CD32','#FFD700','#8A2BE2','#FF4500','#2E8B57','#D2691E','#FF1493','#00CED1'] * (num_streams // 3 + 1)

    # Create a figure with subplots, stacked vertically with a shared x-axis
    fig = make_subplots(rows=num_streams, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Add traces for each data stream
    for i, data in enumerate(data_streams):
        if isinstance(data[0],tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in data], mode='lines', name=labels[i] , text = [x[0] for x in data], line=dict(color=colors[i])), row=i + 1, col=1)
        else:
            fig.add_trace(go.Scatter(y=data, mode='lines', name=labels[i], line=dict(color=colors[i])), row=i + 1, col=1)
    # Update layout with shared hover mode
    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        hovermode='x unified',  # Enables showing all values across subplots at the same x-coordinate
        height=height * num_streams  # Adjust height dynamically based on the number of subplots
    )

    # Export the plot to an HTML file and automatically open it
    pio.write_html(fig, file=output_file, auto_open=True)


def _graph(y_data, x_data=None, text_data=None):
    """
    Creates an interactive graph using Plotly with optional x_data and text annotations.

    Parameters:
    y_data : list or array
        Required y-axis data.
    x_data : list or array, optional
        Optional x-axis data. If not provided, the index of y_data is used as x-axis data.
    text_data : list or array, optional
        Optional text annotations to display for each point.
    """
    # If x_data is not provided, use the index of y_data
    if x_data is None:
        x_data = list(range(len(y_data)))

    # Create the scatter plot
    fig = go.Figure()

    # Add scatter plot to the figure with optional text annotations
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers+lines',  # you can also use just 'markers' or 'lines'
        text=text_data,  # This will display the text when hovering over the points
        hoverinfo='text' if text_data is not None else 'x+y',  # Show text if available, otherwise show x and y values
        marker=dict(size=10)  # You can customize the marker size, color, etc.
    ))

    # Set the title and axis labels
    fig.update_layout(
        title='G',
        hovermode='x unified',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        showlegend=False  # No need for a legend
    )

    # Show the interactive plot
    fig.show()


def plotter_d(streams,bools, labels=None, colors=None):
    num_streams = len(streams)

    if labels is None:
        labels = [f"Data {i + 1}" for i in range(num_streams)]

    if colors is None:
        colors = ["#E12729", "#F37324", "#F8C01B", "#72B043", "#007F4E", "#C8C8C8"] * (num_streams // 3 + 1)

    fig = make_subplots()
    fig.update_layout(
        title_text="plots",
        xaxis_title="Index",
        hovermode='x unified'  # Enables showing all values across subplots at the same x-coordinate
    )
    # First stream
    if isinstance(streams[0][0], tuple):
        fig.add_trace(go.Scatter(y=[x[1] for x in streams[0]], mode='lines', text=[x[0] for x in streams[0]], name=labels[0], line=dict(color=colors[0])))
    else:
        fig.add_trace(go.Scatter(y=streams[0], mode='lines', name=labels[0], line=dict(color=colors[0])))
    fig.update_yaxes(title_text=labels[0], secondary_y=False)

    # Second stream
    if num_streams > 1 and bools[1]==1:
        if isinstance(streams[1][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[1]], mode='lines', text=[x[0] for x in streams[1]], name=labels[1], line=dict(color=colors[1])))
        else:
            fig.add_trace(go.Scatter(y=streams[1], mode='lines', name=labels[1], line=dict(color=colors[1])))
        fig.update_layout(yaxis3=dict(title=labels[1], titlefont=dict(color=colors[1]), tickfont=dict(color=colors[1]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][1]['yaxis'] = 'y3'


    # Third stream
    if num_streams > 2 and bools[2]==1:
        if isinstance(streams[2][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[2]], mode='lines', text=[x[0] for x in streams[2]], name=labels[2], line=dict(color=colors[2])))
        else:
            fig.add_trace(go.Scatter(y=streams[2], mode='lines', name=labels[2], line=dict(color=colors[2])))
        fig.update_layout(yaxis4=dict(title=labels[2], titlefont=dict(color=colors[2]), tickfont=dict(color=colors[2]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][2]['yaxis'] = 'y4'

    # Fourth stream
    if num_streams > 3  and bools[3]==1:
        if isinstance(streams[3][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[3]], mode='lines', text=[x[0] for x in streams[3]], name=labels[3], line=dict(color=colors[3])))
        else:
            fig.add_trace(go.Scatter(y=streams[3], mode='lines', name=labels[3], line=dict(color=colors[3])))
        fig.update_layout(yaxis5=dict(title=labels[3], titlefont=dict(color=colors[3]), tickfont=dict(color=colors[3]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][3]['yaxis'] = 'y5'

    # Fifth stream
    if num_streams > 4  and bools[4]==1:
        if isinstance(streams[4][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[4]], mode='lines', text=[x[0] for x in streams[4]], name=labels[4], line=dict(color=colors[4])))
        else:
            fig.add_trace(go.Scatter(y=streams[4], mode='lines', name=labels[4], line=dict(color=colors[4])))
        fig.update_layout(yaxis6=dict(title=labels[4], titlefont=dict(color=colors[4]), tickfont=dict(color=colors[4]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][4]['yaxis'] = 'y6'

    # Sixth stream
    if num_streams > 5 and bools[5]==1:
        if isinstance(streams[5][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[5]], mode='lines', text=[x[0] for x in streams[5]], name=labels[5], line=dict(color=colors[5])))
        else:
            fig.add_trace(go.Scatter(y=streams[5], mode='lines', name=labels[5], line=dict(color=colors[5])))
        fig.update_layout(yaxis7=dict(title=labels[5], titlefont=dict(color=colors[5]), tickfont=dict(color=colors[5]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][5]['yaxis'] = 'y7'

    # Seventh stream
    if num_streams > 6  and bools[6]==1:
        if isinstance(streams[6][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[6]], mode='lines', text=[x[0] for x in streams[6]], name=labels[6], line=dict(color=colors[6])))
        else:
            fig.add_trace(go.Scatter(y=streams[6], mode='lines', name=labels[6], line=dict(color=colors[6])))
        fig.update_layout(yaxis8=dict(title=labels[6], titlefont=dict(color=colors[6]), tickfont=dict(color=colors[6]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][6]['yaxis'] = 'y8'

    # Eighth stream
    if num_streams > 7  and bools[7]==1:
        if isinstance(streams[7][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[7]], mode='lines', text=[x[0] for x in streams[7]], name=labels[7], line=dict(color=colors[7])))
        else:
            fig.add_trace(go.Scatter(y=streams[7], mode='lines', name=labels[7], line=dict(color=colors[7])))
        fig.update_layout(yaxis9=dict(title=labels[7], titlefont=dict(color=colors[7]), tickfont=dict(color=colors[7]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][7]['yaxis'] = 'y9'

    # Ninth stream
    if num_streams > 8 and bools[8]==1:
        if isinstance(streams[8][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[8]], mode='lines', text=[x[0] for x in streams[8]], name=labels[8], line=dict(color=colors[8])))
        else:
            fig.add_trace(go.Scatter(y=streams[8], mode='lines', name=labels[8], line=dict(color=colors[8])))
        fig.update_layout(yaxis10=dict(title=labels[8], titlefont=dict(color=colors[8]), tickfont=dict(color=colors[8]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][8]['yaxis'] = 'y10'

    # Tenth stream
    if num_streams > 9 and bools[9]==1:
        if isinstance(streams[9][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[9]], mode='lines', text=[x[0] for x in streams[9]], name=labels[9], line=dict(color=colors[9])))
        else:
            fig.add_trace(go.Scatter(y=streams[9], mode='lines', name=labels[9], line=dict(color=colors[9])))
        fig.update_layout(yaxis11=dict(title=labels[9], titlefont=dict(color=colors[9]), tickfont=dict(color=colors[9]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][9]['yaxis'] = 'y11'

    pio.write_html(fig, file='plot.html', auto_open=True)


def plotter(streams, labels=None, colors=None ,filename = 'plot.html'):
    num_streams = len(streams)

    if labels is None:
        labels = [f"Data {i + 1}" for i in range(num_streams)]

    if colors is None:
        colors = ['#FF6347','#4682B4','#32CD32','#FFD700','#8A2BE2','#FF4500','#2E8B57','#D2691E','#FF1493','#00CED1'] * (num_streams // 3 + 1)

    fig = make_subplots()
    fig.update_layout(
        title_text=" ",
        xaxis_title="Index",
        hovermode='x unified'  # Enables showing all values across subplots at the same x-coordinate
    )
    # First stream
    if isinstance(streams[0][0], tuple):
        fig.add_trace(go.Scatter(y=[x[1] for x in streams[0]], mode='lines', text=[x[0] for x in streams[0]], name=labels[0], line=dict(color=colors[0])))
    else:
        fig.add_trace(go.Scatter(y=streams[0], mode='lines', name=labels[0], line=dict(color=colors[0])))
    fig.update_yaxes(title_text=labels[0], secondary_y=False)

    # Second stream
    if num_streams > 1:
        if isinstance(streams[1][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[1]], mode='lines', text=[x[0] for x in streams[1]], name=labels[1], line=dict(color=colors[1])))
        else:
            fig.add_trace(go.Scatter(y=streams[1], mode='lines', name=labels[1], line=dict(color=colors[1])))
        fig.update_layout(yaxis3=dict(title=labels[1], titlefont=dict(color=colors[1]), tickfont=dict(color=colors[1]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][1]['yaxis'] = 'y3'


    # Third stream
    if num_streams > 2:
        if isinstance(streams[2][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[2]], mode='lines', text=[x[0] for x in streams[2]], name=labels[2], line=dict(color=colors[2])))
        else:
            fig.add_trace(go.Scatter(y=streams[2], mode='lines', name=labels[2], line=dict(color=colors[2])))
        fig.update_layout(yaxis4=dict(title=labels[2], titlefont=dict(color=colors[2]), tickfont=dict(color=colors[2]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][2]['yaxis'] = 'y4'

    # Fourth stream
    if num_streams > 3:
        if isinstance(streams[3][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[3]], mode='lines', text=[x[0] for x in streams[3]], name=labels[3], line=dict(color=colors[3])))
        else:
            fig.add_trace(go.Scatter(y=streams[3], mode='lines', name=labels[3], line=dict(color=colors[3])))
        fig.update_layout(yaxis5=dict(title=labels[3], titlefont=dict(color=colors[3]), tickfont=dict(color=colors[3]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][3]['yaxis'] = 'y5'

    # Fifth stream
    if num_streams > 4:
        if isinstance(streams[4][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[4]], mode='lines', text=[x[0] for x in streams[4]], name=labels[4], line=dict(color=colors[4])))
        else:
            fig.add_trace(go.Scatter(y=streams[4], mode='lines', name=labels[4], line=dict(color=colors[4])))
        fig.update_layout(yaxis6=dict(title=labels[4], titlefont=dict(color=colors[4]), tickfont=dict(color=colors[4]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][4]['yaxis'] = 'y6'

    # Sixth stream
    if num_streams > 5:
        if isinstance(streams[5][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[5]], mode='lines', text=[x[0] for x in streams[5]], name=labels[5], line=dict(color=colors[5])))
        else:
            fig.add_trace(go.Scatter(y=streams[5], mode='lines', name=labels[5], line=dict(color=colors[5])))
        fig.update_layout(yaxis7=dict(title=labels[5], titlefont=dict(color=colors[5]), tickfont=dict(color=colors[5]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][5]['yaxis'] = 'y7'

    # Seventh stream
    if num_streams > 6:
        if isinstance(streams[6][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[6]], mode='lines', text=[x[0] for x in streams[6]], name=labels[6], line=dict(color=colors[6])))
        else:
            fig.add_trace(go.Scatter(y=streams[6], mode='lines', name=labels[6], line=dict(color=colors[6])))
        fig.update_layout(yaxis8=dict(title=labels[6], titlefont=dict(color=colors[6]), tickfont=dict(color=colors[6]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][6]['yaxis'] = 'y8'

    # Eighth stream
    if num_streams > 7:
        if isinstance(streams[7][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[7]], mode='lines', text=[x[0] for x in streams[7]], name=labels[7], line=dict(color=colors[7])))
        else:
            fig.add_trace(go.Scatter(y=streams[7], mode='lines', name=labels[7], line=dict(color=colors[7])))
        fig.update_layout(yaxis9=dict(title=labels[7], titlefont=dict(color=colors[7]), tickfont=dict(color=colors[7]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][7]['yaxis'] = 'y9'

    # Ninth stream
    if num_streams > 8:
        if isinstance(streams[8][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[8]], mode='lines', text=[x[0] for x in streams[8]], name=labels[8], line=dict(color=colors[8])))
        else:
            fig.add_trace(go.Scatter(y=streams[8], mode='lines', name=labels[8], line=dict(color=colors[8])))
        fig.update_layout(yaxis10=dict(title=labels[8], titlefont=dict(color=colors[8]), tickfont=dict(color=colors[8]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][8]['yaxis'] = 'y10'

    # Tenth stream
    if num_streams > 9:
        if isinstance(streams[9][0], tuple):
            fig.add_trace(go.Scatter(y=[x[1] for x in streams[9]], mode='lines', text=[x[0] for x in streams[9]], name=labels[9], line=dict(color=colors[9])))
        else:
            fig.add_trace(go.Scatter(y=streams[9], mode='lines', name=labels[9], line=dict(color=colors[9])))
        fig.update_layout(yaxis11=dict(title=labels[9], titlefont=dict(color=colors[9]), tickfont=dict(color=colors[9]), anchor="x", overlaying="y", side="right", position=1))
        fig['data'][9]['yaxis'] = 'y11'

    pio.write_html(fig, file=filename,auto_open=True)


import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio


def plot_candlestick_chart_to_html(df, output_file='candlestick_chart.html', title="Options OHLC Candlestick Chart"):
    """
    Function to plot a candlestick chart using Plotly from a DataFrame containing OHLC data and save it as HTML.

    Parameters:
    - df: pandas DataFrame containing 'date_timestamp', 'open', 'high', 'low', 'close' columns.
    - output_file: The name of the output HTML file (default: 'candlestick_chart.html').
    - title: Title of the chart (default: "Options OHLC Candlestick Chart")

    Returns:
    - None. Saves the candlestick chart as an HTML file.
    """
    # Ensure the 'date_timestamp' column is of datetime type
    df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['date_timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    )])

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified',  # Add unified hover mode
        xaxis_rangeslider_visible=False,  # Hide the range slider (optional)
        template="plotly_dark",  # Optional: Dark theme
    )

    # Save the figure as an HTML file
    pio.write_html(fig, file=output_file, auto_open=True)

# Example usage:
# plot_candlestick_chart_to_html(df, 'my_candlestick_chart.html')



def append_file_contents(source_file, target_file):
    # Open the source file in read mode and the target file in append mode
    with open(source_file, 'r') as src, open(target_file, 'a') as tgt:
        # Read from source and append to target
        tgt.write(src.read())

def clean_file(file_path):
    # Open the file in write mode, which automatically clears its contents
    with open(file_path, 'w') as file:
        pass  # Just open and close to clean the file


def filter_dataframe(df, filter_dict):
    """
    Filters a DataFrame based on the provided dictionary of conditions.

    Parameters:
    df : pandas.DataFrame
        The DataFrame to be filtered.
    filter_dict : dict
        A dictionary where keys are column names and values are the values to filter by.

    Returns:
    pandas.DataFrame
        The filtered DataFrame.
    """
    # Apply the filter conditions from the dictionary
    filtered_df = df
    for key, value in filter_dict.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    return filtered_df
