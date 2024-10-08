from datetime import datetime, time, timedelta
from datetime import datetime, date
import psycopg2
import pandas as pd
import numpy as np
import math
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from Demos.RegCreateKeyTransacted import keyname
from adodbapi.apibase import onIronPython
from pandas.core.nanops import nanmax

import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import FileLink, display

#
# def syn_future(call_strikes, put_strikes,fut_close, risk_free_rate = 0.12):
#     """
#     Calculate the synthetic future price using put-call parity.
#
#     Parameters:
#     call_price (float): The price of the call option.
#     put_price (float): The price of the put option.
#     strike_price (float): The strike price of the options.
#     expiry_date (str): The expiration date in 'YYYY-MM-DD' format.
#     risk_free_rate (float): The annual risk-free interest rate (e.g., 0.05 for 5%).
#
#     Returns:
#     float: The synthetic future price.
#     """
#
#     call_strike_values = call_strikes['strike']
#     put_strike_values = put_strikes['strike']
#     # Find common strikes between both DataFrames
#     common_strikes = set(call_strike_values).intersection(set(put_strike_values))
#
#     # Filter both DataFrames to retain only the rows with common strikes
#     strikes_cf = call_strikes[call_strikes['strike'].isin(common_strikes)]
#     strikes_pf = put_strikes[put_strikes['strike'].isin(common_strikes)]
#     ci, pi = np.argmin(abs(strikes_cf['strike'] - fut_close)), np.argmin(abs(strikes_pf['strike'] - fut_close))
#     indexx = ci
#     min_val = abs(strikes_pf.iloc[pi]['close'] - strikes_cf.iloc[pi]['close'])
#     for i in range(max(0, ci - 5), min(ci + 5, min(len(strikes_cf), len(strikes_pf)))):
#         if abs(abs(strikes_pf.iloc[i]['close'] - strikes_cf.iloc[i]['close'])) < min_val:
#             min_val = abs(strikes_pf.iloc[i]['close'] - strikes_cf.iloc[i]['close'])
#             indexx = i
#
#     call_price, put_price, strike_price, expiry_date = strikes_cf.iloc[indexx]['close'] , strikes_pf.iloc[indexx]['close'] , strikes_cf.iloc[indexx]['strike'] , strikes_cf.iloc[indexx]['expiry']
#
#     # Convert expiry_date to a datetime object
#     expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
#
#     # Get the current date
#     current_date = datetime.now()
#
#     # Calculate the time to expiration in years
#     time_to_expiration = (expiry_date - current_date).days / 365.0
#
#     # Calculate the present value of the strike price
#     present_value_strike = strike_price * math.exp(-risk_free_rate * time_to_expiration)
#
#     # Calculate the synthetic future price
#     synthetic_future_price = call_price - put_price + present_value_strike
#
#     return synthetic_future_price
#

import math
from datetime import datetime, date, time

from requests import options


def syn_future(call_strikes, put_strikes, fut_close, risk_free_rate=0.12):
    """
    Calculate the synthetic future price using put-call parity.

    Parameters:
    call_strikes (DataFrame): DataFrame containing call option data of current datetime .
    put_strikes (DataFrame): DataFrame containing put option data of current datetime .
    fut_close (float): The close price of the future.
    risk_free_rate (float): The annual risk-free interest rate (e.g., 0.12 for 12%).

    Returns:
    float: The synthetic future price.
    """

    call_strike_values = call_strikes['strike']
    put_strike_values = put_strikes['strike']

    # Find common strikes between both DataFrames
    common_strikes = set(call_strike_values).intersection(set(put_strike_values))

    if not common_strikes:
        return -1
    # Filter both DataFrames to retain only the rows with common strikes
    # now both strikes_cf and strikes_pf have same strikes ( only ohlc data are different )
    strikes_cf = call_strikes[call_strikes['strike'].isin(common_strikes)]
    strikes_pf = put_strikes[put_strikes['strike'].isin(common_strikes)]

    # Find the index of the strike price closest to the future close price
    # here both ci and pi are equal as strikes are same set of values for both
    ci = np.argmin(abs(strikes_cf['strike'] - fut_close))
    pi = np.argmin(abs(strikes_pf['strike'] - fut_close))

    indexx = ci  # pick any

    min_val = abs(strikes_pf.iloc[pi]['close'] - strikes_cf.iloc[pi]['close'])
    # finding min difference between close values of 5 above and 5 below available strikes
    for i in range(max(0, ci - 5), min(ci + 5, min(len(strikes_cf), len(strikes_pf)))):
        if abs(abs(strikes_pf.iloc[i]['close'] - strikes_cf.iloc[i]['close'])) < min_val:
            min_val = abs(strikes_pf.iloc[i]['close'] - strikes_cf.iloc[i]['close'])
            indexx = i
    # now indexx is our index of strike having min difference between call and put close values
    call_price = strikes_cf.iloc[indexx]['close']
    put_price = strikes_pf.iloc[indexx]['close']
    strike_price = strikes_cf.iloc[indexx]['strike']
    expiry_date = strikes_cf.iloc[indexx]['expiry']

    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
    elif isinstance(expiry_date, date):
        expiry_date = datetime.combine(expiry_date, time(15, 29))

    # Get the current date
    current_date = datetime.now()

    # Calculate the time to expiration in years
    time_to_expiration = (expiry_date - current_date).total_seconds() / (365.0 * 24 * 3600)

    if time_to_expiration < 0:
        time_to_expiration = 0
    # Calculate the present value of the strike price
    present_value_strike = strike_price * math.exp(-risk_free_rate * time_to_expiration)

    # Calculate the synthetic future price
    synthetic_future_price = call_price - put_price + present_value_strike
    return synthetic_future_price


def txt_to_html(txt_file, html_file):
    # Open the txt file to read and html file to write
    with open(txt_file, "r") as txt, open(html_file, "w") as html:
        # Start the HTML structure
        html.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Logs</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
                th { background-color: #f2f2f2; }
                .success { color: green; }
                .error { color: red; }
                .highlight { background-color: #f9f9f9; }
                .log-entry { padding: 10px 0; border-bottom: 1px solid #ddd; }
                .log-timestamp { font-weight: bold; }
            </style>
        </head>
        <body>
        <h1>Trading Strategy Logs</h1>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Log Message</th>
                </tr>
            </thead>
            <tbody>
        """)

        # Read each line from the txt file
        for line in txt:
            line = line.strip()  # Remove any leading/trailing whitespace
            if line:  # If the line is not empty
                # If the line contains a timestamp, treat it as a separate log entry
                if "streak" in line or "delta" in line or "Movement" in line or "Shifting" in line or "End of day" in line:
                    html.write(f"""
                        <tr class="highlight">
                            <td class="log-timestamp">{line.split()[0]} {line.split()[1]}</td>
                            <td>{line}</td>
                        </tr>
                    """)
                else:
                    # For all other logs without a timestamp, print normally
                    html.write(f"""
                        <tr>
                            <td></td>
                            <td>{line}</td>
                        </tr>
                    """)

        # End the HTML structure
        html.write("""
            </tbody>
        </table>
        </body>
        </html>
        """)

    print(f"HTML log file generated: {html_file}")


def syn_future(call_strikes, put_strikes, fut_close, risk_free_rate=0.12):
    """
    Calculate the synthetic future price using put-call parity.

    Parameters:
    call_strikes (DataFrame): DataFrame containing call option data.
    put_strikes (DataFrame): DataFrame containing put option data.
    fut_close (float): The close price of the future.
    risk_free_rate (float): The annual risk-free interest rate (e.g., 0.12 for 12%).

    Returns:
    float: The synthetic future price.
    """
    call_strike_values = call_strikes['strike']
    put_strike_values = put_strikes['strike']

    # Find common strikes between both DataFrames
    common_strikes = set(call_strike_values).intersection(set(put_strike_values))

    if not common_strikes:
        raise ValueError(
            f"No common strikes found between call and put options at {call_strike_values.iloc[0]['date_timestamp']}.")

    # Filter both DataFrames to retain only the rows with common strikes
    strikes_cf = call_strikes[call_strikes['strike'].isin(common_strikes)]
    strikes_pf = put_strikes[put_strikes['strike'].isin(common_strikes)]

    # Find the index of the strike price closest to the future close price
    ci = np.argmin(abs(strikes_cf['strike'] - fut_close))
    pi = np.argmin(abs(strikes_pf['strike'] - fut_close))

    indexx = ci
    min_val = abs(strikes_pf.iloc[pi]['close'] - strikes_cf.iloc[pi]['close'])

    for i in range(max(0, ci - 5), min(ci + 5, min(len(strikes_cf), len(strikes_pf)))):
        if abs(abs(strikes_pf.iloc[i]['close'] - strikes_cf.iloc[i]['close'])) < min_val:
            min_val = abs(strikes_pf.iloc[i]['close'] - strikes_cf.iloc[i]['close'])
            indexx = i

    call_price = strikes_cf.iloc[indexx]['close']
    put_price = strikes_pf.iloc[indexx]['close']
    strike_price = strikes_cf.iloc[indexx]['strike']
    expiry_date = strikes_cf.iloc[indexx]['expiry']

    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
    elif isinstance(expiry_date, date):
        expiry_date = datetime.combine(expiry_date, time(15, 29))

    # Get the current date
    current_date = datetime.now()

    # Calculate the time to expiration in years
    time_to_expiration = (expiry_date - current_date).total_seconds() / (365.0 * 24 * 3600)

    if time_to_expiration < 0:
        time_to_expiration = 0
    # Calculate the present value of the strike price
    present_value_strike = strike_price * math.exp(-risk_free_rate * time_to_expiration)

    # Calculate the synthetic future price
    synthetic_future_price = call_price - put_price + present_value_strike
    return synthetic_future_price


def GetallOptionsClosetoDelta(options, delta, delta_range, date_timestamp, opt_type, file):
    delta = abs(delta)
    if opt_type == 'PE':
        temp = options[(options['delta'].notna()) & (options['date_timestamp'] == date_timestamp) & (
                    abs(options['delta'] + delta) <= delta_range)].sort_values(by='delta')
        if temp.empty:
            return -1, temp
        abs_diff = np.abs(temp['delta'] - delta)
        print(f"deltas {opt_type}", file=file)
        print(temp['delta'], file=file)
        key = np.argmin(abs_diff)
        print(key, file=file)
        if key - 1 >= 0 and temp.iloc[key]['delta'] > delta:
            key -= 1
        option = temp.iloc[key]
        return key, option
    else:
        temp = options[(options['delta'].notna()) & (options['date_timestamp'] == date_timestamp) & (
                    abs(options['delta'] - delta) <= delta_range)].sort_values(by='delta')
        if temp.empty:
            return -1, temp
        abs_diff = np.abs(temp['delta'] - delta)
        print(f"deltas {opt_type}", file=file)
        print(temp['delta'], file=file)
        key = np.argmin(abs_diff)
        print(key, file=file)
        if key + 1 < len(temp) and temp.iloc[key]['delta'] < delta:
            key += 1
        option = temp.iloc[key]
        return key, option
    return None


def compact(trade):
    formatted_string = f"""
    Trade Details:
    ---------------
    Position: {trade['pos']} | Sharpe Ratio: {trade['sharpie_ratio']} | Drawdown: {trade['drawdown']} | Sell_id: {trade['sell_id']} | Buy_id: {trade['buy_id']}
    Entry: {trade['entry_time']} | Strike: {trade['strike']} | Entry Price: {trade['entry_price']} | Exit: {trade['exit_time']} | Exit Price: {trade['exit_price']}
    Capital: {trade['capital']} | Rollover: {trade['rollover']} | PnL: {trade['pnl']} | PnL Sum: {trade['pnl_sum']}
    Transaction Cost: {trade['transaction_cost']} | Slippage: {trade['slippage']}
    Stop Loss: {trade['stop_loss']} | Profit Cap: {trade['profit_cap']} | Entry_Delta: {trade['entry_delta']} | Exit_Delta: {trade['exit_delta']}
    Logs: {trade['logs']}
    """
    return formatted_string


def shifting(dt, Option, Option_, io, id, selling, buying, type_, capital, rollover, initial_capital, obj, execs,
             trades, row, file, net_pnl):
    capital, rollover = buy(Option, buying, io, id, obj, row, type_, capital, rollover, initial_capital)
    execs.append({
        'id': id, 'time': dt, 'opt_type': 'PE', 'log': f"Buying to Shift", 'delta': Option['delta'],
        'strike': Option['strike'], 'price': Option[buying], 'pnl': 0, 'trade_type': 'BUYING',
    })
    id += 1
    print(f"Buying {type_} for shifting : at {dt}", file=file)
    print(compact(obj), file=file)
    trades.append(obj.copy())
    net_pnl.append(net_pnl[-1] + obj['pnl'])
    obj.update({'pos': 0, 'movement': [], 'sharpie_ratio': 0, 'drawdown': 0, 'entry_time': None, 'strike': None,
                'entry_price': 0.0, 'exit_time': None, 'capital': initial_capital, 'rollover': 0, 'exit_price': None,
                'pnl': 0, 'type': None, 'pnl_sum': 0, 'transaction_cost': 0, 'slippage': 0, 'logs': [], 'stop_loss': 0,
                'profit_cap': 0, 'entry_delta': 0.0, 'exit_delta': 0.0})

    print(f"shifting the {type_} option from delta {Option['delta']} to {Option_['delta']}", file=file)
    capital, rollover = sell(Option_, selling, io, id, obj, row, type_, capital, rollover, initial_capital)
    execs.append({
        'id': id, 'time': dt, 'opt_type': type_, 'log': f"Shifted", 'delta': Option_['delta'],
        'strike': Option_['strike'], 'price': Option_[selling], 'pnl': 0, 'trade_type': 'SELLING',
    })
    id += 1
    print(f"{type_} sold Successfully", file=file)
    return id, capital, rollover


def generate_trades_syn(eh, em, id, futures, ce, pe, gen_call, gen_put, io=0, buying='open', selling='close',
                        DeltaLB=0.2, delta_range=0.3, DeltaUB=0.5, initial_capital=1000,
                        sharpie_per_trade_index='open'):
    trades, call_iv, put_iv, execs, running_pnl, net_pnl, capital, rollover, streak, iv, civ = [], [], [], [], [], [
        0], initial_capital, 0, 0, [], []
    put = {'pos': 0, 'sell_id': 0, 'entry_time': None, 'strike': None, 'entry_price': 0, 'iv': [],
           'buy_id': 0, 'exit_time': None, 'exit_price': None, 'capital': initial_capital, 'rollover': 0,
           'pnl': 0, 'type': None, 'pnl_sum': 0, 'transaction_cost': 0, 'slippage': 0, 'logs': [], 'sharpie_ratio': 0,
           'drawdown': 0,
           'stop_loss': 0, 'profit_cap': 0, 'entry_delta': 0.0, 'exit_delta': 0.0, 'movement': []}

    call, positions, difference, call_movement, put_movement, delta_call, delta_put, cc = put.copy(), [], [], [], [], [], [], 0
    UpTrand, DownTrand = True, False
    with open(f"_Logs.txt", "w") as file:
        for i in range(1, len(futures) - 1):
            row = futures.iloc[i]
            dt, row_i = row['date_timestamp'], futures.iloc[i - 1]
            delta_put.append((dt, put['entry_delta']));
            delta_call.append((dt, call['entry_delta']));
            positions.append((dt, put['pos']))
            print("\n", file=file);
            print("`", cc, "`", file=file);
            print("Time", dt, f"Spot = {row['close']}", file=file);
            cc += 1
            c_iv, p_iv = gen_call[(gen_call['date_timestamp'] == dt)], gen_put[(gen_put['date_timestamp'] == dt)]
            civ_ind, piv_ind = np.argmin(np.abs(c_iv['strike'] - c_iv['syn_fut'])), np.argmin(
                np.abs(p_iv['strike'] - p_iv['syn_fut']))
            iv.append((c_iv.iloc[civ_ind]['iv'] + p_iv.iloc[piv_ind]['iv']) / 2)
            if len(iv) < 3:
                civ.append(np.mean(iv))
            else:
                mn = (np.mean(iv[-3:-1]))
                civ.append(mn)
            if row['date_timestamp'].date() != row_i['date_timestamp'].date():
                streak = 0
            if abs(row['close'] - row_i['close']) * 100 < 2 * 1 * row_i['close']:
                streak += 1
            else:
                streak = 0
            print(f"streak: {streak}", file=file)
            # if dt.time() == time(eh,em,0):
            if streak >= 1 and dt.time() <= time(15, 00, 0):
                streak = 0
                if put['pos'] == 0 and call['pos'] == 0:
                    Cindex, Call_Option = GetallOptionsClosetoDelta(gen_call, DeltaLB, delta_range / 2, dt, 'CE', file)
                    if Cindex == -1:
                        print(
                            f"No Call Available in the range of [{DeltaLB - delta_range}, {DeltaLB + delta_range}] at Timestamp = {dt}",
                            file=file)
                    else:
                        Pindex, Put_Option = GetallOptionsClosetoDelta(gen_put, Call_Option['delta'], delta_range / 2,
                                                                       dt, 'PE', file)
                        if Pindex == -1:
                            print(
                                f"No Put Available in the range of [{-Call_Option['delta'] - delta_range}, {-Call_Option['delta'] + delta_range}] (near to call option delta) at Timestamp = {dt}",
                                file=file)
                        else:
                            print(
                                f"entry with call strike {Call_Option['strike']} and put strike = {Put_Option['strike']} at dt = {dt}",
                                file=file)
                            capital, rollover = sell(Call_Option, selling, io, id, call, row, 'CE', capital, rollover,
                                                     initial_capital)
                            execs.append({
                                'id': id, 'time': dt, 'opt_type': 'CE', 'log': f"Initial Entry",
                                'delta': Call_Option['delta'], 'strike': Call_Option['strike'],
                                'price': Call_Option[selling], 'pnl': 0, 'trade_type': 'SELLING',
                            })
                            id += 1
                            print(f"Selling Call for taking entry :", dt, file=file)
                            print(compact(call), file=file)
                            capital, rollover = sell(Put_Option, selling, io, id, put, row, 'PE', capital, rollover,
                                                     initial_capital)
                            execs.append({
                                'id': id, 'time': dt, 'opt_type': 'PE', 'log': f"Initial Entry",
                                'delta': Put_Option['delta'], 'strike': Put_Option['strike'],
                                'price': Put_Option[selling], 'pnl': 0, 'trade_type': 'SELLING',
                            })
                            id += 1
                            print(f"Selling Put for taking entry :", dt, file=file)
                            print(compact(put), file=file)
                else:
                    print("position is not closed yet", file=file)

            if call['pos'] == -1 and put['pos'] == -1:
                # Add checks for empty DataFrames before accessing their rows
                print("Call and put have short Positions", file=file)
                if not gen_call[
                    (gen_call['date_timestamp'] == dt) & (gen_call['strike'] == call['strike'])].empty and not gen_put[
                    (gen_put['date_timestamp'] == dt) & (gen_put['strike'] == put['strike'])].empty:

                    Call_Option = \
                    gen_call[(gen_call['date_timestamp'] == dt) & (gen_call['strike'] == call['strike'])].iloc[0]
                    Put_Option = gen_put[(gen_put['date_timestamp'] == dt) & (gen_put['strike'] == put['strike'])].iloc[
                        0]

                    print("Current Deltas: ", Call_Option['delta'], Put_Option['delta'],
                          round(Call_Option['delta'] + Put_Option['delta'], 3), file=file)

                    if abs(Call_Option['delta'] + Put_Option['delta']) > delta_range:
                        CallD, PutD = Call_Option['delta'], Put_Option['delta']

                        if CallD > PutD:
                            if CallD > DeltaUB:
                                DownTrand = True
                                UpTrand = False
                                Cindex_, Call_Option_ = GetallOptionsClosetoDelta(gen_call, Put_Option['delta'],
                                                                                  delta_range, dt, 'CE', file)
                                if Cindex_ == -1:
                                    print(
                                        f"No call Options Available in the range of [{Put_Option['delta'] - delta_range}, {Put_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                        file=file)
                                else:
                                    id, capital, rollover = shifting(dt, Call_Option, Call_Option_, io, id, selling,
                                                                     buying, 'CE', capital, rollover, initial_capital,
                                                                     call, execs, trades, row, file, net_pnl)
                            else:
                                if PutD >= DeltaLB:
                                    if UpTrand:
                                        Pindex_, Put_Option_ = GetallOptionsClosetoDelta(gen_put, Call_Option['delta'],
                                                                                         delta_range, dt, 'PE', file)
                                        if Pindex_ == -1:
                                            print(
                                                f"No Put Options Available in the range of [{-Call_Option['delta'] - delta_range}, {-Call_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                                file=file)
                                        else:
                                            id, capital, rollover = shifting(dt, Put_Option, Put_Option_, io, id,
                                                                             selling, buying, 'PE', capital, rollover,
                                                                             initial_capital, put, execs, trades, row,
                                                                             file, net_pnl)
                                    else:
                                        Cindex_, Call_Option_ = GetallOptionsClosetoDelta(gen_call, Put_Option['delta'],
                                                                                          delta_range, dt, 'CE', file)
                                        if Cindex_ == -1:
                                            print(
                                                f"No call Options Available in the range of [{Put_Option['delta'] - delta_range}, {Put_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                                file=file)
                                        else:
                                            id, capital, rollover = shifting(dt, Call_Option, Call_Option_, io, id,
                                                                             selling, buying, 'CE', capital, rollover,
                                                                             initial_capital, call, execs, trades, row,
                                                                             file, net_pnl)

                                else:
                                    DownTrand = False
                                    UpTrand = True
                                    Pindex_, Put_Option_ = GetallOptionsClosetoDelta(gen_put, Call_Option['delta'],
                                                                                     delta_range, dt, 'PE', file)
                                    if Pindex_ == -1:
                                        print(
                                            f"No Put Options Available in the range of [{-Call_Option['delta'] - delta_range}, {-Call_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                            file=file)
                                    else:
                                        id, capital, rollover = shifting(dt, Put_Option, Put_Option_, io, id, selling,
                                                                         buying, 'PE', capital, rollover,
                                                                         initial_capital, put, execs, trades, row, file,
                                                                         net_pnl)
                        else:
                            if PutD > DeltaUB:
                                DownTrand = True
                                UpTrand = False
                                Pindex_, Put_Option_ = GetallOptionsClosetoDelta(gen_put, Call_Option['delta'],
                                                                                 delta_range, dt, 'PE', file)
                                if Pindex_ == -1:
                                    print(
                                        f"No Put Options Available in the range of [{-Call_Option['delta'] - delta_range}, {-Call_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                        file=file)
                                else:
                                    id, capital, rollover = shifting(dt, Put_Option, Put_Option_, io, id, selling,
                                                                     buying, 'PE', capital, rollover, initial_capital,
                                                                     put, execs, trades, row, file, net_pnl)

                            else:
                                if CallD >= DeltaLB:
                                    if UpTrand:
                                        Cindex_, Call_Option_ = GetallOptionsClosetoDelta(gen_call, Put_Option['delta'],
                                                                                          delta_range, dt, 'CE', file)
                                        if Cindex_ == -1:
                                            print(
                                                f"No call Options Available in the range of [{Put_Option['delta'] - delta_range}, {Put_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                                file=file)
                                        else:
                                            id, capital, rollover = shifting(dt, Call_Option, Call_Option_, io, id,
                                                                             selling, buying, 'CE', capital, rollover,
                                                                             initial_capital, call, execs, trades, row,
                                                                             file, net_pnl)
                                    else:
                                        Pindex_, Put_Option_ = GetallOptionsClosetoDelta(gen_put, Call_Option['delta'],
                                                                                         delta_range, dt, 'PE', file)
                                        if Pindex_ == -1:
                                            print(
                                                f"No Put Options Available in the range of [{-Call_Option['delta'] - delta_range}, {-Call_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                                file=file)
                                        else:
                                            id, capital, rollover = shifting(dt, Put_Option, Put_Option_, io, id,
                                                                             selling, buying, 'PE', capital, rollover,
                                                                             initial_capital, put, execs, trades, row,
                                                                             file, net_pnl)

                                else:
                                    DownTrand = False
                                    UpTrand = True
                                    Cindex_, Call_Option_ = GetallOptionsClosetoDelta(gen_call, Put_Option['delta'],
                                                                                      delta_range, dt, 'CE', file)
                                    if Cindex_ == -1:
                                        print(
                                            f"No call Options Available in the range of [{Put_Option['delta'] - delta_range}, {Put_Option['delta'] + delta_range}] at Timestamp = {dt}",
                                            file=file)
                                    else:
                                        id, capital, rollover = shifting(dt, Call_Option, Call_Option_, io, id, selling,
                                                                         buying, 'CE', capital, rollover,
                                                                         initial_capital, call, execs, trades, row,
                                                                         file,
                                                                         net_pnl)

                        # if (Call_Option['delta'] > -Put_Option['delta'] and Call_Option['delta'] <= delta_upper_bound) or (-Put_Option['delta'] > Call_Option['delta'] and -Put_Option['delta'] >= delta_upper_bound):
                        #     Pindex_, Put_Option_ = GetallOptionsClosetoDelta(gen_put, Call_Option['delta'], delta_range, dt,'PE',file)
                        #     if Pindex_ == -1:
                        #         print(f"No Put Options Available in the range of [{-Call_Option['delta'] - delta_range}, {-Call_Option['delta'] + delta_range}] at Timestamp = {dt}",file=file)
                        #     else:
                        #         id,capital,rollover = shifting(dt,Put_Option,Put_Option_,io,id,selling,buying,'PE',capital,rollover,initial_capital,put,execs,trades,row,file,net_pnl)
                        #
                        #
                        # elif (-Put_Option['delta'] > Call_Option['delta'] and -Put_Option['delta'] <= delta_upper_bound) or (Call_Option['delta'] > -Put_Option['delta'] and Call_Option['delta'] >= delta_upper_bound):
                        #     Cindex_, Call_Option_ = GetallOptionsClosetoDelta(gen_call, Put_Option['delta'], delta_range, dt,'CE',file)
                        #     if Cindex_ == -1:
                        #         print(f"No call Options Available in the range of [{Put_Option['delta'] - delta_range}, {Put_Option['delta'] + delta_range}] at Timestamp = {dt}",file=file)
                        #     else:
                        #         id,capital,rollover = shifting(dt,Call_Option,Call_Option_,io,id,selling,buying,'CE',capital,rollover,initial_capital,call,execs,trades,row,file,net_pnl)

                else:
                    if gen_call[(gen_call['date_timestamp'] == dt) & (gen_call['strike'] == call['strike'])].empty:
                        print(f"No Availability of call strike = {call['strike']} at dt = {dt} for Delta Shifting",
                              file=file)
                    if gen_put[(gen_put['date_timestamp'] == dt) & (gen_put['strike'] == put['strike'])].empty:
                        print(f"No Availability of put with strike {put['strike']} at dt = {dt} for Delta Shifting",
                              file=file)

            else:
                print(f"No Open Positions", file=file)

            if put['pos'] == -1 and call['pos'] == -1:
                pe_filtered = pe[(pe['date_timestamp'] == dt) & (pe['strike'] == put['strike'])]
                ce_filtered = ce[(ce['date_timestamp'] == dt) & (ce['strike'] == call['strike'])]
                if not pe_filtered.empty:
                    put_iv.append(pe_filtered.iloc[0]['iv'])
                    tm_pe = pe_filtered.iloc[0]['close']
                    put['movement'].append(tm_pe)
                    print("Put Movement : ", len(put['movement']), tm_pe, file=file)
                    put_movement.append(put['strike'])
                else:
                    put_movement.append(put_movement[-1])
                    print(f"Put movement not found for updating at strike {put['strike']} {dt}", file=file)

                if not ce_filtered.empty:
                    call_iv.append(ce_filtered.iloc[0]['iv'])
                    tm_ce = ce_filtered.iloc[0]['close']
                    call['movement'].append(tm_ce)
                    print("Call Movement : ", len(call['movement']), tm_ce, file=file)
                    call_movement.append(call['strike'])
                else:
                    call_movement.append(call_movement[-1])
                    print(f"Call movement not found for updating at strike {call['strike']}  {dt}", file=file)
            else:
                call_iv.append(0)
                put_iv.append(0)
                call_movement.append(call['strike'])
                put_movement.append(put['strike'])

            if row['date_timestamp'].time() >= time(15, 0, 0):
                if row['date_timestamp'].time() == time(15, 0, 0):
                    print(f"End of day reached at dt {dt} ", file=file)
                if put['pos'] == -1:
                    if not gen_put[(gen_put['date_timestamp'] == dt) & (gen_put['strike'] == put['strike'])].empty:
                        Put_Option = \
                        gen_put[(gen_put['date_timestamp'] == dt) & (gen_put['strike'] == put['strike'])].iloc[0]
                        capital, rollover = buy(Put_Option, buying, io, id, put, row, "PE", capital, rollover,
                                                initial_capital)
                        execs.append({
                            'id': id, 'time': dt, 'opt_type': 'PE', 'log': f"EoD Sq_off",
                            'delta': Put_Option['delta'], 'strike': Put_Option['strike'],
                            'price': Put_Option[selling], 'pnl': 0, 'trade_type': 'BUYING',
                        })
                        id += 1
                        print(f"Put squared of at EoD {dt}", file=file)
                        print(compact(put), file=file)
                        trades.append(put.copy())
                        net_pnl.append(put['pnl'] + net_pnl[-1])
                        put.update({'pos': 0, 'movement': [], 'sharpie_ratio': 0, 'drawdown': 0, 'entry_time': None,
                                    'strike': None, 'entry_price': 0.0, 'exit_time': None, 'capital': initial_capital,
                                    'rollover': 0, 'exit_price': None, 'pnl': 0, 'type': None, 'pnl_sum': 0,
                                    'transaction_cost': 0.0, 'slippage': 0, 'logs': [], 'stop_loss': 0, 'profit_cap': 0,
                                    'entry_delta': 0.0, 'exit_delta': 0.0})
                    else:
                        print(f"Not able to find put with strike {put['strike']} at dt = {dt}", file=file)
                if call['pos'] == -1:
                    if not gen_call[(gen_call['date_timestamp'] == dt) & (gen_call['strike'] == call['strike'])].empty:

                        Call_Option = \
                        gen_call[(gen_call['date_timestamp'] == dt) & (gen_call['strike'] == call['strike'])].iloc[0]
                        capital, rollover = buy(Call_Option, buying, io, id, call, row, "CE", capital, rollover,
                                                initial_capital)
                        execs.append({
                            'id': id, 'time': dt, 'opt_type': 'CE', 'log': f"EoD Sq_off",
                            'delta': Call_Option['delta'], 'strike': Call_Option['strike'],
                            'price': Call_Option[buying], 'pnl': 0, 'trade_type': 'BUYING',
                        })
                        id += 1
                        print(f"Call squared off at EoD {dt}", file=file)
                        print(compact(call), file=file)
                        trades.append(call.copy())
                        net_pnl.append(call['pnl'] + net_pnl[-1])
                        call.update({'pos': 0, 'movement': [], 'sharpie_ratio': 0, 'drawdown': 0, 'entry_time': None,
                                     'strike': None, 'entry_price': 0, 'exit_time': None, 'capital': initial_capital,
                                     'rollover': 0, 'exit_price': None, 'pnl': 0, 'type': None, 'pnl_sum': 0.0,
                                     'transaction_cost': 0, 'slippage': 0, 'logs': [], 'stop_loss': 0, 'profit_cap': 0,
                                     'entry_delta': 0.0, 'exit_delta': 0.0})
                    else:
                        print(f'didnt able to find call with strike {call["strike"]} at dt = {dt}', file=file)

            if put['pos'] != 0 and call['pos'] != 0:
                Call_Sq_Off = call['movement'][0] - call['movement'][-1] + (
                            call['movement'][-1] + call['movement'][0]) * (215 / 100000)
                Put_Sq_Off = put['movement'][0] - put['movement'][-1] + (put['movement'][-1] + put['movement'][0]) * (
                            215 / 100000)
                running_pnl.append((row['date_timestamp'], net_pnl[-1] + Call_Sq_Off + Put_Sq_Off))
                print("Running Pnl Update:", round(Call_Sq_Off, 3), round(Put_Sq_Off, 3), file=file)
            else:
                running_pnl.append((row['date_timestamp'], net_pnl[-1]))
            difference.append(call['entry_delta'] + put['entry_delta'])

    if put['pos'] == -1 or call['pos'] == -1:
        if put['pos'] == -1:
            Put_Option = \
            gen_put[(gen_put['strike'] == put['strike'])].sort_values(by='date_timestamp', ascending=False).iloc[0]
            capital, rollover = buy(Put_Option, buying, io, id, put, futures.iloc[-1], "PE", capital, rollover,
                                    initial_capital)
            execs.append({
                'id': id, 'time': Put_Option['date_timestamp'], 'opt_type': 'PE', 'log': f"Final Sq_off",
                'delta': Put_Option['delta'],
                'strike': Put_Option['strike'], 'price': Put_Option[buying], 'pnl': 0, 'trade_type': 'BUYING',
            })
            id += 1

            net_pnl.append(put['pnl'] + net_pnl[-1])
            trades.append(put.copy())

        if call['pos'] == -1:
            Call_Option = \
            gen_call[(gen_call['strike'] == call['strike'])].sort_values(by='date_timestamp', ascending=False).iloc[0]
            capital, rollover = buy(Call_Option, buying, io, id, call, futures.iloc[-1], "CE", capital, rollover,
                                    initial_capital)
            execs.append({
                'id': id, 'time': Call_Option['date_timestamp'], 'opt_type': 'CE', 'log': f"Final Sq_off",
                'delta': Call_Option['delta'],
                'strike': Call_Option['strike'], 'price': Call_Option[buying], 'pnl': 0, 'trade_type': 'BUYING',
            })
            id += 1
            trades.append(call.copy())
            net_pnl.append(call['pnl'] + net_pnl[-1])

    running_pnl.append((futures.iloc[-1]['date_timestamp'], net_pnl[-1]))

    return id, execs, trades, running_pnl, net_pnl, capital, rollover, positions, difference, call_movement, put_movement, delta_put, delta_call, call_iv, put_iv, civ


def sell(option, selling, io, id, obj, row, opt_type, capital, rollover, initial_capital):
    sell_val = option[selling]
    capital += sell_val
    if capital >= initial_capital:
        rollover += capital - initial_capital
        capital = initial_capital
    obj.update({
        'pos': -1, 'sell_id': id,
        'entry_time': row['date_timestamp'],
        'strike': option['strike'],
        'entry_price': sell_val,
        'movement': [],
        'entry_delta': option['delta']
    })
    obj['movement'].append(sell_val)
    return capital, rollover


def buy(option, buying, io, id, obj, row, type, capital, rollover, initial_capital):
    if capital <= option[buying]:
        # Log the insufficient capital scenario
        obj['logs'].append(f"Capital is not sufficient for buying at timestamp {row['date_timestamp']}")
    else:
        # Proceed with buying
        capital -= option[buying]
        obj.update(
            {'pos': 0, 'exit_time': row['date_timestamp'], 'exit_price': option[buying], 'exit_delta': option['delta'],
             'transaction_cost': round((obj['entry_price'] + option[buying]) * (115 / 100000), 3),
             'slippage': round((obj['entry_price'] + option[buying]) * (1 / 1000), 3), 'buy_id': id,
             'pnl': round(
                 obj['entry_price'] - option[buying] - (obj['entry_price'] + option[buying]) * (115 / 100000) - (
                             obj['entry_price'] + option[buying]) * (1 / 1000), 3),
             'type': type})
        obj['movement'].append(option[buying])
        # Calculate sharpie ratio and drawdown, assuming the existence of those functions
        sharpie_ratio = sharpie_per_trade(obj)
        drawdown_val = drawdown_trade(obj['movement'])  # Assuming you have a `drawdown_trade` function
        obj.update({'sharpie_ratio': round(sharpie_ratio, 3), 'drawdown': round(drawdown_val, 3)})
    return capital, rollover


def drawdown(arr):
    max_val = -10000000000
    max_index, i = 0, 0
    drawdown_val = -10000000000
    while i < len(arr):
        if arr[i]['pnl_sum'] > max_val:
            max_val = arr[i]['pnl_sum']
            while i < len(arr) and arr[i]['pnl_sum'] <= max_val:
                drawdown_val = max(drawdown_val, max_val - arr[i]['pnl_sum'])
                i += 1

    return drawdown_val


def sharpie_straddle(trade_, risk_free_per_min=6 / (365 * 24 * 60 * 100)):
    trades = pd.DataFrame(trade_)
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])

    trades['entry_time'] = pd.to_datetime(trades['entry_time'])

    # Extract the date part from 'exit_time'
    trades['exit_date'] = trades['exit_time'].dt.date

    # Group by 'exit_date' and calculate daily PnL and daily Capital (sum of exit prices)
    daily_stats = trades.groupby('exit_date').agg(
        daily_pnl=('pnl', 'sum'),
        daily_capital=('exit_price', 'sum')
    ).reset_index()

    # Replace daily_capital with 1 if it is 0
    daily_stats['daily_capital'] = daily_stats['daily_capital'].replace(0, 1)

    # The rest of your function can continue as before
    # Calculate daily return (daily_pnl / daily_capital)
    daily_stats['daily_return'] = daily_stats['daily_pnl'] / daily_stats['daily_capital']

    # Calculate the average daily return
    average_daily_return = daily_stats['daily_return'].mean()

    # Calculate the standard deviation of daily returns
    if len(daily_stats['daily_return']) > 1:
        std_dev_daily_return = daily_stats['daily_return'].std()
    else:
        std_dev_daily_return = 1
    # Set the annual risk-free rate (e.g., 3% annual risk-free rate)
    annual_risk_free_rate = 0.06

    # Convert the annual risk-free rate to a daily rate
    daily_risk_free_rate = annual_risk_free_rate / 365

    # Calculate the Sharpe Ratio using the daily risk-free rate
    sharpe_ratio = (average_daily_return - daily_risk_free_rate) / std_dev_daily_return

    return sharpe_ratio, daily_stats


def sharpie_straddle_Combine(trade_, risk_free_per_min=12 / (365 * 24 * 60 * 100)):
    tl = []
    for it in trade_.values():
        tl.extend(it)
    trades = pd.DataFrame(tl)
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])

    trades['entry_time'] = pd.to_datetime(trades['entry_time'])

    # Extract the date part from 'exit_time'
    trades['exit_date'] = trades['exit_time'].dt.date

    # Group by 'exit_date' and calculate daily PnL and daily Capital (sum of exit prices)
    daily_stats = trades.groupby('exit_date').agg(
        daily_pnl=('pnl', 'sum'),
        daily_capital=('exit_price', 'sum')
    ).reset_index()

    # Replace daily_capital with 1 if it is 0
    daily_stats['daily_capital'] = daily_stats['daily_capital'].replace(0, 1)

    # The rest of your function can continue as before
    # Calculate daily return (daily_pnl / daily_capital)
    daily_stats['daily_return'] = daily_stats['daily_pnl'] / daily_stats['daily_capital']

    # Calculate the average daily return
    average_daily_return = daily_stats['daily_return'].mean()

    # Calculate the standard deviation of daily returns
    if len(daily_stats['daily_return']) > 1:
        std_dev_daily_return = daily_stats['daily_return'].std()
    else:
        std_dev_daily_return = 1
    # Set the annual risk-free rate (e.g., 3% annual risk-free rate)
    annual_risk_free_rate = 0.06

    # Convert the annual risk-free rate to a daily rate
    daily_risk_free_rate = annual_risk_free_rate / 365

    # Calculate the Sharpe Ratio using the daily risk-free rate
    sharpe_ratio = (average_daily_return - daily_risk_free_rate) / std_dev_daily_return

    return sharpe_ratio, daily_stats


def sharpie_trades(trades):
    entry_per_day, exit_per_day, percentage_pnl = {}, {}, {}

    for trade in trades:
        if trade['entry_time'] not in entry_per_day: entry_per_day[trade['entry_time']] = 0
        if trade['exit_time'] not in exit_per_day: exit_per_day[trade['exit_time']] = 0
        entry_per_day[trade['entry_time']] += trade['entry_price']
        exit_per_day[trade['exit_time']] += trade['exit_price']

    for day in entry_per_day:
        if day not in exit_per_day: exit_per_day[day] = 0
        if day not in entry_per_day: entry_per_day[day] = 0
        percentage_pnl[day] = ((exit_per_day[day] - entry_per_day[day]) / entry_per_day[day]) * 100
    arr = np.array(list(percentage_pnl.values()))
    mean, std_dev, risk_free_rate = np.mean(arr), np.std(arr), 0.0

    sharpie_ratio = (mean - risk_free_rate) / std_dev

    return sharpie_ratio


def lower_bound(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = low + (high - low) // 2
        if arr['strike'].iloc[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def drawdown_trade(arr):
    max_val = -10000000000
    max_index, i = 0, 0
    drawdown_val = -10000000000
    while i < len(arr):
        if arr[i] > max_val:
            max_val = arr[i]
            while i < len(arr) and arr[i] <= max_val:
                drawdown_val = max(drawdown_val, max_val - arr[i])
                i += 1

    return drawdown_val


def sharpie_per_trade(obj, risk_free_per_min=12 / (365 * 24 * 60 * 100)):
    return (obj['entry_price'] - obj['exit_price'] - obj['slippage'] - obj['transaction_cost'] - obj[
        'entry_price'] * risk_free_per_min * (len(obj['movement']) - 1)) / np.std(obj['movement'])


def to_excl(trade_list, columns, filename):
    df = pd.DataFrame(trade_list)
    df = df[columns]
    df.to_excel(f'{filename}.xlsx')


def print_trades_to_html(trades, columns_to_print, filename="trades_output.html"):
    # Start the HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            th {
                cursor: pointer;
            }
        </style>
        <script>
            function sortTable(n) {
                var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                table = document.getElementById("tradesTable");
                switching = true;
                dir = "asc"; // Set the sorting direction to ascending:
                while (switching) {
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];

                        let xValue = x.innerHTML;
                        let yValue = y.innerHTML;

                        // Convert values to numeric or date if applicable
                        let xNumber = parseFloat(xValue.replace(/[^0-9.-]/g, ''));
                        let yNumber = parseFloat(yValue.replace(/[^0-9.-]/g, ''));

                        let xDate = new Date(xValue);
                        let yDate = new Date(yValue);

                        if (!isNaN(xNumber) && !isNaN(yNumber)) {
                            // Numeric sorting
                            if (dir === "asc") {
                                if (xNumber > yNumber) {
                                    shouldSwitch = true;
                                    break;
                                }
                            } else if (dir === "desc") {
                                if (xNumber < yNumber) {
                                    shouldSwitch = true;
                                    break;
                                }
                            }
                        } else if (!isNaN(xDate.getTime()) && !isNaN(yDate.getTime())) {
                            // Date sorting
                            if (dir === "asc") {
                                if (xDate > yDate) {
                                    shouldSwitch = true;
                                    break;
                                }
                            } else if (dir === "desc") {
                                if (xDate < yDate) {
                                    shouldSwitch = true;
                                    break;
                                }
                            }
                        } else {
                            // String sorting
                            if (dir === "asc") {
                                if (xValue.toLowerCase() > yValue.toLowerCase()) {
                                    shouldSwitch = true;
                                    break;
                                }
                            } else if (dir === "desc") {
                                if (xValue.toLowerCase() < yValue.toLowerCase()) {
                                    shouldSwitch = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (shouldSwitch) {
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    } else {
                        if (switchcount === 0 && dir === "asc") {
                            dir = "desc";
                            switching = true;
                        }
                    }
                }
            }
        </script>
    </head>
    <body>
        <table id="tradesTable">
            <thead>
                <tr>
    """

    # Add table headers with sorting
    header_cols = []
    if columns_to_print.get('pos', 0):
        header_cols.append('pos')
    if columns_to_print.get('sell_id', 0):
        header_cols.append('sell_id')
    if columns_to_print.get('buy_id', 0):
        header_cols.append('buy_id')
    if columns_to_print.get('signal_time', 0):
        header_cols.append('signal time')
    if columns_to_print.get('long_ema', 0):
        header_cols.append('long_ema')
    if columns_to_print.get('short_ema', 0):
        header_cols.append('short_ema')
    if columns_to_print.get('entry_time', 0):
        header_cols.append('entry time')
    if columns_to_print.get('strike', 0):
        header_cols.append('strike')
    if columns_to_print.get('entry_price', 0):
        header_cols.append('entry price')
    if columns_to_print.get('exit_time', 0):
        header_cols.append('exit time')
    if columns_to_print.get('exit_price', 0):
        header_cols.append('exit price')
    if columns_to_print.get('pnl', 0):
        header_cols.append('pnl')
    if columns_to_print.get('type', 0):
        header_cols.append('type')
    if columns_to_print.get('pnl_sum', 0):
        header_cols.append('pnl_sum')
    if columns_to_print.get('price', 0):
        header_cols.append('price')
    if columns_to_print.get('logs', 0):
        header_cols.append('logs')
    if columns_to_print.get('movement', 0):
        header_cols.append('movement')
    if columns_to_print.get('sharpie_ratio', 0):
        header_cols.append('sharpie_ratio')
    if columns_to_print.get('drawdown', 0):
        header_cols.append('drawdown')
    if columns_to_print.get('transaction_cost', 0):
        header_cols.append('transaction_cost')
    if columns_to_print.get('slippage', 0):
        header_cols.append('slippage')
    if columns_to_print.get('capital', 0):
        header_cols.append('capital')
    if columns_to_print.get('rollover', 0):
        header_cols.append('rollover')
    if columns_to_print.get('entry_delta', 0):
        header_cols.append('entry_delta')
    if columns_to_print.get('exit_delta', 0):
        header_cols.append('exit_delta')
    if columns_to_print.get('time', 0):
        header_cols.append('time')

    # Add clickable header elements for sorting
    for index, col in enumerate(header_cols):
        html_content += f"<th onclick=\"sortTable({index})\">{col}</th>"

    html_content += """
                </tr>
            </thead>
            <tbody>
    """

    # Add table rows for each trade
    for i, trade in zip(range(len(trades)), trades):
        html_content += "<tr>"
        if columns_to_print.get('pos', 0):
            html_content += f"<td>{trade['pos']}</td>"
        if columns_to_print.get('sell_id', 0):
            html_content += f"<td>{trade['sell_id']}</td>"
        if columns_to_print.get('buy_id', 0):
            html_content += f"<td>{trade['buy_id']}</td>"
        if columns_to_print.get('signal_time', 0):
            html_content += f"<td>{trade['signal_time']}</td>"
        if columns_to_print.get('long_ema', 0):
            html_content += f"<td>{trade['long_ema']}</td>"
        if columns_to_print.get('short_ema', 0):
            html_content += f"<td>{trade['short_ema']}</td>"
        if columns_to_print.get('entry_time', 0):
            html_content += f"<td>{trade['entry_time']}</td>"
        if columns_to_print.get('strike', 0):
            html_content += f"<td>{trade['strike']}</td>"
        if columns_to_print.get('entry_price', 0):
            html_content += f"<td>{trade['entry_price']}</td>"
        if columns_to_print.get('exit_time', 0):
            html_content += f"<td>{trade['exit_time']}</td>"
        if columns_to_print.get('exit_price', 0):
            html_content += f"<td>{trade['exit_price']}</td>"
        if columns_to_print.get('pnl', 0):
            html_content += f"<td>{round(trade['pnl'], 2)}</td>"
        if columns_to_print.get('type', 0):
            html_content += f"<td>{trade['type']}</td>"
        if columns_to_print.get('pnl_sum', 0):
            html_content += f"<td>{trade['pnl_sum']}</td>"
        if columns_to_print.get('price', 0):
            html_content += f"<td>{trade['price']}</td>"
        if columns_to_print.get('logs', 0):
            html_content += f"<td>{trade['logs']}</td>"
        if columns_to_print.get('movement', 0):
            html_content += f"<td>{trade['movement']}</td>"
        if columns_to_print.get('sharpie_ratio', 0):
            html_content += f"<td>{trade['sharpie_ratio']}</td>"
        if columns_to_print.get('drawdown', 0):
            html_content += f"<td>{trade['drawdown']}</td>"
        if columns_to_print.get('transaction_cost', 0):
            html_content += f"<td>{trade['transaction_cost']}</td>"
        if columns_to_print.get('slippage', 0):
            html_content += f"<td>{trade['slippage']}</td>"
        if columns_to_print.get('capital', 0):
            html_content += f"<td>{trade['capital']}</td>"
        if columns_to_print.get('rollover', 0):
            html_content += f"<td>{trade['rollover']}</td>"
        if columns_to_print.get('entry_delta', 0):
            html_content += f"<td>{round(trade['entry_delta'], 3)}</td>"
        if columns_to_print.get('exit_delta', 0):
            html_content += f"<td>{round(trade['exit_delta'], 3)}</td>"
        if columns_to_print.get('time', 0):
            html_content += f"<td>{trade['exit_time'] - trade['entry_time']}</td>"
        html_content += "</tr>"

    # Close the table and HTML structure
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(filename, "w") as file:
        file.write(html_content)

    # Display the link to the file in Jupyter notebook
    from IPython.display import display, FileLink
    display(FileLink(filename))


def print_execs_to_html(trades, columns_to_print, filename="trades_output.html"):
    # Start the HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            th {
                cursor: pointer;
            }
        </style>
        <script>
    function sortTable(n) {
        var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
        table = document.getElementById("tradesTable");
        switching = true;
        dir = "asc"; // Set the sorting direction to ascending

        // Fetch all rows from the table
        var rowsArray = Array.from(table.rows).slice(1); // Exclude the header row

        // Sort the rows based on the clicked column (n)
        rowsArray.sort(function (rowA, rowB) {
            var x = rowA.getElementsByTagName("TD")[n].innerHTML;
            var y = rowB.getElementsByTagName("TD")[n].innerHTML;

            let xValue = x.trim();
            let yValue = y.trim();

            // Convert values to numeric or date if applicable
            let xNumber = parseFloat(xValue.replace(/[^0-9.-]/g, ''));
            let yNumber = parseFloat(yValue.replace(/[^0-9.-]/g, ''));

            let xDate = new Date(xValue);
            let yDate = new Date(yValue);

            if (!isNaN(xNumber) && !isNaN(yNumber)) {
                // Numeric sorting
                return (dir === "asc") ? xNumber - yNumber : yNumber - xNumber;
            } else if (!isNaN(xDate.getTime()) && !isNaN(yDate.getTime())) {
                // Date sorting
                return (dir === "asc") ? xDate - yDate : yDate - xDate;
            } else {
                // String sorting
                return (dir === "asc") ? xValue.localeCompare(yValue) : yValue.localeCompare(xValue);
            }
        });

        // If sorted in ascending order, change to descending for next click
        if (dir === "asc") {
            dir = "desc";
        } else {
            dir = "asc";
        }

        // Remove all existing rows from the table body
        let tableBody = table.getElementsByTagName("tbody")[0];
        while (tableBody.firstChild) {
            tableBody.removeChild(tableBody.firstChild);
        }

        // Append the sorted rows to the table
        rowsArray.forEach(function (row) {
            tableBody.appendChild(row);
        });
    }
    </script>
    </head>
    <body>
        <table id="tradesTable">
            <thead>
                <tr>
    """

    # Add table headers with sorting
    header_cols = []
    if columns_to_print.get('id', 0):
        header_cols.append('id')
    if columns_to_print.get('time', 0):
        header_cols.append('time')
    if columns_to_print.get('strike', 0):
        header_cols.append('strike')
    if columns_to_print.get('price', 0):
        header_cols.append('price')
    if columns_to_print.get('pnl', 0):
        header_cols.append('pnl')
    if columns_to_print.get('trade_type', 0):
        header_cols.append('trade type')
    if columns_to_print.get('opt_type', 0):
        header_cols.append('opt. type')
    if columns_to_print.get('log', 0):
        header_cols.append('log')
    if columns_to_print.get('transaction_cost', 0):
        header_cols.append('transaction cost')
    if columns_to_print.get('slippage', 0):
        header_cols.append('slippage')
    if columns_to_print.get('delta', 0):
        header_cols.append('delta')

    # Add clickable header elements for sorting
    for index, col in enumerate(header_cols):
        html_content += f"<th onclick=\"sortTable({index})\">{col}</th>"

    html_content += """
                </tr>
            </thead>
            <tbody>
    """

    # Add table rows for each trade
    for i, trade in zip(range(len(trades)), trades):
        # Determine row color based on trade type
        row_color = ""
        if trade.get('trade_type', '').lower() == 'selling':
            row_color = 'style="background-color:#ffcccc;"'  # Red for SELLING
        elif trade.get('trade_type', '').lower() == 'buying':
            row_color = 'style="background-color:#ccffcc;"'  # Green for BUYING

        html_content += f"<tr {row_color}>"
        if columns_to_print.get('id', 0):
            html_content += f"<td>{trade.get('id', '')}</td>"
        if columns_to_print.get('time', 0):
            html_content += f"<td>{trade.get('time', '')}</td>"
        if columns_to_print.get('strike', 0):
            html_content += f"<td>{trade.get('strike', 0)}</td>"
        if columns_to_print.get('price', 0):
            html_content += f"<td>{trade.get('price', 0)}</td>"
        if columns_to_print.get('pnl', 0):
            html_content += f"<td>{round(trade.get('pnl', 0), 2)}</td>"
        if columns_to_print.get('trade_type', 0):
            html_content += f"<td>{trade.get('trade_type', '')}</td>"
        if columns_to_print.get('opt_type', 0):
            html_content += f"<td>{trade.get('opt_type', '')}</td>"
        if columns_to_print.get('log', 0):
            html_content += f"<td>{trade.get('log', '')}</td>"
        if columns_to_print.get('transaction_cost', 0):
            html_content += f"<td>{trade.get('transaction_cost', 0)}</td>"
        if columns_to_print.get('slippage', 0):
            html_content += f"<td>{trade.get('slippage', 0)}</td>"
        if columns_to_print.get('delta', 0):
            html_content += f"<td>{round(trade.get('delta', 0.0), 3)}</td>"
        html_content += "</tr>"

    # Close the table and HTML structure
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(filename, "w") as file:
        file.write(html_content)

    # Display the link to the file in Jupyter notebook
    from IPython.display import display, FileLink
    display(FileLink(filename))


def print_trades(trades, columns_to_print, filename="trades_output.txt"):
    # Write to the specified file
    with open(filename, "w") as file:
        for i, trade in zip(range(len(trades)), trades):
            output = []
            output.append(f" {i} ")
            if columns_to_print.get('pos', 0):
                output.append(f"pos {trade['pos']}")
            if columns_to_print.get('signal_time', 0):
                output.append(f"signal time {trade['signal_time']}")
            if columns_to_print.get('long_ema', 0):
                output.append(f"long_ema {trade['long_ema']}")
            if columns_to_print.get('short_ema', 0):
                output.append(f"short_ema {trade['short_ema']}")
            if columns_to_print.get('entry_time', 0):
                output.append(f"entry time {trade['entry_time']}")
            if columns_to_print.get('strike', 0):
                output.append(f"strike {trade['strike']}")
            if columns_to_print.get('entry_price', 0):
                output.append(f"entry price {trade['entry_price']}")
            if columns_to_print.get('exit_time', 0):
                output.append(f"exit time {trade['exit_time']}")
            if columns_to_print.get('exit_price', 0):
                output.append(f"exit price {trade['exit_price']}")
            if columns_to_print.get('pnl', 0):
                output.append(f"pnl {trade['pnl']}")
            if columns_to_print.get('type', 0):
                output.append(f"type {trade['type']}")
            if columns_to_print.get('pnl_sum', 0):
                output.append(f"pnl_sum {trade['pnl_sum']}")
            if columns_to_print.get('price', 0):
                output.append(f"price {trade['price']}")
            if columns_to_print.get('logs', 0):
                output.append(f"logs {trade['logs']}")
            if columns_to_print.get('movement', 0):
                output.append(f"movement {trade['movement']}")
            if columns_to_print.get('sharpie_ratio', 0):
                output.append(f"sharpie_ratio {trade['sharpie_ratio']}")
            if columns_to_print.get('drawdown', 0):
                output.append(f"drawdown {trade['drawdown']}")
            if columns_to_print.get('transaction_cost', 0):
                output.append(f"transaction_cost {trade['transaction_cost']}")
            if columns_to_print.get('slippage', 0):
                output.append(f"slippage {trade['slippage']}")
            if columns_to_print.get('capital', 0):
                output.append(f"capital {trade['capital']}")
            if columns_to_print.get('rollover', 0):
                output.append(f"rollover {trade['rollover']}")

            file.write(', '.join(output) + '\n')

    # Display the link to the file in Jupyter notebook
    display(FileLink(filename))
