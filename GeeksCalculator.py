import QuantLib as ql
import pandas as pd
from datetime import datetime

from scipy.stats import false_discovery_control


def has_time_component(date_str):
    try:
        # Try to parse with date and time
        datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        # If it fails, check if it's just a date
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return False
        except ValueError:
            raise ValueError("Invalid date format")

def calc(spot, strike, interest_rate, option_price, expiry_datetime, dividend_yield, timestamp,
                           option_type='CE'):
    """
    Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho) based on the input fields and a specific timestamp with minute precision.

    Parameters:
    - spot: Spot price of the underlying asset
    - strike: Strike price of the option
    - interest_rate: Risk-free interest rate (as a percentage, e.g., 10 for 10%)
    - option_price: The current market price of the option
    - expiry_datetime: Expiry date and time of the option (in string format 'YYYY-MM-DD HH:MM:SS')
    - dividend_yield: Dividend yield (as a percentage, e.g., 2 for 2%)
    - timestamp: The specific time (datetime object) at which to calculate the Greeks (with minute precision)
    - option_type: 'call' or 'put'

    Returns:
    A dictionary with the calculated option Greeks (Delta, Gamma, Vega, Theta, Rho) and the implied volatility or an error.
    """
    spot = int(spot)
    strike = int(strike)
    option_price = float(option_price)
    # Convert percentages to decimals
    interest_rate = interest_rate / 100
    dividend_yield = dividend_yield / 100

    # Parse expiry date and timestamp to datetime objects
    expiry_datetime = pd.to_datetime(expiry_datetime)  # Ensures input is a datetime object
    expiry_datetime = expiry_datetime.replace(hour=15, minute=30)

    timestamp_datetime = pd.to_datetime(timestamp)  # Ensure timestamp is also a datetime object

    # Calculate the total difference between expiry and timestamp in minutes
    time_difference = (expiry_datetime - timestamp_datetime).total_seconds() / 60  # Minutes difference


    if time_difference < 0:
        return {'error': 'Timestamp is past the expiry date and time.'}

    # Convert the time difference to fractional days
    days_to_expiry = time_difference / (24 * 60)  # Minutes to fractional days
    print(days_to_expiry)
    # Set the evaluation date to the given timestamp (QuantLib works on day level, but we manage the precision)
    ql.Settings.instance().evaluationDate = ql.Date(timestamp_datetime.day, timestamp_datetime.month,timestamp_datetime.year)

    # Create the spot price handle
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

    # Set up interest rate and dividend yield term structures
    day_count = ql.Actual365Fixed()
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql.Settings.instance().evaluationDate, interest_rate, day_count))
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql.Settings.instance().evaluationDate, dividend_yield, day_count))

    # Create a guess for the implied volatility (initial guess)
    initial_volatility_guess = 0.20  # 20% initial guess
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(ql.Settings.instance().evaluationDate, ql.NullCalendar(), initial_volatility_guess,
                            day_count))

    # Set up the Black-Scholes-Merton process
    bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

    # Define the option payoff (Call or Put)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'CE' else ql.Option.Put, strike)
    exercise = ql.EuropeanExercise(ql.Date(expiry_datetime.day, expiry_datetime.month, expiry_datetime.year))

    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

    # Try to calculate the implied volatility based on the market price of the option
    try:
        implied_volatility = option.impliedVolatility(option_price, bs_process, 1e-4, 100)
    except RuntimeError as e:
        return {
            'error': f"Implied volatility calculation failed: {str(e)}",
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None
        }

    # Update the volatility term structure with the implied volatility
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(ql.Settings.instance().evaluationDate, ql.NullCalendar(), implied_volatility, day_count))
    bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

    # Reassign the pricing engine with the updated process
    option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

    # Calculate the Greeks
    delta = option.delta()
    gamma = option.gamma()
    vega = option.vega()
    theta = option.thetaPerDay()  # Theta is per day
    rho = option.rho()

    # Return the Greeks and the implied volatility as a dictionary
    return {
        'implied_volatility': implied_volatility,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho,
        'days_to_expiry': days_to_expiry
    }


# # Example usage:
#
# # Input values based on the fields in the UI (these are sample values)
# spot = 150.0  # Spot price of the underlying asset
# strike = 150  # Strike price of the option
# interest_rate = 6  # Interest rate (6%)
# option_price = 30  # Option price (market price)
# expiry_datetime = '2024-09-15 15:30:00'  # Expiry date and time in 'YYYY-MM-DD HH:MM:SS' format
# dividend_yield = 2  # Dividend yield (2%)
# timestamp = pd.Timestamp('2024-09-09 09:30:00')  # Timestamp for which we calculate Greeks
#
# # Call the function for a call option
# greeks = calculate_greeks_input(spot, strike, interest_rate, option_price, expiry_datetime, dividend_yield, timestamp,
#                                 option_type='call')
# print(greeks)
