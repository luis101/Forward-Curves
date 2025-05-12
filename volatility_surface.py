

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def option_chains(ticker):

    asset = yf.Ticker(ticker)
    expirations = asset.options
    chains = pd.DataFrame()

    for expiration in expirations:

        # tuple of two dataframes
        opt = asset.option_chain(expiration)

        calls = opt.calls
        calls['optionType'] = "call"

        puts = opt.puts
        puts['optionType'] = "put"

        chain = pd.concat([calls, puts])
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        chains = pd.concat([chains, chain])

    chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1

    return chains


class VolatilitySurface:
    def __init__(self):
        """
        Initialize a volatility surface with methods for creating, 
        interpolating, and analyzing implied volatilities.
        """
        self.strikes = []
        self.expirations = []
        self.volatilities = []
    
    def add_market_data(self, strike, expiration, volatility):
        """
        Add market observed option volatility data point.
        
        :param strike: Option strike price
        :param expiration: Time to expiration (in years)
        :param volatility: Implied volatility percentage
        """
        #self.strikes.append(strike)
        #self.expirations.append(expiration)
        #self.volatilities.append(volatility)
        self.strikes = strike
        self.expirations = expiration
        self.volatilities = volatility
    
    def interpolate_surface(self, num_strikes=1000, num_expirations=100):
        """
        Interpolate a smooth volatility surface using griddata.
        
        :return: Interpolated grid of volatilities
        """
        # Convert lists to numpy arrays
        strikes = np.array(self.strikes)
        expirations = np.array(self.expirations)
        volatilities = np.array(self.volatilities) 
        
        # Create grid for interpolation
        strike_range = np.linspace(strikes.min(), strikes.max(), num_strikes)
        expiration_range = np.linspace(expirations.min(), expirations.max(), num_expirations)
        
        strike_grid, expiration_grid = np.meshgrid(strike_range, expiration_range)
        
        # Interpolate volatilities
        interpolated_vols = griddata(
            (strikes, expirations), 
            volatilities, 
            (strike_grid, expiration_grid), 
            method='cubic'
        )
        
        return strike_grid, expiration_grid, interpolated_vols
    
    def calculate_risk_metrics(self):
        """
        Calculate key risk metrics from the volatility surface.
        
        :return: Dictionary of risk metrics
        """
        # Compute basic statistical properties
        risk_metrics = {
            'mean_volatility': np.mean(self.volatilities),
            'volatility_std_dev': np.std(self.volatilities),
            'max_volatility': np.max(self.volatilities),
            'min_volatility': np.min(self.volatilities)
        }
        
        # Compute volatility smile characteristics
        smile_metrics = self.analyze_volatility_smile()
        risk_metrics.update(smile_metrics)
        
        return risk_metrics
    
    def analyze_volatility_smile(self):
        """
        Analyze the volatility smile characteristics.
        
        :return: Dictionary of smile metrics
        """
        # Group volatilities by strike levels
        strikes = np.array(self.strikes)
        vols = np.array(self.volatilities)
        
        # Categorize strikes
        moneyness_levels = {
            'out_of_the_money_puts': vols[strikes < np.median(strikes)],
            'at_the_money_options': vols[np.abs(strikes - np.median(strikes)) < np.std(strikes)],
            'out_of_the_money_calls': vols[strikes > np.median(strikes)]
        }
        
        smile_metrics = {
            'put_vol_skew': np.mean(moneyness_levels['out_of_the_money_puts']),
            'atm_vol_mean': np.mean(moneyness_levels['at_the_money_options']),
            'call_vol_skew': np.mean(moneyness_levels['out_of_the_money_calls'])
        }
        
        return smile_metrics
    
    def plot_3d_surface(self):
        """
        Visualize the volatility surface in 3D.
        """
        # Interpolate the surface
        strike_grid, expiration_grid, interpolated_vols = self.interpolate_surface()

        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot with color mapping
        surf = ax.plot_surface(
            strike_grid, 
            expiration_grid, 
            interpolated_vols, 
            cmap='viridis',  # Color gradient
            edgecolor='none',  # Remove edge lines
            alpha=0.8,  # Slight transparency
            linewidth=0
        )
        
        # Scatter plot of original data points
        ax.scatter(
            self.strikes, 
            self.expirations, 
            self.volatilities, 
            color='red', 
            s=50, 
            label='Original Data Points'
        )
        
        # Customize the plot
        ax.set_title('3D Volatility Surface', fontsize=16)
        ax.set_xlabel('Strike Price', fontsize=12)
        ax.set_ylabel('Tenors (Days)', fontsize=12)
        ax.set_zlabel('Implied Volatility (%)', fontsize=12)
        
        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Rotate the plot for better visualization
        ax.view_init(30, 45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_surface(self):
        """
        Visualize the volatility surface.
        """
        strike_grid, expiration_grid, interpolated_vols = self.interpolate_surface()
        
        plt.figure(figsize=(12, 8))        
        contour = plt.contourf(
            strike_grid, 
            expiration_grid, 
            interpolated_vols, 
            levels=20, 
            cmap='viridis'
        )
        plt.colorbar(contour, label='Implied Volatility (%)')
        plt.title('Volatility Surface')
        plt.xlabel('Strike Price')
        plt.ylabel('Tenor (Days)')
        plt.show()

        plt.show()


# Example usage and demonstration
def main():
    
    # Create volatility surface
    vol_surface = VolatilitySurface()
    
    # Add market data (strike, expiration, volatility)
    options = option_chains("SPY")
    
    calls = options[options["optionType"] == "call"]
    calls = calls.sort_values(by=["daysToExpiration", "strike"])
    calls = calls.drop_duplicates(subset=["daysToExpiration", "strike"])
    
    #strikes = calls["strike"]
    #expirations = calls["daysToExpiration"]
    #volatilities = calls["impliedVolatility"]
    
    vol_surface.add_market_data(calls["strike"], calls["daysToExpiration"], calls["impliedVolatility"])
    
    # Calculate risk metrics
    risk_metrics = vol_surface.calculate_risk_metrics()
    print("Risk Metrics:", risk_metrics)
    
    # Plot the volatility surface
    vol_surface.plot_3d_surface()

if __name__ == "__main__":
    main()


"""
# Create an interpolation function using RectBivariateSpline
vol_surface = RectBivariateSpline(calls["daysToExpiration"], calls["strike"], calls["impliedVolatility"])

surface = (
    calls[['daysToExpiration', 'strike', 'impliedVolatility']]
    .pivot_table(values='impliedVolatility', index='strike', columns='daysToExpiration')
    .dropna()
)

fig = go.Figure(data=[go.Surface(x=surface.columns.values, y=surface.index.values, z=surface.values)])


# Create a grid for plotting
strike_grid = np.linspace(min(calls["strike"],), max(calls["strike"],), 100)
expiry_grid = np.linspace(min(calls["daysToExpiration"]), max(calls["daysToExpiration"]), 100)

# Generate the volatility surface data
vol_surface_data = vol_surface(strike_grid, expiry_grid)

# Plot the volatility surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(surface.columns.values, surface.index.values)
"""

