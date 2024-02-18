from formulas import NetPresentValue, MarketRisks, DevProjectCalcs 
import math 
import numpy as np 
from scipy.stats import norm 


class BlackScholesRealEstate: 
	"""Attempts to calculate an overall value for a dev site or property using a 
	an adaptation of the Black-Scholes option valuation model.

	General formula is C = S * N(d1) - K * e^(-rT) * N(d2):

	C = Black-Scholes value 
	S = Underlying Asset Value
	K = Exercise Price 
	r = Risk-free rate 
	T = time 
	d1  =  (ln(S/K)+(r+σ^(2) / 2)T) / (σ√T)
	d2 = d1 - σ√T

	N(d1) and N(d2) = Values from cumulative distribution function. Represent the probabilities 
	associated with the d1 and d2 values under the standard normal distribution. They 
	transform the inputs into probabilities used to determine property's theoretical value. 
	"""

	
	def __init__(self, underlying_asset_value: float, exercise_price: float, 
		rate: float, time: int, volatility: float) -> None: 
		self.underlying_asset_value = underlying_asset_value
		self.exercise_price = exercise_price 
		self.rate = rate 
		self.time = time 
		self.volatility = volatility 

	def __repr__(self): 
		return (f"BlackScholesRealEstate(underlying_asset_value={self.underlying_asset_value}, "
                f"exercise_price={self.exercise_price}, rate={self.rate}, "
                f"time={self.time}, volatility={self.volatility})")

	def __eq__(self, other): 
		if not isinstance(other, BlackScholesRealEstate): 
			return NotImplemented 
		return (self.underlying_asset_value == other.underlying_asset_value and
                self.exercise_price == other.exercise_price and
                self.rate == other.rate and
                self.time == other.time and
                self.volatility == other.volatility)


	def calcNd1Nd2(self) -> tuple[float, float, float, float]: 
		"""N(d1) and N(d2) are functions that give the probability 
		that a random draw from a standard normal distribution will be less that d1 and 
		d2, respectively."""

		# Calculate d1 and d2
		d1 = (np.log(self.underlying_asset_value / self.exercise_price) + 
			(self.rate + (self.volatility**2) / 2) * self.time) / (self.volatility * np.sqrt(self.time))
		d2 = d1 - self.volatility * np.sqrt(self.time)

		# Calculate N(d1) and N(d2)
		N_d1 = norm.cdf(d1)
		N_d2 = norm.cdf(d2)

		return d1, d2, N_d1, N_d2

	def black_scholes(self) -> float: 
		"""# Utilize the 'calcNd1Nd2' method to get d1, d2, N(d1), and N(d2)."""
		d1, d2, N_d1, N_d2, = self.calcNd1Nd2()

		# Calculate the Black-Scholes value using the formuls
		C = self.underlying_asset_value * N_d1 - self.exercise_price * math.exp(-self.rate * self.time) * N_d2 

		return C 


bs1 = BlackScholesRealEstate(999, 750, .03, 5, 0.01552389778590584)
print(bs1.black_scholes())




