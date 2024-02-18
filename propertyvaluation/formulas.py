import math 
from scipy.optimize import newton
import numpy as np 

class NetPresentValue:
	"""Calculates the net present value of a series of future cash flows."""

	def __init__(self, rate: float, num_payments: int, CF0: float, 
		cash_flows: list[float]) -> None:
		self.rate = rate
		self.num_payments = num_payments
		self.CF0 = CF0
		self.cash_flows = cash_flows 

	def __repr__(self): 
		return (f"NetPresentValue(rate={self.rate}, num_payments={self.num_payments}, "
		f"CF0={self.CF0}, cash_flows={self.cash_flows}")


	def __eq__(self, other): 
		"""Defines how to evaluate equality between two instances of the same class."""
		if not isinstance(other, NetPresentValue): 
			return NotImplemented 
		return (self.rate == other.rate and 
                self.num_payments == other.num_payments and 
                self.CF0 == other.CF0 and 
                self.cash_flows == other.cash_flows)


	def calculate_npv(self) -> float: 
		# Adjust for initial investment by directly subtracting it. 
		npv = self.CF0 
		for t, cf in enumerate(self.cash_flows, start=1): 
			npv += cf / (1 + self.rate)**t 
		return npv 

	def calc_profitability_index(self) -> float:
		""" The Profitability Index helps to determine whether the potential profit 
		from an investment project justifies the initial costs. PI > 1 can mean investment 
		is potentiall attractive."""
		if self.CF0 == 0:
			raise ValueError("Initial cash flow (CFO) cannot be 0 when calculating \
				the profitability index.")
		npv = self.calculate_npv() 
		p_index = npv / self.CF0
		return abs(p_index)


class FVCashFlows:
	"""Calculates the future value of a series of cash outflows and inflows."""

	def __init__(self, rate: float, num_payments: int, CF0: float,  
		cash_flows: list[float]) -> None: 
		self.rate = rate
		self.num_payments = num_payments
		self.CF0 = CF0
		self.cash_flows = cash_flows 


	def __repr__(self): 
		return f"FVCashFlows(rate={self.rate}, num_payments={self.num_payments}, \
		CF0={self.CF0}, cash_flows={self.cash_flows})" 


	def __eq__(self, other): 
		"""Defines how to evaluate equality between two instances of the same class."""
		if not isinstance(other, NetPresentValue): 
			return NotImplemented 
		return (self.rate == other.rate and 
                self.num_payments == other.num_payments and 
                self.CF0 == other.CF0 and 
                self.cash_flows == other.cash_flows)


	def calculate_fv(self) -> float: 
		fv = self.CF0 * (1 + self.rate)**self.num_payments 
		for t, cf in enumerate(self.cash_flows, start=1): 
			# Corrected to adjust for each cash flow's timing.
			fv += cf * (1 + self.rate)**(self.num_payments - t) 
		return fv 

	def calculate_reversion_value(self) -> float: 
		"""Calculate the reversion value (present value of the future value)."""
		fv = self.calculate_fv()
		reversion_value = fv / (1 + self.rate)**self.num_payments
		return reversion_value

	"""
	If the IRR is greater than the minimum RRR (Required rate of return - typically 
	the cost of capital) on a project or investment, then the project can be pursued.
	"""

class RRR:
	"""Calculates the required rate of return using the CAPM and 
	Income Approach Methods."""

	def __init__(self, risk_free_rate: float, market_rate: float, beta: float, 
		NOI: float, debt: float, property_value: float, growth_rate: float) -> None: 
		self.risk_free_rate = risk_free_rate
		self. market_rate = market_rate 
		self.beta = beta 
		self.NOI = NOI 
		self.debt = debt 
		self.property_value = property_value 
		self.growth_rate = growth_rate

	def rrr_capm(self, risk_free_rate: float, market_rate: float, beta: float) -> float:
		"""Calculates required rate of return using CAPM method."""
		risk_premium = self.market_rate - self.risk_free_rate
		capm = self.beta * risk_premium + self.risk_free_rate
		return capm 

	def rrr_income(self, NOI: float, debt: float, property_value: float, 
		growth_rate: float) -> float: 
		"""Calculates required rate of return using Income Approach 
		(Band of Investment). This is more often compared to a company's 
		IRR."""
		cap = (self.NOI - self.debt) / self.property_value
		rrr = cap + self.growth_rate
		return rrr 

class IRR:
	"""Calculates the internal rate of return on an investment and compares it to 
	the minimum required rate of return (RRR)."""

	def __init__(self, rate: float, CF0: float, 
		cash_flows: list[float]) -> None:
		self.rate = rate
		self.CF0 = CF0 
		self.cash_flows = (CF0,) + tuple(cash_flows) # convert cash flows to a tuple. 
		self.n = len(self.cash_flows) 

	def __repr__(self): 
		return f"IRR(rate={self.rate}, CF0={self.CF0}, cash_flows={self.cash_flows})"


	def calculate_irr(self) -> float:
		"""Calculates the IRR of an investment using scipy module."""
		def npv(rate):
			npv = 0 
			for t, cf in enumerate(self.cash_flows):
				npv += cf / (1 + rate)**t 
			return npv 

		try:
			irr = newton(npv, self.rate)  # Use self.rate as the initial guess for IRR
			return irr 
		except Exception as e: 
			print("Error calculating IRR:", e)
			return None

	def calculate_mirr(self, reinvestment_rate: float, finance_rate: float) -> float: 
		"""Calculates the MIRR of an investment."""
		positive_cash_flows = [cf for cf in self.cash_flows if cf > 0]
		negative_cash_flows = [cf for cf in self.cash_flows if cf < 0]

		future_value_positive = sum([cf * (1 + reinvestment_rate)**(i + 1) 
			for i, cf in enumerate(positive_cash_flows)])
		present_value_negative = sum([cf / (1 + finance_rate)**(i + 1) 
			for i, cf in enumerate(negative_cash_flows)])

		mirr = (future_value_positive / abs(present_value_negative))**(1 / self.n) -1 
		return mirr 


class CapRate:
	"""Calculate cap rate of an investment."""

	def __init__(self, NOI: float, property_value: float) -> None:
		self.NOI = NOI 
		self.property_value = property_value 

	def cap_rate(self) -> float: 
		return self.NOI / self.property_value


class CashOnCashReturn: 
	"""Calculates the annual cash flow relative to the initial investment, 
	expressed as a percentage. Provides insight into the immediate returns 
	generated from an investment."""

	def __init__(self, cash_flows: list[float], total_investment: float) -> None: 
		self.cash_flows = cash_flows
		self.total_investment = total_investment

	def cash_on_cash(self) -> float: 
		return self.cash_flows / self.total_investment 


class GrossRentMultiplier: 
	"""GRM compares a property's purchase price to its gross rental income.
	It provides a quick estimate of how long it would take to recover the 
	investment based on rental income alone."""

	def __init__(self, property_price: float, gross_rental_income: float) -> None: 
		self.property_price = property_price
		self.gross_rental_income = gross_rental_income 

	def gross_rent_multiplier(self) -> float: 
		return self.property_price / self.gross_rental_income


class BreakEvenRatio: 

	def __init__(self, operating_expenses: float, debt: float, 
		gross_operating_income: float) -> None: 
		self.operating_expenses = operating_expenses
		self.debt = debt 
		self.gross_operating_income = gross_operating_income
		

	def break_even_ratio(self) -> float: 
		"""Ratio that determines the minimum occupancy rate required to cover 
		all operating expenses and debt service. Helps assess the risk associated
		with vacancy rates."""
		if self.gross_operating_income == 0: 
			raise ValueError("Gross operating income cannot be zero.")
		be_ratio = (self.operating_expenses + self.debt) / self.gross_operating_income
		return be_ratio


class MarketRisks: 
	"""Class with only static methods do not need __init__. Static methods are intended 
	to be independent of any particular instance of the class (i.e. an instance of MarketRisks 
	isn't required to use its calc_volatility static method)"""

	@staticmethod
	def calc_volatility(percent_change_prop: list[float], 
		percent_change_cons: list[float], 
		percent_change_rent: list[float]) -> float: 
		"""Calculate an adjusted overall volatility considering directional sensitivity 
		for changes in property values, construction costs, and rental income."""

		# Calculate mean changes for each category 
		mean_change_prop = np.mean(percent_change_prop) 
		mean_change_const = np.mean(percent_change_cons) 
		mean_change_rent = np.mean(percent_change_rent) 

		# Calculate the standard deviations. 
		std_dev_prop = np.std(percent_change_prop) 
		std_dev_const = np.std(percent_change_cons) 
		std_dev_rent = np.std(percent_change_rent)

		# Adjust standard deviations based on mean change direction 
		# Positive mean changes in property and rent signify reduced risk, 
		# reflected by decreasing the std deviation 
		# Positive mean changes in construction costs signify increased risk, 
		# reflected by increasing the std deviation
		adj_std_dev_prop = std_dev_prop * (1 if mean_change_prop > 0 else 1.5)
		adj_std_dev_const = std_dev_const * (1.5 if mean_change_const > 0 else 1)
		adj_std_dev_rent = std_dev_rent * (1 if mean_change_rent > 0 else 1.5)

		# Calculate adjusted overall volatility as the average of adjusted standard deviations. 
		volatility = (adj_std_dev_prop + adj_std_dev_const + adj_std_dev_rent) / 3

		return volatility


class DevProjectCalcs:

	def __init__(self, npv: float, time: int, rate: float, annual_cash_flow: float, 
		land_cost: float, construction_cost: float, other_costs, 
		volatility: float = None) -> None: 
		self.npv = npv 
		self.time = time 
		self.rate = rate 
		self.annual_cash_flow = annual_cash_flow
		self.land_cost = land_cost 
		self.construction_cost = construction_cost 
		self.other_costs = other_costs
		self.volatility = volatility 

	def __repr__(self): 
		return (f"DevProjectCalcs(npv={self.npv}, time={self.time}, rate={self.rate}, "
                f"annual_cash_flow={self.annual_cash_flow}, land_cost={self.land_cost}, "
                f"construction_cost={self.construction_cost}, other_costs={self.other_costs}, "
                f"volatility={self.volatility})")

	def calc_underlying_asset_value(self) -> float: 
		"""Current value of the projected value of completed development project.""" 
		adjustment_factor = math.exp(-self.volatility**2 * self.time / 2)
		underlying_asset_value = self.npv * adjustment_factor

		return underlying_asset_value 

	def calc_exercise_price(self) -> float: 
		"""Cost of acquiring land + construction costs + other development expenses."""
		exercise_price = self.land_cost + self.construction_cost + self.other_costs 

		return exercise_price 

	def calc_cash_flow_yield(self) -> float: 
		""" Reflects income generated from a property relative to its value. Takes into 
		account the cash flow an investor actually receives, which makes it a practical 
		measure for evaluating an investment's performance."""
		if self.land_cost == 0:
			raise ValueError("Land cost cannot be 0 when calculating cash flow yield.")
		cash_flow_yield = self.annual_cash_flow / self.land_cost

		return cash_flow_yield 



npv1 = NetPresentValue(0.05, 3, 0, [100, 200, 300])
npv2 = NetPresentValue(0.05, 3, -250, [100, 200, 300])
fv1 = FVCashFlows(0.05, 5, -500, [200, 200, 200, 200, 200])
irr1 = IRR(0.10, -200, [100, 200, 500])

npv3 = NetPresentValue(0.10, 5, -500, [100, 200, 300, 400, 500])
fv2 = FVCashFlows(0.10, 5, -500, [100, 200, 300, 400, 500])

print(npv1.calculate_npv())
print(npv2.calculate_npv())
print(npv2.calc_profitability_index())
print(fv1.calculate_fv())
print(fv1.calculate_reversion_value())
print(irr1.calculate_irr())
print(irr1.calculate_mirr(0.12, 0.08))
print(npv3.calculate_npv())
print(fv2.calculate_fv())
print(fv2.calculate_reversion_value())

risks = MarketRisks()
print(risks.calc_volatility([.04, .05, -.03, .02, .06], [.02, .02, 0, 0, 0], 
	[.01, .01, .01, .01, .01]))

devCalcs = DevProjectCalcs(1000, 5, .08, 1200,  500, 200, 50, 0.00735893197662858)

print(devCalcs.calc_underlying_asset_value())
print(devCalcs.calc_exercise_price())
print(devCalcs.calc_cash_flow_yield())


