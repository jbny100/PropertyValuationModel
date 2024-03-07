# make_simulation_params.py

from scipy.optimize import newton
import math 


discount_rate = .055

historical_discount_rates = [.0329, .0309, .0293, .0328, .04, .0339, .0261, .0227, .0332, .0575]

time_years = 30 

cf0 = - 2133390

cash_flows = [198350, 222828, 242259, 256815, 277322, 298419, 320126, 342460, 359018, 
	3546699, 134787, 158973, 183897, 201986, 228376, 255570, 283594, 312472, 333252, 
	363827, 165301, 197945, 231578, 255613, 291208, 327882, 365669, 404602, 432164, 473369]

cash_flows_excluding_debt = [566590, 590186, 608743, 622434, 642085, 662335, 683203, 
704706, 720442, 743212, 766675, 790855, 815772, 833854, 860237, 887424, 915441, 944313, 
	965086, 995655, 1027157, 1059620, 1093074, 1116932, 1152350, 1188851, 
	1226466, 1265228, 1292622, 1333660, 1373669]

risk_free_rate = .043

historical_risk_free_rate = [.02, .0176, .0231, .0288, .0265, .0155, .014, .0195, .0395, .043]

market_rate = .2

historical_market_rates = [.1139, -.0073, .0954, .1942, -.0643, .2888, .1626, .2689, -.1944, .2423] 

reinvestment_rate = .1 

finance_rate = .0575

beta = 1

operating_expenses = [292368, 300474, 308636, 322466, 331163, 340110, 349316, 358788, 374957, 
	385049, 395433, 406117, 417110, 436013, 447727, 459778, 472177, 484934, 507039, 520634, 
	534620, 549010, 563816, 589664, 605443, 621677, 638378, 655560, 685791, 704105, 725228]

projected_noi = [566590, 590186, 608743, 622434, 642085, 662335, 683203, 704706, 720442, 
	743212, 766675, 790855, 815772, 833854, 860237, 887424, 915441, 944313, 
	965086, 995655, 1027157, 1059620, 1093074, 1116932, 1152350, 1188851, 
	1226466, 1265228, 1292622, 1333660, 1373669]

total_debt = [368240, 367358, 366484, 365620, 364763, 363916, 363077, 362246, 361424, 360610, 
	631888, 631882, 631875, 631868, 631861, 631854, 631848, 631841, 631834, 631828, 861856, 
	861675, 861495, 861318, 861143, 860969, 860797, 860626, 860458, 860291]

loan_amount = 7525000

property_purchase_price = 1573000

property_value_growth_rate = .05

rental_growth_rate = .03

vacancy_rates = [.08, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, 
	.07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07, .07] 

property_value_last_year = 24248356

pre_tax_income = [858958, 890660, 917379, 944901, 973248, 1002445, 1032519, 1063494, 1095399, 
	1128261, 1162109, 1196972, 1232881, 1269868, 1307964, 1347202, 1387619, 1429247, 1472125, 
	1516288, 1561777, 1608630, 1656889, 1706596, 1757794, 1810527, 1864843, 1920789, 1978412, 
	2037765, 2098898]

gross_operating_income = pre_tax_income

construction_costs = 5395033 

other_costs = 2690357

total_investment = property_purchase_price + construction_costs + other_costs

gross_rental_income = [933650, 961660, 990509, 1020225, 1050831, 1082356, 1114827, 1148272, 1182720, 1218201, 1254748, 
	1292390, 1331162, 1371097, 1412229, 1454596, 1498234, 1543181, 1589477, 1637161, 1686276, 1736864, 1788970, 1842639, 
	1897918, 1954856, 2013501, 2073906, 2136124, 2200207, 2266214]

capex = 9298198.5

inflation_rate = [0.8, 0.7, 2.1, 2.1, 1.9, 2.3, 1.4, 7, 6.5, 3.4, 3.1]

unemployment_rate = .058 

gdp_growth_rates = [.0184, .0229, .0271, .0167, .0224, .0295, .0229, -.0277, .0595, .0206]

cost_contingency = .15





simulation_params = {} 


def add_npv(): 

	npv = cf0 + sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, 
		start=1))

	simulation_params['npv'] = npv 

	return simulation_params


def add_profitability_index(): 

	if cf0 == 0:
		raise ValueError("Initial cash flow (CFO) cannot be 0 when calculating \
			the profitability index.")
	simulation_params['profitability_index'] = simulation_params['npv'] / cf0 

	return simulation_params 


def add_future_value(): 

	fv_cf0 = cf0 * (1 + discount_rate) ** time_years 
	fv_cash_flows = [cf * (1 + discount_rate) ** (time_years - t) for t, cf in 
	enumerate(cash_flows, start=1)]
	fv = fv_cf0 + sum(fv_cash_flows)

	simulation_params['fv'] = fv 

	return simulation_params 


def add_rrr_capm(): 

	risk_premium = market_rate - risk_free_rate
	capm = beta * risk_premium + risk_free_rate 

	simulation_params['rrr_capm'] = capm 

	return simulation_params


def add_rrr_income(): 

	mean_total_debt = sum(total_debt) / len(total_debt)

	mean_projected_noi = sum(projected_noi) / len(projected_noi)

	cap = (mean_projected_noi - mean_total_debt) / property_purchase_price 
	rrr_income = cap * property_value_growth_rate

	simulation_params['rrr_income'] = rrr_income 

	return simulation_params 


def npv(discount_rate, cash_flows, initial_investment):
	return cf0 + sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, 
		start=1))

def npv2(discount_rate, unlevered_cash_flows, initial_investment):
	return cf0 + sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(unlevered_cash_flows, 
		start=1))


def add_levered_irr(): 

	def npv_rate_only(discount_rate): 
		return npv(discount_rate, cash_flows, cf0)

	initial_guess = discount_rate

	try: 
		irr = newton(npv_rate_only, initial_guess)
		simulation_params['levered_irr'] = irr 
	except RuntimeError as e: 
		print(f"Error calculating IRR: {e}")

		simulation_params['levered_irr'] = None 

	return simulation_params


def add_unlevered_irr(): 

	def npv_rate_only(discount_rate): 
		return npv2(discount_rate, cash_flows_excluding_debt, cf0)

	initial_guess = discount_rate

	try: 
		irr = newton(npv_rate_only, initial_guess)
		simulation_params['unlevered_irr'] = irr 
	except RuntimeError as e: 
		print(f"Error calculating IRR: {e}")

		simulation_params['unlevered_irr'] = None 

	return simulation_params


def add_levered_mirr(): 

	positive_cash_flows = [cf for cf in cash_flows if cf > 0]
	negative_cash_flows = [cf0] + [cf for cf in cash_flows if cf < 0] 

	future_value_positive = sum([cf * (1 + reinvestment_rate)**(i + 1) 
		for i, cf in enumerate(positive_cash_flows)])
	present_value_negative = sum([cf / (1 + finance_rate)**(i + 1) 
		for i, cf in enumerate(negative_cash_flows)])

	if len(cash_flows) == 0: 
		raise ValueError("Cash flows list is empty.")

	mirr = (future_value_positive / abs(present_value_negative))**(1 / len(cash_flows)) -1 

	simulation_params['levered_mirr'] = mirr 

	return simulation_params


def add_unlevered_mirr(): 

	positive_cash_flows = [cf for cf in cash_flows_excluding_debt if cf > 0]
	negative_cash_flows = [cf0] + [cf for cf in cash_flows_excluding_debt if cf < 0] 

	future_value_positive = sum([cf * (1 + reinvestment_rate)**(i + 1) 
		for i, cf in enumerate(positive_cash_flows)])
	present_value_negative = sum([cf / (1 + finance_rate)**(i + 1) 
		for i, cf in enumerate(negative_cash_flows)])

	if len(cash_flows_excluding_debt) == 0: 
		raise ValueError("Cash flows list is empty.")

	unlevered_mirr = (future_value_positive / abs(present_value_negative))**(1 / len(cash_flows_excluding_debt)) -1 

	simulation_params['unlevered_mirr'] = unlevered_mirr 

	return simulation_params


def add_cap_rate(): 

	cap_rate = projected_noi[0] / total_investment

	simulation_params['cap_rate'] = cap_rate 

	return simulation_params 


def add_reversion_value(): 

	rv = projected_noi[-1] / simulation_params['cap_rate']

	simulation_params['reversion_value'] = rv 

	return simulation_params


def add_capex(): 

	capex = (construction_costs + other_costs) * (1 + cost_contingency) 

	simulation_params['capex'] = capex 

	return simulation_params



def add_cash_on_cash_return(): 

	mean_cash_flow = sum(cash_flows) / len(cash_flows)

	cash_on_cash_return = mean_cash_flow / total_investment

	simulation_params['cash_on_cash_return'] = cash_on_cash_return

	return simulation_params


def add_gross_rent_multiplier(): 

	gross_rent_multiplier = property_purchase_price / gross_rental_income[1]

	simulation_params['rent_multiplier'] = gross_rent_multiplier 


def add_break_even_ratio(): 

	mean_gross_operating_income = sum(gross_operating_income) / len(gross_operating_income)
	mean_total_debt = sum(total_debt) / len(total_debt)

	if mean_gross_operating_income == 0: 
		raise ValueError("Gross operating income cannot be zero.")
	be_ratio = (operating_expenses[1] + mean_total_debt) / mean_gross_operating_income 

	simulation_params['break_even_ratio'] = be_ratio

	return simulation_params 


def add_vacancy_rate():

	avg_vacancy_rate = vacancy_rates 

	simulation_params['vacancy_rate'] = avg_vacancy_rate 

	return simulation_params


def add_inflation_rate():

	avg_inflation_rate = inflation_rate

	simulation_params['inflation_rate'] = avg_inflation_rate

	return simulation_params 


def add_unemployment_rate(): 

	avg_unemployment_rate = unemployment_rate 

	simulation_params['unemployment_rate'] = avg_unemployment_rate 

	return simulation_params 


def add_gdp_growth_rate(): 

	avg_gdp_growth_rate = gdp_growth_rates

	simulation_params['gdp_growth_rate'] = avg_gdp_growth_rate 

	return simulation_params




def main():
	add_npv()
	add_profitability_index()
	add_future_value()
	add_rrr_capm()
	add_rrr_income()
	add_levered_irr()
	add_unlevered_irr()
	add_levered_mirr()
	add_unlevered_mirr()
	add_cap_rate()
	add_reversion_value()
	add_capex()
	add_cash_on_cash_return()
	add_gross_rent_multiplier()
	add_break_even_ratio()
	add_vacancy_rate()
	add_inflation_rate()
	add_unemployment_rate()
	add_gdp_growth_rate()

	return simulation_params
	

main()
print(simulation_params)








