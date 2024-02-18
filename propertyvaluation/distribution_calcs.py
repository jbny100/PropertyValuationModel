
# distribution_calcs.py

import numpy as np 
import numpy_financial as npf 
from scipy.optimize import minimize
from scipy.stats import norm, t 
import math 

import make_simulation_params 
from make_simulation_params import discount_rate, historical_discount_rates, cf0 
from make_simulation_params import time_years, cash_flows, cash_flows_excluding_debt
from make_simulation_params import risk_free_rate, historical_risk_free_rate, market_rate
from make_simulation_params import historical_market_rates, reinvestment_rate, finance_rate
from make_simulation_params import gross_operating_income, operating_expenses, noi, total_debt
from make_simulation_params import loan_amount, property_purchase_price, property_value_growth_rate
from make_simulation_params import rental_growth_rate, vacancy_rate, pre_tax_income 
from make_simulation_params import total_investment, gross_rental_income, capex
from make_simulation_params import inflation_rate, unemployment_rate, gdp_growth
from make_simulation_params import construction_costs, other_costs, cost_contingency 

dist_calcs = {
	'npv': {'distribution': 'normal'}, 
	'profitability_index': {'distribution': 'normal'}, 
	'fv': {'distribution': 'lognormal'},
    'rrr-capm': {'distribution': 'normal'},
    'rrr-income': {'distribution': 'normal'},
    'cap_rate': {'distribution': 'normal'},
    'reversion_value': {'distribution': 'lognormal'},
    'capex': {'distribution': 'lognormal'},
    'cash-on-cash_return': {'distribution': 'normal'},
    'rent_multiplier': {'distribution': 'normal'},
    'break_even_ratio': {'distribution': 'normal'},
    'inflation_rate': {'distribution': 'lognormal'},
    'vacancy_rate': {'distribution': 'beta'},
    'gdp_growth_rate': {'distribution': 'normal'},
	
} 

"""
Next three functions are for determining the Mean and STD of the provided 
Discount Rate using the Ornstein-Uhlenbeck distribution method.
"""

def ou_objective(params, historical_discount_rates, time_step=1): 
    """Objective function for OU parameter estimation for discount rate."""
    theta, mu, sigma = params 
    data_diff = np.diff(historical_discount_rates)  # Differences in consecutive data points
    predicted_diff = theta * (mu - historical_discount_rates[:-1]) # ou prediction for changes 

    # Calculate the sum of squared differences, normalized by variance attributed to sigma
    residuals = data_diff - predicted_diff  # differences in consecutive data points
    sum_squared_differences = np.sum(residuals**2) / (sigma**2) + len(data_diff) * np.log(sigma**2)

    return sum_squared_differences


def calc_base_mean_std(discount_rate, theta, mu, sigma, t=1): 
    """Calculate initial avg mean and standard dev of the discount rate modeled by 
    the Ornstein-Uhlenbeck process at time t."""

    # r_0 = base discount rate
    mean_rate_t = discount_rate * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t))

    variance_t = (sigma**2) / (2 * theta) * (1 - np.exp(-2 * theta * t))

    std_t = np.sqrt(variance_t)

    return mean_rate_t, std_t  

def estimate_and_calc_ou(historical_discount_rates, discount_rate, t=1): 
    """Estimate OU parameters and calculate mean & std of Discount Rate.""" 

    # Historical data of discount rates
    discount_rates = np.array(historical_discount_rates)

    # Initial guesses for theta, mu and sigma
    initial_guess = [0.1, np.mean(discount_rates), 0.02] 

    # Perform optimization to estimate parameters
    result = minimize(ou_objective, initial_guess, args=(discount_rates))

    # Extract estimated params  
    theta_est, mu_est, sigma_est = result.x 

    # Use estimated params to calculate base mean and std at time t 
    mean_rate_t, std_t = calc_base_mean_std(discount_rate, theta_est, mu_est, 
        sigma_est, t)

    return theta_est, mu_est, sigma_est, mean_rate_t, std_t 

def generate_discount_rate_scenarios(theta, mu, sigma, t, iterations): 
    """Generate scenarios of simulated discount rates using the Ornstein-Uhlenbeck 
    (OU) process.""" 
    # Mean and variance of the discount rate at time t using OU parameters 
    mean_rate_t = mu * (1 - np.exp(-theta * t)) 
    variance_t = (sigma ** 2) / (2 * theta) * (1 - np.exp(-2 * theta * t)) 


    # Simulate discount rate scenarios
    sim_discount_rates = norm.rvs(loc=mean_rate_t, scale=np.sqrt(variance_t), 
        size=iterations)

    return sim_discount_rates



"""
Calculate the Mean and STD of Cash Flows using a lognormal distribution.
"""



def simulate_cash_flows(cf0, cash_flows, iterations):
    """Simulate cash flow series using lognormal distribution, including cf0."""
    mean_log = np.log(np.mean(cash_flows)) 
    sigma_log = 0.04  # example standard deviation in log-space

    # Simulate future cash flows excluding cf0
    n_periods = len(cash_flows)
    simulated_cash_flows = np.exp(norm.rvs(mean_log, sigma_log, 
        size=(iterations, n_periods))) 

    # Prepend cf0 to each simulated series
    cash_flows_series = np.hstack((np.full((iterations, 1), cf0), simulated_cash_flows))

    return cash_flows_series

def calculate_npv_series(sim_discount_rates, cash_flows_series): 
    """Calculate NPV for each pair of simulated discount rate and cash flow series."""
    npvs = np.array([npf.npv(rate, flows) for rate, 
        flows in zip(sim_discount_rates, cash_flows_series)])
    return npvs


# Number of scenarios to simulate 
iterations = 5
n_periods = len(cash_flows)
mean_beta = 1.0 
std_beta = 0.2 


def run_npvs_sim(cf0, cash_flows, historical_discount_rates, discount_rate, iterations): 
    # Call estimate_and_calc() to get parameters for generate_discount_rate_scenarios
    theta_est, mu_est, sigma_est, mean_rate_t, std_t = estimate_and_calc_ou(historical_discount_rates, 
        discount_rate, t=1)

    simulated_discount_rates = generate_discount_rate_scenarios(theta_est, mu_est, sigma_est, 
        t=1, iterations=iterations)
    simulated_cash_flows = simulate_cash_flows(cf0, cash_flows, iterations)
    # Calculate NPV for each scenario
    npvs = calculate_npv_series(simulated_discount_rates, simulated_cash_flows)
    # Calculate mean and standard deviation of the simulated NPVs 
    mean_npv = np.round(np.mean(npvs), 2) 
    std_npv = np.round(np.std(npvs), 2)

    return mean_npv, std_npv, npvs, simulated_discount_rates, simulated_cash_flows


def simulate_profitability_index(npvs, cf0):
    # Calculate the PI for each NPV value using the fixed CF0. 
    pi_values = npvs / -cf0
    
    # Calculate the mean and standard deviation of the resulting PI values 
    mean_pi = np.round(np.mean(pi_values), 2) 
    std_pi = np.round(np.std(pi_values), 2)
    
    return mean_pi, std_pi, pi_values


def calculate_fv_series(simulated_discount_rates, cash_flows_series, n_periods): 
    """Calculate fv for each pair of simulated discount rate and cash flow series."""
    fvs = np.array([npf.fv(rate, n_periods, 0, -np.sum(flows)) for rate, 
        flows in zip(simulated_discount_rates, cash_flows_series)])

    return fvs 

def run_fv_sim(cf0, cash_flows, historical_discount_rates, discount_rate, iterations): 
    """Run FV simulations."""
    # Use run_npvs_sim to get simulated discount rates and cash flows
    mean_npv, std_npv, npvs, simulated_discount_rates, simulated_cash_flows = run_npvs_sim(cf0, 
        cash_flows, historical_discount_rates, discount_rate, iterations)
    # Calculate FV for each scenario
    fvs = calculate_fv_series(simulated_discount_rates, simulated_cash_flows, len(cash_flows))
    # Calculate mean and std for FV
    mean_fv = np.round(np.mean(fvs), 2)
    std_fv = np.round(np.std(fvs), 2)

    return mean_fv, std_fv, fvs 


def sim_market_return_scenario(historical_market_rates, iterations):
    """Calculate market return scenarios using t-distribution."""

    # Fit the t-distribution to the historical returns data
    df, loc, scale = t.fit(historical_market_rates)
    # Generate random samples
    market_return_scenarios = t.rvs(df, loc=loc, scale=scale, size=iterations)

    return market_return_scenarios 

def sim_risk_free_rate_scenarios(historical_risk_free_rate, iterations): 
    """Simulate risk-free-rate scenarios using a normal distribution.""" 
    # Calculate mean and standard deviation from historical data 
    mean_rate = np.mean(historical_risk_free_rate) 
    std_dev = np.std(historical_risk_free_rate)
    # Simulate future risk-free rates based on the calculated mean and standard deviation
    sim_risk_free_rates = np.random.normal(mean_rate, std_dev, iterations)
    # Ensure non-negativity
    sim_risk_free_rates = np.maximum(sim_risk_free_rates, 0)

    return sim_risk_free_rates



def simulate_betas(mean_beta, std_beta, iterations): 
    """Simulate betas for each scenario."""
    betas = np.random.normal(mean_beta, std_beta, iterations)

    return betas


def simulate_rrr_capm(betas, market_return_scenarios, risk_free_rates, iterations): 
    """Simulate required rate of retrun scenarios using the capm method."""




# Assuming the necessary variables (cf0, cash_flows, etc.) are defined
mean_npv, std_npv, npvs, simulated_discount_rates, simulated_cash_flows = run_npvs_sim(cf0, cash_flows, historical_discount_rates, discount_rate, iterations)

mean_pi, std_pi, pi_values = simulate_profitability_index(npvs, cf0)

mean_fv, std_fv, fvs = run_fv_sim(cf0, cash_flows, historical_discount_rates, discount_rate, 
    iterations)

# Simulate betas 
betas = simulate_betas(mean_beta, std_beta, iterations)
print(betas)

# Simulate risk-free rates 
risk_free_rates = sim_risk_free_rate_scenarios(historical_risk_free_rate, iterations)

# Function to get all NPVs from simulations 
def get_all_npvs(): 
    return npvs

all_npvs = get_all_npvs()
print(all_npvs)


# Function to get all Profitability Index values from simulations 
def get_profitability_index_values(): 
    return pi_values 

all_pi_values = get_profitability_index_values()
print(all_pi_values)


# Function to get all FVs from simulations 
def get_all_fvs(): 
    return fvs

all_fvs = get_all_fvs()
print(all_fvs)








def update_dict(): 
    """Update dist_calcs dictionary with the Mean and STD for all metrics.""" 
    dist_calcs['npv'] = {'distribution': 'normal', 'mean': mean_npv, 'std': std_npv}
    dist_calcs['profitability_index'] = {'distribution': 'normal', 'mean': mean_pi, 'std': std_pi}
    dist_calcs['fv'] = {'distribution': 'lognormal', 'mean': mean_fv, 'std': std_fv}

update_dict()
print(dist_calcs)





