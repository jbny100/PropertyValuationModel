
# distribution_calcs.py

import numpy as np 
import numpy_financial as npf 
from scipy.optimize import minimize
from scipy.stats import norm, t 
import math 

import make_simulation_params 
from make_simulation_params import discount_rate, historical_discount_rates
from make_simulation_params import cf0, time_years, cash_flows, cash_flows_excluding_debt
from make_simulation_params import risk_free_rate, historical_risk_free_rate, market_rate
from make_simulation_params import historical_market_rates, reinvestment_rate, finance_rate
from make_simulation_params import gross_operating_income, operating_expenses, projected_noi
from make_simulation_params import loan_amount, property_purchase_price, property_value_growth_rate
from make_simulation_params import rental_growth_rate, vacancy_rates, pre_tax_income 
from make_simulation_params import total_investment, gross_rental_income, capex
from make_simulation_params import inflation_rate, unemployment_rate, gdp_growth_rates
from make_simulation_params import construction_costs, other_costs, cost_contingency 
from make_simulation_params import property_value_last_year, total_debt 

dist_calcs = {
	'npv': {'distribution': 'normal'}, 
	'profitability_index': {'distribution': 'normal'}, 
	'fv': {'distribution': 'lognormal'},
    'rrr_capm': {'distribution': 'normal'},
    'rrr_income': {'distribution': 'normal'}, 
    'levered_irr': {'distribution': 'none'}, 
    'unlevered_irr': {'distribution': 'none'}, 
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

    # Calculate the simple mean of historical rates
    simple_mean_mu = np.mean(historical_discount_rates)

    # Adjust mu slightly upwards based on expectations
    adjusted_mu = simple_mean_mu + 0.01

     # Initial guesses for theta, mu and sigma
    initial_guess = [0.05, adjusted_mu, 0.02]

    # Perform optimization to estimate parameters
    result = minimize(ou_objective, initial_guess, args=(discount_rates))

    # Extract estimated params  
    theta_est, mu_est, sigma_est = result.x 

    # Use estimated params to calculate base mean and std at time t 
    mean_rate_t, std_t = calc_base_mean_std(discount_rate, theta_est, mu_est, 
        sigma_est, t)

    return theta_est, mu_est, sigma_est, mean_rate_t, std_t 

def generate_discount_rate_scenarios(theta_est, mu_est, sigma_est, iterations, t=1): 
    """Generate scenarios of simulated discount rates using the Ornstein-Uhlenbeck 
    (OU) process.""" 
    # Mean and variance of the discount rate at time t using OU parameters 
    mean_rate_t = mu_est * (1 - np.exp(-theta_est * t)) 
    variance_t = (sigma_est ** 2) / (2 * theta_est) * (1 - np.exp(-2 * theta_est * t)) 


    # Simulate discount rate scenarios
    sim_discount_rates = norm.rvs(loc=mean_rate_t, scale=np.sqrt(variance_t), 
        size=iterations)

    return sim_discount_rates



"""
Calculate the Mean and STD of Cash Flows using a lognormal distribution.

In the below simulate_cash_flows() function I would like to return the means and stds of 
each cash_flows_series in addition to the cash_flows_series.
"""



def simulate_cash_flows(cf0, cash_flows, iterations):
    """Simulate cash flow series using lognormal distribution, including cf0."""
    mean_log = np.log(np.mean(cash_flows)) 
    sigma_log = 0.04  # example standard deviation in log-space

    # Simulate future cash flows excluding cf0
    n_periods = len(cash_flows)
    simulated_cash_flows = np.exp(np.random.normal(mean_log, sigma_log, 
        size=(iterations, n_periods))) 

    # Prepend cf0 to each simulated series
    cash_flows_series = np.hstack((np.full((iterations, 1), cf0), simulated_cash_flows))

    # Calculate mean and std for each cash flow series
    cash_flows_means = np.mean(cash_flows_series, axis=1)
    cash_flows_stds = np.std(cash_flows_series, axis=1)

    return cash_flows_series, cash_flows_means, cash_flows_stds

def simulate_cash_flows_excluding_debt(cash_flows_excluding_debt, iterations): 
    """Simulate cash flow series excluding debt payments using lognormal distribution."""
    mean_log = np.log(np.mean(cash_flows_excluding_debt))
    sigma_log = 0.04  # example standard deviation in log-space

    # Simulate future cash flows excluding debt
    n_periods = len(cash_flows_excluding_debt)
    simulated_cash_flows_excluding_debt = np.exp(norm.rvs(mean_log, sigma_log, 
        size=(iterations, n_periods)))

    return simulated_cash_flows_excluding_debt


def calculate_npv_series(sim_discount_rates, cash_flows_series): 
    """Calculate NPV for each pair of simulated discount rate and cash flow series."""
    npvs = np.array([npf.npv(rate, flows) for rate, flows in zip(sim_discount_rates, 
        cash_flows_series)])
    return npvs 

# Number of scenarios to simulate 
iterations = 5
n_periods = len(cash_flows)
mean_beta = 1.0 
std_beta = 0.2 

def run_npv_simulation(cf0, cash_flows, historical_discount_rates, discount_rate, iterations): 
    # Call estimate_and_calc_ou() to get OU parameters for discount rate scenarios
    theta_est, mu_est, sigma_est, mean_rate_t, std_t = estimate_and_calc_ou(historical_discount_rates, 
        discount_rate, t=1)

    # Generate discount rate scenarios
    simulated_discount_rates = generate_discount_rate_scenarios(theta_est, mu_est, sigma_est, 
        t=1, iterations=iterations)

    # Generate cash flows series and discard means and stds for NPV calculation
    cash_flows_series, _, _ = simulate_cash_flows(cf0, cash_flows, iterations)

    # Calculate NPVs using the simulated discount rates and cash flows
    npvs = calculate_npv_series(simulated_discount_rates, cash_flows_series)

    # Calculate mean and standard deviation of the simulated NPVs 
    mean_npv = np.round(np.mean(npvs), 2) 
    std_npv = np.round(np.std(npvs), 2)

    return mean_npv, std_npv, npvs, simulated_discount_rates, cash_flows_series



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
    mean_npv, std_npv, npvs, simulated_discount_rates, simulated_cash_flows = run_npv_simulation(cf0, 
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
    """Returns an array of simulated RRRs based on CAPM for each scenario."""
    # Ensure the input arrays have the correct length matching iterations
    assert len(betas) == len(market_return_scenarios) == len(risk_free_rates) == iterations
    # Calculate RRR for each scenario using CAPM formula with condition
    rrr_capm_scenarios = np.where(
        risk_free_rates >= market_return_scenarios,
        risk_free_rates + betas * (0.10),  # Assuming a fixed market premium of 10% in scenarios where RFR >= Market Returns
        risk_free_rates + betas * (market_return_scenarios - risk_free_rates)
    )

    # Calculate mean and std of RRR CAPM scenarios
    mean_rrr_capm = np.mean(rrr_capm_scenarios)
    std_rrr_capm = np.std(rrr_capm_scenarios)

    return rrr_capm_scenarios, mean_rrr_capm, std_rrr_capm 

"""
The next three functions return scenario values for the inputs that go into calculating 
the Required Rate of Return (rrr) using the income approach: NOI, Total Debt, and 
Property Growth Rate.
"""

def simulate_noi_scenarios(projected_noi, iterations): 
    """Simulate Net Operating Income scenarios using a lognormal distribution and return 
    the simulated NOIs along with their mean and standard deviations"""
    # Calculate the log of the initial NOIs to estimate params for lognormal distribution
    log_projected_noi = np.log(projected_noi)

    # Calculate mean and standard deviation of log-transformed initial NOIs
    mean_log_noi = np.mean(log_projected_noi)
    std_log_noi = np.std(log_projected_noi)

    # Simulate NOI scenarios using the lognormal distribution
    noi_scenarios = np.exp(np.random.normal(mean_log_noi, std_log_noi, iterations))

    mean_noi = np.mean(noi_scenarios)
    std_noi = np.std(noi_scenarios) 

    return noi_scenarios, mean_noi, std_noi 

def calc_mortgage_payments(discount_rate, loan_amount, time_years):
    """Calculate montly loan interest payments."""

    monthly_interest_rate = discount_rate / 12
    total_periods = time_years * 12

    # Calculate monthly payment
    monthly_payment = npf.pmt(monthly_interest_rate, total_periods, -loan_amount)

    return monthly_payment

def calc_mortgage2_payments(discount_rate, loan_amount, time_years):
    """Calculate montly loan interest payments on additional mortgage if applicable."""
    monthly_interest_rate = discount_rate / 12
    total_periods = time_years * 12

    # Calculate monthly payment
    monthly_payment2 = npf.pmt(monthly_interest_rate, total_periods, -loan_amount)

    return monthly_payment2

def simulate_property_growth_rates(property_value_growth_rate, iterations): 
    """Simulates property growth rates using a lognormal distribution."""
    # Generate lognormal distribution of growth rates
    mean_growth_log = np.log(1 + property_value_growth_rate)  # Assuming a mean growth rate of 5.75%, converted to log-scale
    std_growth_log = 0.1  # # Assumed standard deviation in log-scale

    growth_rates = np.exp(np.random.normal(mean_growth_log, std_growth_log, iterations))

    return growth_rates, mean_growth_log, std_growth_log

growth_rates, mean_growth_log, std_growth_log = simulate_property_growth_rates(property_value_growth_rate, iterations)


def simulate_rrr_income(cash_flows_series, property_value_last_year, time_years, iterations):
    """Calculate RRR using the income approach directly from cash flow series, including 
    property sale in the final year."""
    rrr_income_scenarios = np.zeros(iterations)

    for i in range(iterations):
        # Copy the series to avoid altering the original data
        adjusted_cash_flows = np.copy(cash_flows_series[i])
        # Add the property sale value as the final inflow
        adjusted_cash_flows[-1] += property_value_last_year

        try:
            rrr = npf.irr(adjusted_cash_flows)
            rrr_income_scenarios[i] = rrr
        except:
            rrr_income_scenarios[i] = np.nan  # Handle non-convergence or other issues

    # Filter out NaN values before calculating mean and std to avoid NaN results
    valid_rrr = rrr_income_scenarios[~np.isnan(rrr_income_scenarios)]
    mean_rrr_income = np.mean(valid_rrr) if len(valid_rrr) > 0 else np.nan
    std_rrr_income = np.std(valid_rrr) if len(valid_rrr) > 0 else np.nan

    return rrr_income_scenarios, mean_rrr_income, std_rrr_income


def simulate_levered_irr(cash_flows_series, iterations): 
    """Simulate levered IRRs 

    Returns:
    - levered_irrs: An array containing the levered IRR for each scenario.
    - mean_levered_irr: The mean of the simulated levered IRRs.
    - std_levered_irr: The standard deviation of the simulated levered IRRs.
    """ 
    levered_irrs = np.zeros(iterations)
    for i in range(iterations): 
        try: 
            # Calculate th IRR for the current cash flow serries.
            irr = npf.irr(cash_flows_series[i])
            levered_irrs[i] = irr 
        except Exception as e: 
            print(f"Error calculating IRR for scenario {i+1}: {e}")
            levered_irrs[i] = np.nan # Use NaN to indicate a calculation error

    # Calculate mean and std, excluding any NaN values
    valid_irrs = levered_irrs[~np.isnan(levered_irrs)]
    mean_levered_irr = np.nanmean(valid_irrs)
    std_levered_irr = np.nanstd(valid_irrs)

    return levered_irrs, mean_levered_irr, std_levered_irr


def simulate_unlevered_irr(total_investment, simulated_cash_flows_excluding_debt, 
    iterations): 
    """Simulated unlevered IRRs for each scenario.

    Returns: 
    - unlevered_irrs: An array containing the unlevered IRR for each scenario.
    - mean_unlevered_irr: The mean of the simulated unlevered IRRs.
    - std_unlevered_irr: The standard deviation of the simulated unlevered IRRs.
    """
    unlevered_irrs = []
    for simulated_cash_flows in simulated_cash_flows_excluding_debt:
        # Combine initial investment with operational cash flows for IRR calculation
        cash_flow_series = np.concatenate(([-total_investment], simulated_cash_flows))
        try: 
            # Calculate the IRR for the current cash flow series excluding debt
            irr = npf.irr(cash_flow_series)
            unlevered_irrs.append(irr)
        except ValueError: 
            # Handle cases where IRR calculation does not converge
            unlevered_irrs.append(np.nan)
    # Filter out any nan values and calculate mean and std deviation
    unlevered_irrs = np.array(unlevered_irrs)
    valid_irrs = unlevered_irrs[~np.isnan(unlevered_irrs)]

    mean_unlevered_irr = np.mean(valid_irrs) if valid_irrs.size else np.nan
    std_unlevered_irr = np.std(valid_irrs) if valid_irrs.size else np.nan

    return unlevered_irrs, mean_unlevered_irr, std_unlevered_irr

def simulate_cap_rate(noi_scenarios, total_investment, iterations): 
    """Simulates cap rates based on NOI scenarios and total investment.

    Returns: 
    - cap_rates: An array containing the cap rate for each scenario.
    - mean_cap_rate: The mean of the simulated cap rates.
    - std_cap_rate: The standard deviation of the simulated cap rates.
    """
    simulated_cap_rates = noi_scenarios / total_investment
    mean_cap_rate = np.mean(simulated_cap_rates)
    std_cap_rate = np.std(simulated_cap_rates) 

    return simulated_cap_rates, mean_cap_rate, std_cap_rate 

def simulate_reversion_value(projected_noi, mean_cap_rate, iterations): 
    """Simulates reversion values based on the final year's NOI and the mean cap rate.

    Returns:
        Tuple[np.ndarray, float, float]: A tuple containing the array of simulated reversion 
        values, the mean of these reversion values, and their standard deviation.
    """
    # Assuming final year's NOI does not vary, use the last value from projected_noi
    final_year_noi = projected_noi[-1]

    # Simulate NOI for the final year using a dist that reflects potential variability
    # For simplicity we use the mean of the final year's NOI with a fixed std
    std_dev_noi = final_year_noi * 0.05  # Assuming a 5% standard deviation
    simulated_final_noi = np.random.normal(final_year_noi, std_dev_noi, iterations)

    # Calculate reversion values for each scenario and round to nearest whole number
    simulated_reversion_values = np.round(simulated_final_noi / mean_cap_rate, 0)

    # Calculate the mean and standard deviation of the simulated reversion values

    mean_reversion_value = np.mean(simulated_reversion_values)
    std_reversion_value = np.std(simulated_reversion_values)

    return simulated_reversion_values, mean_reversion_value, std_reversion_value


def simulate_capex(construction_costs, other_costs, cost_contingency, iterations): 
    """Simulate CAPEX values incorporating variability in construction costs, 
    other costs, and cost contingency.

    Returns: 
        Tuple[np.ndarray, float, float]: A tuple containing the array of simulated 
        CAPEX values, the mean of these CAPEX values, and their standard deviation.

    """
    # Base total costs before contingency
    base_total_costs = construction_costs + other_costs

    # Simulating cost contingency percentages from 0 to the maximum specified
    simulated_contingencies = np.random.uniform(0, cost_contingency, iterations)

    # Applying the simulated contingencies to the base total costs
    simulated_capex_values = base_total_costs * (1 + simulated_contingencies)

    # Round values to nearest whole number
    simulated_capex_values = np.round(simulated_capex_values)

    # If costs log-normally distributed, convert to log-space for mean and std calc
    log_simulated_capex = np.log(simulated_capex_values)
    mean_log_capex = np.mean(log_simulated_capex)
    std_log_capex = np.std(log_simulated_capex)

    # Convert mean and std back from log-space to original scale and 
    # round to the nearest whole number
    mean_capex = np.round(np.exp(mean_log_capex))
    std_capex = np.round((np.exp(std_log_capex**2) - 1) * \
        np.exp(2*mean_log_capex + std_log_capex**2))

    return simulated_capex_values, mean_capex, std_capex 

def simulate_cash_on_cash_return(cash_flows_means, total_investment, iterations): 
    """Calculate cash_on_cash return estimates for each iteration scenario in a 
    simulation.

    Returns: 
        Tuple[np.ndarray, float, float]: A tuple containing the array of simulated 
        cash-on-cash return values, the mean of these returns, and their standard deviation.
    """
    # Calculate cash-on-cash return for each scenario
    simulated_cash_on_cash_returns = np.round(cash_flows_means / total_investment, 3)

    # Calculate the mean and standard deviation of the simulated cash-on-cash return values
    cash_on_cash_return_mean = np.round(np.mean(simulated_cash_on_cash_returns), 3)
    cash_on_cash_return_std = np.round(np.std(simulated_cash_on_cash_returns), 3)

    return simulated_cash_on_cash_returns, cash_on_cash_return_mean, cash_on_cash_return_std 


def simulate_rental_income(gross_rental_income, iterations):
    """Simulate rental income values using a lognormal distribution."""
    # Log-transform the base gross rental income values
    log_gross_rental_income = np.log(gross_rental_income)
    
    # Calculate mean and std of the log-transformed values
    mean_log = np.mean(log_gross_rental_income)
    std_log = np.std(log_gross_rental_income)
    
    # Simulate rental income values
    simulated_rental_incomes = np.exp(np.random.normal(mean_log, std_log, iterations))
    
    # Calculate mean and std of the simulated rental income
    rental_income_mean_values = np.mean(simulated_rental_incomes)
    rental_income_std_values = np.std(simulated_rental_incomes)
    
    return simulated_rental_incomes, np.round(rental_income_mean_values, 2), np.round(rental_income_std_values, 2) 


def simulate_rent_multiplier(property_purchase_price, simulated_rental_incomes, iterations):
    """Simulates rent multiplier scenarios based on simulated rental incomes."""
    simulated_rent_multipliers = np.zeros(iterations)

    # Ensure simulated_rental_incomes is an array with length equal to iterations
    print(f"Expected iterations: {iterations}, Actual simulated_rental_incomes length: {len(simulated_rental_incomes)}")

    # Calculate rent multipliers for each iteration
    for i in range(iterations): 
        simulated_rent_multipliers[i] = property_purchase_price / simulated_rental_incomes[i]
    
    # Calculate mean and std dev of simulated rent multipliers, round to 2 decimal places
    rent_multiplier_mean = np.round(np.mean(simulated_rent_multipliers), 2)
    rent_multiplier_std = np.round(np.std(simulated_rent_multipliers), 2) 

    # Rounding to 2 decimal places
    simulated_rent_multipliers = np.round(simulated_rent_multipliers, 2)
    rent_multiplier_mean = round(rent_multiplier_mean, 2)
    rent_multiplier_std = round(rent_multiplier_std, 2)
    
    return simulated_rent_multipliers, rent_multiplier_mean, rent_multiplier_std 


def simulate_pre_tax_income(pre_tax_income, iterations): 
    """Simulate pre-tax income scenarios using lognormal distribution based on 
    initial base values and return imulated pre-tax incomes along with their mean and 
    standard deviation for each scenario.

    Returns:
        np.ndarray: Simulated pre-tax incomes for each iteration.
        float: Mean of the simulated pre-tax incomes.
        float: Standard deviation of the simulated pre-tax incomes.
    """
    # Log-transform the initial pre-tax income values to estimate parameters
    log_pre_tax_income = np.log(pre_tax_income)
    mean_log = np.mean(log_pre_tax_income)
    std_log = np.std(log_pre_tax_income)

    # Simulate pre-tax income scenarios using the lognormal distribution
    simulated_pre_tax_income = np.exp(np.random.normal(mean_log, std_log, iterations))

    # Calculate the mean and standard deviation of the simulated pre-tax incomes
    simulated_pre_tax_mean = np.mean(simulated_pre_tax_income)
    simulated_pre_tax_std = np.std(simulated_pre_tax_income)

    # Round the mean and standard deviation to 2 decimal places
    simulated_pre_tax_mean = np.round(simulated_pre_tax_mean, 2)
    simulated_pre_tax_std = np.round(simulated_pre_tax_std, 2)

    return simulated_pre_tax_income, simulated_pre_tax_mean, simulated_pre_tax_std 


def simulate_total_debt(total_debt, iterations): 
    """Simulate total debt scenarios using a lognormal distribution, based on initial 
    base values. Return simulated debt values along with their mean and standard deviation for 
    each scenario.

    Returns:
        np.ndarray: Simulated total debts for each iteration.
        float: Mean of the simulated total debts.
        float: Standard deviation of the simulated total debts.
    """

    # Log-transform the initial total debt values to estimate parameters
    log_total_debt = np.log(total_debt)
    mean_log = np.mean(log_total_debt)
    std_log = np.std(log_total_debt)

    # Simulate total debt scenarios using the lognormal distribution
    simulated_total_debt = np.exp(np.random.normal(mean_log, std_log, iterations))

    # Calculate the mean and standard deviation of the simulated total debts
    simulated_total_debt_mean = np.mean(simulated_total_debt)
    simulated_total_debt_std = np.std(simulated_total_debt)

    # Round the mean and standard deviation to 2 decimal places
    simulated_total_debt_mean = np.round(simulated_total_debt_mean, 2)
    simulated_total_debt_std = np.round(simulated_total_debt_std, 2)

    return simulated_total_debt, simulated_total_debt_mean, simulated_total_debt_std

def simulate_operating_expenses(operating_expenses, iterations): 
    """Simulate operating expense scenarios using a lognormal distribution, based on 
    initial base values. Return simulated operating expenses along with their mean and 
    standard deviation for each scenario.

    Returns:
        np.ndarray: Simulated operating expenses for each iteration.
        float: Mean of the simulated operating expenses.
        float: Standard deviation of the simulated operating expenses.
    """
    # Log-transform the initial operating expenses to estimate parameters
    log_operating_expenses = np.log(operating_expenses)
    mean_log = np.mean(log_operating_expenses)
    std_log = np.std(log_operating_expenses)

    # Simulate operating expenses scenarios using the lognormal distribution
    simulated_operating_expenses = np.exp(np.random.normal(mean_log, std_log, iterations))

    # Calculate the mean and standard deviation of the simulated operating expenses
    simulated_operating_expenses_mean = np.mean(simulated_operating_expenses)
    simulated_operating_expenses_std = np.std(simulated_operating_expenses)

    # Round the mean and standard deviation to 2 decimal places
    simulated_operating_expenses_mean = np.round(simulated_operating_expenses_mean, 2)
    simulated_operating_expenses_std = np.round(simulated_operating_expenses_std, 2) 

    return simulated_operating_expenses, simulated_operating_expenses_mean, simulated_operating_expenses_std



def simulate_be_ratio(simulated_pre_tax_income, simulated_total_debt, 
    simulated_operating_expenses, iterations): 
    print(f"Pre-tax Income: {simulated_pre_tax_income}, Length: {len(simulated_pre_tax_income)}")
    print(f"Total Debt: {simulated_total_debt}, Length: {len(simulated_total_debt)}")
    print(f"Operating Expenses: {simulated_operating_expenses}, Length: {len(simulated_operating_expenses)}")
    """
    Simulates break-even ratio scenarios based on simulated pre-tax income, total debt, 
    and operating expenses. 
    
    Returns:
        np.ndarray: Simulated break-even ratio for each iteration.
        float: Mean of the simulated break-even ratios.
        float: Standard deviation of the simulated break-even ratios.
    """
    # Initialize an array to hold the simulated break-even ratios
    simulated_be_ratios = np.zeros(iterations)

    # Iterate over each scenario
    for i in range(iterations):
        # Calculate break-even ratio for each scenario
        # Note: Ensure the calculation is correct as per your specific formula
        simulated_be_ratios[i] = (operating_expenses[i] + total_debt[i]) / pre_tax_income[i]

    # Calculate mean and standard deviation of simulated break-even ratios
    mean_be_ratio = np.mean(simulated_be_ratios)
    std_be_ratio = np.std(simulated_be_ratios)

    # Round results to 2 decimal places
    simulated_be_ratios = np.round(simulated_be_ratios, 2)
    mean_be_ratio = round(mean_be_ratio, 2)
    std_be_ratio = round(std_be_ratio, 2)

    return simulated_be_ratios, mean_be_ratio, std_be_ratio


def simulate_inflation_rates(inflation_rate, iterations): 
    """Simulates inflation rate scenarios using the lognormal distribution. Return simulated inflation rates
    along with their mean and standard deviation for each scenario.

    Returns:
    tuple: Contains simulated inflation rates, their mean, and standard deviation
    """
    # Log-transform the historical inflation rates
    # np.log1p for log(1+x) to handle zero inflation rates
    log_inflation_rate = np.log1p(inflation_rate) 

    # Calculate the mean and standard deviation of the log-transformed rates
    mean_log = np.mean(log_inflation_rate)
    std_log = np.std(log_inflation_rate)

    # Simulate log-normal inflation rates
    simulated_log_inflation = np.random.normal(mean_log, std_log, iterations) 
    # Reverse the log(1+x) transform
    simulated_inflation_rates = np.expm1(simulated_log_inflation) 

    # Calculate mean and standard deviation of the simulated inflation rates
    simulated_inflation_rates_mean = np.mean(simulated_inflation_rates)
    simulated_inflation_rates_std = np.std(simulated_inflation_rates)

    # Round results to 2 decimal places
    simulated_inflation_rates = np.round(simulated_inflation_rates, 2)
    simulated_inflation_rates_mean = round(simulated_inflation_rates_mean, 2)
    simulated_inflation_rates_std = round(simulated_inflation_rates_std, 2)

    return simulated_inflation_rates, simulated_inflation_rates_mean, simulated_inflation_rates_std 


def simulate_vacancy_rates(vacancy_rates, iterations): 
    """Simulates vacancy rate scenarios using the beta distribution. Return simulated 
    along with their mean and standard deviation for each scenario.

    Return: 
    tuple: Contains simulated vacancy rates, their mean, and standard deviation.
    """
    # Convert rates to a numpy array for operations
    vacancy_rates = np.array(vacancy_rates)

    # Estimate parameters of the Beta distribution from data
    alpha, beta = estimate_beta_params(vacancy_rates)

    # Simulate Beta Distribution 
    simulated_vacancy_rates = np.random.beta(alpha, beta, size=iterations)

    # Calculate mean and standard deviation of the simulated vacancy rates
    simulated_vacancy_rates_mean = np.mean(simulated_vacancy_rates)
    simulated_vacancy_rates_std = np.std(simulated_vacancy_rates)

    # Round results to 4 decimal places for precision
    simulated_vacancy_rates = np.round(simulated_vacancy_rates, 4)
    simulated_vacancy_rates_mean = round(simulated_vacancy_rates_mean, 4)
    simulated_vacancy_rates_std = round(simulated_vacancy_rates_std, 4)

    return simulated_vacancy_rates, simulated_vacancy_rates_mean, simulated_vacancy_rates_std

def estimate_beta_params(data): 
    """Estimate Beta distribution parameters from data.

    Returns: 
        tuple: Estimated alpha and beta parameters.
    """
    mean_data = data.mean()
    var_data = data.var()

    # Use method of moments to estimate parameters
    alpha = mean_data * ((mean_data * (1 - mean_data)) / var_data - 1)
    beta = (1 - mean_data) * ((mean_data * (1 - mean_data)) / var_data - 1)

    return alpha, beta 


def simulate_gdp_growth_rates(gdp_growth_rate, iterations): 
    """Simulate GDP growth rate scenarios using a t-distribution. 

    Returns:
        tuple: Contains simulated GDP growth rates, their mean, and standard deviation
    """
    # Estimate parameters for t-distribution from historical data
    df, loc, scale = t.fit(gdp_growth_rates)

    # Mean and standard deviation of historical data
    mean_growth = np.mean(gdp_growth_rate)
    std_growth = np.std(gdp_growth_rate)

    # Simulate GDP Growth Rates
    simulated_gdp_growth_rates = np.random.normal(mean_growth, std_growth, size=iterations)

    # Calculate mean and standard deviation of the simulated GDP growth rates
    simulated_gdp_growth_mean = np.mean(simulated_gdp_growth_rates)
    simulated_gdp_growth_std = np.std(simulated_gdp_growth_rates)

    # Round results to 4 decimal places for precision
    simulated_gdp_growth_rates = np.round(simulated_gdp_growth_rates, 4)
    simulated_gdp_growth_mean = round(simulated_gdp_growth_mean, 4)
    simulated_gdp_growth_std = round(simulated_gdp_growth_std, 4)

    return simulated_gdp_growth_rates, simulated_gdp_growth_mean, simulated_gdp_growth_std


# Simulate cash_flows_series and only pass cash_flows_series parameter to output
cash_flows_series, cash_flows_means, cash_flows_stds = simulate_cash_flows(cf0, cash_flows, 
    iterations)

# Simulate discount rates
theta_est, mu_est, sigma_est, mean_rate_t, std_t = estimate_and_calc_ou(historical_discount_rates, 
    discount_rate, t=1)
sim_discount_rates = generate_discount_rate_scenarios(theta_est, mu_est, sigma_est, 
    iterations, t=1)

# Simulate NPV series results
npvs = calculate_npv_series(sim_discount_rates, cash_flows_series)

mean_npv, std_npv, npvs, simulated_discount_rates, cash_flows_series = run_npv_simulation(cf0, cash_flows, 
    historical_discount_rates, discount_rate, iterations)

mean_pi, std_pi, pi_values = simulate_profitability_index(npvs, cf0)

mean_fv, std_fv, fvs = run_fv_sim(cf0, cash_flows, historical_discount_rates, discount_rate, 
    iterations)

# Simulate betas 
betas = simulate_betas(mean_beta, std_beta, iterations)

# Simulate risk-free rates 
risk_free_rates = sim_risk_free_rate_scenarios(historical_risk_free_rate, iterations)

# Simulate market return scenarios
market_return_scenarios = sim_market_return_scenario(historical_market_rates, iterations)

# Simulate rrr_capm values
rrr_capm_scenarios, mean_rrr_capm, std_rrr_capm = simulate_rrr_capm(betas, 
    market_return_scenarios, risk_free_rates, iterations)


# Return noi scenarios
noi_scenarios, mean_noi, std_noi = simulate_noi_scenarios(projected_noi, iterations)

# Simulate monthly debt payments
monthly_payment = calc_mortgage_payments(discount_rate, loan_amount, time_years)
monthly_payment2 = 0

# Simulate property growth rate scenarios
growth_rates = simulate_property_growth_rates(discount_rate, iterations)

# Simulate rrr_income values
rrr_income_scenarios, mean_rrr_income, std_rrr_income = simulate_rrr_income(cash_flows_series, 
    property_value_last_year, time_years, iterations)

# Simulate levered and unlevered IRR scenario values
levered_irrs, mean_levered_irr, std_levered_irr = simulate_levered_irr(cash_flows_series, 
    iterations)

simulated_cash_flows_excluding_debt = simulate_cash_flows_excluding_debt(cash_flows_excluding_debt, 
    iterations)

unlevered_irrs, mean_unlevered_irr, std_unlevered_irr = simulate_unlevered_irr(total_investment, 
    simulated_cash_flows_excluding_debt, iterations)

# Simulate cap rate scenario values
simulated_cap_rates, mean_cap_rate, std_cap_rate = simulate_cap_rate(noi_scenarios, 
    total_investment, iterations)

# Simulate reversion value scenarios 
simulated_reversion_values, mean_reversion_value, std_reversion_value = simulate_reversion_value(projected_noi, mean_cap_rate, iterations)

# Simulate capex value scenarios 
simulated_capex_values, mean_capex, std_capex = simulate_capex(construction_costs, 
    other_costs, cost_contingency, iterations)

# Simulate cash-on-cash return values 
simulated_cash_on_cash_returns, cash_on_cash_return_mean, cash_on_cash_return_std = simulate_cash_on_cash_return(cash_flows_means, 
    total_investment, iterations)

# Simulate rent multiplier return values using simulated rental income scenario
simulated_rental_incomes, rental_income_mean_values,rental_income_std_values = simulate_rental_income(gross_rental_income, iterations)

simulated_rent_multipliers, rent_multiplier_mean, rent_multiplier_std = simulate_rent_multiplier(property_purchase_price, simulated_rental_incomes, iterations)

# Simulate break-even return values using simulated pre-tax income, simulated total debt, and simulated operating expense scenarios
simulated_pre_tax_income, simulated_pre_tax_mean, simulated_pre_tax_std = simulate_pre_tax_income(pre_tax_income, iterations)
simulated_total_debt, simulated_total_debt_mean, simulated_total_debt_std = simulate_total_debt(total_debt, iterations)
simulated_operating_expenses, simulated_operating_expenses_mean, simulated_operating_expenses_std = simulate_operating_expenses(operating_expenses, iterations)
simulated_be_ratios, be_ratio_mean, be_ratio_std = simulate_be_ratio(simulated_pre_tax_income, simulated_total_debt, simulated_operating_expenses, iterations)


# Simulate all inflation rate scenario values.
simulated_inflation_rates, simulated_inflation_rates_mean, simulated_inflation_rates_std = simulate_inflation_rates(inflation_rate, iterations)


# Simulate vacancy rate scenario values
simulated_vacancy_rates, simulated_vacancy_rates_mean, simulated_vacancy_rates_std = simulate_vacancy_rates(vacancy_rates, iterations)

# Simulate GDP growth rate scenario values
simulated_gdp_growth_rates, simulated_gdp_growth_mean, simulated_gdp_growth_std = simulate_gdp_growth_rates(gdp_growth_rates, iterations)


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

# Function to get all rrr_capm values from simulations 
def get_all_rrr_capm(): 
    return rrr_capm_scenarios

all_rrr_capm = get_all_rrr_capm()
print(all_rrr_capm)

# Function to get all rrr_income values from simulations 
def get_all_rrr_income(): 
    return rrr_income_scenarios

all_rrr_income = get_all_rrr_income()
print(all_rrr_income)

# Function to get all levered_irr values from simulations
def get_all_levered_irrs():
    return levered_irrs

all_levered_irrs = get_all_levered_irrs()
print(all_levered_irrs)

# Function to get all unlevered_irr values from simulations
def get_all_unlevered_irrs():
    return unlevered_irrs

all_unlevered_irrs = get_all_unlevered_irrs()
print(all_unlevered_irrs)

# Function to get all cap rate value scenarios from simulations
def get_all_cap_rates():
    return simulated_cap_rates

all_cap_rates = get_all_cap_rates()
print(all_cap_rates)

# Function to get all reversion value scenarios from simulations
def get_all_reversion_values():
    return simulated_reversion_values

all_reversion_values = get_all_reversion_values()
print(all_reversion_values)


# Function to get all capex value scenarios from simulations
def get_all_capex_values():
    return simulated_capex_values

all_capex_values = get_all_capex_values()
print(all_capex_values)


# Function to get all cash_on_cash value scenarios from simulations
def get_all_cash_on_cash_values():
    return simulated_cash_on_cash_returns

all_cash_on_cash_values = get_all_cash_on_cash_values()
print(all_cash_on_cash_values)


# Function to get all rent_multiplier value scenarios from simulations
def get_all_rent_multiplier_values():
    return simulated_rent_multipliers

all_rent_multiplier_values = get_all_rent_multiplier_values()
print(all_rent_multiplier_values)


# Function to get all break-even-ratio scenario values from simulations
def get_all_be_ratio_values():
    return simulated_be_ratios

all_be_ratio_values = get_all_be_ratio_values()
print(all_be_ratio_values)


# Function to get all vacancy rate scenario values from simulations
def get_all_vacancy_rate_values():
    return simulated_vacancy_rates

all_vacancy_rate_values = get_all_vacancy_rate_values()
print(all_vacancy_rate_values)

# Function to get all gdp growth scenario values from simulations
def get_all_gdp_growth_rates():
    return simulated_gdp_growth_rates

all_gdp_growth_rates = get_all_gdp_growth_rates()
print(all_gdp_growth_rates)



def update_dict(): 
    """Update dist_calcs dictionary with the Mean and STD for all metrics.""" 
    dist_calcs['npv'] = {'distribution': 'normal', 'mean': mean_npv, 'std': std_npv}
    dist_calcs['profitability_index'] = {'distribution': 'normal', 'mean': mean_pi, 'std': std_pi}
    dist_calcs['fv'] = {'distribution': 'lognormal', 'mean': mean_fv, 'std': std_fv}
    dist_calcs['rrr_capm'] = {'distribution': 'normal', 'mean': mean_rrr_capm, 'std': std_rrr_capm}
    dist_calcs['rrr_income'] = {'distribution': 'normal', 'mean': mean_rrr_income, 'std': std_rrr_income}
    dist_calcs['levered_irr'] = {'distribution': 'none', 'mean': mean_levered_irr, 'std': std_levered_irr}
    dist_calcs['unlevered_irr'] = {'distribution': 'none', 'mean': mean_unlevered_irr, 'std': std_unlevered_irr}
    dist_calcs['cap_rate'] = {'distribution': 'lognormal', 'mean': mean_cap_rate, 'std': std_cap_rate}
    dist_calcs['reversion_value'] = {'distribution': 'lognormal', 'mean': mean_reversion_value, 'std': std_reversion_value}
    dist_calcs['capex'] = {'distribution': 'lognormal', 'mean': mean_capex, 'std': mean_capex}
    dist_calcs['cash-on-cash_return'] = {'distribution': 'normal', 'mean': cash_on_cash_return_mean, 'std': cash_on_cash_return_std}
    dist_calcs['rent_multiplier'] = {'distribution': 'normal', 'mean': rent_multiplier_mean, 'std': rent_multiplier_std}
    dist_calcs['break_even_ratio'] = {'distribution': 'normal', 'mean': be_ratio_mean, 'std': be_ratio_std}
    dist_calcs['inflation_rate'] = {'distribution': 'lognormal', 'mean': simulated_inflation_rates_mean, 'std': simulated_inflation_rates_std}
    dist_calcs['vacancy_rate'] = {'distribution': 'beta', 'mean': simulated_vacancy_rates_mean, 'std': simulated_vacancy_rates_std}
    dist_calcs['gdp_growth_rate'] = {'distribution': 'normal', 'mean': simulated_gdp_growth_mean, 'std': simulated_gdp_growth_std}

update_dict()
print(dist_calcs)





