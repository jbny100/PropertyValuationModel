# run_simulations.py

import numpy as np 
import numpy_financial as npf 
from scipy.optimize import minimize
from scipy.stats import norm, t 
import math 
from pprint import pprint

import make_simulation_params 
from make_simulation_params import discount_rate, historical_discount_rates
from make_simulation_params import cf0, time_years, cash_flows, cash_flows_excluding_debt
from make_simulation_params import risk_free_rate, historical_risk_free_rate, market_rate
from make_simulation_params import historical_market_rates, reinvestment_rate, finance_rate
from make_simulation_params import gross_operating_income, operating_expenses, projected_noi
from make_simulation_params import loan_amount, loan_amount2, property_purchase_price, property_value_growth_rate
from make_simulation_params import rental_growth_rate, vacancy_rates, pre_tax_income 
from make_simulation_params import total_investment, gross_rental_income, capex
from make_simulation_params import inflation_rate, unemployment_rate, gdp_growth_rates
from make_simulation_params import construction_costs, other_costs, cost_contingency 
from make_simulation_params import property_value_last_year, total_debt 


# Number of scenarios to simulate 
iterations = 5
n_periods = len(cash_flows)

# Initial assumptions for mean and std of beta value
mean_beta = 1.0 
std_beta = 0.2 


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
    try: 
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
    except ValueError as e:
        print(f"ValueError occurred in simulate_cash_flows: {e}")
    except OverflowError as e: 
        print(f"OverflowError occurred in simulate_csah_flows: {e}")

def simulate_cash_flows_excluding_debt(cash_flows_excluding_debt, iterations): 
    """Simulate cash flow series excluding debt payments using lognormal distribution."""
    try: 
        mean_log = np.log(np.mean(cash_flows_excluding_debt))
        sigma_log = 0.04  # example standard deviation in log-space

        # Simulate future cash flows excluding debt
        n_periods = len(cash_flows_excluding_debt)
        simulated_cash_flows_excluding_debt = np.exp(norm.rvs(mean_log, sigma_log, 
            size=(iterations, n_periods)))

        return simulated_cash_flows_excluding_debt
    except ValueError as e:
        print(f"ValueError occurred in simulate_cash_flows: {e}")
    except OverflowError as e: 
        print(f"OverflowError occurred in simulate_csah_flows: {e}")


def calculate_npv_series(sim_discount_rates, cash_flows_series):
    """Calculate NPV for each pair of simulated discount rate and cash flow series."""
    npvs = np.empty(len(sim_discount_rates))  # Initialize the NPVs array
    npvs.fill(np.nan)  # Pre-fill with NaNs to handle potential errors gracefully

    for i, (rate, flows) in enumerate(zip(sim_discount_rates, cash_flows_series)):
        try:
            npvs[i] = npf.npv(rate, flows)
        except Exception as e:
            print(f"Error calculating NPV for scenario {i}: {e}")
    return npvs


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
    npvs = np.round(calculate_npv_series(simulated_discount_rates, cash_flows_series))

    # Calculate mean and standard deviation of the simulated NPVs 
    mean_npv = np.round(np.mean(npvs)) 
    std_npv = np.round(np.std(npvs))

    return mean_npv, std_npv, npvs, simulated_discount_rates, cash_flows_series



def simulate_profitability_index(npvs, cf0):
    # Calculate the PI for each NPV value using the fixed CF0. 
    pi_values = np.round(npvs / - cf0, 2)
    
    # Calculate the mean and standard deviation of the resulting PI values 
    mean_pi = np.round(np.mean(pi_values), 2) 
    std_pi = np.round(np.std(pi_values), 2)
    
    return mean_pi, std_pi, pi_values


def calculate_fv_series(simulated_discount_rates, cash_flows_series, n_periods):
    """Calculate fv for each pair of simulated discount rate and cash flow series."""
    fvs = np.empty(len(simulated_discount_rates))  # Initialize the FVs array
    fvs.fill(np.nan)  # Pre-fill with NaNs to handle potential errors gracefully

    for i, (rate, flows) in enumerate(zip(simulated_discount_rates, cash_flows_series)):
        try:
            fvs[i] = npf.fv(rate, n_periods, 0, -np.sum(flows))
        except Exception as e:
            print(f"Error calculating FV for scenario {i}: {e}")
    return fvs 

def run_fv_sim(cf0, cash_flows, historical_discount_rates, discount_rate, iterations):
    """Run FV simulations."""
    # Use run_npv_simulation to get simulated discount rates and cash flows
    mean_npv, std_npv, npvs, simulated_discount_rates, simulated_cash_flows = run_npv_simulation(cf0, 
        cash_flows, historical_discount_rates, discount_rate, iterations)
    # Calculate FV for each scenario
    fvs = calculate_fv_series(simulated_discount_rates, simulated_cash_flows, len(cash_flows))
    fvs = np.round(fvs)  # Round FV calculations after handling potential errors
    # Calculate mean and std for FV
    mean_fv = np.round(np.mean(fvs[np.isnan(fvs) == False]))  # Ignore NaN values for mean calculation
    std_fv = np.round(np.std(fvs[np.isnan(fvs) == False]))  # Ignore NaN values for std calculation

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
    try: 
        assert len(betas) == len(market_return_scenarios) == len(risk_free_rates) == iterations
        # Calculate RRR for each scenario using CAPM formula with condition
        rrr_capm_scenarios = np.round(np.where(
            risk_free_rates >= market_return_scenarios,
            risk_free_rates + betas * (0.10),  # Assuming a fixed market premium of 10% in scenarios where RFR >= Market Returns
            risk_free_rates + betas * (market_return_scenarios - risk_free_rates)), 3)

        # Calculate mean and std of RRR CAPM scenarios
        mean_rrr_capm = np.round(np.mean(rrr_capm_scenarios), 3)
        std_rrr_capm = np.round(np.std(rrr_capm_scenarios), 3)

        return rrr_capm_scenarios, mean_rrr_capm, std_rrr_capm 
    except AssertionError: 
        print("Input arrays must have the same length matching the number of iterations.")
        return None, None, None 

"""
The next three functions return scenario values for the inputs that go into calculating 
the Required Rate of Return (rrr) using the income approach: NOI, Total Debt, and 
Property Growth Rate.
"""

def simulate_noi_scenarios(projected_noi, iterations): 
    """Simulate Net Operating Income scenarios using a lognormal distribution and return 
    the simulated NOIs along with their mean and standard deviations"""
    # Calculate the log of the initial NOIs to estimate params for lognormal distribution
    try: 
        log_projected_noi = np.log(projected_noi)

        # Calculate mean and standard deviation of log-transformed initial NOIs
        mean_log_noi = np.mean(log_projected_noi)
        std_log_noi = np.std(log_projected_noi)

        # Simulate NOI scenarios using the lognormal distribution
        noi_scenarios = np.exp(np.random.normal(mean_log_noi, std_log_noi, iterations))

        mean_noi = np.mean(noi_scenarios)
        std_noi = np.std(noi_scenarios) 

        return noi_scenarios, mean_noi, std_noi 
    except ValueError as e: 
        print(f"Attempting to take log of non-positive number in simulate_noi_scenarios: {e}")

def calc_mortgage_payments(discount_rate, loan_amount, time_years):
    """Calculate monthly loan interest payments."""
    if discount_rate <= 0 or loan_amount <= 0 or time_years <= 0: 
        raise ValueError("All parameters must be positive")
    try: 
        monthly_interest_rate = discount_rate / 12
        total_periods = time_years * 12

        # Calculate monthly payment
        monthly_mortgage_payment = npf.pmt(monthly_interest_rate, total_periods, -loan_amount)

        return monthly_mortgage_payment
    except Exception as e: 
        print(f"Unexpected error in calc_mortgage_payments: {e}")


def calc_mortgage2_payments(discount_rate, loan_amount2, time_years):
    """Calculate montly loan interest payments on additional mortgage if applicable."""
    if discount_rate <= 0 or loan_amount2 <= 0 or time_years <= 0:
        raise ValueError("All parameters must be positive")
    try: 
        monthly_interest_rate = discount_rate / 12
        total_periods = time_years * 12

        # Calculate monthly payment
        monthly_mortgage2_payment = npf.pmt(monthly_interest_rate, total_periods, -loan_amount2)

        return monthly_mortgage2_payment

    except Exception as e: 
        print(f"Unexpected error in calc_mortgage2_payments: {e}")


def simulate_property_growth_rates(property_value_growth_rate, iterations): 
    """Simulates property growth rates using a lognormal distribution."""
    # Generate lognormal distribution of growth rates
    mean_growth_log = np.log(1 + property_value_growth_rate)  # Assuming a mean growth rate of 5.75%, converted to log-scale
    std_growth_log = 0.1  # # Assumed standard deviation in log-scale

    growth_rates = np.exp(np.random.normal(mean_growth_log, std_growth_log, iterations))

    return growth_rates, mean_growth_log, std_growth_log


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
            rrr_income_scenarios[i] = npf.irr(adjusted_cash_flows)
        except Exception as e: 
            print(f"Error calculating IRR for scenario {i}: {e}")
            rrr_income_scenarios[i] = np.nan  # Handle non-convergence or other issues

    # Filter out NaN values before calculating mean and std to avoid NaN results
    valid_rrr = rrr_income_scenarios[~np.isnan(rrr_income_scenarios)]
    mean_rrr_income = np.mean(valid_rrr) if len(valid_rrr) > 0 else np.nan
    std_rrr_income = np.std(valid_rrr) if len(valid_rrr) > 0 else np.nan

    # Round the results
    rrr_income_scenarios = np.round(rrr_income_scenarios, 3)
    mean_rrr_income = np.round(mean_rrr_income, 3)
    std_rrr_income = np.round(std_rrr_income, 3)

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

    # Round the results
    levered_irrs = np.round(levered_irrs, 3)
    mean_levered_irr = np.round(mean_levered_irr, 3)
    std_levered_irr = np.round(std_levered_irr, 3)

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

    # Round the results
    unlevered_irrs = np.round(unlevered_irrs, 3)
    mean_unlevered_irr = np.round(mean_unlevered_irr, 3)
    std_unlevered_irr = np.round(std_unlevered_irr, 3)

    return unlevered_irrs, mean_unlevered_irr, std_unlevered_irr


def simulate_cap_rate(noi_scenarios, total_investment, iterations): 
    """Simulates cap rates based on NOI scenarios and total investment.

    Returns: 
    - cap_rates: An array containing the cap rate for each scenario.
    - mean_cap_rate: The mean of the simulated cap rates.
    - std_cap_rate: The standard deviation of the simulated cap rates.
    """
    try: 
        if total_investment <= 0: 
            raise ValueError("Total investment must be greater than 0 to avoid division by zero.")
        simulated_cap_rates = np.round(noi_scenarios / total_investment, 4)
        mean_cap_rate = np.round(np.mean(simulated_cap_rates), 4)
        std_cap_rate = np.round(np.std(simulated_cap_rates), 4) 

        return simulated_cap_rates, mean_cap_rate, std_cap_rate
    except Exception as e: 
        print(f"An error occurred in simulate_cap_rate: {e}")


def simulate_reversion_value(projected_noi, mean_cap_rate, iterations): 
    """Simulates reversion values based on the final year's NOI and the mean cap rate.

    Returns:
        Tuple[np.ndarray, float, float]: A tuple containing the array of simulated reversion 
        values, the mean of these reversion values, and their standard deviation.
    """
    # Assuming final year's NOI does not vary, use the last value from projected_noi
    try: 
        if mean_cap_rate <= 0: 
            raise ValueError("Mean cap rate must be greater than 0 to avoid division by zero.")
        final_year_noi = projected_noi[-1]

        # Simulate NOI for the final year using a dist that reflects potential variability
        # For simplicity we use the mean of the final year's NOI with a fixed std
        std_dev_noi = final_year_noi * 0.05  # Assuming a 5% standard deviation
        simulated_final_noi = np.random.normal(final_year_noi, std_dev_noi, iterations)

        # Calculate reversion values for each scenario and round to nearest whole number
        simulated_reversion_values = np.round(simulated_final_noi / mean_cap_rate, 0)

        # Calculate the mean and standard deviation of the simulated reversion values

        mean_reversion_value = np.round(np.mean(simulated_reversion_values))
        std_reversion_value = np.round(np.std(simulated_reversion_values))

        return simulated_reversion_values, mean_reversion_value, std_reversion_value
    except Exception as e: 
        print(f"An error occurred in simulate_reversion_value: {e}")
        return None, None, None 


def simulate_capex(construction_costs, other_costs, cost_contingency, iterations): 
    """Simulate CAPEX values incorporating variability in construction costs, 
    other costs, and cost contingency.

    Returns: 
        Tuple[np.ndarray, float, float]: A tuple containing the array of simulated 
        CAPEX values, the mean of these CAPEX values, and their standard deviation.

    """
    # Base total costs before contingency
    try: 
        if construction_costs < 0 or other_costs < 0 or cost_contingency < 0: 
            raise ValueError("Construction costs, other costs, and cost contingency must be non-negative.")
        base_total_costs = construction_costs + other_costs

        # Simulating cost contingency percentages from 0 to the maximum specified
        simulated_contingencies = np.random.uniform(0, cost_contingency, iterations)

        # Applying the simulated contingencies to the base total costs
        simulated_capex_values = base_total_costs * (1 + simulated_contingencies)
        # Round values to nearest whole number
        simulated_capex_values = np.round(simulated_capex_values)

        # If costs log-normally distributed, convert to log-space for mean and std calc
        log_simulated_capex = np.log(simulated_capex_values + 1)  # Adding 1 to handle log(0) case
        mean_log_capex = np.mean(log_simulated_capex)
        std_log_capex = np.std(log_simulated_capex)

        # Convert mean and std back from log-space to original scale and 
        # round to the nearest whole number
        mean_capex = np.round(np.exp(mean_log_capex) -1)  # Subtracting 1 to reverse the addition for log(0)
        std_capex = np.round((np.exp(std_log_capex**2) - 1) * \
            np.exp(2*mean_log_capex + std_log_capex**2))

        return simulated_capex_values, mean_capex, std_capex 
    except Exception as e: 
        print(f"An error occurred in simulate_capex: {e}")
        return None, None, None 

def simulate_cash_on_cash_return(cash_flows_means, total_investment, iterations): 
    """Calculate cash_on_cash return estimates for each iteration scenario in a 
    simulation.

    Returns: 
        Tuple[np.ndarray, float, float]: A tuple containing the array of simulated 
        cash-on-cash return values, the mean of these returns, and their standard deviation.
    """
    # Calculate cash-on-cash return for each scenario
    try: 
        if total_investment <= 0: 
            raise ValueError("Total investment must be greater than 0 to calculate cash-on-cash return.")
        simulated_cash_on_cash_returns = np.round(cash_flows_means / total_investment, 3)

        # Calculate the mean and standard deviation of the simulated cash-on-cash return values
        cash_on_cash_return_mean = np.round(np.mean(simulated_cash_on_cash_returns), 3)
        cash_on_cash_return_std = np.round(np.std(simulated_cash_on_cash_returns), 3)

        return simulated_cash_on_cash_returns, cash_on_cash_return_mean, cash_on_cash_return_std 
    except Exception as e: 
        print(f"An error occurred in simulate_cash_on_cash_return: {e}")
        return None, None, None 


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
    try:
        if property_purchase_price <= 0:
            raise ValueError("Property purchase price must be greater than 0.")

        simulated_rent_multipliers = np.zeros(iterations)

        # Calculate rent multipliers for each iteration
        for i in range(iterations):
            if simulated_rental_incomes[i] <= 0:
                raise ValueError(f"Simulated rental income at index {i} must be greater than 0.")
            simulated_rent_multipliers[i] = np.round(property_purchase_price / simulated_rental_incomes[i], 2)
        
        # Calculate mean and std dev of simulated rent multipliers
        rent_multiplier_mean = np.round(np.mean(simulated_rent_multipliers), 2)
        rent_multiplier_std = np.round(np.std(simulated_rent_multipliers), 2)

        return simulated_rent_multipliers, rent_multiplier_mean, rent_multiplier_std
    except ValueError as e:
        print(f"Value error in simulate_rent_multiplier: {e}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error in simulate_rent_multiplier: {e}")
        return None, None, None

# Let's add try-except blocks for the following simulate_be_ratio() function:

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
    #print(f"Pre-tax Income: {simulated_pre_tax_income}, Length: {len(simulated_pre_tax_income)}")
    #print(f"Total Debt: {simulated_total_debt}, Length: {len(simulated_total_debt)}")
    #print(f"Operating Expenses: {simulated_operating_expenses}, Length: {len(simulated_operating_expenses)}")
    """
    Simulates break-even ratio scenarios based on simulated pre-tax income, total debt, 
    and operating expenses. 
    
    Returns:
        np.ndarray: Simulated break-even ratio for each iteration.
        float: Mean of the simulated break-even ratios.
        float: Standard deviation of the simulated break-even ratios.
    """
    try: 
        # Initialize an array to hold the simulated break-even ratios
        simulated_be_ratios = np.zeros(iterations)

        # Iterate over each scenario
        for i in range(iterations): 
            if simulated_pre_tax_income[i] <= 0: 
                raise ValueError(f"Pre-tax income at index {i} must be greater than 0.")
            # Calculate break-even ratio for each scenario
            simulated_be_ratios[i] = (simulated_operating_expenses[i] + simulated_total_debt[i]) / simulated_pre_tax_income[i]

        # Calculate mean and standard deviation of simulated break-even ratios
        mean_be_ratio = np.mean(simulated_be_ratios)
        std_be_ratio = np.std(simulated_be_ratios)

        # Round results to 2 decimal places
        simulated_be_ratios = np.round(simulated_be_ratios, 2)
        mean_be_ratio = round(mean_be_ratio, 2)
        std_be_ratio = round(std_be_ratio, 2)

        return simulated_be_ratios, mean_be_ratio, std_be_ratio
    except ValueError as e: 
        print(f"Value error in simulate_be_ratio: {e}")
        return None, None, None 
    except ZeroDivisionError as e: 
        print(f"Division by zero in simulate_be_ratio: {e}")
        return None, None, None 
    except Exception as e: 
        print(f"Unexpected error in simulate_be_ratio: {e}")
        return None, None, None 


def simulate_inflation_rates(inflation_rate, iterations): 
    """Simulates inflation rate scenarios using the lognormal distribution. Return simulated inflation rates
    along with their mean and standard deviation for each scenario.

    Returns:
    tuple: Contains simulated inflation rates, their mean, and standard deviation
    """
    try: 
        # Log-transform the historical inflation rates
        # np.log1p for log(1+x) to handle zero inflation rates
        log_inflation_rate = np.log1p(inflation_rate)

        # Ensure that the log transformation was successful
        if np.any(np.isnan(log_inflation_rate)): 
            raise ValueError("Log transformation resulted in NaN values.") 

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

    except ValueError as e: 
        print(f"Value error in simulate_inflation_rates: {e}")
        return None, None, None 
    except Exception as e: 
        print(f"Unexpected error in simulate_inflation_rates: {e}")
        return None, None, None 


def simulate_vacancy_rates(vacancy_rates, iterations): 
    """Simulates vacancy rate scenarios using the beta distribution. Return simulated 
    along with their mean and standard deviation for each scenario.

    Return: 
    tuple: Contains simulated vacancy rates, their mean, and standard deviation.
    """
    # Convert rates to a numpy array for operations
    try: 
        vacancy_rates = np.array(vacancy_rates)

        # Ensure vacancy rates are within the valid range for beta distribution
        if np.any(vacancy_rates < 0) or np.any(vacancy_rates > 1): 
            raise ValueError("Vacancy rates must be between 0 and 1 for beta distribution.")

        # Estimate parameters of the Beta distribution from data
        alpha, beta = estimate_beta_params(vacancy_rates)
        if alpha is None or beta is None: 
            raise ValueError("Failed to estimate beta parameters.")

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
    except ValueError as e: 
        print(f"Value error in simulate_vacancy_rates: {e}")
        return None, None, None 
    except Exception as e: 
        print(f"Unexpected error in simulate_vacancy_rates: {e}")


def estimate_beta_params(data): 
    """Estimate Beta distribution parameters from data.

    Returns: 
        tuple: Estimated alpha and beta parameters.
    """
    try: 
        mean_data = data.mean()
        var_data = data.var()
        if var_data == 0: 
            raise ValueError("Variance cannot be zero for beta parameter estimation.")

        # Use method of moments to estimate parameters
        alpha = mean_data * ((mean_data * (1 - mean_data)) / var_data - 1)
        beta = (1 - mean_data) * ((mean_data * (1 - mean_data)) / var_data - 1)

        if alpha <= 0 or beta <= 0: 
            raise ValueError("Estimated alpha and beta must be positive.")

        return alpha, beta 
    except ValueError as e: 
        print(f"Value error in estimate_beta_params: {e}")
        return None, None 
    except Exception as e: 
        print(f"Unexpected error in estimate_beta_params: {e}")
        return None, None 



def simulate_gdp_growth_rates(gdp_growth_rate, iterations): 
    """Simulate GDP growth rate scenarios using a t-distribution. 

    Returns:
        tuple: Contains simulated GDP growth rates, their mean, and standard deviation
    """
    try:
        if len(gdp_growth_rates) == 0:
            raise ValueError("gdp_growth_rates cannot be empty.")

        # Estimate parameters for t-distribution from historical data
        df, loc, scale = t.fit(gdp_growth_rates)

        # Simulate GDP Growth Rates using the t-distribution parameters
        simulated_gdp_growth_rates = t.rvs(df, loc=loc, scale=scale, size=iterations)

        # Calculate mean and standard deviation of the simulated GDP growth rates
        simulated_gdp_growth_mean = np.mean(simulated_gdp_growth_rates)
        simulated_gdp_growth_std = np.std(simulated_gdp_growth_rates)

        # Round results to 4 decimal places for precision
        simulated_gdp_growth_rates = np.round(simulated_gdp_growth_rates, 4)
        simulated_gdp_growth_mean = round(simulated_gdp_growth_mean, 4)
        simulated_gdp_growth_std = round(simulated_gdp_growth_std, 4)

        return simulated_gdp_growth_rates, simulated_gdp_growth_mean, simulated_gdp_growth_std
    except ValueError as e:
        print(f"Value error in simulate_gdp_growth_rates: {e}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error in simulate_gdp_growth_rates: {e}")
        return None, None, None


def run_all_simulations(cf0, cash_flows, historical_discount_rates, discount_rate, 
    property_purchase_price, gross_rental_income, pre_tax_income, total_debt, 
    operating_expenses, iterations): 
    """encapsulate the logic for setting up and running all simulations"""

    simulation_inputs = {}
    dist_calcs = {}

    # Attempt each simulation block individually, catch errors, and continue
    try: 
        # Simulate cash flows series and associated metrics
        cash_flows_series, cash_flows_means, cash_flows_stds = simulate_cash_flows(cf0, 
            cash_flows, iterations)
        simulation_inputs['cash_flows_series'] = {'values': cash_flows_series, 'mean': cash_flows_means, 'std': cash_flows_stds}
    except Exception as e: 
        print(f"Error in simulating cash flows: {e}")

    try: 
        # Estimate and calculate OU parameters, then simulate the discount rates
        theta_est, mu_est, sigma_est, mean_rate_t, std_t = estimate_and_calc_ou(historical_discount_rates, 
            discount_rate, t=1)
        sim_discount_rates = generate_discount_rate_scenarios(theta_est, mu_est, sigma_est, iterations, t=1)
        simulation_inputs['discount rates'] = {'values': sim_discount_rates}
    except Exception as e: 
        print(f"Error in simulating discount rates: {e}")

    try: 
        # Run NPV simulation and calculate profitability index and future value
        mean_npv, std_npv, npvs, simulated_discount_rates, cash_flows_series = run_npv_simulation(cf0, cash_flows, 
            historical_discount_rates, discount_rate, iterations)
        dist_calcs['npv'] = {'values': npvs, 'mean': mean_npv, 'std': std_npv}
    except Exception as e: 
        print(f"Error in simulating npvs: {e}")

    try: 
        mean_pi, std_pi, pi_values = simulate_profitability_index(npvs, cf0)
        dist_calcs['profitability_index'] = {'values': pi_values, 'mean': mean_pi, 'std': std_pi}
    except Exception as e: 
        print(f"Error in simulating profitability index values: {e}")

    try: 
        mean_fv, std_fv, fvs = run_fv_sim(cf0, cash_flows, historical_discount_rates, discount_rate, 
            iterations)
        dist_calcs['fv'] = {'values': fvs, 'mean': mean_fv, 'std': std_fv}
    except Exception as e: 
        print(f"Error in simulating fvs: {e}")

    try: 
        # Simulate market parameters like betas, risk-free rates, and market returns
        betas = simulate_betas(mean_beta, std_beta, iterations) 
        simulation_inputs['betas'] = {'values': betas}
    except Exception as e: 
        print(f"Error in simulating betas: {e}")

    try: 
        risk_free_rates = sim_risk_free_rate_scenarios(historical_risk_free_rate, iterations)
        simulation_inputs['risk_free_rates'] = {'values': risk_free_rates}
    except Exception as e: 
        print(f"Error in simulating risk-free rates: {e}")

    try: 
        market_return_scenarios = sim_market_return_scenario(historical_market_rates, iterations)
        simulation_inputs['market_return_scenarios'] = {'values': market_return_scenarios}
    except Exception as e: 
        print(f"Error in simulating market return scenarios: {e}")

    try: 
        rrr_capm_scenarios, mean_rrr_capm, std_rrr_capm = simulate_rrr_capm(betas, 
            market_return_scenarios, risk_free_rates, iterations)
        dist_calcs['rrr_capm'] = {'values': rrr_capm_scenarios, 'mean': mean_rrr_capm, 'std': std_rrr_capm}
    except Exception as e: 
        print(f"Error in simulating rrr-capm values: {e}")

    try: 
        # Simulate rrr_income values
        rrr_income_scenarios, mean_rrr_income, std_rrr_income = simulate_rrr_income(cash_flows_series, 
            property_value_last_year, time_years, iterations)
        dist_calcs['rrr_income'] = {'values': rrr_income_scenarios, 'mean': mean_rrr_income, 'std': std_rrr_income}
    except Exception as e: 
        print(f"Error in simulating rrr-income values: {e}")

    try: 
        # Simulate monthly debt payments
        monthly_mortgage_payment = calc_mortgage_payments(discount_rate, loan_amount, time_years)
        simulation_inputs['monthly_mortgage_payment'] = {'values': monthly_mortgage_payment}
    except Exception as e: 
        print(f"Error in calculating monthly mortgage payment: {e}")

    try: 
        monthly_mortgage2_payment = calc_mortgage2_payments(discount_rate, loan_amount2, time_years)
        simulation_inputs['monthly_mortgage2_payments'] = {'values': monthly_mortgage2_payment}
    except Exception as e: 
        print(f"Error handing monthly mortgage2 payments: {e}")

    try: 
        # Simulate property growth rate scenarios
        growth_rates, mean_growth_log, std_growth_log = simulate_property_growth_rates(property_value_growth_rate, iterations)
        dist_calcs['simulated_growth_rates'] = {'values': growth_rates, 'mean': mean_growth_log, 'std': std_growth_log}
    except Exception as e: 
        print(f"Error in simulating property growth rates: {e}")

    try: 
        # Simulate levered IRR scenario values
        levered_irrs, mean_levered_irr, std_levered_irr = simulate_levered_irr(cash_flows_series, 
            iterations)
        dist_calcs['levered_irr'] = {'values': levered_irrs, 'mean': mean_levered_irr, 'std': std_levered_irr}
    except Exception as e: 
        print(f"Error simulating levered IRR scenario values: {e}")

    try: 
        # Simulate cash clows excluding debt
        simulated_cash_flows_excluding_debt = simulate_cash_flows_excluding_debt(cash_flows_excluding_debt, 
            iterations)
        simulation_inputs['cash_flows_excluding_debt'] = {'values': simulated_cash_flows_excluding_debt}
    except Exception as e: 
        print(f"Error simulating cash flows excluding debt: {e}")

    try: 
        # Simulate unlevered IRR scenario values 
        unlevered_irrs, mean_unlevered_irr, std_unlevered_irr = simulate_unlevered_irr(total_investment, 
            simulated_cash_flows_excluding_debt, iterations)
        dist_calcs['unlevered_irr'] = {'values': unlevered_irrs, 'mean': mean_unlevered_irr, 'std': std_unlevered_irr}
    except Exception as e: 
        print(f"Error simulating unlevered IRR scenario values: {e}")

    try: 
        # Return noi scenarios
        noi_scenarios, mean_noi, std_noi = simulate_noi_scenarios(projected_noi, iterations)
        simulation_inputs['noi_scenarios'] = {'values': noi_scenarios, 'mean': mean_noi, 'std': std_noi}
    except Exception as e: 
        print(f"Error simulating noi scenario values: {e}")

    try: 
        # Simulate cap rate scenario values 
        simulated_cap_rates, mean_cap_rate, std_cap_rate = simulate_cap_rate(noi_scenarios, 
            total_investment, iterations)
        dist_calcs['cap_rate'] = {'values': simulated_cap_rates, 'mean': mean_cap_rate, 'std': std_cap_rate}
    except Exception as e: 
        print(f"Error calculating cap rate scenario values: {e}")

    try: 
        # Simulate reversion value scenarios 
        simulated_reversion_values, mean_reversion_value, std_reversion_value = simulate_reversion_value(projected_noi, 
            mean_cap_rate, iterations)
        dist_calcs['reversion_value'] = {'values': simulated_reversion_values, 'mean': mean_reversion_value, 'std': std_reversion_value}
    except Exception as e: 
        print(f"Error calculating reversion value scenarios: {e}")

    try: 
        # Simulate capex value scenarios 
        simulated_capex_values, mean_capex, std_capex = simulate_capex(construction_costs, 
            other_costs, cost_contingency, iterations)
        dist_calcs['capex'] = {'values': simulated_capex_values, 'mean': mean_capex, 'std': mean_capex}
    except Exception as e: 
        print(f"Error simulating capex value scenarios: {e}")

    try: 
        # Simulate cash-on-cash return values 
        simulated_cash_on_cash_returns, cash_on_cash_return_mean, cash_on_cash_return_std = simulate_cash_on_cash_return(cash_flows_means, 
            total_investment, iterations)
        dist_calcs['cash-on-cash_return'] = {'values': simulated_cash_on_cash_returns, 'mean': cash_on_cash_return_mean, 'std': cash_on_cash_return_std}
    except Exception as e: 
        print(f"Error simulating cash-on-cash return scenarios: {e}")

    try: 
        # Simulate rental income scenario 
        simulated_rental_incomes, rental_income_mean_values, rental_income_std_values = simulate_rental_income(gross_rental_income, iterations)
        simulation_inputs['simulated_rental_incomes'] = {'values': simulated_rental_incomes, 'mean': rental_income_mean_values, 'std': rental_income_std_values}
    except Exception as e: 
        print(f"Error simulating rental income scenarios: {e}")

    try: 
        # Simulate rent multiplier 
        simulated_rent_multipliers, rent_multiplier_mean, rent_multiplier_std = simulate_rent_multiplier(property_purchase_price, simulated_rental_incomes, iterations)
        dist_calcs['simulated_rent_multipliers'] = {'values': simulated_rent_multipliers, 'mean': rent_multiplier_mean, 'std': rent_multiplier_std}
    except Exception as e: 
        print(f"Error simulating rent multiplier scenario values: {e}")

    try: 
        # Simulate pre-tax income values
        simulated_pre_tax_income, simulated_pre_tax_mean, simulated_pre_tax_std = simulate_pre_tax_income(pre_tax_income, iterations)
        simulation_inputs['simulated_pre_tax_income'] = {'values': simulated_pre_tax_income, 'mean': simulated_pre_tax_mean, 'std': simulated_pre_tax_std}
    except Exception as e: 
        print(f"Error simulating pre-tax income scenario values: {e}")

    try: 
        # Simulate total debt values
        simulated_total_debt, simulated_total_debt_mean, simulated_total_debt_std = simulate_total_debt(total_debt, iterations)
        simulation_inputs['simulated_total_debt'] = {'values': simulated_total_debt, 'mean': simulated_total_debt_mean, 'std': simulated_total_debt_std}
    except Exception as e: 
        print(f"Error simulating total debt values: {e}")

    try: 
        # Simulate operating expense scenarios
        simulated_operating_expenses, simulated_operating_expenses_mean, simulated_operating_expenses_std = simulate_operating_expenses(operating_expenses, iterations)
        simulation_inputs['simulated_operating_expenses'] = {'values': simulated_operating_expenses, 'mean': simulated_operating_expenses_mean, 'std': simulated_operating_expenses_std}
    except Exception as e: 
        print(f"Error simulating operating expense scenarios: {e}")

    try: 
        # Simulate break-even return values
        simulated_be_ratios, be_ratio_mean, be_ratio_std = simulate_be_ratio(simulated_pre_tax_income, simulated_total_debt, simulated_operating_expenses, iterations)
        dist_calcs['break_even_ratio'] = {'values': simulated_be_ratios, 'mean': be_ratio_mean, 'std': be_ratio_std}
    except Exception as e: 
        print(f"Error simulating break-even return values: {e}")

    try: 
        # Simulate all inflation rate scenario values
        simulated_inflation_rates, simulated_inflation_rates_mean, simulated_inflation_rates_std = simulate_inflation_rates(inflation_rate, iterations)
        dist_calcs['simulated_inflation_rates'] = {'values': simulated_inflation_rates, 'mean': simulated_inflation_rates_mean, 'std': simulated_inflation_rates_std}
    except Exception as e: 
        print(f"Error simulating inflation rate scenarios: {e}")

    try: 
        # Simulate vacancy rate scenario values
        simulated_vacancy_rates, simulated_vacancy_rates_mean, simulated_vacancy_rates_std = simulate_vacancy_rates(vacancy_rates, iterations)
        dist_calcs['vacancy_rates'] = {'values': simulated_vacancy_rates, 'mean': simulated_vacancy_rates_mean, 'std': simulated_vacancy_rates_std}
    except Exception as e: 
        print(f"Error simulating vacancy rate scenarios: {e}")

    try: 
        # Simulate GDP growth rate scenario values 
        simulated_gdp_growth_rates, simulated_gdp_growth_mean, simulated_gdp_growth_std = simulate_gdp_growth_rates(gdp_growth_rates, iterations)
        dist_calcs['simulated_gdp_growth_rates'] = {'values': simulated_gdp_growth_rates, 'mean': simulated_gdp_growth_mean, 'std': simulated_gdp_growth_std}
    except Exception as e: 
        print(f"Error simulating GDP growth rate scenarios: {e}")
    

    return simulation_inputs, dist_calcs

simulation_inputs, dist_calcs = run_all_simulations(cf0, cash_flows, historical_discount_rates, discount_rate, 
    property_purchase_price, gross_rental_income, pre_tax_income, total_debt, 
    operating_expenses, iterations)

# print(simulation_inputs)

# print(dist_calcs)
# pprint(dist_calcs)

# Print the 'dist_calcs' dictionary
print("Distribution Calculations:")
for key, value in dist_calcs.items():
    print(f"{key}:")
    for sub_key, sub_value in value.items():
        print(f"  {sub_key}: {sub_value}")

'''
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
'''
