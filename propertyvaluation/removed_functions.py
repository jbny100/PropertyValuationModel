def log_params_cash_flows(cf0, cash_flows):
    """Calculate mean_log and sigma_log for initial cash flow scenario."""
    total_cash_flows = [cf0] + cash_flows
    if len(total_cash_flows) == 0 or sum(total_cash_flows) <= 0:
        raise ValueError("Total cash flows must be positive for lognormal distribution parameters calculation.")

    mean_cash_flow = sum(total_cash_flows) / len(total_cash_flows) 
    if mean_cash_flow <= 0: 
        raise ValueError("Mean cash flow must be positive for log calculation.")

    deviations = [x - mean_cash_flow for x in total_cash_flows]
    squared_deviations = [x**2 for x in deviations]
    mean_squared_deviations = sum(squared_deviations) / len(total_cash_flows)
    standard_deviation = math.sqrt(mean_squared_deviations)

    mean_log = np.log(mean_cash_flow) - 0.5  # adjusted for lognormal dist.
    sigma_log = np.sqrt(np.log(1 + standard_deviation**2))

    return mean_log, sigma_log


def sim_cash_on_cash_returns(cash_flows_series, total_investment, iterations):
    """Return cash-on-cash return scenarios using the mean of each cash_flows scenario. 
    Will use return result in beta calculation."""
    mean_cash_flow_series = np.mean(cash_flows_series)

    cash_on_cash_returns = mean_cash_flows_series / total_investment

    return cash_on_cash_returns

def sim_investment_betas(cash_on_cash_returns, historical_market_rates, iterations):
    """Return simulated betas."""
    mean_returns = np.mean(cash_on_cash_returns)
    std_returns = np.std(cash_on_cash_returns)

    # Create DataFram for regression analysis.
    df = pd.DataFrame({'cash_on_cash_returns': cash_on_cash_returns, 
        'historical_market_rates': historical_market_rates})

     # Specify model
    X = df['historical_market_rates']  # Independent variable (S&P 500 returns)
    y = df['cash_on_cash_returns']  # Dependent variable (Investment returns)
    X = sm.add_constant(X)  # Add constant term for intercept

    # Fit model
    model = sm.OLS(y, X).fit()

    # Extract beta coefficient 
    betas = model.params['historical_market_rates']

    return betas 

cash_on_cash_returns = sim_cash_on_cash_returns(pre_tax_income, total_investment, iterations)
print(cash_on_cash_returns)
betas = sim_investment_betas(cash_on_cash_returns, historical_market_rates, iterations)


def estimate_and_calc_ou(historical_discount_rates, discount_rate, t=1): 
    """Estimate OU parameters and calculate mean & std of Discount Rate.""" 

    # Historical data of discount rates
    discount_rates = np.array(historical_discount_rates)

    # Initial guesses for theta, mu and sigma
    initial_guess = [0.2, np.mean(discount_rates), 0.02] 

    # Perform optimization to estimate parameters
    result = minimize(ou_objective, initial_guess, args=(discount_rates))

    # Extract estimated params  
    theta_est, mu_est, sigma_est = result.x 

    # Use estimated params to calculate base mean and std at time t 
    mean_rate_t, std_t = calc_base_mean_std(discount_rate, theta_est, mu_est, 
        sigma_est, t)

    return theta_est, mu_est, sigma_est, mean_rate_t, std_t 


def calc_debt_payments(discount_rate, loan_amount, n_periods):
    """Calculate montly loan interest payments."""
    

def run_all_simulations(cf0, cash_flows, historical_discount_rates, discount_rate, 
    property_purchase_price, gross_rental_income, pre_tax_income, total_debt, 
    operating_expenses, iterations): 
    """encapsulate the logic for setting up and running all simulations"""

    # Simulate cash flows series and associated metrics
    cash_flows_series, cash_flows_means, cash_flows_stds = simulate_cash_flows(cf0, 
        cash_flows, iterations)

    # Estimate and calculate OU parameters, then simulate the discount rates
    theta_est, mu_est, sigma_est, mean_rate_t, std_t = estimate_and_calc_ou(historical_discount_rates, 
    discount_rate, t=1)
    sim_discount_rates = generate_discount_rate_scenarios(theta_est, mu_est, sigma_est, iterations, t=1)

    # Run NPV simulation and calculate profitability index and future value
    mean_npv, std_npv, npvs, simulated_discount_rates, cash_flows_series = run_npv_simulation(cf0, cash_flows, 
        historical_discount_rates, discount_rate, iterations)

    mean_pi, std_pi, pi_values = simulate_profitability_index(npvs, cf0)

    mean_fv, std_fv, fvs = run_fv_sim(cf0, cash_flows, historical_discount_rates, discount_rate, 
        iterations)


    # Simulate market parameters like betas, risk-free rates, and market returns
    betas = simulate_betas(mean_beta, std_beta, iterations) 

    risk_free_rates = sim_risk_free_rate_scenarios(historical_risk_free_rate, iterations)

    market_return_scenarios = sim_market_return_scenario(historical_market_rates, iterations)

    rrr_capm_scenarios, mean_rrr_capm, std_rrr_capm = simulate_rrr_capm(betas, 
        market_return_scenarios, risk_free_rates, iterations)

    # Simulate rrr_income values
    rrr_income_scenarios, mean_rrr_income, std_rrr_income = simulate_rrr_income(cash_flows_series, 
        property_value_last_year, time_years, iterations)

    # Simulate monthly debt payments
    monthly_mortgage_payment = calc_mortgage_payments(discount_rate, loan_amount, time_years)
    monthly_mortgage_payment2 = 0

    # Simulate property growth rate scenarios
    growth_rates, mean_growth_log, std_growth_log = simulate_property_growth_rates(property_value_growth_rate, iterations)


    # Simulate levered and unlevered IRR scenario values
    levered_irrs, mean_levered_irr, std_levered_irr = simulate_levered_irr(cash_flows_series, 
        iterations)

    simulated_cash_flows_excluding_debt = simulate_cash_flows_excluding_debt(cash_flows_excluding_debt, 
        iterations)

    unlevered_irrs, mean_unlevered_irr, std_unlevered_irr = simulate_unlevered_irr(total_investment, 
        simulated_cash_flows_excluding_debt, iterations)

    # Return noi scenarios
    noi_scenarios, mean_noi, std_noi = simulate_noi_scenarios(projected_noi, iterations)

    # Simulate cap rate scenario values
    simulated_cap_rates, mean_cap_rate, std_cap_rate = simulate_cap_rate(noi_scenarios, 
        total_investment, iterations)

    # Simulate reversion value scenarios 
    simulated_reversion_values, mean_reversion_value, std_reversion_value = simulate_reversion_value(projected_noi, 
        mean_cap_rate, iterations)

    # Simulate capex value scenarios 
    simulated_capex_values, mean_capex, std_capex = simulate_capex(construction_costs, 
        other_costs, cost_contingency, iterations)

    # Simulate cash-on-cash return values 
    simulated_cash_on_cash_returns, cash_on_cash_return_mean, cash_on_cash_return_std = simulate_cash_on_cash_return(cash_flows_means, 
        total_investment, iterations)

    # Simulate rent multiplier return values using simulated rental income scenario
    simulated_rental_incomes, rental_income_mean_values, rental_income_std_values = simulate_rental_income(gross_rental_income, iterations)

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


    simulation_inputs = {}
    simulation_inputs['betas'] = {'values': betas}
    simulation_inputs['risk_free_rates'] = {'values': risk_free_rates}
    simulation_inputs['market_return_scenarios'] = {'values': market_return_scenarios}
    simulation_inputs['monthly_mortgage_payment'] = {'values': monthly_mortgage_payment}
    simulation_inputs['monthly_mortgage_payment2'] = {'values': monthly_mortgage_payment2}
    simulation_inputs['cash_flows_series'] = {'values': cash_flows_series, 'mean': cash_flows_means, 'std': cash_flows_stds}
    simulation_inputs['cash_flows_excluding_debt'] = {'values': simulated_cash_flows_excluding_debt}
    simulation_inputs['noi_scenarios'] = {'values': noi_scenarios, 'mean': mean_noi, 'std': std_noi}
    simulation_inputs['simulated_rental_incomes'] = {'values': simulated_rental_incomes, 'mean': rental_income_mean_values, 'std': rental_income_std_values}
    simulation_inputs['simulated_pre_tax_income'] = {'values': simulated_pre_tax_income, 'mean': simulated_pre_tax_mean, 'std': simulated_pre_tax_std}
    simulation_inputs['simulated_total_debt'] = {'values': simulated_total_debt, 'mean': simulated_total_debt_mean, 'std': simulated_total_debt_std}
    simulation_inputs['simulated_operating_expenses'] = {'values': simulated_operating_expenses, 'mean': simulated_operating_expenses_mean, 'std': simulated_operating_expenses_std}


    dist_calcs = {} 
    dist_calcs['npv'] = {'values': npvs, 'mean': mean_npv, 'std': std_npv}
    dist_calcs['profitability_index'] = {'values': pi_values, 'mean': mean_pi, 'std': std_pi}
    dist_calcs['fv'] = {'values': fvs, 'mean': mean_fv, 'std': std_fv}
    dist_calcs['rrr_capm'] = {'values': rrr_capm_scenarios, 'mean': mean_rrr_capm, 'std': std_rrr_capm}
    dist_calcs['rrr_income'] = {'values': rrr_income_scenarios, 'mean': mean_rrr_income, 'std': std_rrr_income}
    dist_calcs['levered_irr'] = {'values': levered_irrs, 'mean': mean_levered_irr, 'std': std_levered_irr}
    dist_calcs['unlevered_irr'] = {'values': unlevered_irrs, 'mean': mean_unlevered_irr, 'std': std_unlevered_irr}
    dist_calcs['cap_rate'] = {'values': simulated_cap_rates, 'mean': mean_cap_rate, 'std': std_cap_rate}
    dist_calcs['reversion_value'] = {'values': simulated_reversion_values, 'mean': mean_reversion_value, 'std': std_reversion_value}
    dist_calcs['capex'] = {'values': simulated_capex_values, 'mean': mean_capex, 'std': mean_capex}
    dist_calcs['cash-on-cash_return'] = {'values': simulated_cash_on_cash_returns, 'mean': cash_on_cash_return_mean, 'std': cash_on_cash_return_std}
    dist_calcs['simulated_rent_multipliers'] = {'values': simulated_rent_multipliers, 'mean': rent_multiplier_mean, 'std': rent_multiplier_std}
    dist_calcs['break_even_ratio'] = {'values': simulated_be_ratios, 'mean': be_ratio_mean, 'std': be_ratio_std}
    dist_calcs['simulated_inflation_rates'] = {'values': simulated_inflation_rates, 'mean': simulated_inflation_rates_mean, 'std': simulated_inflation_rates_std}
    dist_calcs['vacancy_rates'] = {'values': simulated_vacancy_rates, 'mean': simulated_vacancy_rates_mean, 'std': simulated_vacancy_rates_std}
    dist_calcs['simulated_gdp_growth_rates'] = {'values': simulated_gdp_growth_rates, 'mean': simulated_gdp_growth_mean, 'std': simulated_gdp_growth_std}
    

    return simulation_inputs, dist_calcs







