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