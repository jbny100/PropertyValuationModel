## Monte Carlo Simuluation for Real Estate Assets

This program conducts a Monte Carlo simulation to analyze various types of real estate investments. It calculates key financial metrics across 10,000 scenarios to determine the viability and risk of real estate investments under varying economic conditions.

## Features 

Property Type Specific Simulation - Tailors simulations and analyses to different types of properties such as Value-Add, Mixed-Use, Development Site, Income-Generating, Affordable Housing, Multifamily, Office Building, Warehouse/Industrial, and Retail.

Distribution-Based Inputs - Utilizes various probability distributions (Ornstein-Uhlenbeck, Normal, Lognormal, T-Distribution, Uniform, Beta) to model uncertainties in market conditions and property-specific factors.

Comprehensive Financial Metrics - Calculates Net Present Value (NPV), Profitability Index, Future Value (FV), Required Rate of Return (RRR) using both CAPM and Income approaches, Internal Rate of Return (levered and unlevered), Cap Rate, Reversion Value, CAPEX, Cash-on-Cash Return, Rent Multiplier, Break-even Ratio, Vacancy Rate, and more.

Dynamic Thershold Analysis - Applies customizable thresholds for triggering alerts based on the simulation outcomes, enhancing decision-making accuracy.

## Structure

Modules: 

monte_carlo.py - Main module which defines the property classes and runs the simulations.

run_simulations.py - Contains functions to execute the simulation logic for all financial metrics and scenarios, addes results to dictionaries and imports those dictionaries into mont_carlo.py

make_simulation_params.py - Sets up initial parameters and calculates base values required for the simulations.

Classes: 

RealEstateProperty - Base class for all property types.

ValueAddProperty, MixedUse, DevelopmentSite, etc. - Derived classes that represent specific property types with tailored simulation logic.

## Installation

Clone the repository - git clone [repository-url] 
Navigate to the directory - cd [repository-directory]
Install required Python package - pip install -r requirements.txt

## Contributing

Contributions to the SEO Report Generator are welcome. Please fork the repository and submit a pull request with your enhancements.

