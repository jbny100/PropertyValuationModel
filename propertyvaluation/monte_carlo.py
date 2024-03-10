
# monte_carlo.py

from make_simulation_params import simulation_params 
import numpy as np 
import numpy_financial as npf
from scipy.stats import norm, lognorm, beta
from scipy.optimize import newton 
from run_simulations import dist_calcs
from run_simulations import simulation_inputs
import run_simulations


class RealEstateProperty:
    
    def __init__(self, name, initial_investment, dist_calcs=dist_calcs): 
    	self.name = name
    	self.initial_investment = initial_investment
    	self.sim_params = sim_params  # Dict to hold simulation parameters.


    def simulate_metric(self, metric_name): 

    	params = self.sim_params.get(metric_name, {})



    def run_simulation(self, iterations=10000): 
        pass



    def analyze_data(self, results): 
        pass




    # Additional methods for calculating metrics like IRR, MIRR based on simulated cash flows
    # would be implemented here, potentially using scipy.optimize.newton for root-finding.



class ValueAddProperty(RealEstateProperty):
    # Unique attributes and methods for value-add
    pass 

class MixedUse(RealEstateProperty):
    # Unique attributes and methods for mixed-use
    pass

class DevelopmentSite(RealEstateProperty):
    # Unique attributes and methods for development sites
    pass

class MarketRateIncomeGenerating(RealEstateProperty):
    # Unique attributes and methods for market-rate properties
    pass

class AffordableHousingIncomeGenerating(RealEstateProperty):
    # Unique attributes and methods for affordable housing
    pass

class MultiFamily(RealEstateProperty):
    # Unique attributes and methods for multi-family
    pass

class OfficeBuilding(RealEstateProperty):
    # Unique attributes and methods for office buildings
    pass

class WarehouseIndustrial(RealEstateProperty):
    # Unique attributes and methods for warehouse/industrial
    pass 

class Retail(RealEstateProperty): 
	# Unique attributes and methods for retail
	pass






