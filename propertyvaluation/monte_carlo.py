
# monte_carlo.py

import numpy as np 
import numpy_financial as npf
from scipy.stats import norm, lognorm, beta
from scipy.optimize import newton 
from run_simulations import thresholds
from run_simulations import dist_calcs
from run_simulations import simulation_inputs
import run_simulations
from pprint import pprint


class RealEstateProperty:
    
    def __init__(self, thresholds=thresholds, dist_calcs=dist_calcs): 
        self.dist_calcs = dist_calcs  # Dict to hold simulation parameters.
        self.thresholds = thresholds  # Dict holding custom thresholds for analysis.

    def run_simulation(self): 
        # Extract the mean at std values from dist_calcs
        results = {metric: {"mean": details["mean"], "std": details["std"]} for metric, 
        details in self.dist_calcs.items()}

        print("\nSimulation Results:")
        for key, value in results.items(): 
            print(f"\n{key:}")
            for sub_key, sub_value in value.items(): 
                print(f"    {sub_key}: {sub_value}")


        return results 


    def analyze_data(self):
        analysis_results = {
            'risk_vs_return': self._analyze_risk_vs_return(),
            'leverage_impact': self._analyze_leverage_impact(),
            'cap_rate_alert': self._analyze_cap_rate(),
            'cash_on_cash_alert': self._analyze_cash_on_cash_return(),
            'growth_vs_stability': self._analyze_growth_vs_stability(),
            'expense_ratios': self._analyze_expense_ratios(),
            'inflation_and_market_dynamics': self._analyze_inflation_and_market_dynamics(),
        }
        # Filter out None results from analyses that don't trigger any alert or note.
        analysis_results = {k: v for k, v in analysis_results.items() if v is not None}
        print()
        return analysis_results

    def _analyze_risk_vs_return(self):
        if (self.dist_calcs['rrr_capm']['mean'] > self.thresholds['threshold_high_rrr'] or \
            self.dist_calcs['rrr_income']['mean'] > self.thresholds['threshold_high_rrr']) and \
            (self.dist_calcs['npv']['mean'] > self.thresholds['threshold_npv_good'] and \
            self.dist_calcs['profitability_index']['mean'] > self.thresholds['threshold_profitability_index_good']):
            return "High return accompanies high risk, requiring careful risk assessment."

    def _analyze_leverage_impact(self):
        leverage_difference = self.dist_calcs['levered_irr']['mean'] - self.dist_calcs['unlevered_irr']['mean']
        if leverage_difference > self.thresholds['threshold_leverage_impact']:
            return f"Leverage increases IRR by {leverage_difference}, indicating significant" 
            " impact but higher risk."

    def _analyze_cap_rate(self):
        if self.dist_calcs['cap_rate']['mean'] < self.thresholds['threshold_cap_rate_low']:
            return "Cap rate below average for investment type, indicating lower return on investment."

    def _analyze_cash_on_cash_return(self):
        if self.dist_calcs['cash-on-cash_return']['mean'] < self.thresholds['target_cash_on_cash_return']:
            return "Cash-on-cash return below target, indicating lower liquidity."

    def _analyze_growth_vs_stability(self):
        # Combine the growth vs. stability analysis into a single step.
        analysis = ""
        if self.dist_calcs['simulated_growth_rates']['mean'] < self.thresholds['threshold_growth_low']:
            analysis += "Properties in stable areas suggest lower growth but more immediate returns. "
        if self.dist_calcs['simulated_growth_rates']['mean'] > self.thresholds['threshold_growth_high']:
            analysis += "Properties in high-growth areas might show lower initial cap rates but promise higher future values."
        return analysis if analysis else None

    def _analyze_expense_ratios(self):
        if self.dist_calcs['break_even_ratio']['mean'] > self.thresholds['threshold_break_even_high'] or \
           self.dist_calcs['vacancy_rates']['mean'] > self.thresholds['threshold_vacancy_high']:
            return "High break-even or vacancy rates suggest operational inefficiencies or" 
            " market misalignments."

    def _analyze_inflation_and_market_dynamics(self):
        if self.dist_calcs['simulated_inflation_rates']['mean'] > self.thresholds['threshold_inflation_high'] and \
           self.dist_calcs['simulated_gdp_growth_rates']['mean'] < self.thresholds['threshold_gdp_growth_low']:
            return "High inflation with low GDP growth may erode real returns and indicate" 
            " weaker market conditions."




    # Additional methods for calculating metrics like IRR, MIRR based on simulated cash flows
    # would be implemented here, potentially using scipy.optimize.newton for root-finding.



class ValueAddProperty(RealEstateProperty):

    def _analyze_growth_vs_stability(self):
        """Value-add properties may emphasize renovation potential and increased 
        operational efficiency."""
        if self.dist_calcs['simulated_growth_rates']['mean'] > self.thresholds['threshold_growth_high']:
            return "Renovation and repositioning can significantly enhance this property's value in high-growth areas."
        return super()._analyze_growth_vs_stability()

    def print_asset_desc(self):
        print("\nProperties with leases that include CPI (Consumer Price Index) escalations" 
            " or those in markets with strong rent growth potential (e.g., Value-Add, Market-Rate" 
            " Income-Generating) may be positively correlated with inflation\n")


class MixedUse(RealEstateProperty):
    # Unique attributes and methods for mixed-use
    pass

class DevelopmentSite(RealEstateProperty):
    # Unique attributes and methods for development sites
    
    def print_asset_desc(self):
        print("""
            Highly leveraged properties (e.g., Development Sites requiring construction 
            financing) or those considering refinancing during the investment horizon may 
            be more sensitive to finance rate fluctuations.
            """)

class MarketRateIncomeGenerating(RealEstateProperty):
    # Unique attributes and methods for market-rate properties
    
    def print_asset_desc(self):
        print(dedent("""
            Properties with leases that include CPI (Consumer Price Index) escalations or 
            those in markets with strong rent growth potential (e.g., Value-Add, Market-Rate 
            Income-Generating) may be positively correlated with inflation.
            """))

class AffordableHousingIncomeGenerating(RealEstateProperty):
    def _analyze_cash_on_cash_return(self):
        # Affordable housing may have different expectations for cash-on-cash return
        if self.dist_calcs['cash-on-cash_return']['mean'] < self.thresholds['target_cash_on_cash_return']:
            return "Despite lower liquidity, the social impact and stable, government-backed income streams offer unique value."
        return super()._analyze_cash_on_cash_return()

    def print_asset_desc(self):
        print("\nFixed-income properties (e.g., some Affordable Housing) may have limited" 
         " ability to adjust rents in response to inflation.\n")


class MultiFamily(RealEstateProperty):
    # Unique attributes and methods for multi-family

    def print_asset_desc(self):
        print("\nGenerally, properties with shorter lease terms (e.g., Multifamily, Retail)"
        " are more sensitive to vacancy rate fluctuations than those with longer lease terms.\n")
        
class OfficeBuilding(RealEstateProperty):
    # Unique attributes and methods for office buildings
    
    def print_asset_desc(self):
        print("\nGenerally, properties with longer lease terms (e.g., office buildings)"
        " are less sensitive to vacancy rate fluctuations than those with shorter lease terms.\n")

class WarehouseIndustrial(RealEstateProperty):
    # Unique attributes and methods for warehouse/industrial
    
    def print_asset_desc(self):
        print("Income Stability - Industrial leases are often longer-term, providing stable" 
            " income streams but may include escalations tied to inflation or specific indices.\n\n" 

            "Key Factors of Warehouse/Industrial Space: \n\n"

            " The rise of e-commerce has significantly increased the demand for warehouse and" 
            " distribution centers, impacting the growth potential and valuation of these properties.\n\n" 
            
            " The ability to accommodate a variety of tenants, including light manufacturing," 
            " distribution, and data centers, affects the property's marketability.\n\n" 

            " Industrial properties may face specific regulatory challenges, including environmental" 
            " regulations and zoning laws that can impact usage and expansion opportunities.\n\n" 

            " may have different maintenance and operational cost structures compared to residential" 
            " or office properties, influenced by the industrial activities they accommodate.\n")

class Retail(RealEstateProperty): 
	# Unique attributes and methods for retail
	
    def print_asset_desc(self):
        print("\nGenerally, properties with shorter lease terms (e.g., Multifamily, Retail)"
        " are more sensitive to vacancy rate fluctuations than those with longer lease terms.\n")


# sim_results = ValueAddProperty()
# sim_results = OfficeBuilding()
# sim_results = AffordableHousingIncomeGenerating()
# sim_results = MarketRateIncomeGenerating()
sim_results = DevelopmentSite()
#sim_results = WarehouseIndustrial()

property_desc = sim_results.print_asset_desc()

simulation_results = sim_results.run_simulation()

analysis_results = sim_results.analyze_data()

pprint(analysis_results)

















