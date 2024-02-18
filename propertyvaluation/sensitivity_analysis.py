from formulas import NetPresentValue, FVCashFlows, IRR

class SensitivityAnalysis:
    def __init__(self, npv: NetPresentValue, fv: FVCashFlows, irr: IRR, 
        original_params: dict, scenario_1_mods: dict, scenario_2_mods: dict): 
        self.npv = npv
        self.fv = fv
        self.irr = irr
        self.original_params = original_params
        self.scenario_1_mods = scenario_1_mods
        self.scenario_2_mods = scenario_2_mods

    def __repr__(self): 
        return (f"SensitivityAnalysis(npv={self.npv}, fv={self.fv}, "
                f"irr={self.irr})")

    def __eq__(self, other): 
        if not isinstance(other, SensitivityAnalysis): 
            return NotImplemented 
        return (self.npv == other.npv and self.fv == other.fv and
                self.irr == other.irr)


    def run_sensitivity_analysis(self) -> dict:
        # Calculate results for the original scenario
        original_npv = NetPresentValue(**self.original_params) 
        original_fv = FVCashFlows(**self.original_params)

        # Extract only the expected arguments from IRR
        irr_params = {key: self.original_params[key] for key in ['rate', 'CF0', 'cash_flows']}
        original_irr = IRR(**irr_params) 

        
        original_results = {
            "NPV": original_npv.calculate_npv(),
            "FV": original_fv.calculate_fv(),
            "IRR": original_irr.calculate_irr(),   
        }

        # Apply modifications for scenario 1
        scenario_1_params = self.apply_modifications(self.original_params, 
            self.scenario_1_mods)
        npv_scenario_1 = NetPresentValue(**scenario_1_params)
        fv_scenario_1 = FVCashFlows(**scenario_1_params)
        # Adjust the IRR for scenario 1
        scenario_1_irr_params = {key: scenario_1_params[key] for key in ['rate', 'CF0', 'cash_flows']}
        irr_scenario_1 = IRR(**scenario_1_irr_params)

        
        scenario_1_results = {
            "NPV": npv_scenario_1.calculate_npv(),
            "FV": fv_scenario_1.calculate_fv(),
            "IRR": irr_scenario_1.calculate_irr(), 
        }

        # Apply modifications for Scenario 2
        scenario_2_params = self.apply_modifications(self.original_params, 
            self.scenario_2_mods)
        npv_scenario_2 = NetPresentValue(**scenario_2_params)
        fv_scenario_2 = FVCashFlows(**scenario_2_params)
        # Adjust the IRR for scenario 2
        scenario_2_irr_params = {key: scenario_2_params[key] for key in ['rate', 'CF0', 'cash_flows']}
        irr_scenario_2 = IRR(**scenario_2_irr_params)

        
        scenario_2_results = {
            "NPV": npv_scenario_2.calculate_npv(),
            "FV": fv_scenario_2.calculate_fv(),
            "IRR": irr_scenario_2.calculate_irr(),
        }

        # Combine all results into a dictionary 
        results = { 
            "original": original_results, 
            "scenario_1": scenario_1_results, 
            "scenario_2": scenario_2_results, 
        }

        return results 

    def apply_modifications(self, original_params: dict, modifications: dict) -> dict: 
        """Switches out original_params for scenario_params so we can calculate 
        npv, fv and irr under different scenarios"""
        modified_params = original_params.copy()
        for key, value in modifications.items(): 
            modified_params[key] = value 
        return modified_params 


# Example variables
rate = 0.05  # Example discount rate.
num_payments = 5  # Default number of payment periods.
CF0 = -1200000  # Cash outflow
cash_flows = [4000, 4000, 4000, 4000, 1800000]  # Starting cash flows

original_params = {
    'rate': rate, 
    'num_payments': num_payments, 
    'CF0': CF0, 
    'cash_flows': cash_flows, 

}

scenario_1_mods = {
    'rate': rate + 0.01,
    'CF0': CF0 * 1.2, 
    'cash_flows': [cf * 0.9 for cf in cash_flows],
}

scenario_2_mods = {
    'rate': rate - 0.01, 
    'CF0': CF0 * 0.9, 
    'cash_flows': [cf * 1.1 for cf in cash_flows], 
}

# Instantiate NetPresentValue, FVCashFlows, and IRR using original_params

npv_instance = NetPresentValue(rate=original_params['rate'], 
    num_payments=original_params['num_payments'], CF0=original_params['CF0'], 
    cash_flows=original_params['cash_flows'])

fv_instance = FVCashFlows(rate=original_params['rate'], 
    num_payments=original_params['num_payments'], CF0=original_params['CF0'], 
    cash_flows=original_params['cash_flows'])

irr_instance = IRR(rate=original_params['rate'], CF0=original_params['CF0'], 
    cash_flows=original_params['cash_flows'])


# Create a SensitivityAnalysis instance with the created objects and scenario dictionaries

saInstance = SensitivityAnalysis(npv=npv_instance, fv=fv_instance, irr=irr_instance,
    original_params=original_params, scenario_1_mods=scenario_1_mods, 
    scenario_2_mods=scenario_2_mods)

# Run the sensitivity analysis and get the results
results = saInstance.run_sensitivity_analysis()

from pprint import pprint 
pprint(results)









