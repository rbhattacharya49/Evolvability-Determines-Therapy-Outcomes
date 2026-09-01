import numpy as np

class BaseEvoModel:
    def __init__(self, growth_rate=0.35, carrying_capacity=100, carrying_capacity_std=50, evolvability_value=0.1, evolvability_cost=0.1, 
                 therapy_max_eff=0, therapy_start_time=600, cycle_duration=100):
        """
        Base class for evolutionary models, defining common parameters and methods.
        """
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.carrying_capacity_std = carrying_capacity_std
        self.evolvability_cost = evolvability_cost
        self.evolvability_value = evolvability_value
        self.therapy_max_eff = therapy_max_eff
        self.therapy_start_time = therapy_start_time
        self.cycle_duration = cycle_duration

    def get_params(self):
        """
        Retrieve the model parameters as a dictionary
        """
        return {
            'growth_rate': self.growth_rate,
            'carrying_capacity': self.carrying_capacity,
            'carrying_capacity_std': self.carrying_capacity_std,
            'evolvability_value': self.evolvability_value,
            'evolvability_cost': self.evolvability_cost,
            'therapy_max_eff': self.therapy_max_eff,
            'therapy_start_time': self.therapy_start_time,
            'cycle_duration': self.cycle_duration
        }
    
    def set_params(self, **kwargs):
        """
        Update model parameters dynamically.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} is not valid for this model.")
            
    def apply_extinction(self, threshold=1.0):
        """
        Apply extinction logic to the population if it falls below a threshold.
        """
        y_cut = self.y.copy()
        extinct = y_cut[0, :] < threshold  # Check population values only

        if np.any(extinct):
            first_extinct_idx = np.argmax(extinct)
            y_cut[0, first_extinct_idx:] = 0  # Zero both rows from that point on

        self.y = y_cut
        
    #This just has some additional functions to get eqb poimts 
    def get_special_values(self):
        """
        Extracts equilibrium, minimum, and maximum values from model output `y`.
        
        """
        
        output = self.y.copy()

        
        result = {
            'Equilibrium Population Size': output[0, -1],
            'Equilibrium Strategy 1 Size': output[1, -1],
            'Minimum Population Size': np.min(output[0]),
            'Minimum Strategy 1 Size': np.min(output[1]),
            'Maximum Population Size': np.max(output[0]),
            'Maximum Strategy 1 Size': np.max(output[1])

            
            
        }

        if output.shape[0] == 3:
            result.update({
                'Equilibrium Strategy 2 Size': output[2, -1],
                'Minimum Strategy 1 Size': np.min(output(y[2])),
                'Maximum Strateg2 2 Size': np.max(output(y[2]))
            })
                
                
        if self.facultative_evolvability:
            evolvability_1 = self.evo_1.copy()
            evolvability_2 = self.evo_2.copy()
            result.update({
                'Evolvability 1 Equilibrium': evolvability_1[-1],
                'Minimum Evolvability 1': np.min(evolvability_1),
                'Minimum Evolvability 1': np.max(evolvability_1),
                
                
                'Evolvability 2 Equilibrium': evolvability_2[-1],
                'Minimum Evolvability 2': np.min(evolvability_2),
                'Minimum Evolvability 2': np.max(evolvability_2)
            })


        return result
    
    
    def get_specific_values(self, x, y=None):
        """
        Extracts specific values or range of values from model output `y`
        
        if x is an integer, and y is undefined then return population, strategy, and evolvability values at index n
        
        if x and y are defined then then arrays containing population, strategy, 
        evolvability values from index x to index y will be returned. 
        
        if x is defined and y is 'end' then all values from x till the end will be returned.
        
        """
        output = self.y.copy()
        
        if self.facultative:
            evolvability_1 = self.evo_1.copy()
            evolvability_2 = self.evo_2.copy()
                
        result = {}

        # Determine y range
        if y is None:
            result['Population'] = output[0, x]
            result['Strategy 1'] = output[1, x]
            if output.shape[0] == 3:
                result['Strategy 2'] = output[2, x]
            if self.facultative_evolvability:
                result['Evolvability 1'] = evolvability_1[x]
                result['Evolvability 2'] = evolvability_2[x]
        else:
            if y == 'end':
                y = output.shape[1]
            result['Population'] = output[0, x:y]
            result['Strategy 1'] = output[1, x:y]
            if output.shape[0] == 3:
                result['Strategy 2'] = output[2, x:y]
            if self.facultative_evolvability:
                result['Evolvability 1'] = evolvability_1[x:y]
                result['Evolvability 2'] = evolvability_2[x:y]

        return result
        
        
if __name__ == "__main__":
    """
    Test the BaseEvoModel class with default and updated parameters.
    """
    # Test BaseEvoModel
    model = BaseEvoModel()
    print(model.get_params())

    model.set_params(pop_size=200, growth_rate=0.4)
    print(model.get_params())