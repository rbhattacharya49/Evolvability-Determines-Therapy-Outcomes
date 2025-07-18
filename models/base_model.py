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
        Retrieve the model parameters as a dictionary.
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
        
        
        
if __name__ == "__main__":
    """
    Test the BaseEvoModel class with default and updated parameters.
    """
    # Test BaseEvoModel
    model = BaseEvoModel()
    print(model.get_params())

    model.set_params(pop_size=200, growth_rate=0.4)
    print(model.get_params())