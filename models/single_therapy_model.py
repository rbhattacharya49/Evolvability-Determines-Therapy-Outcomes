import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.base_model import BaseEvoModel

class SingleTherapyModel(BaseEvoModel):
    def __init__(self, init, t_max=3000,
                 efficacy=0.15, efficacy_std=np.sqrt(6), evolvability_scaling_factor=1.67,
                 cycle_duration=50, intermittent=True, facultative_evolvability=True,
                 **kwargs):
        """
        Initialize the SingleTherapyModel with parameters for therapy dynamics and evolvability.
        """
        super().__init__(**kwargs)

        self.init = init
        self.t_max = t_max
        self.cycle_duration = cycle_duration
        self.intermittent = intermittent
        self.facultative_evolvability = facultative_evolvability

        self.efficacy = efficacy
        self.efficacy_std = efficacy_std
        self.evolvability_scaling_factor = evolvability_scaling_factor

    def evolvability(self, v, s, mu, sigma_t, k_r):
        """
        Calculate the evolvability based on therapy parameters and strategy.
        """
        if not self.facultative_evolvability:
            return self.evolvability_value
        return self.evolvability_value + k_r * s * np.exp(-((v - mu)**2) / sigma_t**2)
    
    def get_evolvability_scaling_factor(self, HIGH_EVO):
        if self.facultative_evolvability:
            self.evolvability_scaling_factor = (HIGH_EVO - self.evolvability_value) / self.efficacy

    def model(self, t, y):
        """
        Define the system of differential equations for population size (x) and strategy (v).
        """
        x, v = y
        
        #k_r functions as scaling factor or baseline evolvabiliy or is simply 0 
        if t < self.therapy_start_time:
            s = 0
            sigma_t = self.efficacy_std
            k_r = 0
        elif self.intermittent:
            if ((t - self.therapy_start_time) // self.cycle_duration) % 2 == 0:  #logic for cycling therapy
                s = self.efficacy
                sigma_t = self.efficacy_std
                k_r = self.evolvability_scaling_factor
            else:
                s = 0
                sigma_t = self.efficacy_std
                k_r = 0
        else:
            s = self.efficacy
            sigma_t = self.efficacy_std
            k_r = self.evolvability_scaling_factor

        mu = self.therapy_max_eff
        m = self.evolvability_value

        def G(v, x):
            """
            Growth function incorporating therapy effects and evolvability costs.
            """
            sigma_k = self.carrying_capacity_std
            K_m = self.carrying_capacity
            r = self.growth_rate
            d = self.evolvability_cost

            term1 = (r / (K_m * np.exp(-v**2 / sigma_k**2))) * (K_m * np.exp(-v**2 / sigma_k**2) - x)
            term2 = s * np.exp(-(v - mu)**2 / sigma_t**2)
            term3 = d * self.evolvability(v, s, mu, sigma_t,k_r)
            
            return term1 - term2 - term3

        def dG(v, x):
            """
            Derivative of the growth function with respect to strategy (v).
            """
            sigma_k = self.carrying_capacity_std
            K_m = self.carrying_capacity
            r = self.growth_rate
            d = self.evolvability_cost

            d_term1 = (2 * d * k_r * s * (v - mu) * np.exp(-((v - mu) ** 2) / sigma_t ** 2)) / sigma_t ** 2
            d_term2 = (2 * s * (v - mu) * np.exp(-((v - mu) ** 2) / sigma_t ** 2)) / sigma_t ** 2
            d_term3 = (2 * r * v * x * np.exp(v ** 2 / sigma_k ** 2)) / (K_m * sigma_k ** 2)
            return d_term1 + d_term2 - d_term3

        g = G(v, x)
        dg = dG(v, x)
        evol = self.evolvability(v, s, mu, sigma_t, k_r)

        dx = x * g
        dv = evol * dg
        return [dx, dv]

    def run(self):
        """
        Run the simulation using the defined model and store results.
        """
        t_eval = np.arange(0, self.t_max + 1)
        sol = solve_ivp(self.model, [0, self.t_max], self.init, t_eval=t_eval)

        if sol.success:
            self.t = sol.t
            self.y = sol.y

            # Store evolvability values during simulation
            self.evo = []
            for t, vi in zip(self.t, self.y[1, :]):
                if t < self.therapy_start_time:
                    s = 0
                    sigma_t = self.efficacy_std
                    k_r = 0
                elif self.intermittent:
                    if ((t - self.therapy_start_time) // self.cycle_duration) % 2 == 0:
                        s = self.efficacy
                        sigma_t = self.efficacy_std
                        k_r = self.evolvability_scaling_factor
                    else:
                        s = 0
                        sigma_t = self.efficacy_std
                        k_r = 0
                else:
                    s = self.efficacy
                    sigma_t = self.efficacy_std
                    k_r = self.evolvability_scaling_factor

                mu = self.therapy_max_eff
                self.evo.append(self.evolvability(vi, s, mu, sigma_t, k_r))

# #######################################################
# TEST CODE    
# #######################################################            

if __name__ == "__main__":
    """
    Test the SingleTherapyModel with example parameters and plot results.
    """
    init = [20, 0.01]
    model = SingleTherapyModel(
    init=init,
    t_max=6000,
    efficacy=0.15,
    efficacy_std=np.sqrt(6),
    evolvability_scaling_factor=1.67,
    cycle_duration=100,
    therapy_max_eff=0,
    therapy_start_time=600,
    carrying_capacity_std=10,
    facultative_evolvability=True,
    evolvability_value = 0.1
)
    model.run()

    v = model.y[1, :]
    x = model.y[0, :]
    evo = model.evo

    plt.figure()
    plt.plot(model.t, x, 'b', linewidth=3)
    plt.xlabel("Time")
    plt.ylabel("Population Size: x")
    plt.title("Intermittent Chemotherapy: Population")
    plt.grid(True)

    plt.figure()
    plt.plot(model.t, v, 'r', linewidth=3)
    plt.xlabel("Time")
    plt.ylabel("Strategy v")
    plt.title("Intermittent Chemotherapy: Strategy")
    plt.grid(True)

    plt.figure()
    plt.plot(model.t, evo, 'g', linewidth=3, label='Facultative Evolvability')
    plt.xlabel("Time")
    plt.ylabel("Evolvability $\sigma_g^2$")
    plt.title("Intermittent Chemotherapy: Evolvability")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

