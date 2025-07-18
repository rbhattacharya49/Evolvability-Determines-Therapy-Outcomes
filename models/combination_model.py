import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.base_model import BaseEvoModel

class CombinationModel(BaseEvoModel):
    def __init__(self, init, t_max=2500,
                 efficacy_1=0.15, efficacy_2=0.3,
                 efficacy_std_1=np.sqrt(6), efficacy_std_2=np.sqrt(2),
                 evolvability_value=0.3, evolvability_cost=0.12,
                 evolvability_scaling_factor_1=1.67, evolvability_scaling_factor_2=0.84,
                 facultative_evolvability=False, covariance=0.005,
                 **kwargs):
        """
        Initialize the DoubleBindModel with parameters for dual therapy dynamics.
        """
        super().__init__(evolvability_value=evolvability_value,
                         evolvability_cost=evolvability_cost, **kwargs)

        self.init = init
        self.t_max = t_max
        self.facultative_evolvability = facultative_evolvability

        self.efficacy_1 = efficacy_1
        self.efficacy_2 = efficacy_2
        self.efficacy_std_1 = efficacy_std_1
        self.efficacy_std_2 = efficacy_std_2
        self.evolvability_scaling_factor_1 = evolvability_scaling_factor_1
        self.evolvability_scaling_factor_2 = evolvability_scaling_factor_2
        self.covariance = covariance
    
        
    def get_evolvability_scaling_factor(self, HIGH_EVO):
        self.evolvability_scaling_factor_1 = (HIGH_EVO - self.evolvability_value) / (self.efficacy_1)
        self.evolvability_scaling_factor_2 = (HIGH_EVO - self.evolvability_value) / (self.efficacy_2)

    def model(self, t, y):
        """
        Define the system of differential equations for population size (x) and strategies (v1, v2).
        """
        x, v1, v2 = y

        covar = self.covariance
        if v1 < 0.006: v1 = 0.006
        if v2 < 0.006: v2 = 0.006

        # Therapy switching logic
        if t < self.therapy_start_time:
            s1, s2 = 0, 0
            covar = 0
            v1, v2 = self.init[1:]
        else:
            s1, s2 = self.efficacy_1, self.efficacy_2
            
            
        mu = self.therapy_max_eff
        sigma_k = self.carrying_capacity_std
        K_m = self.carrying_capacity
        r = self.growth_rate
        d1 = d2 = self.evolvability_cost
        k0 = self.evolvability_value
        k_r1, k_r2 = self.evolvability_scaling_factor_1, self.evolvability_scaling_factor_2

        sigma_t1 = self.efficacy_std_1
        sigma_t2 = self.efficacy_std_2

        exp_k = np.exp(-(v1**2 + v2**2) / sigma_k**2)
        exp_t1 = np.exp(-((v1 - mu)**2) / sigma_t1**2)
        exp_t2 = np.exp(-((v2 - mu)**2) / sigma_t2**2)

        # === Effective evolvability values depending on facultative status ===
        if self.facultative_evolvability:
            k1 = k0 + k_r1 * s1 * exp_t1
            k2 = k0 + k_r2 * s2 * exp_t2
        else:
            k1 = k2 = k0

        def G(v1, v2, x):
            """
            Growth function incorporating dual therapy effects and evolvability costs.
            """
            growth_term = (r / (K_m * exp_k)) * (K_m * exp_k - x)
            therapy_term_1 = s1 * exp_t1
            therapy_term_2 = s2 * exp_t2
            cost_term = d1 * k1 + d2 * k2
            return growth_term - (therapy_term_1 + therapy_term_2 + cost_term)

        def dG(v1, v2, x):
            """
            Derivatives of the growth function with respect to strategies (v1, v2).
            """
            exp_common = np.exp((v1**2 + v2**2) / sigma_k**2)

            if self.facultative_evolvability:
                # === Derivative w.r.t v1 ===
                dGdv1_term1 = (2 * d1 * k_r1 * s1 * (v1 - mu) * exp_t1) / sigma_t1**2
                dGdv1_term2 = (2 * s1 * (v1 - mu) * exp_t1) / sigma_t1**2
                dGdv1_term3 = (2 * r * x * v1 * exp_common) / (K_m * sigma_k**2)
                dGdv1 = dGdv1_term1 + dGdv1_term2 - dGdv1_term3

                # === Derivative w.r.t v2 ===
                dGdv2_term1 = (2 * d2 * k_r2 * s2 * (v2 - mu) * exp_t2) / sigma_t2**2
                dGdv2_term2 = (2 * s2 * (v2 - mu) * exp_t2) / sigma_t2**2
                dGdv2_term3 = (2 * r * x * v2 * exp_common) / (K_m * sigma_k**2)
                dGdv2 = dGdv2_term1 + dGdv2_term2 - dGdv2_term3
            else:
                # === Constant Evolvability (no evolvability terms in derivative) ===
                dGdv1 = (2 * s1 * (v1 - mu) * exp_t1) / sigma_t1**2 - (2 * r * x * v1 * exp_common) / (K_m * sigma_k**2)
                dGdv2 = (2 * s2 * (v2 - mu) * exp_t2) / sigma_t2**2 - (2 * r * x * v2 * exp_common) / (K_m * sigma_k**2)

            return dGdv1, dGdv2
        
        # === Compute dynamics ===
        g = G(v1, v2, x)
        dGdv1, dGdv2 = dG(v1, v2, x)

        dx = x * g
        dv1 = k1 * dGdv1 - covar * k2 * dGdv2
        dv2 = k2 * dGdv2 - covar * k1 * dGdv1

        return [dx, dv1, dv2]



    def run(self):
        """
        Run the simulation using the defined model and store results.
        """
        t_eval = np.arange(0, self.t_max + 1)
        sol = solve_ivp(self.model, [0, self.t_max], self.init, t_eval=t_eval)
        self.t = sol.t
        self.y = sol.y

        # Track evolvabilities
        v1, v2 = self.y[1], self.y[2]
        mu = self.therapy_max_eff
        k0 = self.evolvability_value
        sigma_t1, sigma_t2 = self.efficacy_std_1, self.efficacy_std_2
        k_r1, k_r2 = self.evolvability_scaling_factor_1, self.evolvability_scaling_factor_2

        self.evo_1 = []
        self.evo_2 = []

        for t, vi1, vi2 in zip(self.t, v1, v2):
            if t < self.therapy_start_time:
                s1, s2 = 0, 0
            else:
                s1, s2 = self.efficacy_1, self.efficacy_2

            if self.facultative_evolvability:
                e1 = k0 + k_r1 * s1 * np.exp(-(vi1 - mu)**2 / sigma_t1**2)
                e2 = k0 + k_r2 * s2 * np.exp(-(vi2 - mu)**2 / sigma_t2**2)
            else:
                e1 = e2 = k0

            self.evo_1.append(e1)
            self.evo_2.append(e2)

if __name__ == "__main__":
    """
    Test the DoubleBindModel with example parameters and plot results.
    """
    init = [20, 0.01, 0.01]

    model = DoubleBindModel(
        init=init,
        t_max=5000,
        cycle_duration = 500,
        efficacy_1=0.15,
        efficacy_2=0.3,
        efficacy_std_1=np.sqrt(6),
        efficacy_std_2=np.sqrt(2),
        evolvability_value=0.1,
        evolvability_cost=0.12,
        evolvability_scaling_factor_1=1.67,
        evolvability_scaling_factor_2=0.84,
        facultative_evolvability=True,
        carrying_capacity_std=10,
        carrying_capacity=100,
        growth_rate=0.3,
        therapy_max_eff=0,
        therapy_start_time=600,
        
    )

    model.run()

    t = model.t
    x = model.y[0]
    v1 = model.y[1]
    v2 = model.y[2]
    k1 = model.evo_1
    k2 = model.evo_1

    # Plot population
    plt.figure()
    plt.plot(t, x, label='Population Size', linewidth=2, color='purple')
    plt.xlabel("Time")
    plt.ylabel("Population Size")
    plt.title("Double Bind Model: Population Dynamics")
    plt.grid(True)
    plt.legend()

    # Plot strategies
    plt.figure()
    plt.plot(t, v1, label='v_c (chemo strategy)', color='blue')
    plt.plot(t, v2, label='v_t (targeted strategy)', color='red')
    plt.xlabel("Time")
    plt.ylabel("Strategy")
    plt.title("Double Bind Model: Strategy Dynamics")
    plt.legend()
    plt.grid(True)

    # Plot evolvabilities
    plt.figure()
    plt.plot(t, k1, label='k1 (chemo evolvability)', color='blue', linestyle='--')
    plt.plot(t, k2, label='k2 (targeted evolvability)', color='red', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Evolvability")
    plt.title("Double Bind Model: Evolvability Dynamics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
