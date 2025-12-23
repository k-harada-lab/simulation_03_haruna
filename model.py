import numpy as np
from tqdm import tqdm


class LuxMarchesiModel:
    def __init__(self, state, params):
        self.state = state.copy()
        self.params = params.copy()
        self.last_prices = [self.state["p"]]
        self.price_change = 0

    def excessDemand(self):
        """
        Function to compute the excess demand of the asset
        """
        n_p = self.state["n_p"]
        n_m = self.state["n_m"]
        n_f = self.state["n_f"]
        p_f = self.state["p_f"]
        p = self.state["p"]

        t_c = self.params["t_c"]
        gamma = self.params["gamma"]

        D_c = (n_p - n_m)*t_c
        D_f = n_f*gamma*(p_f - p)/p
        D = D_c + D_f

        return D
 
    def changeFundamental(self):
        """
        Function to change the fundamental value of the asset
        """
        sigma_eps = self.params["sigma_eps"]
        p_f = self.state["p_f"]
        dt = self.params["dt"]

        eps = np.random.normal(0, sigma_eps, 1)
        p_f = p_f * np.exp(eps)
        
        self.state["p_f"] = p_f[0]
        return
    
    def changePrice(self):
        """
        Changes the price of the asset
        """
        # Extract the parameters
        p = self.state["p"]
        #mu = self.params["mu"]
        beta = self.params["beta"]
        dt = self.params["dt"]

        # Calculate excess demand
        D = self.excessDemand()
        mu = np.random.normal(0, self.params["sigma_mu"], 1)[0]
        x = beta*(D + mu)

        # Update the price
        rand = np.random.uniform(0,1,1)
        delta_p = 0.001*p
        if (x > 0) and (rand < np.max([0, x])):
            p = p + delta_p
        elif (x < 0) and (rand < -np.min([0, x])):
            p = p - delta_p
        self.state["p"] = p
    
        if len(self.last_prices) > 0.2/dt:
            self.last_prices.pop(0)
        self.last_prices.append(p)
        self.price_change = (p - self.last_prices[0])/0.2 #### 
        
        return
              
    def updateGroups(self):
        """
        Fluxes between the fundamentalists and the noise traders
        """
        # Extract the parameters
        R = self.params["R"]
        p_f = self.state["p_f"]
        p = self.state["p"]
        n_p = self.state["n_p"]
        n_m = self.state["n_m"]
        n_f = self.state["n_f"]
        price_change = self.price_change
        alpha_3 = self.params["alpha_3"]
        v_2 = self.params["v_2"]
        s = self.params["s"]
        dt = self.params["dt"]

        # Compute other parameters
        r = R*p_f
        N = n_p + n_m + n_f
        threshold = 0.008 * N

        # Compute the forcing terms
        x = (r + price_change/v_2)/p - R
        y = s*np.abs((p_f - p)/p)
        U_21 = alpha_3*(x-y)
        U_22 = alpha_3*(-x-y)

        # Compute the probabilities
        pi_pf = v_2 * (n_p / N) * np.exp(U_21) * dt
        pi_fp = v_2 * (n_f / N) * np.exp(-U_21) * dt
        pi_mf = v_2 * (n_m / N) * np.exp(U_22) * dt
        pi_fm = v_2 * (n_f / N) * np.exp(-U_22) * dt

        # Compute the fluxes
        n_pf = np.random.binomial(n_f, pi_pf)
        n_fp = np.random.binomial(n_p, pi_fp)
        n_mf = np.random.binomial(n_f, pi_mf)
        n_fm = np.random.binomial(n_p, pi_fm)

        # Corrections to avoid vanishing groups
        if n_p - n_fp < threshold:
          n_fp = 0  
        if n_f - n_pf - n_mf < threshold:
          n_pf = 0  
          n_mf = 0  
        if n_m - n_fm < threshold:
          n_fm = 0

        # Update the groups
        self.state["n_p"] = n_p + n_pf - n_fp
        self.state["n_f"] = n_f + n_fp + n_fm - n_pf - n_mf
        self.state["n_m"] = n_m + n_mf - n_fm

        return
  
    def updateChartists(self):
        """
        Changes of opinion of the noise traders
        """
        p = self.state["p"]
        n_p = self.state["n_p"]
        n_m = self.state["n_m"]
        n_f = self.state["n_f"]
        price_change = self.price_change
        alpha_1 = self.params["alpha_1"]
        alpha_2 = self.params["alpha_2"]
        v_1 = self.params["v_1"]
        dt = self.params["dt"]

        # Compute the forcing terms
        x = (n_p - n_m)/(n_p + n_m) # opinion index
        N = n_p + n_m + n_f
        price_trend = price_change/p
        U_1 = alpha_1*x + alpha_2*price_trend/v_1
        
        # Compute the probabilities
        pi_pm = v_1*(n_p + n_m)/N*np.exp(U_1)  * dt
        pi_mp = v_1*(n_p + n_m)/N*np.exp(-U_1) * dt

        # Compute the fluxes
        n_pm = np.random.binomial(n_m, pi_pm) # From m to p
        n_mp = np.random.binomial(n_p, pi_mp) # From p to m

        # Corrections to avoid vanishing groups
        if n_p - n_mp < 0.008 * N:
            n_mp = 0
        if n_m - n_pm < 0.008 * N:
            n_pm = 0

        # Update the groups
        self.state["n_p"] = n_p + n_pm - n_mp
        self.state["n_m"] = n_m + n_mp - n_pm

        return
    
    def oneStep(self):
        """
        Defines one full step of the model
        """
        self.updateChartists()
        self.updateGroups()
        self.changePrice()
        self.changeFundamental()
        return
    
    def simulate(self, t, deltaT):
        """
        Simulate the model for a given time period
        """
        n = int(t/deltaT)
        delay = int(deltaT/self.params["dt"])
        
        # Initialize history
        prices = np.zeros(n)
        fundamentals = np.zeros(n)
        num_p = np.zeros(n)
        num_m = np.zeros(n)
        num_f = np.zeros(n)
        
        # Simulate the model
        for i in tqdm(range(n)):
            for _ in range(delay):
                # Simulate one step
                self.updateChartists()
                self.updateGroups()
                self.changePrice()
            # Update the fundamental
            self.changeFundamental()

            # Store the data
            prices[i] = self.state["p"]
            fundamentals[i] = self.state["p_f"]
            num_p[i] = self.state["n_p"]
            num_m[i] = self.state["n_m"]
            num_f[i] = self.state["n_f"]

        # Create history data structure
        history = {"prices": prices, "fundamentals": fundamentals, 
                   "optimists": num_p, "pessimists": num_m, "fundamentalists": num_f}
        
        return history
    
    def __str__(self):
        return str(self.state)
    