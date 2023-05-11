from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class OLGModelClass():

    def __init__(self,do_print=False):
        """ create the model """

        if do_print: print('initializing the model:')
        #Creating Namespaces:
        self.par = SimpleNamespace() #parameters and start values
        self.sim = SimpleNamespace() #simulated values

        if do_print: print('calling .setup()')
        self.setup() #calls setup

        if do_print: print('calling .allocate()')
        self.allocate() #calls allocate
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.beta = 0.97 # discount factor
        par.n = 0.05 # population growth

        # b. firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.3 # capital weight

        # c. government
        par.tau_w = 0 # wage tax

        # d. start values and amount of simulations
        par.K_ini = 0.1 # initial capital stock
        par.L_ini = 1.0 # initial Population
        par.simT = 20 # length of simulation (how many time periods)

    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        household = ['C1','C2','savings']
        firm = ['K','Y','L', 'k']
        prices = ['w','r']

        # b. creates sim. arrays for all the variables that are the length of the simulation
        allvarnames = household + firm + prices
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self,do_print=True):
        """ simulate model """

        t0 = time.time() #creates time variable

        par = self.par
        sim = self.sim 
        
        # a. initial values
        sim.K[0] = par.K_ini
        sim.L[0] = par.L_ini

        # b. Simulate the model
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t)
            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect') #optimizes the euler error wrt. s
            s = result.root
            # iii. log optimal savingsrate
            sim.savings[t]=s

            # iiii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ find bracket for s to search in """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min
    upper = s_max

    it = 0
    
    #finds the values for which the euler error changes sign from + to -, as the correct s must be within those brackets
    while it < maxiter:
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t)

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. check conditions
        valid = not np.isnan(value)
        correct_sign = np.sign(value)*sign_max < 0
        
        # iii. next step
        if valid and correct_sign:
            s_min = s
            s_max = upper
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # too low s -> increase lower bound
            lower = s
        else: # too high s -> increase upper bound
            upper = s

        # iv. increment
        it += 1

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # next period

    # b. Euler equation
    LHS = sim.C1[t]**(-1)
    RHS = (1+sim.r[t+1])*par.beta * sim.C2[t+1]**(-1)

    #calculates euler-error
    return LHS-RHS

def simulate_before_s(par,sim,t):
    """ simulate forward """
    
    if t==0:
        sim.K[t] = par.K_ini
        sim.L[t] = par.L_ini
        #sim.k[t] = 1
    if t > 0:
        sim.L[t] = sim.L[t-1]*(1+par.n)
    
    # i. production
    sim.Y[t] = sim.K[t]**par.alpha * (sim.L[t])**(1-par.alpha)

    # ii. factor prices
    sim.r[t] = par.alpha * sim.K[t]**(par.alpha-1) * (sim.L[t])**(1-par.alpha)
    sim.w[t] = (1-par.alpha) * sim.K[t]**(par.alpha) * (sim.L[t])**(-par.alpha)

    # consumption
    sim.C2[t] = (1+sim.r[t])*(sim.K[t])

def simulate_after_s(par,sim,t,s):
    """ simulate forward """
    sim.k[t] = sim.K[t]/sim.L[t]
    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s) * sim.L[t]
    
    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t]
    sim.K[t+1] = sim.K[t]+I