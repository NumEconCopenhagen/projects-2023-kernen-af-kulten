from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
from IPython.display import display, Math

def OLG_analytical(print_output=True):
    ##solving model for k*
    ##we define the parameters and functions:
    from IPython.display import display, Math

    # Define parameters
    beta = sm.symbols('beta')
    alpha = sm.symbols('alpha')
    n = sm.symbols('n')
    rt = sm.symbols('r_t')
    wt = sm.symbols('w_t')
    Kt = sm.symbols('K_t')
    Lt = sm.symbols('L_t')

    rho = sm.symbols('rho')
    kt = sm.symbols('k_t')
    kt_1 = sm.symbols('k_{t+1}')
    k = sm.symbols('k*')
    # Define production function
    f = Kt**alpha * Lt**(1-alpha)

    # Solve for wage and MPK
    mpl = sm.diff(f, Lt)

    #substitute in kt
    mpl_sub = mpl.subs(Kt**(alpha)*Lt**(1-alpha)*Lt**(-1), kt**alpha)
    w_eq = sm.Eq(wt, mpl_sub)

    # Define household utility
    c1t, c2t_1 = sm.symbols('c_{1t} c_{2t+1}')
    wt, rt_1 = sm.symbols('w_t r_{t+1}')
    st = sm.symbols('s_t')
    c1t = wt - st
    c2t_1 = (1+rt_1)*st
    U = sm.log(c1t) + beta*sm.log(c2t_1)

    # Find the derivative of U with respect to st
    dU = sm.diff(U, st)

    #solve for s_t
    s_eq = sm.Eq(0, dU)
    st_path = sm.solve(s_eq, st)[0]


    # Define the equation
    st_path_sub=st_path.subs(wt, mpl_sub)

    #we know that k_(t+1)=s_t/(1+n)
    kt_1 = st_path_sub/(1+n)

    #As we are looking for steady state we can set k_(t+1)=k_t
    kt_1_sub = kt_1.subs(kt, k)
    ss_solve = sm.Eq(k, kt_1_sub)

    k_ss = sm.solve(ss_solve, k)

    #Analytical answer with our chosen parameter values
    f_kss = sm.lambdify((alpha, n, beta), k_ss)

    if print_output:
        # Print the resulting equation
        print("wage equation for w_t:")
        display(Math(sm.latex(w_eq)))

        print("differentiated utility:")
        display(Math(sm.latex(dU)))

        print('s_t:')
        display(Math(sm.latex(st_path)))

        print('s_t with w_t inserted:')
        display(Math(sm.latex(st_path_sub)))

        print('steady state for capital per capita')
        display(Math(sm.latex(k_ss)))

        print('steady state for capital per capita, numerically')
        print(f_kss(0.3, 0.05, 0.97))
    
    #returns the theoretical solution, so it can be used for the comparisons to the analytical
    return f_kss




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

def first_results():
    model = OLGModelClass()
    par = model.par 
    sim = model.sim 

    s_guess = 0.4
    
    #Simulating some periods of the model manually
    simulate_before_s(par,sim,t=0)
    print("Consumption by old people in t=0:", f'{sim.C2[0] = : .4f}')

    simulate_after_s(par,sim,s=s_guess,t=0)
    print("Consumption by young people in t=0:", f'{sim.C1[0] = : .4f}')

    simulate_before_s(par,sim,t=1)
    print("Consumption by old people in t=1:", f'{sim.C2[1] = : .4f}')

    simulate_after_s(par,sim,s=s_guess,t=1)
    print("Consumption by young people in t=1:", f'{sim.C1[1] = : .4f}')

    #Calculating the Euler-error
    LHS_Euler = sim.C1[0]**(-1)
    RHS_Euler = (1+sim.r[1])*par.beta * sim.C2[1]**(-1)
    print(f'euler-error from period 0 to 1: {LHS_Euler-RHS_Euler:.8f}')

    #Check if euler-error goes to 0:
    model.simulate()
    LHS_Euler = sim.C1[18]**(-1)
    RHS_Euler = (1+sim.r[19])*par.beta * sim.C2[19]**(-1)
    print("euler error after model has been simulated", LHS_Euler-RHS_Euler)

    #save SS for this version
    sim.k_orig=sim.k.copy()

#code for figure showing change in beta
def model_beta06():
    model = OLGModelClass()
    par = model.par 
    sim = model.sim 

    model.simulate()
    #save SS for this version
    k_orig=sim.k.copy()

    #New parameter change of beta:
    par.beta = 0.60
    # New steady state
    f_kss_func = OLG_analytical(print_output=False)
    f_kss_06 = f_kss_func(0.3, 0.05, 0.60)
    f_kss_orig = f_kss_func(0.3, 0.05, 0.97)


    #Simulate model and log k
    model.simulate()
    k_new1 = model.sim.k

    #Visualisation
    fig = plt.figure(figsize=(6,6/1.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_new1,label=r'$k_{t}$, beta=0.60 new')
    ax.plot(k_orig,label=r'$k_{t}$ , old')
    ax.axhline(f_kss_orig,ls='--',color='black',label='analytical steady state (beta=0.97)')
    ax.axhline(f_kss_06,ls='--',color='black',label='analytical steady state (beta=0.60)')
    ax.legend(frameon=True,fontsize=12)
    ax.set_title("Numerical solutions for various parameter changes")
    ax.annotate(f'SS_old: {f_kss_orig}', xy=(3, 0.19), xycoords='data', color='red')
    ax.annotate(f'SS_new: {f_kss_06}', xy=(3, 0.15), xycoords='data', color='red')
    fig.tight_layout()

#code for figure showing change in alpha
def model_alpha035():
    model = OLGModelClass()
    par = model.par 
    sim = model.sim 

    #Back to original parameter for beta:
    par.beta = 0.97
    model.simulate()
    #save SS for this version
    k_orig=sim.k.copy()


    #New parameter change
    par.alpha = 0.35

    # New steady state
    f_kss_func = OLG_analytical(print_output=False)
    f_kss_035 = f_kss_func(0.35, 0.05, 0.97)
    f_kss_orig = f_kss_func(0.3, 0.05, 0.97)


    #Simulate model and log k
    model.simulate()
    k_new1 = model.sim.k

    #Visualisation
    fig = plt.figure(figsize=(6,6/1.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_new1,label=r'$k_{t}$, alpha=0.35 new')
    ax.plot(k_orig,label=r'$k_{t}$ , alpha=0.3 old')
    ax.axhline(f_kss_orig,ls='--',color='black',label='analytical steady state (alpha=0.3)')
    ax.axhline(f_kss_035,ls='--',color='black',label='analytical steady state (alpha=0.35)')
    ax.legend(frameon=True,fontsize=12)
    ax.set_title("Numerical solutions for various parameter changes")
    ax.annotate(f'SS_old: {f_kss_orig}', xy=(3, 0.19), xycoords='data', color='red')
    ax.annotate(f'SS_new: {f_kss_035}', xy=(3, 0.15), xycoords='data', color='red')
    fig.tight_layout()

#code for figure showing change in n
def model_n01():
    model = OLGModelClass()
    par = model.par 
    sim = model.sim 

    #Back to original parameter for beta and alpha:
    par.beta = 0.97
    par.alpha = 0.3
    model.simulate()
    #save SS for this version
    k_orig=sim.k.copy()


    #New parameter change
    par.n = 0.10

    # New steady state
    f_kss_func = OLG_analytical(print_output=False)
    f_kss_01 = f_kss_func(0.3, 0.10, 0.97)
    f_kss_orig = f_kss_func(0.3, 0.05, 0.97)


    #Simulate model and log k
    model.simulate()
    k_new1 = model.sim.k

    #Visualisation
    fig = plt.figure(figsize=(6,6/1.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_new1,label=r'$k_{t}$, n=0.1 new')
    ax.plot(k_orig,label=r'$k_{t}$ , n=0.05 old')
    ax.axhline(f_kss_orig,ls='--',color='black',label='analytical steady state (n=0.1)')
    ax.axhline(f_kss_01,ls='--',color='black',label='analytical steady state (n=0.05)')
    ax.legend(frameon=True,fontsize=12)
    ax.set_title("Numerical solutions for various parameter changes")
    ax.annotate(f'SS_old: {f_kss_orig}', xy=(3, 0.195), xycoords='data', color='red')
    ax.annotate(f'SS_new: {f_kss_01}', xy=(3, 0.17), xycoords='data', color='red')
    fig.tight_layout()

#code for figure showing introduction of tax
def taxes():
    model = OLGModelClass()
    par = model.par 
    sim = model.sim 

    par.n = 0.05
    par.alpha = 0.3
    par.beta = 0.97

    model.simulate()
    #save SS for this version
    k_orig=sim.k.copy()
    s_orig=sim.savings.copy()

    #There is no change to the analytical steady state capital per capita, as we assume that the funds of the social system has the same interest rate as the private sector
    #Thus, there is no need to recalculate the analytical SS. The 20% tax is thus just a lower limit on savings, having no effect, as s_opt = 49 % > 20 %.
    #Note that the savings_rate given below is of disposable income, and the new savings rate is therefore 36%, rather than 49-20 = 29 %.
    f_kss_func = OLG_analytical(print_output=False)
    f_kss_orig = f_kss_func(0.3, 0.05, 0.97)
    #initialize a wage tax
    par.tau_w = 0.2

    #Model simulation and log k
    model.simulate()
    k_new1 = model.sim.k

    #Visualisation
    fig = plt.figure(figsize=(6,6/1.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_new1,label=r'$k_{t}$, tau_w=20% new')
    ax.plot(k_orig,label=r'$k_{t}$ , tau_w=0 old')
    ax.axhline(f_kss_orig,ls='--',color='black',label='analytical steady state')
    ax.legend(frameon=True,fontsize=12)
    ax.set_title("Numerical solutions for various parameter changes")
    ax.annotate(f'SS_old: {f_kss_orig}', xy=(3, 0.195), xycoords='data', color='red')
    ax.annotate(f'SS_new: {f_kss_orig}', xy=(3, 0.17), xycoords='data', color='red')
    print(f'savings_rate_old[18]: {s_orig[18]}')
    print(f'savings_rate_new[18]: {sim.savings[18]}')
    fig.tight_layout()
