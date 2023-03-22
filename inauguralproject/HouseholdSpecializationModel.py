from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        #creates two different nu's for men and women
        par.nu_M=0.001
        par.nu_F=0.001

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        # create new variables for question 5
        sol.LM_vec_ny = np.zeros(par.wF_vec.size)
        sol.HM_vec_ny = np.zeros(par.wF_vec.size)
        sol.LF_vec_ny = np.zeros(par.wF_vec.size)
        sol.HF_vec_ny = np.zeros(par.wF_vec.size)

        sol.beta0_ny = np.nan
        sol.beta1_ny = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production (fixed by ALB)
        if par.sigma==0:
            H=min(HM, HF)
        elif par.sigma==1:
            H=HM**(1-par.alpha)*HF**par.alpha
        else:
            H=((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False, ratio=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations

        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()
        

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        opt.ratio = opt.HF/opt.HM #added a ratio variable


        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. Sets the start guesses at the optimal for the discrete optimization
        LM_guess=[4.5]
        LF_guess=[4.5]
        HM_guess=[4.5]
        HF_guess=[4.5]


        # b. calculate utility (negative as we want to maximize)
        def objective_function(x):
            LM, HM, LF, HF = x
            if LM + HM > 24 or LF + HF > 24: #create restriction so it isn't possible to work more that 24 hours
                return -np.inf
            return -self.calc_utility(LM, HM, LF, HF)

        # d. find maximizing argument using Nelder-Mead
        res = optimize.minimize(objective_function, [LM_guess, HM_guess, LF_guess, HF_guess], method="Nelder-Mead")

        if not res.success:
            print("Optimization failed.")
        #saves the optimized values
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]
        opt.ratio = opt.HF / opt.HM

         # e. print
        if do_print:
            for k, v in opt.__dict__.items():
                print(f"{k} = {v:6.4f}")

        return opt
       


    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        #Creates the wF vector needed
        par.wF_vec = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # Calculates the HF and HM for each of the wF values and saves the results in the sol. vectors
        for j, wage in enumerate(par.wF_vec):
            par.wF = wage
            model_solution=self.solve()
            sol.HF_vec[j]=model_solution.HF
            sol.HM_vec[j]=model_solution.HM

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ minimize error between model results and targets """

        par = self.par
        sol = self.sol

        # a. define error function
        def error_function(alpha_sigma):
            alpha, sigma = alpha_sigma.ravel()  # flatten the 2D array
            par.alpha, par.sigma = alpha, sigma
            self.solve_wF_vec() #runs the solve_wF_vec function
            self.run_regression() #runs the run_regression function
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        # b. find minimizing argument
        res = optimize.minimize(error_function, [par.alpha, par.sigma], method="Nelder-Mead")

        if not res.success:
            print("Optimization failed.")

        # d. print results
        print(f"Optimal alpha: {par.alpha}")
        print(f"Optimal sigma: {par.sigma}")

    #Creates the framework for question 5
    #exactly the same utility function as before except men and women have a different disutility-parameter of work from home (so far set to the same value as before)
    def calc_utility_ny(self,LM,HM,LF,HF):
        """ calculate utility when nu is different for men and women"""

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production (fixed by ALB)
        if par.sigma==0:
            H=min(HM, HF)
        elif par.sigma==1:
            H=HM**(1-par.alpha)*HF**par.alpha
        else:
            H=((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu_M*TM**epsilon_/epsilon_+par.nu_F*TF**epsilon_/epsilon_
        
        return utility - disutility

    def solve_ny(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. Sets the start guesses at the optimal for the discrete optimization
        LM_guess=[4.5]
        LF_guess=[4.5]
        HM_guess=[4.5]
        HF_guess=[4.5]


        # b. calculate utility (negative as we want to maximize)
        def objective_function(x):
            LM, HM, LF, HF = x
            if LM + HM > 24 or LF + HF > 24: #create restriction so it isn't possible to work more that 24 hours
                return -np.inf
            return -self.calc_utility_ny(LM, HM, LF, HF)

        # d. find maximizing argument using Nelder-Mead
        res = optimize.minimize(objective_function, [LM_guess, HM_guess, LF_guess, HF_guess], method="Nelder-Mead")

        if not res.success:
            print("Optimization failed.")
        #saves the optimized values
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]
        opt.ratio = opt.HF / opt.HM

         # e. print
        if do_print:
            for k, v in opt.__dict__.items():
                print(f"{k} = {v:6.4f}")

        return opt
       


    def solve_wF_vec_ny(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        #Creates the wF vector needed
        par.wF_vec = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # Calculates the HF and HM for each of the wF values and saves the results in the sol. vectors
        for j, wage in enumerate(par.wF_vec):
            par.wF = wage
            model_solution=self.solve_ny()
            sol.HF_vec_ny[j]=model_solution.HF
            sol.HM_vec_ny[j]=model_solution.HM
            sol.LF_vec_ny[j]=model_solution.LF
            sol.LM_vec_ny[j]=model_solution.LM

    def run_regression_ny(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec_ny/sol.HM_vec_ny)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0_ny,sol.beta1_ny = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate_opg5(self,nu_M=None,sigma=None):
        """ minimize error between model results and targets """

        par = self.par
        sol = self.sol

        # a. define error function
        def error_function(nu_M_sigma):
            nu_M, sigma = nu_M_sigma.ravel()  # flatten the 2D array
            par.nu_M, par.sigma = nu_M, sigma
            self.solve_wF_vec_ny() #runs the solve_wF_vec function
            self.run_regression_ny() #runs the run_regression function
            return (par.beta0_target - sol.beta0_ny)**2 + (par.beta1_target - sol.beta1_ny)**2

        # b. find minimizing argument
        res = optimize.minimize(error_function, [par.nu_M, par.sigma], method="Nelder-Mead")

        if not res.success:
            print("Optimization failed.")

        # d. print results
        print(f"Optimal nu_M: {par.nu_M}")
        print(f"Optimal sigma: {par.sigma}")