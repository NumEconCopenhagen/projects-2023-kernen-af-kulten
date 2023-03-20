
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

        #eksistererer i dictionary par 
        # b. preferences
        par.rho = 2.0 
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5
       

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

        #eksisterer i dictionary sol
        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,args):
        LM,HM,LF,HF = args
        """ calculate utility """

        par = self.par #siger at par er lig den par fra tidligere
        sol = self.sol #siger at sol er lig den sol fra tidligere

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
    #finder nettonytten

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
        u = self.calc_utility([LM,HM,LF,HF])
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        #opt.LF = LF[j]
        opt.HF = HF[j]
        opt.ratio = opt.HF/opt.HM

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    


    def solve(self,do_print=False):
        """ solve model continously """
        def constraint1(args):
            x = args[:4]
            return x[0] + x[1] - 24
        def constraint2(args):
            x = args[:4]
            return x[2] + x[3] - 24
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        from scipy import optimize
        x0 = [1, 1, 1, 1]
        cons = [{'type': 'ineq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2}]
        print(optimize.minimize(self.calc_utility, x0, method="SLSQP",constraints=cons))
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        opt.ratio = opt.HF/opt.HM


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
            if LM + HM > 24 or LF + HF > 24:
                return -np.inf
            return -self.calc_utility(LM, HM, LF, HF)

        # d. find maximizing argument
        res = optimize.minimize(objective_function, [LM_guess, HM_guess, LF_guess, HF_guess], method="Nelder-Mead")

        if not res.success:
            print("Optimization failed.")

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
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        # b. find minimizing argument
        res = optimize.minimize(error_function, [par.alpha, par.sigma], method="Nelder-Mead")

        if not res.success:
            print("Optimization failed.")

        # d. print results
        print(f"Optimal alpha: {par.alpha}")
        print(f"Optimal sigma: {par.sigma}")

