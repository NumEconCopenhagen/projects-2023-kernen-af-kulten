#importing relevant packages
from types import SimpleNamespace
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar
import sympy as sm
import matplotlib.pyplot as plt
from IPython.display import display, Math
import math as math
import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits import mplot3d
import pandas as pd


class Assignment1:
    def __init__(self):
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()        
        
        #define the parameters
        par.alpha = sm.symbols('alpha', domain = sm.Interval(0,1),real=True) #weight of private consumption
        par.kappa = sm.symbols('kappa', positive = True,real=True) #free private consumption
        par.v = sm.symbols('v', positive = True,real=True) #disutility of labor
        par.w = sm.symbols('w', positive = True,real=True) #wage
        par.tau = sm.symbols('tau', domain = sm.Interval(0,1),real=True) #tax
        par.C = sm.symbols('C', positive=True,real=True) #private consumption
        par.G = sm.symbols('G', positive=True,real=True) #government consumption
        par.L = sm.symbols('L', domain = sm.Interval(0, 24),real=True) #labor
        par.w_tilde = sm.symbols(r'\tilde{w}', positive = True,real=True)

        #equations
        par.V_eq=sm.log(par.C**(par.alpha)*par.G**(1-par.alpha))-par.v*par.L**2/2
        par.C_eq =par.kappa + (1-par.tau)*par.w*par.L
    
        par.alpha_val = 0.5
        par.kappa_val = 1.0
        par.v_val = 1/(2*16**2)
        par.tau_val = 0.30
        par.w_val = 1

        sol.tau_star=0.3

    def as1_1(self, print_output=True, save = False):
        par = self.par
        sol = self.sol
        #Substitute in C
        V_eq_sub = par.V_eq.subs(par.C, par.C_eq)
        #differentiate
        V_diff_L = sm.Eq(sm.diff(V_eq_sub, par.L), 0)
        V_diff_L_sub = V_diff_L.subs((1-par.tau)*par.w, par.w_tilde)
        
        #Solve for L
        L_sol = sm.solve(V_diff_L_sub, par.L)
        
        if print_output==True:
            print("L_star:")
            display(Math(sm.latex(L_sol)))
        
        if save == True:
            return V_eq_sub
            
    def as1_2(self, print_output=True, save=False):
        par = self.par
        sol = self.sol
    # We rewrite the solution, so it is only the positive solution we use
        L_star = (-par.kappa + (par.kappa**2+4*par.alpha/par.v*par.w_tilde**2)**(1/2))/(2*par.w_tilde)

        #we substitute (1-tau)*w in again, and lambdify the function
        L_star_sub=L_star.subs(par.w_tilde, (1-par.tau)*par.w)
        display(Math(sm.latex(L_star_sub)))
        f_Lstar = sm.lambdify((par.w, par.alpha, par.kappa, par.v, par.tau), L_star_sub)

        # Define the range of w values
        w_values = np.linspace(0.1, 50, 1000)
        par.alpha_val = 0.5
        par.kappa_val = 1.0
        par.v_val = 1/(2*16**2)
        par.tau_val = 0.30
        par.w_val = 1
        # Calculate L_star for each w value
        L_star_values = f_Lstar(w_values, par.alpha_val, par.kappa_val, par.v_val, par.tau_val)
        
        if print_output==True:
            # Plot the function
            plt.plot(w_values, L_star_values)
            plt.xlabel('w')
            plt.ylabel('L_star')
            plt.title('L_star vs. w')
            plt.grid(True)
            plt.show()

        if save == True:
            return L_star_sub
    
    def as1_3(self, print_output=True, save=False):
        par = self.par
        sol = self.sol
        w_values = np.linspace(0.1, 50, 1000)
        par.alpha_val = 0.5
        par.kappa_val = 1.0
        par.v_val = 1/(2*16**2)
        par.tau_val = 0.30
        par.w_val = 1
        # I substitute in this version of G into the utility function:
        L_star_sub=self.as1_2(print_output=False, save=True)
        V_eq_sub=self.as1_1(print_output=False, save=True)
        G_q3 = par.tau*par.w*L_star_sub*((1-par.tau)*par.w)

        utility_q3 = V_eq_sub.subs(par.G, G_q3)

        f_utility = sm.lambdify((par.w, par.alpha, par.kappa, par.v, par.tau, par.L), utility_q3)

        # Define the range of w values
        L_values = np.linspace(0, 24, 1000)
        tau_values = np.linspace(0.0001, 1, 1000)
        X, Y = np.meshgrid(tau_values, L_values)

        # Calculate L_star for each w value
        Utility_values = f_utility(par.w_val, par.alpha_val, par.kappa_val, par.v_val, X, Y)

        if print_output==True:
            print("Utility:")
            display(Math(sm.latex(utility_q3)))
            # Create a figure and an axes object
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            # Create the surface plot
            ax.plot_surface(X, Y, Utility_values, cmap='viridis')

            # Customize the plot
            ax.set_xlabel('tau')
            ax.set_ylabel('Labor')
            ax.set_zlabel('Utility')
            ax.set_title('Utility for different tau and labor values')

            # Show the plot
            plt.show()
        
        if save==True:
            return utility_q3

    def as1_4(self, print_output=True, save=False):
        par = self.par
        sol = self.sol
        par.alpha_val = 0.5
        par.kappa_val = 1.0
        par.v_val = 1/(2*16**2)
        par.tau_val = 0.30
        par.w_val = 1
        utility_q3=self.as1_3(print_output=False, save=True)
        L_star_sub=self.as1_2(print_output=False, save=True)                
        # Define the utility_q4 function
        utility_q4 = - utility_q3.subs(par.L, L_star_sub) #make it negative as we use minimize
        
        # Create a lambda function with fixed values for w, alpha, kappa, and v
        f_utility_q4 = sm.lambdify(par.tau, utility_q4.subs({par.w: par.w_val, par.alpha: par.alpha_val, par.kappa: par.kappa_val, par.v: par.v_val}))
        
        def test(tau):
            return f_utility_q4(tau).item()  # Convert the result to a scalar instead of an array
        
        
        bounds = [(0, 1)]
        tau_star = optimize.minimize(test, par.tau_val, method="Nelder-Mead", bounds=bounds)
        
        sol.tau_star = tau_star.x[0]
        # Generate values for tau
        tau_values = np.linspace(0, 1, 1000)
        
        # Calculate utility for each tau value
        utility_values = [test(t) for t in tau_values]
        
        if print_output==True:
            print("optimal tau:", tau_star.x[0])  # Print the optimized variable value instead of the function valueprint(tau_star.fun)
            # Plot the utility graph
            plt.plot(tau_values, utility_values, label='Utility')
            plt.scatter(tau_star.x[0], tau_star.fun, color='red', label='Optimized Tau')
            plt.xlabel('Tau')
            plt.ylabel('Utility')
            plt.legend()

            # Add text annotation for optimal value
            optimal_tau = tau_star.x[0]
            optimal_utility = tau_star.fun
            plt.text(optimal_tau, optimal_utility, f'Optimal Tau: {optimal_tau:.4f}', 
                     ha='center', va='bottom', color='red')
            plt.show()
        



    def as1_5(self):

        # Define the objective function
        def objective_function_1(G, w, tau):
            # Constants
            epsilon = 1.000  
            rho = 1.001  
            sigma = 1.001  
            nu = 1/(2*16**2)  
            kappa = 1  
            alpha = 0.5  
            
            # Define the value function
            def value_function_1(L):
                C = kappa + (1 - tau) * w * L
                return (((alpha * C ** ((sigma - 1) / sigma) + (1 - alpha) * G ** ((sigma - 1) / sigma)) ** (sigma / (sigma-1))) ** (1 - rho) - 1) / (1 - rho) - nu * (L ** (1 + epsilon)) / (1 + epsilon)
            
            # Find the optimal labor supply
            result = minimize_scalar(lambda L: -value_function_1(L), bounds=(0, 24), method='bounded')
            L_star = result.x
            
            # Calculate the equation G = tau * w * L_star
            equation = G-tau*w*L_star
            
            return -equation

        # Solve for G
        tau = 0.338  # Given value for tau
        w = 1  # Choose an appropriate value for w
        result = minimize_scalar(lambda G: abs(objective_function_1(G, w, tau)), method='bounded', bounds=(0, 100))  # Adjust the bounds as necessary
        optimal_G = result.x

        print("Optimal G for rho = 1.001 and sigma = 1.001:", optimal_G)
        

        # Define the objective function
        def objective_function_2(G, w, tau):
            # Constants
            epsilon = 1.000  
            rho = 1.5  
            sigma = 1.5  
            nu = 1/(2*16**2)  
            kappa = 1  
            alpha = 0.5  
            # Define the value function
            def value_function_2(L):
                C = kappa + (1 - tau) * w * L
                return (((alpha * C ** ((sigma - 1) / sigma) + (1 - alpha) * G ** ((sigma - 1) / sigma)) ** (sigma / (sigma-1))) ** (1 - rho) - 1) / (1 - rho) - nu * (L ** (1 + epsilon)) / (1 + epsilon)
            
            # Find the optimal labor supply
            result = minimize_scalar(lambda L: -value_function_2(L), bounds=(0, 24), method='bounded')
            L_star = result.x
            
            # Calculate the equation G = tau * w * L_star
            equation = G-tau*w*L_star
            
            return -equation

        # Solve for G
        tau = 0.338  # Given value for tau
        w = 1  # Choose an appropriate value for w
        result = minimize_scalar(lambda G: abs(objective_function_2(G, w, tau)), method='bounded', bounds=(0, 100))  # Adjust the bounds as necessary
        optimal_G = result.x

        print("Optimal G for rho = 1.5 and sigma = 1.5:", optimal_G)


    def as1_6(self):

                
        # Define the objective function
        def totalobj(tau):

            def objective_function(G, w, tau):
                # Constants
                epsilon = 1.000  
                rho = 1.001  
                sigma = 1.001  
                nu = 1/(2*16**2)  
                kappa = 1  
                alpha = 0.5  
                # Define the value function
                def value_function(L):
                    C = kappa + (1 - tau) * w * L
                    return (((alpha * C ** ((sigma - 1) / sigma) + (1 - alpha) * G ** ((sigma - 1) / sigma)) ** (sigma / (sigma-1))) ** (1 - rho) - 1) / (1 - rho) - nu * (L ** (1 + epsilon)) / (1 + epsilon)
            
                # Find the optimal labor supply
                result = minimize_scalar(lambda L: -value_function(L), bounds=(0, 24), method='bounded')
                L_star = result.x
            
                # Calculate the equation G = tau * w * L_star. The absolute value of this is minimized to secure that the G-equation holds.
                equation = G-tau*w*L_star
            
                return -equation

            # Solve for G
            w = 1 
            result = minimize_scalar(lambda G: abs(objective_function(G, w, tau)), method='bounded', bounds=(0, 100))  # Adjust the bounds as necessary
            optimal_G = result.x
            return objective_function(optimal_G,w,tau)
        
        # Solve for tau
        tau=0.3 #initial value
        result = minimize_scalar(lambda tau: totalobj(tau), method='bounded', bounds=(0, 1))  # Adjust the bounds as necessary
        optimal_tau = result.x
        print(f"Optimal tau for sigma = rho = 1.001: {optimal_tau}")
    
    def as1_6_2(self):

                
        # Define the objective function
        def totalobj(tau):

            def objective_function(G, w, tau):
                # Constants
                epsilon = 1.000  
                rho = 1.5  
                sigma = 1.5  
                nu = 1/(2*16**2)  
                kappa = 1  
                alpha = 0.5  
                # Define the value function
                def value_function(L):
                    C = kappa + (1 - tau) * w * L
                    return (((alpha * C ** ((sigma - 1) / sigma) + (1 - alpha) * G ** ((sigma - 1) / sigma)) ** (sigma / (sigma-1))) ** (1 - rho) - 1) / (1 - rho) - nu * (L ** (1 + epsilon)) / (1 + epsilon)
            
                # Find the optimal labor supply
                result = minimize_scalar(lambda L: -value_function(L), bounds=(0, 24), method='bounded')
                L_star = result.x
            
                # Calculate the equation G = tau * w * L_star. The absolute value of this is minimized to secure that the G-equation holds.
                equation = G-tau*w*L_star
            
                return -equation

            # Solve for G
            w = 1 
            result = minimize_scalar(lambda G: abs(objective_function(G, w, tau)), method='bounded', bounds=(0, 100))  # Adjust the bounds as necessary
            optimal_G = result.x
            return objective_function(optimal_G,w,tau)
        
        # Solve for tau
        tau=0.3 #initial value
        result = minimize_scalar(lambda tau: totalobj(tau), method='bounded', bounds=(0, 1))  # Adjust the bounds as necessary
        optimal_tau = result.x
        print(f"Optimal tau for sigma = rho = 1.5: {optimal_tau}")
    

class Assignment2():
    def __init__(self):
        self.eta = 0.5   # Elasticity of demand
        self.w = 1.0     # Wage
        self.rho = 0.9   # AR(1) parameter for demand shock
        self.sigma_epsilon = 0.1  # Standard deviation of demand shock
        self.iota = 0.01  # Adjustment cost
        self.R = (1 + 0.01) ** (1 / 12)  # Discount factor
        self.K = 1000  # Number of random shock series
        self.delta = 0.05
        np.random.seed(0)
        self.shock_series = np.random.normal(
            loc=-0.5 * self.sigma_epsilon ** 2,
            scale=self.sigma_epsilon,
            size=(self.K, 120)
        )
        self.l_series = np.zeros(120)
        self.kappa_series = np.zeros(120)
        self.l_star_series = np.zeros(120)

    def calculate_profits(self, kappa_t, l):
        return kappa_t * l ** (-self.eta) * l - self.w * l

    def policy1(self, kappa_t, l_t_minus_1,delta,t):
        self.l_star_series[t]=((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)
        if (
            self.calculate_profits(
                kappa_t, ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)
            )
            + self.R
            * self.calculate_profits(
                np.exp(self.rho * np.log(kappa_t) - 0.5 * self.sigma_epsilon ** 2),
                ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta),
            )
            - self.iota
            > self.calculate_profits(kappa_t, l_t_minus_1)
            + self.R
            * self.calculate_profits(
                np.exp(self.rho * np.log(kappa_t) - 0.5 * self.sigma_epsilon ** 2),
                l_t_minus_1,
            )
        ):
            return ((1 - self.eta) * kappa_t / self.w) ** (1 / self.eta)
        else:
            return l_t_minus_1
        
        
    def deltapolicy(self,kappa_t,l_t_minus_1,delta,t):
        l_star = ((1 - self.eta) * kappa_t / self.w)**(1 / self.eta)

        self.l_star_series[t] = l_star
        if np.abs(l_t_minus_1-l_star)>delta:
            return l_star
        else: return l_t_minus_1
    
    def initialpolicy(self,kappa_t,l_t_minus_1,delta,t):
        l_star = ((1 - self.eta) * kappa_t / self.w)**(1 / self.eta)
        self.l_star_series[t] = l_star
        return l_star
    
    def ex_post_value(self, shock_series, policy):
        kappa_t_minus_1 = 1.0
        l_t_minus_1 = 0.00001
        ex_post_value = 0.0

        for t in range(len(shock_series)):
            kappa_t = np.exp(self.rho * np.log(kappa_t_minus_1) + shock_series[t])
            l_t = policy(kappa_t, l_t_minus_1,self.delta,t)
            y = l_t
            self.l_series[t] = l_t
            self.kappa_series[t] = kappa_t
            p_t = kappa_t * y ** (-self.eta)
            profits = p_t * y - self.w * l_t - (l_t != l_t_minus_1) * 1 * self.iota
            ex_post_value += self.R ** (-t) * profits
            kappa_t_minus_1 = kappa_t
            l_t_minus_1 = l_t

        return ex_post_value

    def obj_H(self,delta):
        self.delta = delta
        ex_ante_value = np.mean(
            [self.ex_post_value(self.shock_series[k], self.deltapolicy) for k in range(self.K)]
        )
        return -ex_ante_value

    def plot_graph(self):
        fig, ax1 = plt.subplots()
        ax1.plot(
            np.arange(len(self.shock_series[self.K - 1])),
            self.kappa_series,
            label="Demand (kappa)",
            color="blue",
            linestyle="--",
            alpha=0.5,
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Demand (kappa)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")
        ax1.set_ylim([min(self.kappa_series) - 1, max(self.kappa_series) + 1])

        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(len(self.shock_series[self.K - 1])),
            self.l_series,
            label="Labour hired",
            color="green",
        )
        ax2.plot(
            np.arange(len(self.shock_series[self.K - 1])),
            self.l_star_series,
            label="Optimal labour hired",
            color="red",
        )
        ax2.set_ylabel("Labour", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.legend(loc="upper right")

        plt.title("Labour and Demand")
        plt.show()


class Assignment3:
    def __init__(self):
        
        # create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        #set seed to ensure the same outcome every time even though we use random variables for intial values of x
        par.seed = 2023
        
        #set given settings:
        par.bounds = [-600, 600]
        par.tau = 1e-8 
        par.warmup_iterations = 10 #This is what is refered to as K_ in the pseudo code
        par.max_iterations = 1000 #this is K in the pseudo code
        
        #set up the starting values as random
        par.x_star = np.random.uniform(low=par.bounds[0], high=par.bounds[1], size=2)  # Set start values for x_star
        par.help = 0

        # create empty lists that can be used inside the loop
        sol.k_values = []
        sol.x_k0_values = []
        sol.x_star = par.x_star
        sol.result=[]
        
    # Just the definition of the given functions
    def griewank(self, x):
        return self.griewank_(x[0], x[1])
    
    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1
    
    def refined_global_optimizer(self, bounds, tau, warmup_iterations, max_iterations):
        par = self.par
        sol = self.sol
        
        K = np.arange(max_iterations)
        help = par.help #this is just a help variable to store k
    
        # because we use range(len(K)) we loop over k from 0 which is defult, up to the step before K, which is K-1 as de defult sted is 1.
        for k in range(len(K)):
            x_k = np.random.uniform(low=par.bounds[0], high=par.bounds[1], size=2)  # Set start values for x_k (3.A)
            
            #if in the warmup iterations (3.B)
            x_k0 = np.random.uniform(low=par.bounds[0], high=par.bounds[1], size=2) # we need a random starting value to go off
            if k < warmup_iterations:
                result = optimize.minimize(self.griewank, x_k0, method='BFGS', tol=par.tau) #optimization stated in 3.E
                x_k_star = result.x
                # This part is the implementation of 3.F
                if k == 0 or self.griewank(x_k_star) < self.griewank(sol.x_star):
                    sol.x_star = x_k_star
                # This is 3.G 
                if self.griewank(sol.x_star) < par.tau:
                    break 
            
            #if after the warmup iterations
            else:
                help = k
                sol.k_values.append(help)
                chi_k = 0.5 * 2 / (1 + np.exp((k - warmup_iterations) / 100)) #the expression af chi_k stated in 3.C
                x_k0 = chi_k * x_k + (1 - chi_k) * sol.x_star #the expression af x_k0 stated in 3.D, where we use the x_star found in the warm up face
    
                result = optimize.minimize(self.griewank, x_k0, method='BFGS', tol=par.tau) #optimization stated in 3.E
                x_k_star = result.x
                # This part is the implementation of 3.F
                if k == 0 or self.griewank(x_k_star) < self.griewank(sol.x_star):
                    sol.x_star = x_k_star
                # This is 3.G 
                if self.griewank(sol.x_star) < par.tau:
                    break
                sol.x_k0_values.append(x_k0)
        return result, sol.x_star, sol.x_k0_values, sol.k_values #we return more than x_star even though the alorithem only states to return x_star because we need it for the plot
    
    def figure1(self):
        #set seed to ensure the same outcome every time even though we use random variables for intial values of x
        par = self.par
        sol = self.sol
        np.random.seed(par.seed)
        par.warmup_iterations=10
        sol.result, sol.x_star, sol.x_k0_values, sol.k_values = self.refined_global_optimizer(par.bounds, par.tau, par.warmup_iterations, par.max_iterations)
        
        #print the x_star values
        print(f'x_star = {sol.x_star}')

        #create plot for initial values of x
        # Plotting x values convergence
        #x_k0_values = np.array(sol.x_k0_values)
        x1_values = [x[0] for x in sol.x_k0_values]
        x2_values = [x[1] for x in sol.x_k0_values]
        print(f'Iterations: {len(sol.x_k0_values)}')
        #x2_values = sol.x_k0_values[:, 1]
        
        plt.figure()
        plt.scatter(np.arange(len(x1_values)), x1_values, label='x1')
        plt.scatter(np.arange(len(x2_values)), x2_values, label='x2')
        plt.xlim(0,1000)
        plt.ylim(-600,600)
        plt.xlabel('k')
        plt.ylabel('x value')
        plt.title('Convergence of initial x values')
        plt.legend()
        plt.show()
    
    def figure2(self):
        #set seed to ensure the same outcome every time even though we use random variables for intial values of x
        par = self.par
        sol = self.sol
        np.random.seed(par.seed)
        par.warmup_iterations=100
        sol.result, sol.x_star, sol.x_k0_values, sol.k_values = self.refined_global_optimizer(par.bounds, par.tau, par.warmup_iterations, par.max_iterations)
        
        #print the x_star values
        print(f'x_star = {sol.x_star}')
        #create plot for initial values of x
        # Plotting x values convergence
        #x_k0_values = np.array(sol.x_k0_values)
        x1_values = [x[0] for x in sol.x_k0_values]
        x2_values = [x[1] for x in sol.x_k0_values]
        print(f'Iterations: {len(sol.x_k0_values)}')

        #x2_values = sol.x_k0_values[:, 1]
        
        plt.figure()
        plt.scatter(np.arange(len(x1_values)), x1_values, label='x1')
        plt.scatter(np.arange(len(x2_values)), x2_values, label='x2')
        plt.xlabel('k')
        plt.xlim(0,1000)
        plt.ylim(-600,600)
        plt.ylabel('x value')
        plt.title('Convergence of initial x values')
        plt.legend()
        plt.show()
    
    
    

    

    
    