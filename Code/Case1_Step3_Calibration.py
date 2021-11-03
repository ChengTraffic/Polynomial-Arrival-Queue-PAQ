# -*- coding: utf-8 -*-

# Citation: Cheng, Q., Liu, Z., Guo, J., Wu, X., Pendyala, R., Belezamo, B., 
# & Zhou, X. 2022. Estimating key traffic state parameters through parsimonious
# spatial queue models. Under review with Transportation Research Part C.

import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

class get_data:
    
    def get_useful_data(self, flow_raw, occupancy_raw, speed_raw, lanes_raw, distance_raw, 
                        start_location_index, end_location_index, start_time_index, end_time_index):
        
        flow = flow_raw.iloc[start_location_index:end_location_index, start_time_index:end_time_index]
        speed = speed_raw.iloc[start_location_index:end_location_index, start_time_index:end_time_index]
        occupancy = occupancy_raw.iloc[start_location_index:end_location_index, start_time_index:end_time_index]
        density = self.calculate_density_with_occupancy(occupancy)
        lanes = lanes_raw.iloc[start_location_index:end_location_index, ]
        distance = distance_raw.iloc[start_location_index:end_location_index, ]
        
        return flow, density, speed, lanes, distance
    
    def calculate_density_with_occupancy(self, occupancy):
        # see Eq. (7.2) on page 193 in May, A.D., 1990. Traffic flow fundamentals. Prentice Hall, Inc., New Jersey.
        L = 25
        density = 5280/L*occupancy
        return density
    
    def get_ObsCumulativeDepartureCurve(self, flow, bottleneck_downstream_location_index, start_time_index, end_time_index):
        obsCumulativeDeparture = flow.iloc[bottleneck_downstream_location_index, start_time_index:end_time_index]
        for i in range(1, len(obsCumulativeDeparture)):
            obsCumulativeDeparture[i] = obsCumulativeDeparture.iloc[i-1] + obsCumulativeDeparture.iloc[i]
        return obsCumulativeDeparture
    
    def get_ObsQueue(self, density, lanes, distance, critical_density):
        obsQueueTemp = density*0
        for i in range(density.shape[0]):
            for j in range(density.shape[1]):
                if np.array(density.iloc[i,j] - critical_density) < 0:
                    density.iloc[i,j] = 0
                else:
                    density.iloc[i,j] = density.iloc[i,j] - critical_density
                obsQueueTemp.iloc[i,j] = density.iloc[i,j]*lanes.iloc[i,1]*distance.iloc[i,0]
        obsQueue = np.sum(obsQueueTemp, 0)    # note that this is the physical queue length
        return obsQueue
    
    def get_ObsDealy(self, speed, distance, v_f):
        travel_time_temp = speed*0
        for i in range(speed.shape[0]):
            for j in range(speed.shape[1]):
                travel_time_temp.iloc[i,j] = distance.iloc[i,0]/speed.iloc[i,j]*60    # unit: minute
        travel_time = np.sum(travel_time_temp, 0)
        fftt = np.sum(distance,0)/v_f*60    # unit: minute
        obsDealy = travel_time - np.array(fftt)
        return obsDealy

class Adam_optimization():
    
    def __init__(self, objective, first_order_derivative, bounds, x0):
        self.n_iter = 1000
        self.alpha = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.objective = objective
        self.first_order_derivative = first_order_derivative
        self.bounds = bounds
        self.x0 = x0
    
    def adam(self):
        # keep track of solutions and scores
        solutions = list()
        scores = list()
        # generate an initial point
        x = list(self.x0)
        score = self.objective(x)
        # initialize first and second moments
        m = [0.0 for _ in range(self.bounds.shape[0])]
        v = [0.0 for _ in range(self.bounds.shape[0])]
        # run the gradient descent updates
        for t in range(1, self.n_iter):
            # calculate gradient g(t)
            g = self.first_order_derivative(x)
            # build a solution one variable at a time
            for i in range(self.bounds.shape[0]):
                # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g[i]**2
                # mhat(t) = m(t) / (1 - beta1(t))
                mhat = m[i] / (1.0 - self.beta1**(t+1))
                # vhat(t) = v(t) / (1 - beta2(t))
                vhat = v[i] / (1.0 - self.beta2**(t+1))
                # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                x[i] = x[i] - self.alpha * mhat / (sqrt(vhat) + self.eps)
            # evaluate candidate point
            score = self.objective(x)
            # keep track of solutions and scores
            solutions.append(x.copy())
            scores.append(score)
            # report progress
        # print('Solution: %s, \nOptimal function value: %.5f' %(solutions[np.argmin(scores)], min(scores)))
        return solutions, scores
    
    def plot_iteration_process_adam(self, solutions):
        # sample input range uniformly at 0.1 increments
        xaxis = np.arange(self.bounds[0,0], self.bounds[0,1], 0.1)
        yaxis = np.arange(self.bounds[1,0], self.bounds[1,1], 0.1)
        x, y = np.meshgrid(xaxis, yaxis)
        results = self.objective(x, y)
        solutions = np.asarray(solutions)
        fig, ax = plt.subplots(figsize=(10,6))
        cs = ax.contourf(x, y, results, levels=50, cmap='jet')
        ax.set_xlim(self.bounds[0,0], self.bounds[0,1])
        ax.set_ylim(self.bounds[1,0], self.bounds[1,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        plt.tick_params(labelsize=14)
        plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='k')
        plt.colorbar(cs)
        plt.title('Iteration process')
        fig.savefig('../Figures/Case 1/Iteration process.png', dpi=300, bbox_inches='tight')

class BayesianOptimization():
    
    def __init__(self, max_iters, obj_fun, bounds, x0, n_pre_samples=10, gp_params=None, random_search=False, alpha=1e-8, epsilon=1e-8):
        # Reference for Bayesian optimization: https://github.com/thuijskens/bayesian-optimization
        self.max_iters = max_iters
        self.obj_fun = obj_fun
        self.bounds = bounds
        self.x0 = x0
        self.n_pre_samples = n_pre_samples
        self.gp_params = gp_params
        self.random_search = random_search
        self.alpha = alpha
        self.epsilon = epsilon
    
    def expected_improvement(self, x, gaussian_process, evaluated_loss, greater_is_better=True, n_params=2):
        
        x_to_predict = x.reshape(-1, n_params)
        mean, sigma = gaussian_process.predict(x_to_predict, return_std=True)
        if greater_is_better:
            loss_optimum = np.max(evaluated_loss)
        else:
            loss_optimum = np.min(evaluated_loss)
        scaling_factor = (-1) ** (not greater_is_better)
        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mean - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mean - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] == 0.0
        return -1 * expected_improvement
        
    def sample_next_hyperparameter(self, acquisition_func, gaussian_process, evaluated_loss, bounds, n_restarts, greater_is_better=True):
        
        best_x = None
        best_acquisition_value = 1
        n_params = bounds.shape[0]
        for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
            
            res = minimize(fun=acquisition_func,
                           x0=starting_point.reshape(1, -1),
                           bounds=bounds,
                           method='SLSQP',
                           args=(gaussian_process, evaluated_loss, greater_is_better, n_params))
            
            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x
        
        return best_x
        
    def bayesian_optimization(self):
        
        x_list = []
        y_list = []
        n_params = self.bounds.shape[0]
        
        if self.x0 is None:
            for params in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.n_pre_samples, self.bounds.shape[0])):
                x_list.append(params)
                y_list.append(self.obj_fun(params))
        else:
            for params in self.x0:
                x_list.append(params)
                y_list.append(self.obj_fun(params))
        
        xp = np.array(x_list)
        yp = np.array(y_list)
        
        # Create the GP
        if self.gp_params is not None:
            model = gp.GaussianProcessRegressor(**self.gp_params)
        else:
            kernel = gp.kernels.Matern()
            model = gp.GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, n_restarts_optimizer=10, normalize_y=True, random_state=1)
            
        for n in range(self.max_iters):
            model.fit(xp, yp)
            # Sample next hyperparameter
            if self.random_search:
                x_random = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.random_search, n_params))
                ei = -1 * self.expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
                next_sample = x_random[np.argmax(ei), :]
            else:
                next_sample = self.sample_next_hyperparameter(self.expected_improvement, model, yp, bounds=self.bounds, n_restarts=100, greater_is_better=True)
                
            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            if np.any(np.abs(next_sample - xp) <= self.epsilon):
                next_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.bounds.shape[0])
                
            # Sample loss for new set of parameters
            cv_score = self.obj_fun(next_sample)
    
            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)
    
            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)
    
        return xp, yp
    
class solver():
    
    def __init__(self, t0, t3, v_f, v_mu, critical_occupancy, num_of_lanes_at_bottleneck,
                 flow_raw, occupancy_raw, speed_raw, lanes_raw, distance_raw, 
                 start_location_index, end_location_index, start_time_index, end_time_index):
        self.t0 = t0
        self.t3 = t3
        self.v_f = v_f
        self.v_mu = v_mu
        self.P = t3 - t0
        self.factor_virQueue2phyQueue = 1 - v_mu/v_f
        self.num_of_lanes_at_bottleneck = num_of_lanes_at_bottleneck
        self.start_location_index = start_location_index
        self.end_location_index = start_location_index
        self.start_time_index = start_time_index
        self.end_time_index = end_time_index
        self.get_data = get_data()
        self.flow, self.density, self.speed, self.lanes, self.distance = self.get_data.get_useful_data(flow_raw, occupancy_raw, speed_raw, lanes_raw, distance_raw, 
                                                                                                       start_location_index, end_location_index, start_time_index, end_time_index)
        self.critical_density = self.get_data.calculate_density_with_occupancy(critical_occupancy)
        self.obsCumulativeDeparture = self.get_data.get_ObsCumulativeDepartureCurve(flow_raw, start_location_index-1, start_time_index, end_time_index)
        self.obsQueue = self.get_data.get_ObsQueue(self.density, self.lanes, self.distance, self.critical_density)
        self.obsDelay = self.get_data.get_ObsDealy(self.speed, self.distance, v_f)
        self.t = np.linspace(t0, t3, num=len(self.obsQueue))
        
    def get_mu(self):
        # Since mu is assumed to be a constant and it is bounded at t0 and t3, we can calculate it by the mean of the obsCumulativeDeparture during the congestion period.
        mu = self.obsCumulativeDeparture[-1]/len(self.obsCumulativeDeparture)*12
        return mu
    
    def get_obsQueue(self):
        return self.obsQueue
    
    def get_virtual_queue(self, x):
        gamma, m = x
        virtual_queue = np.zeros(len(self.t))
        for i in range(len(self.t)):
            virtual_queue[i] = gamma*(self.t[i] - self.t0)**2*(0.25*(self.t[i] - self.t0)**2 - 1/3*((3-4*m)/(4-6*m)+m)*(self.t3 - self.t0)*(self.t[i] - self.t0) + 1/2*(3 - 4*m)*m/(4 - 6*m)*((self.t3 - self.t0)**2))
        return virtual_queue
    
    def get_physical_queue(self, x):
        virtual_queue = self.get_virtual_queue(x)
        physical_queue = 1/self.factor_virQueue2phyQueue*virtual_queue
        return physical_queue
    
    def get_delay(self, x):
        mu = self.get_mu()
        virtual_queue = self.get_virtual_queue(x)
        delay = virtual_queue/mu
        return delay
    
    def virtual_queue_first_order_derivative(self, x, t):
        gamma, m = x
        Q_wrt_gamma = (t - self.t0)**2*(0.25*(t - self.t0)**2 - 1/3*((3-4*m)/(4-6*m)+m)*(self.t3 - self.t0)*(t - self.t0) + 1/2*(3 - 4*m)*m/(4 - 6*m)*((self.t3 - self.t0)**2))
        Q_wrt_m = (self.t3 - self.t0)*(self.t3 - t)*(t - self.t0)**2*(6*m**2 - 8*m + 3)/(2*(2 - 3*m)**2)
        return np.asarray([Q_wrt_gamma, Q_wrt_m])
    
    def virtual_queue_second_order_derivative(self, x, t):
        gamma, m = x
        Q_wrt_gamma2 = 0
        Q_wrt_gamma_m = (self.t3 - t)*(t - self.t0)**2 * (6*m**2 - 8*m + 3)/(2*(2 - 3*m)**2)
        Q_wrt_m2 = gamma * (self.t3 - self.t0) * (self.t3 - t) * (t - self.t0)**2 / (2 - 3*m)**3
        return np.asarray([[Q_wrt_gamma2, Q_wrt_gamma_m], [Q_wrt_gamma_m, Q_wrt_m2]])
    
    def physical_queue_first_order_derivative(self, x, t):
        virtual_queue_first_order_derivative = self.virtual_queue_first_order_derivative(x, t)
        physical_queue_first_order_derivative = 1/self.factor_virQueue2phyQueue * virtual_queue_first_order_derivative
        return physical_queue_first_order_derivative
    
    def physical_queue_second_order_derivative(self, x, t):
        virtual_queue_second_order_derivative = self.virtual_queue_second_order_derivative(x, t)
        physical_queue_second_order_derivative = 1/self.factor_virQueue2phyQueue * virtual_queue_second_order_derivative
        return physical_queue_second_order_derivative
    
    def Z_first_order_derivative(self, x):
        Z_first_order_derivative_sum = np.asarray([0., 0.])
        physical_queue = self.get_physical_queue(x)
        for i in range(len(self.t)):
            physical_queue_first_order_derivative = self.physical_queue_first_order_derivative(x, self.t[i])
            Z_first_order_derivative = 2 * (physical_queue[i] - self.obsQueue[i]) * physical_queue_first_order_derivative
            Z_first_order_derivative_sum += Z_first_order_derivative
        return Z_first_order_derivative_sum
    
    def Z_second_order_derivative(self, x):
        Z_second_order_derivative_sum = np.asarray([[0., 0.], [0., 0.]])
        physical_queue = self.get_physical_queue(x)
        for i in range(len(self.t)):
            physical_queue_second_order_derivative = self.physical_queue_second_order_derivative(x, self.t[i])
            Z_second_order_derivative = 2 * (1 + physical_queue[i] - self.obsQueue[i]) * physical_queue_second_order_derivative
            Z_second_order_derivative_sum += Z_second_order_derivative
        return Z_second_order_derivative_sum
    
    def cubic_model_Q(self, x):
        # parameters
        gamma = x[0]
        m = x[1]
        # initialization
        Q_t = np.zeros(len(self.obsQueue))
        # theoretical values
        for i in range(len(self.obsCumulativeDeparture)):
            Q_t[i] = 1/self.factor_virQueue2phyQueue*gamma*(self.t[i] - self.t0)**2*(0.25*(self.t[i] - self.t0)**2 - 1/3*((3-4*m)/(4-6*m)+m)*(self.t3 - self.t0)*(self.t[i] - self.t0) + 1/2*(3 - 4*m)*m/(4 - 6*m)*((self.t3 - self.t0)**2))
        obj_fun = np.sum((Q_t - self.obsQueue)**2)
        return obj_fun
    
    def constraint1(self, x):
        mu = self.get_mu()
        gamma = x[0]
        m = x[1]
        t2 = self.t0 + m*(self.t3 - self.t0)
        t_bar = self.t0 + (3 - 4*m)*(self.t3 - self.t0)/(4 - 6*m)
        inflow_rate = gamma*(self.t - self.t0)*(self.t - t2)*(self.t - t_bar) + mu
        return inflow_rate
    
    def bounds(self):
        bounds = np.asarray([[1, 20], [0.5, 0.666]])  # lower and upper bounds for gamma and m
        return bounds
    
    def initial_value(self):
        x0 = np.asarray([10.0, 0.58])
        return x0
    
    def multiple_initial_values(self):
        bounds = self.bounds()
        gamma_list = list(np.repeat(np.linspace(bounds[0, 0], bounds[0, 1], num=5, endpoint=True), 5))
        m_list = list(np.linspace(bounds[1, 0], bounds[1, 1], num=5, endpoint=True))*5
        multiple_x0 = list(zip([gamma for gamma in gamma_list], [m for m in m_list]))
        return multiple_x0
    
    def calibration_with_newton(self):
        x0 = self.initial_value()
        solution = minimize(self.cubic_model_Q, x0, method='Newton-CG', jac = self.Z_first_order_derivative, hess = self.Z_second_order_derivative, 
                            options={'maxiter': 1000, 'xtol': 1e-12, 'eps': 1e-12, 'disp': False})
        gamma, m = solution.x
        obj = solution.fun
        return gamma, m, obj
    
    def calibration_with_newton_multistart(self):
        multiple_x0 = self.multiple_initial_values()
        fun_value = np.inf
        for x0 in multiple_x0:
            solution = minimize(self.cubic_model_Q, x0, method='Newton-CG', jac = self.Z_first_order_derivative, hess = self.Z_second_order_derivative, 
                                options={'maxiter': 1000, 'xtol': 1e-12, 'eps': 1e-12, 'disp': False})
            if solution.fun < fun_value:
                fun_value = solution.fun
                gamma, m = solution.x
        obj = self.cubic_model_Q([gamma, m])
        return gamma, m, obj
    
    def calibration_with_SLSQP(self):
        x0 = self.initial_value()
        bnds = self.bounds()
        con1 = {'type': 'ineq', 'fun': self.constraint1}
        cons = ([con1])
        solution = minimize(self.cubic_model_Q, x0, method='SLSQP', jac = self.Z_first_order_derivative, bounds=bnds, constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-12, 'disp': False})
        gamma, m = solution.x
        obj = solution.fun
        return gamma, m, obj
    
    def calibration_with_SLSQP_multistart(self):
        bnds = self.bounds()
        con1 = {'type': 'ineq', 'fun': self.constraint1}
        cons = ([con1])
        multiple_x0 = self.multiple_initial_values()
        fun_value = np.inf
        for x0 in multiple_x0:
            solution = minimize(self.cubic_model_Q, x0, method='SLSQP', jac = self.Z_first_order_derivative, bounds=bnds, constraints=cons,
                                options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-12, 'disp': False})
            if solution.fun < fun_value:
                fun_value = solution.fun
                gamma, m = solution.x
        obj = self.cubic_model_Q([gamma, m])
        return gamma, m, obj
    
    def calibration_with_Adam(self):
        adam = Adam_optimization(self.cubic_model_Q, self.Z_first_order_derivative, self.bounds(), self.initial_value())
        solutions, scores = adam.adam()
        gamma, m = solutions[np.argmin(scores)]
        obj = min(scores)
        return gamma, m, obj
    
    def calibration_with_Adam_multistart(self):
        multiple_x0 = self.multiple_initial_values()
        fun_value = np.inf
        for x0 in multiple_x0:
            adam = Adam_optimization(self.cubic_model_Q, self.Z_first_order_derivative, self.bounds(), x0)
            solutions, scores = adam.adam()
            if min(scores) < fun_value:
                fun_value = min(scores)
                gamma, m = solutions[np.argmin(scores)]
        obj = self.cubic_model_Q([gamma, m])
        return gamma, m, obj
    
    def calibration_with_Bayesian_optimization(self):
        max_iters = 200
        obj_fun = self.cubic_model_Q
        bounds = self.bounds()
        gamma_list = list(np.repeat(np.linspace(bounds[0, 0], bounds[0, 1], num=5, endpoint=True), 5))
        m_list = list(np.linspace(bounds[1, 0], bounds[1, 1], num=5, endpoint=True))*5
        x0 = list(zip([gamma for gamma in gamma_list], [m for m in m_list]))
        bo = BayesianOptimization(max_iters, obj_fun, bounds, x0)
        solutions, scores = bo.bayesian_optimization()
        gamma = solutions[np.argmin(scores)][0]
        m = solutions[np.argmin(scores)][1]
        obj = min(scores)
        return gamma, m, obj
    
    def plot_CumulativeDepartureCurve(self, mu):
        # Plot cumulative departure curve
        fig = plt.figure()
        plt.plot(self.t, self.obsCumulativeDeparture,'b:', linewidth = 2, label = 'Observed values')
        plt.plot(self.t, mu*(self.t - self.t0 + 1/12), 'r-', linewidth = 3, label = 'Calibrated values')
        plt.xticks([13+i for i in range(0,8)], 
                    labels=['13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00'], fontsize=10)
        plt.xlabel('Time')
        plt.ylabel('Cumulative number of vehicles')
        plt.legend(loc=0)
        plt.title('Calibration results of the cumulative departure curve')
        fig.savefig('../Figures/Case 1/Calibration of the cumulative departure curve.png', dpi=300, bbox_inches='tight')
        
    def plot_InflowRate(self, mu, gamma, m):
        # Plot inflow rate
        fig = plt.figure()
        inflow_rate = gamma*(self.t - self.t0)*(self.t - self.t0 - m*(self.t3 - self.t0))*(self.t - self.t0 - (3-4*m)*(self.t3 - self.t0)/(4 - 6*m)) + mu
        plt.plot(self.t, inflow_rate/self.num_of_lanes_at_bottleneck, 'r-', linewidth=3, label = 'Inflow rate')
        plt.hlines(mu/self.num_of_lanes_at_bottleneck, self.t0, self.t3, colors = 'b', linestyles = 'dashed', linewidth=2, label = '$\mu$')
        plt.xticks([13+i for i in range(0,8)], 
                    labels=['13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00'], fontsize=10)
        plt.ylabel('Number of vehicles (per hour per lane)', fontsize=12)
        plt.ylim((700, 1150))
        plt.legend(loc=0)
        plt.title('Calibrated arrival rate and $\mu$', fontsize=16)
        fig.savefig('../Figures/Case 1/Inflow rate.png', dpi=300, bbox_inches='tight')
    
    def plot_QueueLength(self, mu, gamma, m):
        # Plot queue length
        fig = plt.figure()
        calPhyQueue = 1/self.factor_virQueue2phyQueue*gamma*(self.t - self.t0)**2*(0.25*(self.t - self.t0)**2 - 1/3*((3-4*m)/(4-6*m) + m)*(self.t3 - self.t0)*(self.t - self.t0) + 1/2*(3-4*m)*m/(4-6*m)*((self.t3 - self.t0)**2))
        plt.scatter(self.t, self.obsQueue, s = 2, marker='o', c='b', edgecolors='b', label='Observed queue')
        plt.plot(self.t, calPhyQueue, 'r-', linewidth=3, label = 'Calibrated physical queue length')
        plt.xticks([13+i for i in range(0,8)], 
                    labels=['13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00'], fontsize=10)
        plt.ylabel('Number of vehicles', fontsize=12)
        plt.legend(loc=2)
        plt.title('Calibration results of the queue length', fontsize=16)
        fig.savefig('../Figures/Case 1/Calibration of queue.png', dpi=300, bbox_inches='tight')
        
    def plot_DelayTime(self, mu, gamma, m):
        # Plot delay time
        fig = plt.figure()
        calDelay = 60*(1/mu*gamma*(self.t - self.t0)**2*(0.25*(self.t - self.t0)**2 - 1/3*((3-4*m)/(4-6*m)+m)*(self.t3-self.t0)*(self.t-self.t0) + 1/2*(3-4*m)*m/(4-6*m)*((self.t3-self.t0)**2))) # unit: minute
        plt.scatter(self.t, self.obsDelay, s = 2, marker='o', c='b', edgecolors='b', label='Observed delay')
        plt.plot(self.t, calDelay, 'r-', linewidth=3, label = 'Calibrated delay')
        plt.xticks([13+i for i in range(0,8)], 
                    labels=['13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00'], fontsize=10)
        plt.xlabel('Time')
        plt.ylabel('Delay time (min)')
        plt.legend(loc=0)
        plt.title('Calibration results of the delay time')
        plt.show()
        fig.savefig('../Figures/Case 1/Calibration of delay.png', dpi=300, bbox_inches='tight')
        
    def plot_MU(self, flow_raw_cpoy, mu):
        fig = plt.figure()
        downstream_flow = flow_raw_cpoy.iloc[self.start_location_index-1, self.start_time_index:self.end_time_index]
        plt.scatter(self.t, 12/self.num_of_lanes_at_bottleneck*downstream_flow, s = 2, marker='o', c='b', edgecolors='b', label='Observed volume')
        plt.hlines(mu/self.num_of_lanes_at_bottleneck, self.t0, self.t3, colors = 'r', linewidth=3, label = '$\mu$')
        plt.xticks([13+i for i in range(0,8)], 
                    labels=['13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00'], fontsize=10)
        plt.ylabel('Volume (per hour per lane)', fontsize=12)
        plt.ylim((600, 1400))
        plt.legend(loc=0)
        plt.title('Calibration results of $\mu$', fontsize=16)
        plt.show()
        fig.savefig('../Figures/Case 1/Calibration of mu.png', dpi=300, bbox_inches='tight')

class get_metrics():
    
    def __init__(self, observations, fit_values):
        self.y_true = observations
        self.y_pred = fit_values
    
    def metrics(self):
        MSE = mean_squared_error(self.y_true, self.y_pred)
        MAE = mean_absolute_error(self.y_true, self.y_pred)
        R2 = r2_score(self.y_true, self.y_pred)
        return MSE, MAE, R2
    
    def relative_error(self, y_pred, y_true):
        relative_error = abs((y_pred-y_true)/y_true)
        return relative_error

if __name__ == '__main__':
    
    # input data
    dataset_dir = '../Dataset/Dataset 1/'
    flow_raw = pd.read_csv(dataset_dir+'flow.csv', index_col=0, header=0)
    flow_raw_copy = pd.read_csv(dataset_dir+'flow.csv', index_col=0, header=0)
    speed_raw = pd.read_csv(dataset_dir+'speed.csv', index_col=0, header=0)
    occupancy_raw = pd.read_csv(dataset_dir+'occupancy.csv', index_col=0, header=0)
    distance_raw = pd.read_csv(dataset_dir+'distance.csv')
    lanes_raw = pd.read_csv(dataset_dir+'lanes.csv', header=0)
    num_of_lanes_at_bottleneck = 4
    critical_occupancy = 0.13   # critical occupancy, which is observed from the flow vs occupancy plot in the data prepare process
    t0 = 10+38/12   # t0=13:10:00, observed from the queue profile generated from step 1
    t3 = 10+117/12  # t3=19:44:59, observed from the queue profile generated from step 1
    v_f = 53        # Observed from the figure of speed and occupancy at the bottleneck generated from step 1
    v_mu = 25       # Observed from the figure of speed and occupancy at different locations generated from step 1
    start_location_index = 5    # Observed from the queue profile generated from step 1
    end_location_index = 15     # Observed from the queue profile generated from step 1
    start_time_index = 38       # Observed from the queue profile generated from step 1
    end_time_index = 117        # Observed from the queue profile generated from step 1
    # instantiation
    solver = solver(t0, t3, v_f, v_mu, critical_occupancy, num_of_lanes_at_bottleneck,
                    flow_raw, occupancy_raw, speed_raw, lanes_raw, distance_raw, 
                    start_location_index, end_location_index, start_time_index, end_time_index)
    mu = solver.get_mu()
    obs_physical_queue = solver.get_obsQueue()
    '''
    # Calibration with Newton's method
    gamma, m, obj = solver.calibration_with_newton()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with Newton method: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    '''
    '''
    # Calibration with multi-statr Newton's method
    gamma, m, obj = solver.calibration_with_newton_multistart()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with multi-start Newton method: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    '''
    '''
    # Calibration with SLSQP
    gamma, m, obj = solver.calibration_with_SLSQP()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with SLSQP: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    '''
    '''
    # Calibration with multi-start SLSQP
    gamma, m, obj = solver.calibration_with_SLSQP_multistart()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with multi-start SLSQP: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    '''
    '''
    # Calibration with Adam
    gamma, m, obj = solver.calibration_with_Adam()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with Adam: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    '''
    # Calibration with multi-start Adam
    gamma, m, obj = solver.calibration_with_Adam_multistart()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with multi-start Adam: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    # Plot results
    solver.plot_CumulativeDepartureCurve(mu)
    solver.plot_InflowRate(mu, gamma, m)
    solver.plot_QueueLength(mu, gamma, m)
    solver.plot_DelayTime(mu, gamma, m)
    solver.plot_MU(flow_raw_copy, mu)
    '''
    # Calibration with Bayesion optimization
    gamma, m, obj = solver.calibration_with_Bayesian_optimization()
    pred_physical_queue = solver.get_physical_queue([gamma, m])
    metrics = get_metrics(obs_physical_queue, pred_physical_queue)
    MSE, MAE, R2 = metrics.metrics()
    relative_error = metrics.relative_error(obj, 409795)
    print('\nCalibration results with Bayesian optimization: ')
    print('Discharge rate = {} veh/h'.format(int(round(mu, 2))))
    print('Shape parameter = {} veh/(h^4)'.format(round(gamma, 3)))
    print('Oversaturation factor = {}'.format(round(m, 3)))
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    print('Relative error = {:.3f}'.format(relative_error))
    '''
    