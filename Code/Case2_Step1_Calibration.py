# -*- coding: utf-8 -*-

# Citation: Cheng, Q., Liu, Z., Guo, J., Wu, X., Pendyala, R., Belezamo, B., 
# & Zhou, X. 2022. Estimating key traffic state parameters through parsimonious
# spatial queue models. Under review with Transportation Research Part C.

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


class Adam_optimization():
    
    def __init__(self, objective, first_order_derivative, bounds, x0):
        self.n_iter = 500
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


class solver():
    
    def __init__(self, dataset_dir):
        
        # Bottleneck location observed from speed data: 23 (35974)
        # t0 = 07:00:00, t_bar = 08:54:59
        # Only single bottleneck (location 23)，during the congestion period
        self.t0_1 = 7                          # t0_1 = 07:00:00, start time of the first bottleneck
        self.t_bar_1 = 8+55/60                 # t_bar_1 = 08:54:59, end time of the first bottleneck
        self.t0_2 = self.t_bar_1               # t0_2 = 08:55:00, start time of the second bottleneck
        self.t3_2 = 10+10/60                   # t_bar_2 = 10:09:59
        self.v_f = 45                          # empirically and observed from the speed data, the average speed at t0
        self.probe_veh_location_start = 23     # LinkID 35974(23) in the probe vehicle detectors corresponds to the linkid HI8027c(9) in the rtms locations
        self.probe_veh_location_end = 35       # Single bottleneck (location 23~34)
        self.probe_veh_time_start = 12
        self.probe_veh_time_end = 50
        self.rtms_time_start = 30              # Congestion period (07:00:00~08:54:59)   08:54:59 is t_bar_1
        self.rtms_time_end = 88
        self.dataset_dir = dataset_dir
        self.time_1 = np.linspace(self.t0_1, self.t_bar_1, num=int((self.t_bar_1 - self.t0_1)*12) + 1)  # for the delay calibration of the first bottleneck
        self.time_2 = np.linspace(self.t0_2, self.t3_2, num=int((self.t3_2 - self.t0_2)*12))            # for the delay calibration of the second bottleneck
        self.num_of_lanes_at_bottleneck = 2
        
        # Load raw data
        # Flow data is obtained from RTMS, while speed data is from probe vehicles.
        self.flow_raw = pd.read_csv(dataset_dir + 'rtms_volume_0608_8027c.csv', index_col=0, header=0)
        self.speed_raw = pd.read_csv(dataset_dir + 'probe_vehicle_speed_0608_morning.csv', index_col=0, header=0)
        self.probe_veh_dict = pd.read_excel(dataset_dir + 'probe_vehicle_dictionary.xlsx', index_col=3, header=0)
        self.rtms_dict = pd.read_excel(dataset_dir + 'rtms_dictionary.xlsx')
        self.probe_veh_distance_raw = self.probe_veh_dict[['Length']]
        self.probe_veh_distance_raw = self.probe_veh_distance_raw.sort_index()
        
        self.flow = self.flow_raw.iloc[self.rtms_time_start:self.rtms_time_end, :]
        self.speed = self.speed_raw.iloc[self.probe_veh_location_start:self.probe_veh_location_end, self.probe_veh_time_start:self.probe_veh_time_end]
        self.distance = self.probe_veh_distance_raw.iloc[self.probe_veh_location_start:self.probe_veh_location_end,:]
    
    def get_mu(self):
        
        # Since mu is assumed to be a constant and it is bounded at t0 and t3, we can calculate it by the mean of the obsCumulativeDeparture during the congestion period.
        obsCumulativeDeparture = self.flow.cumsum()
        mu = obsCumulativeDeparture.iloc[-1,:]/len(obsCumulativeDeparture)*30
        return np.float(mu)
    
    def get_ObsDelay(self):
        
        # Obtain the observed delay based on the speed and distance
        travel_time_sublink = self.speed*0
        for i in range(self.speed.shape[0]):
            for j in range(self.speed.shape[1]):
                travel_time_sublink.iloc[i, j] = (1/1000)*self.distance.iloc[i, 0]/self.speed.iloc[i, j]*60
        travel_time = travel_time_sublink.apply(lambda x: x.sum(), axis=0)
        fftt = self.distance.sum()/1000/self.v_f*60
        obsDelay = travel_time * 0
        for i in range(travel_time.shape[0]):
            obsDelay.iloc[i,] = np.array(travel_time.iloc[i,]) - fftt
        return obsDelay
    
    def get_theoreticalDelay(self, x):
        
        # Parameters
        mu = self.get_mu()
        gamma_1 = x[0]
        gamma_2 = x[1]
        t2_1 = x[2]
        t2_2 = x[3]
        t_bar_2 = x[4]
        w_t = np.zeros(len(self.time_1) + len(self.time_2))
        for i in range(len(self.time_1)):
            w_t[i] = (1/mu)*gamma_1*((self.time_1[i]-self.t0_1)**2)*(0.25*(self.time_1[i]-self.t0_1)**2 + 1/3*(2*self.t0_1-t2_1-self.t_bar_1)*(self.time_1[i]-self.t0_1) + 1/2*(self.t0_1-t2_1)*(self.t0_1-self.t_bar_1))
        for i in range(len(self.time_1), len(self.time_1)+len(self.time_2)):
            w_t[i] = w_t[len(self.time_1)-1] + (1/mu)*gamma_2*((self.time_2[i-len(self.time_1)]-self.t0_2)**2)*(0.25*(self.time_2[i-len(self.time_1)]-self.t0_2)**2 + 1/3*(2*self.t0_2-t2_2-t_bar_2)*(self.time_2[i-len(self.time_1)]-self.t0_2) + 1/2*(self.t0_2-t2_2)*(self.t0_2-t_bar_2))
        w_t = w_t*60    # Unit: minute
        return w_t
    
    def get_metrics(self, x):
        y_true = self.get_ObsDelay()[0:len(self.time_1)]
        y_pred = self.get_theoreticalDelay(x)[0:len(self.time_1)]
        MSE = mean_squared_error(y_true, y_pred)
        MAE = mean_absolute_error(y_true, y_pred)
        R2 = r2_score(y_true, y_pred)
        return MSE, MAE, R2
    
    def cubic_model_W(self, x):
        
        obsDelay = self.get_ObsDelay()
        w_t = self.get_theoreticalDelay(x)
        
        obj_fun = np.sum((w_t - obsDelay)**2)
        
        return obj_fun
    
    def constraint1(self, x):  # inflow should be positive
        mu = self.get_mu()
        gamma_1 = x[0]
        gamma_2 = x[1]
        t2_1 = x[2]
        t2_2 = x[3]
        t_bar_2 = x[4]
        inflow_rate_1 = gamma_1*(self.time_1 - self.t0_1)*(self.time_1 - t2_1)*(self.time_1 - self.t_bar_1) + mu
        inflow_rate_2 = gamma_2*(self.time_2 - self.t0_2)*(self.time_2 - t2_2)*(self.time_2 - t_bar_2) + mu
        return [inflow_rate_1, inflow_rate_2]
    
    def bounds(self):
        t_bar_2 = 12
        bnds_gamma_1 = [0, 2000]
        bnds_gamma_2 = [0, 2000]
        bnds_t2_1 = [self.t0_1, self.t_bar_1]
        bnds_t2_2 = [self.t0_2, self.t3_2]
        bnds_t_bar_2 = [self.t3_2, t_bar_2]
        bnds = np.asarray([bnds_gamma_1, bnds_gamma_2, bnds_t2_1, bnds_t2_2, bnds_t_bar_2])
        return bnds
    
    def initial_value(self):
        x0 = np.array([1126., 90., 7.8, 9.7, 11.2])
        return x0
    
    def multiple_initial_values(self):
        first_col = np.array([20*i for i in range(1,100)]).reshape((99,1))
        other_cols = np.array([90., 7.8, 9.7, 11.2]).reshape((1,4))
        other_cols = np.tile(other_cols, (99,1))
        multiple_x0 = np.hstack((first_col, other_cols))
        return multiple_x0
        
    def multiple_initial_values2(self):
        first_col = np.array([1110+i for i in range(0,20)]).reshape((20,1))
        other_cols = np.array([90., 7.8, 9.7, 11.2]).reshape((1,4))
        other_cols = np.tile(other_cols, (20,1))
        multiple_x0 = np.hstack((first_col, other_cols))
        return multiple_x0
        
    def delay_first_order_derivative(self, x, t):
        mu = self.get_mu()
        gamma_1, gamma_2, t2_1, t2_2, t_bar_2 = x
        if t <= self.t_bar_1:
            W_wrt_gamma_1 = 1/mu*(t - self.t0_1)**2*(0.25*(t - self.t0_1)**2 + 1/3*(2*self.t0_1 - t2_1 -self.t_bar_1)*(t - self.t0_1) + 1/2*(self.t0_1 - t2_1)*(self.t0_1 - self.t_bar_1))
            W_wrt_gamma_2 = 0
            W_wrt_t2_1 = 1/mu*gamma_1*(t - self.t0_1)**2*(-1/3*(t - self.t0_1) - 1/2*(self.t0_1 - self.t_bar_1))
            W_wrt_t2_2 = 0
            W_wrt_t_bar_2 = 0
        else:
            W_wrt_gamma_1 = 0
            W_wrt_gamma_2 = 1/mu*(t - self.t0_2)**2*(0.25*(t - self.t0_2)**2 + 1/3*(2*self.t0_2 - t2_2 -self.t_bar_2)*(t - self.t0_2) + 1/2*(self.t0_2 - t2_2)*(self.t0_2 - t_bar_2))
            W_wrt_t2_1 = 0
            W_wrt_t2_2 = 1/mu*gamma_2*(t - self.t0_2)**2*(-1/3*(t - self.t0_2) - 1/2*(self.t0_2 - t_bar_2))
            W_wrt_t_bar_2 = 1/mu*gamma_2*(t - self.t0_2)**2*(-1/3*(t - self.t0_2) - 1/2*(self.t0_2 - t2_2))
        return np.asarray([W_wrt_gamma_1, W_wrt_gamma_2, W_wrt_t2_1, W_wrt_t2_2, W_wrt_t_bar_2])
    
    def Z_first_order_derivative(self, x):
        obsDelay = self.get_ObsDelay()
        theoreticalDelay = self.get_theoreticalDelay(x)
        t = np.linspace(self.t0_1, self.t_bar_1, num=len(obsDelay))
        Z_first_order_derivative_sum = np.asarray([0., 0., 0., 0., 0.])
        for i in range(len(self.time_1)+len(self.time_2)):
            delay_first_order_derivative = self.delay_first_order_derivative(x, t[i])
            Z_first_order_derivative = 2 * (theoreticalDelay[i] - obsDelay[i]) * delay_first_order_derivative
            Z_first_order_derivative_sum += Z_first_order_derivative
        return Z_first_order_derivative_sum
    
    def calibration_with_Adam(self):
        adam = Adam_optimization(self.cubic_model_W, self.Z_first_order_derivative, self.bounds(), self.initial_value())
        solutions, scores = adam.adam()
        gamma_1, gamma_2, t2_1, t2_2, t_bar_2 = solutions[np.argmin(scores)]
        obj = min(scores)
        return gamma_1, gamma_2, t2_1, t2_2, t_bar_2, obj
    
    def calibration_with_Adam_multistart(self):
        multiple_x0 = self.multiple_initial_values()
        fun_value = np.inf
        for x0 in multiple_x0:
            adam = Adam_optimization(self.cubic_model_W, self.Z_first_order_derivative, self.bounds(), x0)
            solutions, scores = adam.adam()
            if min(scores) < fun_value:
                fun_value = min(scores)
                gamma_1, gamma_2, t2_1, t2_2, t_bar_2 = solutions[np.argmin(scores)]
        obj = self.cubic_model_W([gamma_1, gamma_2, t2_1, t2_2, t_bar_2])
        return gamma_1, gamma_2, t2_1, t2_2, t_bar_2, obj
    
    def get_m_1(self, t2_1):
        m_1 = (t2_1 - self.t0_1)/(self.t_bar_1 - self.t0_1)
        return m_1
        
    def plot_InflowRate(self, mu, gamma_1, t2_1):
        # Plot inflow rate
        t = np.linspace(self.t0_1, self.t_bar_1, num=1000)
        fig = plt.figure()
        inflow_rate = gamma_1*(t - self.t0_1)*(t - t2_1)*(t - self.t_bar_1) + mu
        plt.plot(t, inflow_rate/self.num_of_lanes_at_bottleneck, 'r-', linewidth=3, label = 'Inflow rate')
        plt.hlines(mu/self.num_of_lanes_at_bottleneck, self.t0_1, self.t_bar_1, colors = 'b', linestyles = 'dashed', linewidth=2, label = '$\mu$')
        plt.xticks([7+0.25*i for i in range(0,9)], 
                    labels=['07:00','07:15','07:30','07:45','08:00','08:15','08:30','08:45','09:00'], fontsize=10)
        plt.ylabel('Number of vehicles (per hour per lane)', fontsize=12)
        plt.ylim((700, 1150))
        plt.legend(loc=0)
        plt.title('Calibrated arrival rate and $\mu$', fontsize=16)
        fig.savefig('../Figures/Case 2/Inflow rate.png', dpi=300, bbox_inches='tight')
        
    def plot_DelayTime(self, mu, x):
        # Plot delay time
        fig = plt.figure()
        obsDelay = self.get_ObsDelay()
        theoreticalDelay = self.get_theoreticalDelay(x)
        plt.scatter(self.time_1, obsDelay.iloc[0:len(self.time_1)], s = 2, marker='o', c='b', edgecolors='b', label='Observed delay')
        plt.plot(self.time_1, theoreticalDelay[0:len(self.time_1)], 'r-', linewidth=3, label = 'Calibrated delay')
        plt.xticks([7+0.25*i for i in range(0,9)], 
                    labels=['07:00','07:15','07:30','07:45','08:00','08:15','08:30','08:45','09:00'], fontsize=10)
        plt.ylabel('Delay time (min)', fontsize=12)
        plt.legend(loc=0)
        plt.title('Calibration results of the delay time', fontsize=16)
        plt.show()
        fig.savefig('../Figures/Case 2/Calibration of delay.png', dpi=300, bbox_inches='tight')
        
    def plot_MU(self, mu):
        fig = plt.figure()
        downstream_flow = np.array(self.flow)*30/self.num_of_lanes_at_bottleneck
        t = np.linspace(self.t0_1, self.t_bar_1, num=len(downstream_flow))
        plt.scatter(t, downstream_flow, s = 2, marker='o', c='b', edgecolors='b', label='Observed volume')
        plt.hlines(mu/self.num_of_lanes_at_bottleneck, self.t0_1, self.t_bar_1, colors = 'r', linewidth=3, label = '$\mu$')
        plt.xticks([7+0.25*i for i in range(0,9)], 
                    labels=['07:00','07:15','07:30','07:45','08:00','08:15','08:30','08:45','09:00'], fontsize=10)
        plt.ylabel('Volume (per hour per lane)', fontsize=12)
        plt.ylim((600, 1400))
        plt.legend(loc=0)
        plt.title('Calibration results of $\mu$', fontsize=16)
        plt.show()
        fig.savefig('../Figures/Case 2/Calibration of mu.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    
    dataset_dir = '../Dataset/Dataset 2/'
    solver = solver(dataset_dir)
    mu = solver.get_mu()
    
    gamma_1, gamma_2, t2_1, t2_2, t_bar_2, obj = solver.calibration_with_Adam()
    print('Solution: ')
    print('mu = {:.2f}'.format(mu))
    print('gamma_1 = {:.2f}'.format(gamma_1))
    print('gamma_2 = {:.2f}'.format(gamma_2))
    print('t2_1 = {:.2f}'.format(t2_1))
    print('t2_2 = {:.2f}'.format(t2_2))
    print('t_bar_2 = {:.2f}'.format(t_bar_2))
    print('obj = {:.2f}'.format(obj))
    
    # Calculate m_1
    m_1 = solver.get_m_1(t2_1)
    print('m_1 = {:.3f}'.format(m_1))
    
    # Calculate metrics
    MSE, MAE, R2 = solver.get_metrics(x=[gamma_1, gamma_2, t2_1, t2_2, t_bar_2])
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    
    '''
    # Multistart calibraion with Adam
    gamma_1, gamma_2, t2_1, t2_2, t_bar_2, obj = solver.calibration_with_Adam_multistart()
    print('Solution: ')
    print('mu = {:.2f}'.format(mu))
    print('gamma_1 = {:.2f}'.format(gamma_1))
    print('gamma_2 = {:.2f}'.format(gamma_2))
    print('t2_1 = {:.2f}'.format(t2_1))
    print('t2_2 = {:.2f}'.format(t2_2))
    print('t_bar_2 = {:.2f}'.format(t_bar_2))
    print('obj = {:.2f}'.format(obj))
    # Calculate m_1
    m_1 = solver.get_m_1(t2_1)
    print('m_1 = {:.3f}'.format(m_1))
    # Calculate metrics
    MSE, MAE, R2 = solver.get_metrics(x=[gamma_1, gamma_2, t2_1, t2_2, t_bar_2])
    print('MSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}'.format(MSE, MAE, R2))
    '''
    
    # Plot the results
    solver.plot_InflowRate(mu, gamma_1, t2_1)
    solver.plot_DelayTime(mu, [gamma_1, gamma_2, t2_1, t2_2, t_bar_2])
    solver.plot_MU(mu)
