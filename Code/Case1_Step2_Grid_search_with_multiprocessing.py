# -*- coding: utf-8 -*-

# Citation: Cheng, Q., Liu, Z., Guo, J., Wu, X., Pendyala, R., Belezamo, B., 
# & Zhou, X. 2022. Estimating key traffic state parameters through parsimonious
# spatial queue models. Under review with Transportation Research Part C.

import numpy as np
import pandas as pd
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
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
factor_virQueue2phyQueue = 1 - v_mu/v_f
start_location_index = 5    # Observed from the queue profile generated from step 1
end_location_index = 15     # Observed from the queue profile generated from step 1
start_time_index = 38       # Observed from the queue profile generated from step 1
end_time_index = 117        # Observed from the queue profile generated from step 1
get_data = get_data()
flow, density, speed, lanes, distance = get_data.get_useful_data(flow_raw, occupancy_raw, speed_raw, lanes_raw, distance_raw, 
                                                                 start_location_index, end_location_index, start_time_index, end_time_index)
critical_density = get_data.calculate_density_with_occupancy(critical_occupancy)
obsCumulativeDeparture = get_data.get_ObsCumulativeDepartureCurve(flow_raw, start_location_index-1, start_time_index, end_time_index)
obsQueue = get_data.get_ObsQueue(density, lanes, distance, critical_density)
obsDelay = get_data.get_ObsDealy(speed, distance, v_f)
t = np.linspace(t0, t3, num=len(obsQueue))
mu = 3936

def cubic_model_Q(x):
    # parameters
    gamma = x[0]
    m = x[1]
    # initialization
    Q_t = np.zeros(len(obsQueue))
    # theoretical values
    for i in range(len(obsCumulativeDeparture)):
        Q_t[i] = 1/factor_virQueue2phyQueue*gamma*(t[i] - t0)**2*(0.25*(t[i] - t0)**2 - 1/3*((3-4*m)/(4-6*m)+m)*(t3 - t0)*(t[i] - t0) + 1/2*(3 - 4*m)*m/(4 - 6*m)*((t3 - t0)**2))

    obj_fun = np.sum((Q_t - obsQueue)**2)
    return obj_fun

def plot_grid_search_contourf(gamma, m, fun_value):
    GAMMA, M = np.meshgrid(gamma, m)
    FUN_VALUE = np.reshape(fun_value, (len(m), len(gamma)))
    # norm = cm.colors.Normalize(vmax=np.percentile(FUN_VALUE, 25), vmin=FUN_VALUE.min())
    fig, ax = plt.subplots(figsize=(10,6))
    cs = ax.contourf(GAMMA, M, FUN_VALUE, cmap='jet', levels=np.linspace(FUN_VALUE.min(),np.percentile(FUN_VALUE, 25),100), extend='both')
    ax.set_xlim(1, 20)
    ax.set_ylim(0.5, 0.666)
    ax.set_xlabel('gamma')
    ax.set_ylabel('m')
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.tick_params(labelsize=14)
    fig.colorbar(cs)
    fig.savefig('../Figures/Case 1/contour_frontpage.png', dpi=250)  # results in 160x120 px image
    plt.show()

# Grid search
def main(parameter_pairs):
    
    pool = Pool(10)
    res = pool.map(cubic_model_Q, parameter_pairs)
    pool.close()
    pool.join()
    
    return res

if __name__ == '__main__':
    
    gamma_list = list(np.repeat(np.linspace(1, 20, num=2001, endpoint=True), 167))
    m_list = list(np.linspace(0.5, 0.666, num=167, endpoint=True))*2001
    parameter_pairs = list(zip([gamma for gamma in gamma_list], [m for m in m_list]))
    
    ts = time.time()
    fun_value = main(parameter_pairs)
    np.save('../Dataset/Dataset 1//Grid search value.npy', fun_value)
    np.save('../Dataset/Dataset 1//Grid search parameters.npy', parameter_pairs)
    print('Time = %0.2f seconds' % (time.time()-ts))
    print('Minimal objective value = ', min(fun_value))
    print('Parameters = ', parameter_pairs[np.argmin(fun_value)])
    
    gamma = np.linspace(1, 20, num=2001, endpoint=True)
    m = np.linspace(0.5, 0.666, num=167, endpoint=True)
    # fun_value = np.load('../Dataset/Dataset 1//Grid search value.npy', allow_pickle=True)
    plot_grid_search_contourf(gamma, m, fun_value)

