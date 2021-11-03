# -*- coding: utf-8 -*-

# Citation: Cheng, Q., Liu, Z., Guo, J., Wu, X., Pendyala, R., Belezamo, B., 
# & Zhou, X. 2022. Estimating key traffic state parameters through parsimonious
# spatial queue models. Under review with Transportation Research Part C.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font',family='Times New Roman')


class get_data(object):
    
    def __init__(self, dataset_dir):
        
        self.dataset_dir = dataset_dir
        self.flow_dir = self.dataset_dir + 'Flow'
        self.speed_dir = self.dataset_dir + 'Speed'
        self.occupancy_dir = self.dataset_dir + 'Occupancy'
        
    def load_flow(self):
        
        flow = pd.DataFrame(columns=['Time', 'Postmile (Abs)', 'Postmile (CA)', 'VDS', 'AggFlow', '# Lane Points', '% Observed'])
        for parents, dirnames, filenames in os.walk(self.flow_dir):
            for filename in filenames:
                flow_temp = pd.read_excel(os.path.join(parents, filename))
                flow = flow.append(flow_temp, ignore_index=True)
        # type reset
        flow[['VDS', 'AggFlow', '# Lane Points']] = flow[['VDS', 'AggFlow', '# Lane Points']].astype('int')
        # to datetime
        flow['Time'] = pd.to_datetime(flow['Time'], format='%H:%M').dt.time
        return flow
    
    def load_speed(self):
        
        speed = pd.DataFrame(columns=['Time', 'Postmile (Abs)', 'Postmile (CA)', 'VDS', 'AggSpeed', '# Lane Points', '% Observed'])
        for parents, dirnames, filenames in os.walk(self.speed_dir):
            for filename in filenames:
                speed_temp = pd.read_excel(os.path.join(parents, filename))
                speed = speed.append(speed_temp, ignore_index=True)
        # type reset
        speed[['VDS', 'AggSpeed', '# Lane Points']] = speed[['VDS', 'AggSpeed', '# Lane Points']].astype('int')
        # to datetime
        speed['Time'] = pd.to_datetime(speed['Time'], format='%H:%M').dt.time
        return speed
    
    def load_occupancy(self):
        
        occupancy = pd.DataFrame(columns=['Time', 'Postmile (Abs)', 'Postmile (CA)', 'VDS', 'AggOccupancy', '# Lane Points', '% Observed'])
        for parents, dirnames, filenames in os.walk(self.occupancy_dir):
            for filename in filenames:
                occupancy_temp = pd.read_excel(os.path.join(parents, filename))
                occupancy = occupancy.append(occupancy_temp, ignore_index=True)
        # type reset
        occupancy[['VDS', '# Lane Points']] = occupancy[['VDS', '# Lane Points']].astype('int')
        occupancy[['AggOccupancy']] = occupancy[['AggOccupancy']].astype('float')
        # to datetime
        occupancy['Time'] = pd.to_datetime(occupancy['Time'], format='%H:%M').dt.time
        return occupancy
    
    def average_the_data(self, flow, speed, occupancy):
        """ average the data at the same location and time over multiple weekdays """
        
        flow_avg_temp = flow.groupby(['Time', 'Postmile (Abs)'])[['AggFlow']].mean().reset_index()
        speed_avg_temp = speed.groupby(['Time', 'Postmile (Abs)'])[['AggSpeed']].mean().reset_index()
        occupancy_avg_temp = occupancy.groupby(['Time', 'Postmile (Abs)'])[['AggOccupancy']].mean().reset_index()
        
        # reshape the data
        flow_avg = flow_avg_temp.pivot('Postmile (Abs)','Time','AggFlow')
        flow_avg.columns = pd.date_range("10:00", "20:55", freq="5min").strftime('%H:%M')
        speed_avg = speed_avg_temp.pivot('Postmile (Abs)','Time','AggSpeed')
        speed_avg.columns = pd.date_range("10:00", "20:55", freq="5min").strftime('%H:%M')
        occupancy_avg = occupancy_avg_temp.pivot('Postmile (Abs)','Time','AggOccupancy')
        occupancy_avg.columns = pd.date_range("10:00", "20:55", freq="5min").strftime('%H:%M')
        
        # sort traffic flow from botton to up (which will be intuitive for illustration of the speed profile)
        flow_avg = flow_avg.sort_index(ascending=False)
        speed_avg = speed_avg.sort_index(ascending=False)
        occupancy_avg = occupancy_avg.sort_index(ascending=False)
        
        return flow_avg, speed_avg, occupancy_avg


class plot_data():
    
    def __init__(self, flow, speed, occupancy):
        
        self.flow = flow
        self.speed = speed
        self.occupancy = occupancy
    
    def plot_queue_profile(self):
        
        fig = plt.figure(figsize=(10,5))
        sns.heatmap(self.speed, vmin=0, vmax=60, center=0)
        plt.ylabel('Vehicle detector locations (mile)', fontsize=18)
        plt.title('Speed profile at I-405N', fontsize=22)
        plt.show()
        fig.savefig('../Figures/Case 1/Queue profile.png', dpi=200, bbox_inches='tight')
    
    def plot_speed_occupancy(self):
        
        fig = plt.figure(figsize=(16,14))
        plt.title('Speed and occupancy evolution at different locations')
        plt.xticks([])
        plt.yticks([])
        for i in range(self.speed.shape[0]-2):
            ax1 = fig.add_subplot(5,4,i+1)
            self.speed.iloc[i,].plot(style='b-',fontsize=5, linewidth=2)
            plt.vlines(39,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed")
            plt.vlines(118,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed") 
            plt.hlines(53,5,131, colors = 'g', linewidth=0.5, linestyles = "dashed") 
            ax2 = ax1.twinx()
            self.occupancy.iloc[i,].plot(style='r-',fontsize=5, linewidth=2,)
        fig.savefig('../Figures/Case 1/Speed and occupancy evolution at different locations.png', dpi=300, bbox_inches='tight')
        
    def plot_speed_occupancy_near_bottleneck(self, bottleneck_location_index):
        
        # Speed and occupancy at the bottleneck
        fig = plt.figure()
        plt.title('Speed and occupancy at the bottleneck', fontdict={'size':16})
        ax1 = fig.add_subplot(111)
        self.speed.iloc[bottleneck_location_index,].plot(style='b-',fontsize=8, linewidth=2)
        plt.vlines(39,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed")
        plt.vlines(118,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed")
        plt.hlines(53,5,131, colors = 'g', linewidth=0.5, linestyles = "dashed") # free flow speed = 53 mile/hour
        plt.ylabel('Speed (mile/hour)', fontsize=14)
        plt.text(25, 10, r't0=13:10')
        plt.text(105, 10, r't3=19:45')
        plt.text(43, 55, r'v=53mi/h')
        ax2 = ax1.twinx()
        self.occupancy.iloc[5,].plot(style='r-',fontsize=8, linewidth=2)
        plt.ylabel('Occupancy', fontsize=14)
        plt.savefig('../Figures/Case 1/Speed and occupancy at the bottleneck.png', dpi=300, bbox_inches='tight')
        
        # Speed and occupancy downstream the bottleneck
        fig = plt.figure()
        plt.title('Speed and occupancy downstream the bottleneck', fontdict={'size':16})
        ax1 = fig.add_subplot(111)
        self.speed.iloc[bottleneck_location_index-1,].plot(style='b-',fontsize=8, linewidth=2)
        plt.vlines(39,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed")
        plt.vlines(118,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed") 
        plt.hlines(53,5,131, colors = 'g', linewidth=0.5, linestyles = "dashed") # free flow speed = 53 mile/hour
        plt.ylabel('Speed (mile/hour)', fontsize=14)
        plt.text(25, 10, r't0=13:10')
        plt.text(105, 10, r't3=19:45')
        plt.text(43, 55, r'v=53mi/h')
        ax2 = ax1.twinx()
        self.occupancy.iloc[4,].plot(style='r-',fontsize=8, linewidth=2,)
        plt.ylabel('Occupancy', fontsize=14)
        plt.savefig('../Figures/Case 1/Speed and occupancy downstream the bottleneck.png', dpi=300, bbox_inches='tight')
        
        # Speed and occupancy upstream the bottleneck
        fig = plt.figure()
        plt.title('Speed and occupancy upstream the bottleneck', fontdict={'size':16})
        ax1 = fig.add_subplot(111)
        self.speed.iloc[bottleneck_location_index+1,].plot(style='b-',fontsize=8, linewidth=2)
        plt.vlines(39,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed")
        plt.vlines(118,10,65, colors = 'orange', linewidth=0.5, linestyles = "dashed") 
        plt.hlines(53,5,131, colors = 'g', linewidth=0.5, linestyles = "dashed") # free flow speed = 53 mile/hour
        plt.ylabel('Speed (mile/hour)', fontsize=14)
        plt.text(25, 10, r't0=13:10')
        plt.text(105, 10, r't3=19:45')
        plt.text(43, 55, r'v=53mi/h')
        ax2 = ax1.twinx()
        self.occupancy.iloc[6,].plot(style='r-',fontsize=8, linewidth=2,)
        plt.ylabel('Occupancy', fontsize=14)
        plt.savefig('../Figures/Case 1/Speed and occupancy upstream the bottleneck.png', dpi=300, bbox_inches='tight')
        
    def plot_critical_occupancy(self):
        
        fig = plt.figure(figsize=(12,8))
        for i in range(self.flow.shape[0]-2):
            plt.subplot(5,4,i+1)
            plt.scatter(list(self.occupancy.iloc[i,0:92]), list(self.flow.iloc[i,0:92]), s = 5, marker='o', c='r', edgecolors='r')    # loading from 10:00 to 16:45
            plt.scatter(list(self.occupancy.iloc[i,92:-1]), list(self.flow.iloc[i,92:-1]), s = 5, marker='o', c='r', edgecolors='b')  # unloading from 16:45 to 21:00
            plt.xticks(fontsize = 6)
            plt.yticks(fontsize = 6)
        fig.savefig('../Figures/Case 1/Critical occupancy.png', dpi=300, bbox_inches='tight')


def get_distance(flow):
    
    distance = np.zeros(flow.shape[0])
    distance[0] = 1/2*(np.array(flow.index)[0] - np.array(flow.index)[1])
    distance[-1] = 1/2*(np.array(flow.index)[-2] - np.array(flow.index)[-1])
    for i in range(1, len(distance)-1):
        distance[i] = 1/2*(np.array(flow.index)[i-1] - np.array(flow.index)[i+1])
        
    return distance



if __name__ == '__main__':
    
    # Input data: flow, speed, occupancy. The raw data can be downloaded from PeMS.
    # DateTime: Weekday on April 2019，10:00~20:55
    # Location: I-405N Abs 8.03~14.94
    dataset_dir = '../Dataset/Dataset 1/'
    data = get_data(dataset_dir)
    flow_raw = data.load_flow()
    speed_raw = data.load_speed()
    occupancy_raw = data.load_occupancy()
    flow, speed, occupancy = data.average_the_data(flow_raw, speed_raw, occupancy_raw)
    
    # Save data
    flow.to_csv(dataset_dir + 'flow.csv')
    speed.to_csv(dataset_dir + 'speed.csv')
    occupancy.to_csv(dataset_dir + 'occupancy.csv')
    flow_raw['# Lane Points'].iloc[0:22].to_csv(dataset_dir + 'lanes.csv')
    distance = get_distance(flow)
    np.savetxt(dataset_dir + 'distance.csv', distance, delimiter = ',')
    
    # Plot the data
    plot_data = plot_data(flow, speed, occupancy)
    plot_data.plot_queue_profile()
    plot_data.plot_speed_occupancy()
    plot_data.plot_speed_occupancy_near_bottleneck(bottleneck_location_index = 5)
    # Observe the critical occupancy at each location
    # As for the dataset 1, we have:
    # crit_occ = np.array([0.16,0.11,0.10,0.14,0.10,0.13,0.15,0.12,0.13,0.13,0.13,0.13,0.13,0.12,0.12,0.13,0.12,0.10,0.13,0.10,0.15,0.20])
    plot_data.plot_critical_occupancy()

