# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:30:55 2024
@author: solvehor
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import re
import os, sys
import math


def rearrange_headers(raw_df, time_between_frames):
    """
    Removes unneccessary columns and rows in the raw dataframe, and replaces 
    column names to eg. body_x.
    """
    rearranged_df = raw_df.drop("scorer", axis=1)                               # removes first column
    new_columns = [f"{rearranged_df.iloc[0, i]}_{rearranged_df.iloc[1, i]}" 
                   for i in range(len(rearranged_df.columns))]                  # merges name of bodypart and x/y/likelihood
    rearranged_df.columns = new_columns                                         # replaces column names
    rearranged_df = rearranged_df.drop([0, 1], axis=0)                          # removes the row 0 and 1 (bodypart and x/y/likelihood)
    rearranged_df = rearranged_df.reset_index(drop=True)                        # resets the indexes
    
    # assign a time point for every x,y value
    time_list = []
    time_x = time_between_frames
    for x in rearranged_df['body_x']:
        time_x += time_between_frames
        time_list.append(time_x)
        
    rearranged_df["body_time"] = time_list
    rearranged_df = rearranged_df.apply(pd.to_numeric, errors='raise')          # make values in df numeric (floats)
    return rearranged_df
    


def select_free_behavior(rearranged_df, age, frames_per_sec):
    if age in ["P22", "P23"]:
        free_behavior_df = rearranged_df[:3*60*frames_per_sec]
        
    elif age in  ["P24","P25","P26","P27","P28","P29","P30"]:
        free_behavior_df = rearranged_df[:5*60*frames_per_sec]
        
    else: 
        free_behavior_df = rearranged_df
    return free_behavior_df



def filter_df(rearranged_df):
    """
    Filters out high error likelihood scores from DLC, and removes x- and y-values outside 
    the open field box.
    """
    filtered_df = rearranged_df[rearranged_df['body_likelihood'] > 0.95]
        
    if filtered_df is not None and len(filtered_df['body_x']) > 100:
        xmin = np.percentile(filtered_df['body_x'], 5) 
        xmax = np.percentile(filtered_df['body_x'], 95)
        ymin = np.percentile(filtered_df['body_y'], 5)
        ymax = np.percentile(filtered_df['body_y'], 95)
    
        filtered_df = filtered_df[ (filtered_df['body_x'] > xmin)   &  (filtered_df['body_x'] < xmax) ]
        filtered_df = filtered_df[ (filtered_df['body_y'] > ymin)   &  (filtered_df['body_y'] < ymax) ]
    
    return filtered_df 
    


def scale_df(filtered_df):
    """
    Rescales the x and y position values to centimeters to match the actual size of the 
    open field box.
    """
    scaled_df = filtered_df
    
    
    arena_cms = 50
    xarena_pixels = max(filtered_df['body_x']) - min(filtered_df['body_x'])
    yarena_pixels = max(filtered_df['body_y']) - min(filtered_df['body_y'])
    
    # For converting pixels to cms
    conversion_factor = arena_cms / xarena_pixels
    
    scaled_df['body_x'] = scaled_df['body_x'].multiply(conversion_factor)
    scaled_df['body_y'] = scaled_df['body_y'].multiply(conversion_factor)
    
    scaled_df['body_x'] = scaled_df['body_x'] - min( scaled_df['body_x'])
    scaled_df['body_y'] = scaled_df['body_y'] - min( scaled_df['body_y'])
    
    return scaled_df
    

    
def likelihood_hist(rearranged_df, age, identity):
    """
    Plots histogram of every likelihood score from DLC.
    """
    body_likelihood = rearranged_df["body_likelihood"]
    
    plt.figure(figsize=(8, 6), dpi=500)
    plt.hist(body_likelihood, bins=100, density=True)
    plt.xlabel('likelihood')
    plt.title('Likelihood histogram | ' + str(age), 
              fontsize ='12', fontweight='bold')
    directory = os.path.join(r"Z:\NEW_Analyses\Researchers\Solveig\Spyder\Open field\likelihood", str(identity))    
    file_path = os.path.join(directory, f"likelihood_{identity}_{age}.png")
    plt.savefig(file_path)
    plt.show()
    



def occupancy_map(scaled_df, age, duration, identity): 
    """
    Plots the occupancy map of the rat in the open field as both a trajectory map and
    a heat map (and a smoothened version of the heat map).
    """    
    body_x = scaled_df['body_x']
    body_y = scaled_df['body_y']

    fig, axs = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [1, 1.28,1.28]}, dpi=200)
    fig.suptitle('Occupancy maps | ' + str(age) + ' | ' + str(duration) + ' sec', fontsize=15, fontweight='bold')
    
    # trajectory map:
    axs[0].plot(body_x, body_y, linewidth=0.5)
    axs[0].set_title("Trajectory map", fontsize='13')
    axs[0].set_xlim(0,50)
    axs[0].set_ylim(0,50)
    axs[0].set_xlabel('cm')
    axs[0].set_ylabel('cm')
    axs[0].grid(linestyle='--', linewidth=0.5)


    # heat map: 
    bins = 20
    h, xedges, yedges_ = np.histogram2d(body_x, body_y, bins=bins)
    h = h*0.02  # convert number of observations in bin to duration (s)
    im = axs[1].imshow(h.T, origin="lower", interpolation ="none")              # h.T is transposing the 2D histogram matrix - i.e. making rows columns and columns rows, to flip it along diagonal
    axs[1].set_title("Heat map", fontsize='13')
    axs[1].set_xlabel('cm')
    axs[1].set_ylabel('cm')
    axs[1].grid(linestyle='--', linewidth=0.5)
    # convert axis from number of bins to 50x50 cm:
    n_ticks = 5
    xticks_positions = np.arange(0, bins, bins/n_ticks)
    yticks_positions = np.arange(0, bins, bins/n_ticks)
    x_labels = np.arange(0, 50, 50/n_ticks)
    y_labels = np.arange(0, 50, 50/n_ticks)
    axs[1].set_xticks(xticks_positions)
    axs[1].set_yticks(yticks_positions)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_yticklabels(y_labels)
    # include colorbar:
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('duration (s)', fontsize=10)
    
    
    # smoothed heat map:
    im = axs[2].imshow(h.T, origin="lower", interpolation="gaussian")
    axs[2].set_title("Smoothed heat map", fontsize='13')
    axs[2].set_xlabel('cm')
    axs[2].set_ylabel('cm')
    axs[2].grid(linestyle='--', linewidth=0.5)
    # convert axis from number of bins to 50x50 cm:
    axs[2].set_xticks(xticks_positions)
    axs[2].set_yticks(yticks_positions)
    axs[2].set_xticklabels(x_labels)
    axs[2].set_yticklabels(y_labels)
    # include colorbar:
    cbar = fig.colorbar(im, shrink=0.8)
    cbar.set_label('duration (s)', fontsize=10)
    
    # prepare the figure and plot it:
    fig.suptitle('Occupancy maps  |  ' + str(age) + '  |  Total video time: ' + str(duration) + ' min  | ' + '  Included time in plots: ' + str(round(sum(sum(h))/60, 2)) + ' min' , fontsize=15, fontweight='bold')
    directory = os.path.join(r"Z:\NEW_Analyses\Researchers\Solveig\Spyder\Open field\occupancy", str(identity))    
    file_path = os.path.join(directory, f"occupancy_{identity}_{age}.png")
    plt.savefig(file_path)
    plt.show()


    
def calculate_speed_distance(scaled_df): 
    """
    Calculates speed and total distance moved for rat in the current file.
    """
    body_x = scaled_df['body_x']
    body_y = scaled_df['body_y']    
    body_time = scaled_df['body_time']
    
    duration = len(scaled_df) / 50 
    
    distance_list = []
    speed_list = []
    time_valid = []
    
    # filter out values that extend the frame interval period of 0.02 s:
    for index in range(1, len(body_x)):     
        time = body_time.iloc[index] - body_time.iloc[index-1]
        if time <= 0.5: 
            distance = math.sqrt((body_x.iloc[index] - body_x.iloc[index-1])**2 + 
                             (body_y.iloc[index] - body_y.iloc[index-1])**2)
            
            speed = distance / time
        else:
            distance = math.nan
            speed = math.nan
                
        distance_list.append(distance)
        speed_list.append(speed)
        time_valid.append(body_time.iloc[index])
        
    average_speed = round(np.average(speed_list), 2)
    distance = sum(distance_list) / duration * 5 / 100                              # standarized to distance (in meter) moved in 5 min. 
    movement_list = [speed_list, average_speed, distance, time_valid, duration]
    return movement_list



def speed_map(scaled_df, age, duration, identity):
    """
    Plots speed as a function of time, and a histogram of different speed values.
    Returns the average speed.
    """
    movement_list = calculate_speed_distance(scaled_df)
    speed_list = movement_list[0]
    time_valid = movement_list[3]
    duration = movement_list[4] 
    
    time_axis = np.arange(0, int( ( len(speed_list) + 50 ) / 50) , 0.02)
    time_axis = time_axis[0:len(speed_list)]
    
    # plot speed over time:
    plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot(time_axis, speed_list)   # continous line in plot 
    plt.scatter(time_axis, speed_list,s=2)
    plt.ylim(0, 60)
    plt.xlabel('time (s)')
    plt.ylabel('speed (cm/s)')
    plt.title('Speed over time | ' + str(age) + " | " + str(duration) + " min", fontsize ='12', fontweight='bold')
    directory = os.path.join(r"Z:\NEW_Analyses\Researchers\Solveig\Spyder\Open field\speed over time", str(identity))    
    file_path = os.path.join(directory, f"speed_over_time{identity}_{age}.png")
    plt.savefig(file_path)
    plt.show()
    
    # plot speed histogram:
    speed_list = np.array(speed_list)
    speed_list = speed_list
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(speed_list, 100, density=True)
    plt.xlim(0,60)
    plt.xlabel('speed (cm/s)')
    plt.title('Speed histogram | ' + str(age) + " | " + str(duration) + " min", fontsize ='12', fontweight='bold')
    directory = os.path.join(r"Z:\NEW_Analyses\Researchers\Solveig\Spyder\Open field\speed histogram", str(identity))    
    file_path = os.path.join(directory, f"speed_histogram{identity}_{age}.png")
    plt.savefig(file_path)
    plt.show()
    
def fetch_data(rat):
    """
    Plots likelihood histogram, occupancy maps, 
    """
    identity = rat_ids[rat]
    csv_files = [csv_file for path in folder_path_list for csv_file in path.glob(f'*{rat}*.csv')]           # f and {} makes it understand that it is an actual variabel inside the {}. And * means 'everything else'
    
    ages = []
    speeds = []
    distances = []
    
    for file in csv_files:
                
        file = str(file)
        
        print('Processing: ' + file)
        
        raw_df = pd.read_csv(file)
        frames_per_sec = 50
        time_between_frames = 1/frames_per_sec
          
        # edit dataframe:
        rearranged_df = rearrange_headers(raw_df, time_between_frames)
        
        # general info aout video
        age = re.search(r'_(P\d+)-', str(file)).group(1)                         # "regular expression" 
        print(age)
        # duration = round((len(rearranged_df) / frames_per_sec) / 60, 2)    
        # print(duration)# duration, in minutes
        
        free_behavior_df = select_free_behavior(rearranged_df, age, frames_per_sec)
        
        filtered_df = filter_df(free_behavior_df)
        duration = round((len(filtered_df) / frames_per_sec) / 60, 2)    
        # print(duration)# duration, in minutes

        # plot likelihood
        # likelihood_hist(rearranged_df, age, identity)
        
        if filtered_df is not None and len(filtered_df['body_x']) > 100:         # the df must have a certain number of values after filtering, here decided to be at least 100
            scaled_df = scale_df(filtered_df)
            
            # plotting:  
            occupancy_map(scaled_df, age, duration, identity)  
            speed_map(scaled_df, age, duration, identity)
            
            # append age, speed, distance to lists:
            age_integer = int(age.replace('P', ''))
            
            speed_distance = calculate_speed_distance(scaled_df)
            
            sys.exit()
            average_speed = speed_distance[1]
            distance = speed_distance[2]
            
            ages.append(age_integer)
            speeds.append(average_speed)
            distances.append(distance)
            
    # convert to np.arrays:
    ages = np.array(ages)
    speeds = np.array(speeds)
    distances = np.array(distances)
    
    # sort data (from low to high age, and corresponding speed/distance lists):
    sorting_indices = np.argsort(ages)
    ages = ages[sorting_indices]
    speeds = speeds[sorting_indices]
    distance = distances[sorting_indices]
    
    return ages, speeds, distances

    
def plot_speeds_distances(rats, ages, yvalues, xlabel, ylabel, title, file_path):
    plt.figure(figsize=(8, 6), dpi=300)
    
    for index in range(len(ages)):
        plt.plot(ages[index], yvalues[index], label = rats[index])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize ='12', fontweight='bold')
    plt.legend()
    plt.savefig(file_path)
    plt.show()
    

#______________________


folder_path_list = [Path(r'Z:\NEW_Data\Solveig\188167_m1\open field\free behavior'),
                    Path(r'Z:\NEW_Data\Solveig\188168_m2\open field\free behavior'),
                    Path(r'Z:\NEW_Data\Solveig\188176_f1\open field\free behavior'),
                    Path(r'Z:\NEW_Data\Solveig\188177_f2\open field\free behavior')]

rat_ids = {'m1':'188167m1', 'm2':'188168m2', 'f1':'188176f1', 'f2':'188177f2'}

rats = ["m1", "m2", "f1", "f2"]
ages = []
speeds = []
distances = []

for rat in rats:
    data = fetch_data(rat)
    ages.append(data[0])
    speeds.append(data[1])
    distances.append(data[2])
    
    
# plot speeds:
xlabel = 'age (postnatal day)'
ylabel = 'average speed (cm/s)'
title = 'Average speed across ages'
directory = os.path.join(r"Z:\NEW_Analyses\Researchers\Solveig\Spyder\Open field\speed across ages")    
file_path = os.path.join(directory, "average_speed_plot.png")
plot_speeds_distances(rats, ages, speeds, xlabel, ylabel, title, file_path)

# plot distances:
xlabel = 'age (postnatal day)'
ylabel = 'distance moved (m)'
title = 'Average distance moved across ages'
directory = os.path.join(r"Z:\NEW_Analyses\Researchers\Solveig\Spyder\Open field\distance across ages")    
file_path = os.path.join(directory, "distance_plot.png")
plot_speeds_distances(rats, ages, distances, xlabel, ylabel, title, file_path)


#sys.exit()