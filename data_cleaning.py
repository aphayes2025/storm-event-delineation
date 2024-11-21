#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_cleaning.py

This script processes gauge data from CSV files to calculate Hysteresis indices and plots them.

@author: lizamclatchy
Created on Fri Jan 26 08:28:02 2024
"""

import scipy
from scipy import signal
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from statistics import mode



def separatebaseflow(Q,fc,Pass=4): 
    '''
    Seperate streamflow into baseflow and storm flow using a filter method. 

    Args:
        Q (array): holding discharge data
        fc (float): filter coeffcient 
        Pass (int): Number of times filter pass through 

    Returns:
        tuple(storm_flow, base_flow)
    
    ''' 

    # Initialize bf with NaN vals
    n = len(Q)
    bf = np.empty(n)
    bf[:] = np.nan
    bf_p = Q.copy() # baseflow from the previous pass, initial is the streamflow

    # main loop for filtering passes
    for j in range(1, Pass+1):
        # Set forward and backward pass
        if j% 2 == 1:
            sidx, eidx, incr = 0, n, 1
        else:
            sidx, eidx, incr = n-1, 1, -1
        
        # Set the inital value for baseflow
        bf[sidx] = bf_p[sidx]
        
        # filter loop
        for i in range(sidx + incr,eidx,incr):         
            tmp = fc * bf[i-incr] + (1-fc )* (bf_p[i] + bf_p[i-incr]) / 2
            bf[i] = np.nanmin([tmp, bf_p[i]])     
        
        bf_p = bf.copy()
    
    # calculate stormflowx
    sf = Q - bf
    
    return sf,bf


def find_matches(df, storeQ_dates_df):
    date_matches = df[df['Datetime'].isin(storeQ_dates_df['Peak Discharge Dates'])]
    return date_matches

def event_localmins(dummystart, dummyend, startbound, endbound, COLUMN_NAME):
    actualstart = []
    actualend = []
    #start values
    for i in range(len(dummystart)):
        # Define the range based on the dummystart value + the start bound
        min_range_start = (dummystart[i], dummystart[i] + startbound)
        # Filter rows based on the range
        filtered_df = df.loc[(df['Datetime'] >= min_range_start[0]) & (df['Datetime'] <= min_range_start[1])]
        # Find the row with the minimum value in Column2
        localstartmin = filtered_df.loc[filtered_df[COLUMN_NAME].idxmin()]
        # Extract the corresponding value from Column1
        localstartmin_time = localstartmin['Datetime']
    
        #in theory if the start time is != local min then there is a drop between dummy start and the peak (our event starts here)
        if dummystart[i] != localstartmin_time:
            actualstart.append(localstartmin_time)
        else: 
            actualstart.append(dummystart[i])
    
    #ends
    for j in range(len(dummyend)):
        # Define the range based on the dummyend value - the end bound (find the min value within the end bound)
        min_range_end = (dummyend[j], dummyend[j] - endbound)
        # Filter rows based on the range
        filtered_df = df.loc[(df['Datetime'] >= min_range_end[1]) & (df['Datetime'] <= min_range_end[0])]
        # Find the row with the minimum value in Column2
        localendmin = filtered_df.loc[filtered_df[COLUMN_NAME].idxmin()]
        # Extract the corresponding value from Column1
        localendmin_time = localendmin['Datetime']
    
        #in theory if the start time is != local min then there is a drop between dummy start and the peak (our event starts here)
        if dummyend[j] != localendmin_time:
            actualend.append(localendmin_time)
        else: 
            actualend.append(dummyend[j])
            
    return actualstart, actualend


def event_delineation_boundbased(timestamps, startbound, endbound):
    dummystart = []
    dummyend = []
    i = 0  # Start with the first peak

    while i < len(timestamps) - 1:
        current_window_start = timestamps[i] - startbound
        current_window_end = timestamps[i] + endbound

        while i < len(timestamps) - 1 and timestamps[i+1] <= current_window_end:
            # Peak falls within the current window, extend the window
            current_window_end = timestamps[i+1] + endbound
            i += 1  # Move to the next timestamp

        # Store the current window
        dummystart.append(current_window_start)
        dummyend.append(current_window_end)

        # Move to the next peak and start a new window
        i += 1

    return dummystart, dummyend

def calculate_hysteresis_index(discharge_values, turbidity_values):
    # Split the data into rising and falling limbs
    peak_index = np.argmax(discharge_values)
    rising_limb_discharge = discharge_values[:peak_index + 1]
    falling_limb_discharge = discharge_values[peak_index:]
    rising_limb_turbidity = turbidity_values[:peak_index + 1]
    falling_limb_turbidity = turbidity_values[peak_index:]

    # Avoid division by zero in normalization
    def normalize(values):
        range_val = np.nanmax(values) - np.nanmin(values)
        if range_val == 0:
            return np.full(values.shape, np.nan)  # Return NaN array if range is zero
        return (values - np.nanmin(values)) / range_val
 
    normalized_rising_limb_discharge = normalize(rising_limb_discharge)
    normalized_falling_limb_discharge = normalize(falling_limb_discharge)
    normalized_rising_limb_turbidity = normalize(rising_limb_turbidity)
    normalized_falling_limb_turbidity = normalize(falling_limb_turbidity)
 
    percentiles = np.arange(0.02, 1.02, 0.02)
 
    # Handling unique and sorting, making sure arrays are not empty
    def process_limb(discharge, turbidity):
        if len(discharge) > 0:
            _, indices = np.unique(discharge, return_index=True)
            return discharge[indices], turbidity[indices]
        return np.array([]), np.array([])
 
    normalized_rising_limb_discharge, normalized_rising_limb_turbidity = process_limb(normalized_rising_limb_discharge, normalized_rising_limb_turbidity)
    normalized_falling_limb_discharge, normalized_falling_limb_turbidity = process_limb(normalized_falling_limb_discharge, normalized_falling_limb_turbidity)
 
    # Interpolation, ensuring data is not empty
    def safe_interp(x, xp, fp):
        if len(xp) == 0 or np.all(np.isnan(fp)):
            return np.full(x.shape, np.nan)  # Return NaN array if no valid data
        return np.interp(x, xp, fp, left=np.nan, right=np.nan)
 
    interpolated_ssc_rising = safe_interp(percentiles, normalized_rising_limb_discharge, normalized_rising_limb_turbidity)
    interpolated_ssc_falling = safe_interp(percentiles, normalized_falling_limb_discharge, normalized_falling_limb_turbidity)
 
    HI_values = interpolated_ssc_rising - interpolated_ssc_falling
    return HI_values

def normalize(values):
    values = np.where(np.isnan(values), np.nanmean(values), values)
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    if max_val - min_val == 0:
        return np.zeros_like(values)  
    else:
        return (values - min_val) / (max_val - min_val)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'


if __name__ == "__main__":
    # converting 
    csv = "gauge-data/USGS002_data.csv"
    df = pd.read_csv(csv)
    df['Datetime'] = pd.to_datetime(df["Date"] + " " + df["Time"])
    date_range = pd.date_range(start="2016-10-01", end="2021-09-30", freq='15T')

    date_df = pd.DataFrame({'Datetime': date_range})

    #Merge the original DataFrame with the date DataFrame using an outer join
    merged_df = pd.merge(date_df, df, on='Datetime', how='outer')

    #Check for NaN values
    if merged_df['SSC (mg/L)'].isnull().all():
        merged_df['SSC (mg/L)'].fillna(pd.NA, inplace=True)

    if merged_df['Discharge (cfs)'].isnull().all():
        merged_df['Discharge (cfs)'].fillna(pd.NA, inplace=True)

    #clean
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    #Update df
    df = merged_df.copy()
    df_copy = df.copy()
    df['SSC (mg/L)'].interpolate(method = 'linear', inplace = True)
    df['Discharge (cfs)'].interpolate(method = 'linear', inplace = True)

    # Extract required columns
    Q = df['Discharge (cfs)'].tolist()
    C = df['SSC (mg/L)'].tolist()
    m3s2cfs = 0.0283168 #conversion factor

    # filter data
    smooth_Q = signal.savgol_filter(Q, window_length=121, polyorder=3, mode="nearest")
    smooth_C = signal.savgol_filter(C, window_length=121, polyorder=3, mode="nearest")

    #BASFLOW REMOVAL USING HYDRUN
    fc = 0.998
    sf, bf = separatebaseflow(Q, fc)
    baseline_Q = bf

    #Baseline Subtracted Data
    normalized_Q = smooth_Q - bf
    normalized_C = smooth_C 
    normalized_Qdf = pd.DataFrame(normalized_Q)
    normalized_Cdf = pd.DataFrame(normalized_C)

    # ESTABLISH DATAFRAME
    # Make dataframes that have the the corresponding date values for C&Q values

    m3s2cfs = 0.0283168
    df['Normalized Discharge'] = normalized_Qdf
    df['Normalized SSC'] = normalized_Cdf
    df['Discharge (m^3/s)'] = df['Discharge (cfs)'] * m3s2cfs
    df_copy['Discharge (m^3/s)'] = df_copy['Discharge (cfs)'] * m3s2cfs

    # finding peaks
    timestamp = 15 #timestamps are 15 min long
    tpd = (60/timestamp)*24 #timestamps per day --- number of timestamps/hr times 24hrs

    # Use list of these values that were named above before the dataframe
    dummy_Q = scipy.signal.find_peaks(normalized_Q, prominence=30, distance= tpd) 
    dummy_C = scipy.signal.find_peaks(normalized_C, prominence=15, distance= tpd)

    #NOTE: These peaks are ALL the peaks that occur within the dataset, we will use event delineation to determine if each peak is a stand-alone event
    #OR if it can be grouped with other peaks within the parameters examined below

    #store the values in a function
    storeQ = []
    for peak_index in dummy_Q[1]['prominences']:
        storeQ.append(peak_index)

    storeC = []
    for peak_index in dummy_C[1]['prominences']:
        storeC.append(peak_index)

    #Store the dates corresponding to the peaks for Discharge
    storeQ_dates = [date_range[i] for i in dummy_Q[0] if i < len(date_range)]

    #Store the dates corresponding to the peaks for Sediment
    storeC_dates = [date_range[i] for i in dummy_C[0] if i < len(date_range)]


    #From Linh's paper : Once the peaks were identified,
    #we traced back from each peak to the event start time and estimated the event end time.
    #So now, we have the peaks and the dates associated with the peaks
    actual_peak_Q_values = [df['Normalized Discharge'].iloc[i] for i in dummy_Q[0]]
    actual_peak_C_values = [df ['Normalized SSC'].iloc[i] for i in dummy_C[0]]


    ## Definition of window sizes for start and end data of event definition using given boundaries
    tph = 60/15 #timestamps per hour 

    pd.DataFrame(storeQ_dates).isin(df['Datetime']).all()
    storeQ_dates_df = pd.DataFrame(storeQ_dates)
    storeQ_dates_df['Peak Discharge Dates'] = storeQ_dates_df

    date_matches = find_matches(df, storeQ_dates_df)
    if not date_matches.empty:
        print(date_matches)
    date_matches.reset_index(inplace=True) 

    #date matches a dataframe of indices of all the Q peaks, associated dates, and corresponding CQ values 

    # Convert days into numpy.timedelta64 format for Q and C.
    # This format will be used for arithmetic operations on datetime64[ns] dtype.
    Q_start_bound = np.timedelta64(1, 'D') # 3 days in numpy timedelta format
    Q_end_bound = np.timedelta64(1, 'D') #normally 5

    C_start_bound = np.timedelta64(1, 'D') #normally 2
    C_end_bound = np.timedelta64(1, 'D') #normally 3

    dummystartQ, dummyendQ = event_delineation_boundbased(storeQ_dates, Q_start_bound, Q_end_bound)
    column_name_Q = 'Discharge (cfs)'
    actualstartQ, actualendQ = event_localmins(dummystartQ, dummyendQ, Q_start_bound, Q_end_bound,column_name_Q )

    # Event defintiion for C
    dummystartC, dummyendC = event_delineation_boundbased(storeC_dates, C_start_bound, C_end_bound)
    column_name_C = 'SSC (mg/L)'
    actualstartC, actualendC = event_localmins(dummystartC, dummyendC, C_start_bound, C_end_bound,column_name_C)

    
    # Define the time boundary for matching events
    match_bound = np.timedelta64(36, 'h')

    event_start_time = []  # Store starting time of events
    event_end_time = [] 
    i_start = []  # Store start indices from the main dataframe
    i_end = []  # Store end indices from the main dataframe
    j = 0 

    for i in range(len(actualstartQ)):
        eventmatch =  ((actualstartC[j] <= actualstartQ[i] + match_bound) or (actualstartC[j] <= actualstartQ[i] - match_bound))    
        
        if eventmatch and j < len(actualstartC) == True:
            start_time = min(actualstartQ[i], actualstartC[j])
            end_time = max(actualendQ[i], actualendC[j])
            
            event_start_time.append(start_time)
            event_end_time.append(end_time)
            j += 1
        
        else:
            event_start_time.append(actualstartQ[i])
            event_end_time.append(actualendQ[i])
    
        # Append indices from the main dataframe based on determined start and end times
        i_start.append(df[df['Datetime'] == event_start_time[-1]].index[0])
        i_end.append(df[df['Datetime'] == event_end_time[-1]].index[0])


    max_Q = []
    max_C = []
    max_Q_time = []
    max_C_time = []

    for i, (start, end) in enumerate(zip(i_start, i_end)):
        event_df = df_copy.iloc[start:end]
    
        if not event_df.empty:
            maxQ = event_df['Discharge (m^3/s)'].max()
            max_Q.append(maxQ)
            max_Q_index = event_df['Discharge (m^3/s)'].idxmax()
            max_Q_time.append(df.loc[max_Q_index, 'Datetime'])
            
            maxC = event_df['SSC (mg/L)'].max() 
            max_C.append(maxC)
            max_C_index = event_df['SSC (mg/L)'].idxmax()
            
            if pd.notna(max_C_index):  #Check if the index is not NaN
                max_C_time.append(df.loc[max_C_index, 'Datetime'])
            else:
                max_C_time.append(np.nan)  #If index is NaN, append NaN to max_C_time
        else:
            # Handle the case where event_df is empty
            # For example, you can set default values or skip this event
            pass

    threshold_values = {
        "Threshold70": 211 * m3s2cfs,
        "Threshold57": 43.3 * m3s2cfs,
        "Threshold500": 1150 * m3s2cfs,
        "Threshold002": 128 * m3s2cfs,
        "Threshold97": 80.9 * m3s2cfs,
        "Threshold2200": 391 * m3s2cfs,
        "Threshold87": 127 * m3s2cfs
    }
   
    threshold = threshold_values['Threshold002']



    # =============================================================================
    # EVENT DATA 
    # =============================================================================
    event_df_dic = {}
    for i, (start, end) in enumerate(zip(i_start, i_end)):
        event_df = df_copy.iloc[start:end].copy()  #Extract the DataFrame for the current event
        event_df['event_number'] = i  #Add 'event_number' column
        event_df_dic[f'event_{i}'] = event_df  #Store in the dictionary
    event_index_map = {key: i for i, key in enumerate(event_df_dic.keys())}
    events_above_threshold = {}
    HI_values = []
    for key, event_df in event_df_dic.items():
        max_discharge = event_df['Discharge (m^3/s)'].max()
        if max_discharge >= threshold:
            count = event_df['SSC (mg/L)'].isna().sum()
            if count < len(event_df['SSC (mg/L)']) / 2:
                events_above_threshold[key] = event_df

    # Example usage, you will need to define `events_above_threshold` appropriately
    event_HI_values = {}
    for event_title, event_data in events_above_threshold.items():
        discharge_values = np.array(event_data['Discharge (m^3/s)'])
        turbidity_values = np.array(event_data['SSC (mg/L)'])
        event_HI = calculate_hysteresis_index(discharge_values, turbidity_values)
        if event_title not in event_HI_values:
            event_HI_values[event_title] = []
        event_HI_values[event_title].append(event_HI)
    
    # Ensure all values are not NaN before averaging
    HINew = [np.nanmean(values) if not np.all(np.isnan(values)) else np.nan for values in event_HI_values.values()]

    #=============================================================================
    #Flushing Index Point
    #=============================================================================


    peak_Q_index = []
    flushing_index = []

    for key, event_df in events_above_threshold.items():
        event_df = event_df.dropna(subset=['SSC (mg/L)', 'Discharge (m^3/s)'])  #Drop rows where relevant columns are NaN
        start, end = event_df.index[0], event_df.index[-1]
        peak_index = event_df['Discharge (m^3/s)'].idxmax()
        peak_Q_index.append(peak_index)
        turbidity_values = event_df['SSC (mg/L)'].values  #Convert to numpy array
        ci_norm = normalize(turbidity_values)
        
        start_pos = event_df.index.get_loc(start)
        peak_pos = event_df.index.get_loc(peak_index)

        ci_start = ci_norm[start_pos]
        ci_peak = ci_norm[peak_pos]
        flushing_index.append(ci_peak - ci_start)  


    #=============================================================================
    #EXTRACTING FOR DAYMET
    #=============================================================================

        
    #find max_Q_time, i_start and event_start time for just events_above_threshold    
    max_Q_threshold = []
    max_Q_time_threshold = []
    i_start_threshold = []
    i_end_threshold = []
    event_start_time_threshold = []
    max_C_threshold = []
    event_end_time_threshold = []
    event_numbers = []
    #Iterate over events above threshold
    for key, event_df in events_above_threshold.items():
        #Calculate max_Q and max_Q_time
        maxQ = event_df['Discharge (m^3/s)'].max()
        maxC = event_df['SSC (mg/L)'].max()
        max_Q_threshold.append(maxQ)
        max_C_threshold.append(maxC)
        max_Q_index = event_df['Discharge (m^3/s)'].idxmax()
        max_Q_time_threshold.append(df.loc[max_Q_index, 'Datetime'])
        
        # Find the start index and start time for this event
        start_index = df[df['Datetime'] == event_df.iloc[0]['Datetime']].index[0]
        i_start_threshold.append(start_index)
        event_start_time_threshold.append(event_df.iloc[0]['Datetime'])
        event_numbers.append(event_df['event_number'].iloc[0])
        #calculate the end index for this event
        end_index = start_index + len(event_df) - 1
        i_end_threshold.append(end_index)
        event_end_time = df.loc[end_index, 'Datetime']
        event_end_time_threshold.append(event_end_time)
    #Create a dictionary containing the lists
    data = {
        'event number': event_numbers,
        'max_Q': max_Q_threshold,
        'max_Q_time': max_Q_time_threshold,
        'i_start': i_start_threshold,
        'i_end': i_end_threshold,
        'event_start_time': event_start_time_threshold,
        'HINew': HINew,
        'flushing_index': flushing_index,
    ' max_C': max_C_threshold,
    'event_end_time':event_end_time_threshold
    }

    #Create a DataFrame from the dictionary
    df_events_above_threshold = pd.DataFrame(data)
    # Save the DataFrame to a CSV file

    #df_events_above_threshold.to_csv('events_above_threshold2200.csv')
    #df_events_above_threshold.to_csv('events_above_threshold70.csv')
    #df_events_above_threshold.to_csv('events_above_threshold57.csv')
    #df_events_above_threshold.to_csv('events_above_threshold87.csv')
    #df_events_above_threshold.to_csv('events_above_threshold002.csv')
    #df_events_above_threshold.to_csv('events_above_threshold500.csv')


    # =============================================================================
    # HYSTERESIS X HYDRO X SEDi PLOT
    # =============================================================================


    norm = mcolors.Normalize(vmin=min(i_start), vmax=max(i_end))
    colormap = cm.viridis

    num_segments = len(i_start)
    events_to_plot = []
    event_index_map = {key: i for i, key in enumerate(event_df_dic.keys())}
    df_copy['Date'] = df_copy['Datetime'].dt.date
    
    for key, event_df in event_df_dic.items():
        max_discharge = event_df['Discharge (m^3/s)'].max()
        if max_discharge >= threshold:
            count = event_df['SSC (mg/L)'].isna().sum()
            if count < len(event_df['SSC (mg/L)']) / 2:
                events_to_plot.append(key)
    
    analytic_idx_df = pd.DataFrame()
    analytic_idx_df['Hysteresis Index'] = pd.DataFrame(HINew)
    analytic_idx_df['Flushing Index'] = pd.DataFrame(flushing_index)
    analytic_idx_df = analytic_idx_df.round(5)
    analytic_idx_df.set_index(pd.Index(events_to_plot), inplace=True)
    for event_key in events_to_plot:
        start, end = event_df_dic[event_key].index[0], event_df_dic[event_key].index[-1]
        fig, (ax_combined, ax_hysteresis) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Gauge #87", fontsize=16)
    
        ax_hysteresis.spines['bottom'].set_color('blue')
        ax_hysteresis.spines['left'].set_color('brown') 
        ax_hysteresis.xaxis.label.set_color('blue')
        ax_hysteresis.yaxis.label.set_color('brown')
        ax_combined.set_xlabel('Datetime')
        ax_combined.tick_params(axis='y', colors='blue')
        ax_hysteresis.tick_params(axis='y', colors='brown')
        ax_hysteresis.tick_params(axis='x', colors='blue')
        ax_combined.set_ylabel('Discharge (m^3/s)', color='blue')
        ax_combined.plot(df_copy['Datetime'][start:end], df_copy['Discharge (m^3/s)'][start:end], color='blue', label='Discharge (m^3/s)')
        ax_combined.plot(max_Q_time[event_index_map[event_key]], max_Q[event_index_map[event_key]], color='blue', label=f'{max_Q_time[event_index_map[event_key]]}')
        ax_combined_ssc = ax_combined.twinx()
        ax_combined_ssc.tick_params(colors='brown', which='both') 
        ax_combined_ssc.set_ylabel('SSC (mg/L)', color='brown')
        ax_combined_ssc.spines['left'].set_color('blue')
        ax_combined_ssc.spines['right'].set_color('brown') 
        ax_combined_ssc.plot(df_copy['Datetime'][start:end], df_copy['SSC (mg/L)'][start:end], color='brown', label='SSC (mg/L)')
        ax_combined_ssc.plot(max_C_time[event_index_map[event_key]], max_C[event_index_map[event_key]], color='brown', label=f'{max_C_time[event_index_map[event_key]]}')
        ax_combined.axvline(x=max_Q_time[event_index_map[event_key]], color='blue', linestyle='--')
        ax_combined.axvline(x=max_C_time[event_index_map[event_key]], color='brown', linestyle='--')
        lines, labels = ax_combined.get_legend_handles_labels()
        lines_ssc, labels_ssc = ax_combined_ssc.get_legend_handles_labels()
        ax_combined.legend(lines + lines_ssc, labels + labels_ssc, loc='upper right')
        ax_combined.set_title(f'Hydrograph and Sedigraph for Event #{event_key}')
        ax_combined.set_xticklabels(ax_combined.get_xticklabels(), rotation=45, ha='right')
        date_format = DateFormatter('%Y-%m-%d')  # Customize date format as desired
        ax_combined.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        #Viridis 
        cmap = cm.get_cmap('viridis')
    
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_hysteresis = plt.colorbar(scalar_mappable, ax=ax_hysteresis)
        cbar_hysteresis.set_label('Time Progression')
    
        cbar_hysteresis.set_ticks([])
        # Add label at the bottom
        cbar_hysteresis.ax.text(0.5, -0.1, 'Start', ha='center', va='center', fontsize=10)
    
        # Add label at the top
        cbar_hysteresis.ax.text(0.5, 177777, 'End', ha='center', va='center', fontsize=10)
    
    
        segment_data = df_copy['SSC (mg/L)'][start:end]
    
        cmap_time = cm.get_cmap('viridis')
        norm_time = mcolors.Normalize(vmin=start, vmax=end)
    
        # Plot lines connecting hysteresis loop points
        for j in range(start, end - 1):
            line_color = cmap_time(norm_time(j))
            ax_hysteresis.plot([df_copy['Discharge (m^3/s)'].iloc[j], df_copy['Discharge (m^3/s)'].iloc[j + 1]],
                            [df_copy['SSC (mg/L)'].iloc[j], df_copy['SSC (mg/L)'].iloc[j + 1]], color=line_color)
    
        #Scatter plot for hysteresis loop points with Viridis colormap
        cmap_scatter = cm.get_cmap('viridis')
        norm_scatter = mcolors.Normalize(vmin=start, vmax=end)
        scatter_color = cmap_scatter(norm_scatter(np.arange(start, end)))
        ax_hysteresis.scatter(df_copy['Discharge (m^3/s)'][start:end], df_copy['SSC (mg/L)'][start:end], c=scatter_color, marker='o')
    
    
        ax_hysteresis.set_xlabel('Discharge (m^3/s)')
        ax_hysteresis.set_ylabel('SSC (mg/L)')
        event_index = events_to_plot.index(event_key) + 1  #Event index starts from 1
        ax_hysteresis.set_title(f'Hysteresis Loop for Event #{event_key}')
        ax_hysteresis.set_ylim(0, segment_data.max())
        indices_data = analytic_idx_df.loc[event_key].reset_index()
        metrics_str = '\n'.join(f'{col}: {val}' for col, val in indices_data.values)
        ax_hysteresis.annotate(metrics_str, xy=(0.01, 0.99), xycoords='axes fraction', ha='right', fontsize=10)
        ax_hysteresis.set_ylim(0, segment_data.max())

        plt.tight_layout()
        plt.show()
    # =============================================================================
    # BASELINE PLOT
    # =============================================================================
    baseline_Q_onemonth = baseline_Q[:365*24*4]
    q_onemonth = df['Discharge (cfs)'][:365*24*4]
    dates_onemonth = df['Datetime'][:365*24*4]
    smooth_Q_onemonth = smooth_Q[:365*24*4]
    normalized_Q_plt = normalized_Q[:365*24*4]
    plt.plot(dates_onemonth, q_onemonth, label='Raw Discharge (cfs)')
    plt.plot(dates_onemonth, baseline_Q_onemonth, label='Baseline Discharge (cfs)')
    plt.plot(dates_onemonth, normalized_Q_plt, label = 'Basline Subtracted Discharge (cfs)')
    plt.title('Gauge #01362500 Discharge Data Comparison: 1st Year')
    plt.xlabel('Dates')
    plt.ylabel('Discharge (cfs)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # =============================================================================
    # MAX DISCH PLOT
    # =============================================================================
    valid_max_C_time = [time for time in max_C_time if pd.notna(time)]
    fig, ax1 = plt.subplots()

    ax1.scatter(max_Q_time, max_Q, color='b', label='Max. Discharge (cfs)')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Discharge (m^3/s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.xaxis.set_tick_params(rotation=45)

    ax2 = ax1.twinx()
    ax2.scatter(valid_max_C_time[:365*24*4], max_C[:len(valid_max_C_time)][:365*24*4], color='r', label='Max SSC (mg/L)')
    ax2.set_ylabel('SSC (mg/L)', color='r')

    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Gauge #01362500 Distribution of Maximum Storm Values')
    plt.legend()
    plt.show()

    # =============================================================================
    # Seasons Per Events
    # =============================================================================

    event_mode_seasons = {}

    # Iterate through each event in events_above_thresholds
    for event, df in events_above_threshold.items():
        months = df["Datetime"].dt.month
        seasons = [get_season(month) for month in months]
        mode_season = mode(seasons)
        event_mode_seasons[event] = mode_season
        
    season_counts = {'Winter': 0, 'Spring': 0, 'Summer': 0, 'Fall': 0}

    for mode_season in event_mode_seasons.values():
        season_counts[mode_season] += 1
    for season, count in season_counts.items():
        print(f"{season}: {count}")

    # =============================================================================
    # BiVariate Plot
    # =============================================================================
    max_Q_time = []
    max_Q = []
    for event_title, event_data in events_above_threshold.items():
        discharge_values = event_data['Discharge (m^3/s)']
        maxQ = discharge_values.max()
        max_Q.append(maxQ)
        max_Q_index = discharge_values.idxmax()
        max_Q_time.append(event_data.loc[max_Q_index, 'Datetime'])
    
    seasons_df = pd.DataFrame()
    seasons_df['Flushing Index'] = flushing_index
    seasons_df['Hysteresis Index'] = HINew
    seasons_df['Max. Discharge (m^3/s)'] = max_Q
    seasons = []

    #Assign seasons to the data points based on max_Q_time
    for date in max_Q_time:
        if date.month in [12, 1, 2]:  # Winter dates
            season = 'Winter'
        elif date.month in [3, 4, 5]:  # Spring dates
            season = 'Spring'
        elif date.month in [6, 7, 8]:  # Summer dates
            season = 'Summer'
        else:  # Fall dates
            season = 'Fall'
        
        seasons.append(season)
    seasons_df['Season'] = seasons

    cmap = plt.cm.get_cmap('Blues_r')
    normalize = plt.Normalize(vmin=min(seasons_df['Max. Discharge (m^3/s)']), vmax=max(seasons_df['Max. Discharge (m^3/s)']))
    season_markers = {'Winter': 'o', 'Spring': '^', 'Summer': 's', 'Fall': 'd'}


    fig, ax_combined = plt.subplots()
    ax_combined.set_xlabel('Flushing Index')
    ax_combined.set_ylabel('Hysteresis Index')
    ax_combined.tick_params(axis='y')

    legend_handles = []  #To store handles for the legend
    for season in seasons_df['Season'].unique():
        season_data = seasons_df[seasons_df['Season'] == season]
        colors = cmap(normalize(season_data['Max. Discharge (m^3/s)']))  
        marker = season_markers[season]  

        scatter = ax_combined.scatter(season_data['Flushing Index'], season_data['Hysteresis Index'], c=colors, marker=marker, label=f'{season}')

        legend_handles.append(ax_combined.scatter([], [], color='none', edgecolor='black', marker=marker, label=f'{season}'))
    ax_combined.axhline(0, color='black', linewidth=0.5)
    ax_combined.axvline(0, color='black', linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])  #dummy empty array
    cbar = plt.colorbar(sm, ax=ax_combined, label='Max. Discharge (m^3/s)')
    cbar.set_label('Max. Discharge (m^3/s)')

    total_storms = (len(HINew)) 

    # Modify the legend title to include the total number of storms
    legend = ax_combined.legend(handles=legend_handles, loc='best', title=f'Season\nn = {total_storms}')
    for handle in legend.legendHandles:
        handle.set_color('none')  
        handle.set_edgecolor('black')  

    ax_combined.set_title('Indices Distribution for Gauge #002 from 2016-2021')
    plt.show()

    # =============================================================================
    # Time Series HI/FI Plot
    # =============================================================================

    plot_df = pd.DataFrame({
        'Flushing Index': flushing_index,
        'Hysteresis Index': HINew,
        'Peak Event Time': max_Q_time
    })
    plt.scatter(plot_df['Peak Event Time'], plot_df['Flushing Index'], label='Flushing Index', color='blue')
    plt.scatter(plot_df['Peak Event Time'], plot_df['Hysteresis Index'], label='Hysteresis Index', color='brown')
    plt.xlabel('Time of Maximum Discharge Per Event')
    plt.ylabel('Indices')
    plt.title('Time-Series Representation of HI and FI')
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()