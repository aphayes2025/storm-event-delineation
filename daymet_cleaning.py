#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:18:00 2024

@author: lizamclatchy
"""

from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def get_date_from_day(year, day):
    """
    Converts a specific day of the year into a calendar date.

    Parameters:
        year (int or str): The year for which the day-of-year is provided.
        day (int or str): The day of the year (1-based index).

    Returns:
        str: The date in 'MM/DD/YYYY' format.
    """
    start_of_year = datetime.datetime(int(year), 1, 1)
    date = start_of_year + datetime.timedelta(int(day) - 1)
    return date.strftime('%m/%d/%Y')


def get_prior_max_discharge(event_date, months_prior, df_copy):
        """
        Calculate the maximum discharge prior to a given date over a specified period in months.
        
        Args:
            event_date (datetime): The event date to calculate prior from.
            months_prior (int): Number of months prior to the event date.
            df (DataFrame): DataFrame containing discharge data with a datetime column.
        
        Returns:
            float: The maximum discharge value in the specified period.
        """
        start_date = event_date - pd.DateOffset(months=months_prior)
        filtered_df = df_copy[(df_copy['Datetime'] >= start_date) & (df_copy['Datetime'] < event_date)]
        max_discharge = filtered_df['Discharge (m^3/s)'].max()
        return max_discharge

if __name__ == "__main__":
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
    m3s2cfs = 0.0283168
    df_copy['Discharge (m^3/s)'] = df_copy['Discharge (cfs)'] * m3s2cfs

    csv_threshold = 'event-data/events_above_threshold002.csv'
    df_events_above_threshold = pd.read_csv(csv_threshold)
    #CHange to datetime
    df_events_above_threshold['max_Q_time'] = pd.to_datetime(df_events_above_threshold['max_Q_time'])
    df_events_above_threshold['event_end_time'] = pd.to_datetime(df_events_above_threshold['event_end_time'])
    df_events_above_threshold['event_start_time'] = pd.to_datetime(df_events_above_threshold['event_start_time'])
    #Lists of all relevant parameters
    event_start_time = df_events_above_threshold['event_start_time'].tolist()
    i_start = df_events_above_threshold['i_start'].tolist()
    i_end = df_events_above_threshold['i_end'].tolist()
    max_Q = df_events_above_threshold['max_Q'].tolist()
    max_Q_time = df_events_above_threshold['max_Q_time'].tolist()
    max_C = df_events_above_threshold[' max_C'].tolist()
    event_end_time = df_events_above_threshold['event_end_time'].tolist()
    HINew = df_events_above_threshold['HINew'].tolist()
    flushing_index = df_events_above_threshold['flushing_index'].tolist()
    event_number = df_events_above_threshold['event number'].tolist()

    # =============================================================================
    # STORM ATTRIBUTES
    # =============================================================================   
    randomforest_df = pd.DataFrame()  
    randomforest_df['HInew'] = HINew
    randomforest_df['flushing_index'] = flushing_index
    randomforest_df['Event Number'] = event_number

    # =============================================================================
    # #Season 
    # base theseason on month of the start of each event 
    # one-hot
    # =============================================================================
    winter = [0] * len(i_start)
    spring = [0] * len(i_start)
    summer = [0] * len(i_start)
    autumn = [0] * len(i_start)
    for i in range(len(event_start_time)):
        month = event_start_time[i].month
        x = 1
        if 1 <= month <= 3:
            winter[i] = x
        elif 4 <= month <= 6:
            spring[i] = x
        elif 7 <= month <= 9:
            summer[i] = x
        elif 10 <= month <= 12:
            autumn[i] = x

    #Turn the lists into dataframes
    randomforest_df['Winter'] = pd.DataFrame(winter)
    randomforest_df['Spring'] = pd.DataFrame(spring)
    randomforest_df['Summer'] = pd.DataFrame(summer)
    randomforest_df['Autumn'] = pd.DataFrame(autumn)

    #Leaf on/off ratio
    leaf_index = [] 
    for i in range(len(event_start_time)):
        month = event_start_time[i].month
        if 5 <= month <= 10:
            leaf_index.append(1)  
        else:
            leaf_index.append(0)  

    randomforest_df['Leaf Index'] = leaf_index
    # Goal: import daymet data for randomforesting 
    # isolate daily temperatures and daily precip
    randomforest_df['Max Disch m^3/s'] = pd.DataFrame(max_Q)
    randomforest_df['Max SSC'] = pd.DataFrame(max_C)  
    #Peak ratio
    peak_ratio_list = []
    for i in range(len(max_Q)):
        peak_ratio = max_Q[i] / max_C[i]
        peak_ratio_list.append(peak_ratio)
    randomforest_df['Peak Ratio'] = pd.DataFrame(max_C)  

    #FLOOD INTENSITY
    time2peakQ = []
    time2peakQ_min = []
    for i in range(len(max_Q_time)):
        Qpeaktime = max_Q_time[i] - event_start_time[i]
        time2peakQ.append(Qpeaktime)
        time2peakQ_min.append(Qpeaktime.total_seconds()/60)
        
    floodintensity = []

    for i in range(len(time2peakQ)):
        deltaQ = randomforest_df['Max Disch m^3/s'][i] - df_copy['Discharge (m^3/s)'][i_start[i]]
        deltaTQ_hours = time2peakQ[i].total_seconds()/3600
        deltaTQ = deltaTQ_hours # Use the converted value
        floodintensity.append(deltaQ / deltaTQ)

    randomforest_df['Flood Intensity (cfs/h)'] = pd.DataFrame(floodintensity)

    #Time between discharge peaks
    time_btw_Q_peaks = []
    for i in range(len(max_Q_time)-1):
        diff = max_Q_time[i+1] - max_Q_time[i] # find time diff between discharge peaks
        time_btw_Q_peaks.append(diff)

    # =============================================================================
    # DAYMET ATTRIBUTES
    # =============================================================================   
    df_daymet = pd.read_csv('Daymet002.csv')

    # Apply the function to create the 'Date' column
    df_daymet['Date'] = df_daymet.apply(lambda row: get_date_from_day(row['year'], row['yday']), axis=1)

    #drop the other columns
    df_daymet.drop('year', axis=1, inplace=True)
    df_daymet.drop('yday', axis=1, inplace=True)

    #Now df just contains 'Datetime" and Precip in mm/day
    #Format datetime into actual datetime
    df_daymet['Date'] = pd.to_datetime(df_daymet['Date'])

    # =============================================================================
    # since this data is all daily, convert your start and end events to just DATE no hr,min,sec
    # might need to turn this into Datetiime objects
    # =============================================================================
    eventstart_date = [dt.replace(hour=0, minute=0, second=0) for dt in event_start_time]
    eventend_date = [dt.replace(hour=0, minute=0, second=0) for dt in event_end_time]

    # =============================================================================
    # Antecedant soil moisture
    # days prior in datetime format
    # =============================================================================

    sevendb4 = np.timedelta64(7, 'D')
    threedb4 = np.timedelta64(3, 'D')
    onedb4 = np.timedelta64(1, 'D')
        
    one_day_precip = []
    three_days_precip = []
    seven_days_precip = []

    # Iterate over event start times
    for i in range(len(event_start_time)):
        
        # Extract precipitation data for specific date ranges
        daymet_one_day_range = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i] - onedb4) & (df_daymet['Date'] <= event_start_time[i])]
        daymet_three_days_range = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i] - threedb4) & (df_daymet['Date'] <= event_start_time[i])]
        daymet_seven_days_range = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i] - sevendb4) & (df_daymet['Date'] <= event_start_time[i])]
        
        # Sum the precipitation values for each date range
        one_day_precip_sum = daymet_one_day_range['prcp (mm/day)'].sum()
        three_days_precip_sum = daymet_three_days_range['prcp (mm/day)'].sum()
        seven_days_precip_sum = daymet_seven_days_range['prcp (mm/day)'].sum()
        
        # Store the results in lists
        one_day_precip.append(one_day_precip_sum)
        three_days_precip.append(three_days_precip_sum)
        seven_days_precip.append(seven_days_precip_sum)
        
        
    randomforest_df['1-day Prior Precip (mm)'] = pd.DataFrame(one_day_precip)
    randomforest_df['3-day Prior Precip (mm)'] = pd.DataFrame(three_days_precip)  
    randomforest_df['7-day Prior Precip (mm)'] = pd.DataFrame(seven_days_precip)
        
    # =============================================================================
    # Daily Temperature for each event
    # avg over the event for:
    #   - daily mins 
    #   - daily maxs
    #   - daily avgs
    # =============================================================================
    avg_temp = []
    for i in range(len(df_daymet)):
        avg_temp.append((df_daymet['tmin (deg c)'][i] + df_daymet['tmax (deg c)'][i])/2 )
        
    df_daymet['tavg (deg c)'] = pd.DataFrame(avg_temp)

    min_avg_temp = []
    max_avg_temp = []
    avg_avg_temp = []
    for i in range(len(event_start_time)):
        
        # Extract event specific date range
        daymet_event = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i]) & (df_daymet['Date'] <= event_end_time[i])]
        
        # avg the temperature values for each date range
        min_avg_temp.append(daymet_event['tmin (deg c)'].mean())
        max_avg_temp.append(daymet_event['tmax (deg c)'].mean())
        avg_avg_temp.append(daymet_event['tavg (deg c)'].mean())

        
    randomforest_df['Avg Daily Minimum Temp (deg c)'] = pd.DataFrame(min_avg_temp)
    randomforest_df['Avg Daily Maximum Temp (deg c)'] = pd.DataFrame(max_avg_temp)  
    randomforest_df['Avg Daily Average Temp (deg c)'] = pd.DataFrame(avg_avg_temp)

    avgdayl = []
    for i in range(len(event_start_time)):
        
        # Extract event specific date range
        daymet_event = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i]) & (df_daymet['Date'] <= event_end_time[i])]
        
        # avg the temperature values for each date range
        avgdayl.append(daymet_event['dayl (s)'].mean())

    #Add these means to the RandomForest DataFrame
    randomforest_df['Avg Day Length (s)'] = pd.Series(avgdayl)

    #Find avg vapor pressure over event
    avgvp = []
    for i in range(len(event_start_time)):
        
        # Extract event specific date range
        daymet_event = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i]) & (df_daymet['Date'] <= event_end_time[i])]
        
        # avg the temperature values for each date range
        avgvp.append(daymet_event['vp (Pa)'].mean())

    #Add these means to the RandomForest DataFrame
    randomforest_df['Avg Dvp (Pa)'] = pd.Series(avgvp)

    #FInd avg solar radiation over event
    avgsrad = []
    for i in range(len(event_start_time)):
        
        # Extract event specific date range
        daymet_event = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i]) & (df_daymet['Date'] <= event_end_time[i])]
        
        # avg the temperature values for each date range
        avgsrad.append(daymet_event['srad (W/m^2)'].mean())

    #Add these means to the RandomForest DataFrame
    randomforest_df['Avg Srad (W/m^2)'] = pd.Series(avgsrad)

    #Find avg SWE over event
    avgswe = []
    for i in range(len(event_start_time)):
        
        # Extract event specific date range
        daymet_event = df_daymet.loc[(df_daymet['Date'] >= event_start_time[i]) & (df_daymet['Date'] <= event_end_time[i])]
        
        # avg the temperature values for each date range
        avgswe.append(daymet_event['swe (kg/m^2)'].mean())

    #Add these means to the RandomForest DataFrame
    randomforest_df['Avg Swe (kg/m^2)'] = pd.Series(avgswe)
    randomforest_df['Max Discharge 3 Months Prior'] = df_events_above_threshold['event_start_time'].apply(lambda x: get_prior_max_discharge(x, 3, df_copy))
    randomforest_df['Max Discharge 6 Months Prior'] = df_events_above_threshold['event_start_time'].apply(lambda x: get_prior_max_discharge(x, 6, df_copy))

    #Scale variables
    standard_scaler = StandardScaler()
    randomforest_df_scaled = pd.DataFrame(standard_scaler.fit_transform(randomforest_df), columns=randomforest_df.columns)

    #Correlation plot
    correlation_matrix = randomforest_df.corr()
    plt.figure(figsize=(12, 10))  
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix of Variables (#70)')
    plt.xticks(rotation=45, ha='right')  
    plt.yticks(rotation=0) 
    plt.tight_layout()  
    plt.show()

    randomforest_df.to_csv('randomforest002.csv', index=False)
    randomforest_df_scaled.to_csv('randomforest_scaled002.csv', index=False)
