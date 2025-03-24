# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:18:33 2025
Updated on Fri Mar 21 17:32:-- 2025

This script performs multiple tasks:
    1. It lists all the unique PC6
    2. It lists all the unique PC4s. For each PC4, an existing PC6 is found. 
    3. It finds the coordinates all original and added PC6s, using opencagedata
    
The results are stored in three files:
    1. A PC6 - GPS lookup
    2. A PC4 - PC6 lookup
    3. A copy of the deelrittenbestand, in which two columns are added: (1)
    best laad PC (2) best los PC

Documents 1 and 3 can then be used to make a map, see 'MakeMap_gh.py'
"""

from geopy.geocoders import OpenCage
import time
import pandas as pd
import string
from collections import Counter
import numpy as np
from datetime import datetime
from pathlib import Path
import requests
import sys

# %% 1. User input
api_key = ''
stad = ''
deelritten_bestand = ''
subfolder = Path(stad)/"" # subfolder in which result is saved

# %% 2. Make folder and get data

subfolder.mkdir(parents=True, exist_ok = True)

df_full = pd.read_csv(f"{stad}/{deelritten_bestand}", sep=';', on_bad_lines='skip')
df = df_full

# Delete NaN values and change PC4 type to int, get unique values
c_losPC4 = df['losPC4'].dropna().astype(int)
c_laadPC4 = df['laadPC4'].dropna().astype(int)
c_PC4 = np.concatenate([c_losPC4, c_laadPC4]) 
u_PC4 = pd.Series(c_PC4).unique() 

# Delete NaN values and get unique values PC6
c_losPC6 = df['losPC6'].dropna()
c_laadPC6 = df['laadPC6'].dropna()
c_PC6 = np.concatenate([c_losPC6, c_laadPC6])
u_PC6 = pd.Series(c_PC6).unique()

c_PC6 = pd.Series(c_PC6)

# %% 3. Get coordinates for each PC6
geolocator = OpenCage(api_key=api_key)
URL = "https://api.opencagedata.com/geocode/v1/json"
coordinates = []
print_counter = 1

# check if the remaining requests is enough for the loop
params = {'q': 'Rotterdam, Netherlands','key': api_key}

response = requests.get(URL, params=params)
rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining'))

if rate_limit_remaining < len(u_PC6):
    print(f"Rate limit remaining for today is {rate_limit_remaining}, while {len(u_PC6)} PC6 need to be checked in this loop. Since this is impossible, the script will now exit.")
    sys.exit()
else:
    print(f"Rate limit remaining for today is {rate_limit_remaining}")
    
for postcode in u_PC6:
        print(f"\rWorking on PC6 #{print_counter} out of {len(u_PC6)}", end="", flush=True)
        loc_result = geolocator.geocode(f"{postcode}, Netherlands")
        if loc_result:
            coordinates.append((loc_result.latitude, loc_result.longitude))
        else:
            coordinates.append((None, None))
        print_counter += 1
        time.sleep(1)
print()

# Create a DataFrame with the coordinates, add the PC6 column,  
# reorder the columns, and save
df_locPC6 = pd.DataFrame(coordinates, columns=['Latitude', 'Longitude'])
df_locPC6['PC6'] = u_PC6
df_locPC6 = df_locPC6[['PC6', 'Latitude', 'Longitude']]

current_time = datetime.now().strftime('%Y%m%d_%H%M')
filename = subfolder/f'PC6_GPS_{current_time}.csv'
df_locPC6.to_csv(filename, index=False)

# %% 4. Get coordinates for each PC4
PC4_lu = []
PC4_temp = []

# check remaining requests for the day and define conservatve variable stating
# this limit, to prevent error in loop 
params = {'q': 'Rotterdam, Netherlands','key': api_key}

response = requests.get(URL, params=params)
rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining'))
print(f"Rate limit remaining for today is {rate_limit_remaining}")

counter = 0
counter_max = rate_limit_remaining - 25
flag_max = False

counter_PC4 = 1

for postcode in [pc for pc in u_PC4 if 1000 <= pc <= 9999]: # invalid PC4 otherwise
    print(f"\rWorking on PC4 #{counter_PC4} out of {len(u_PC4)}", end="", flush=True)
    counter_PC4 += 1
    # first check if there is/are PC6 that are part of PC4. If so, determine
    # most common one
    matching_values = c_PC6[c_PC6.str.startswith(str(postcode))]
    counter += 1
    if not matching_values.empty:
        most_common = Counter(matching_values).most_common(1)[0][0]
        PC4_lu.append((postcode, most_common))
    # If not: find a PC6 that could be part of the PC4. Select the first encountered
    # PC6, starting with AA, BA, CA, etc.
    else:
        for attempt in range(26*26):
            counter += 1
            letter_combination = ''.join([string.ascii_uppercase[attempt % 26],string.ascii_uppercase[attempt // 26]])
            full_postcode = f"{postcode}{letter_combination}, Netherlands"
            location = geolocator.geocode(full_postcode)
            time.sleep(1) # stick to requests-per-minute restriction
            if location and ',' in location.address: # if the result does not 
            # include a comma, the tested letter combi does not exist
                found_location = True
                coordinates.append((full_postcode, location.latitude, location.longitude))
                PC4_lu.append((postcode, f"{postcode}{letter_combination}"))
                new_row = pd.DataFrame({'PC6': [f"{postcode}{letter_combination}"],'Latitude': [location.latitude],'Longitude': [location.longitude]})    
                df_locPC6 = pd.concat([df_locPC6, new_row], ignore_index=True)
                break
            
            if counter == counter_max:
                flag_max = True
                break
                            
        if not found_location:
            PC4_lu.append((postcode, None))
                    
        if flag_max == True:
            break
print()
        
# Create and save dataframe
df_PC4_lu = pd.DataFrame(PC4_lu, columns=['PC4', 'corr_PC6'])
filename4 = subfolder/f'PC4_lu_{current_time}.csv'
df_PC4_lu.to_csv(filename4, index=False)

# %% 5. Save updated PC6 dataframe
filename6 = subfolder/f'PC6_GPS_{current_time}.csv' 
df_locPC6.to_csv(filename6, index=False)

# %% 6. Add 'best PC' columns to deelritten file
# when original PC6 is available, include PC6 in column. It not, include the 
# PC6 that is found to be part of the original PC4. 

df_full.columns = df_full.columns.str.strip()
df_full['laadPC'] = df_full['laadPC6'].astype(str)
df_full['losPC'] = df_full['losPC6'].astype(str)

for i in range(len(df_full)):
    if pd.isna(df_full['laadPC6'][i]) and pd.isna(df_full['laadPC4'][i]) == False and df_full['laadPC4'][i] > 999:
        PC6_temp = df_PC4_lu.loc[df_PC4_lu['PC4'] == df_full['laadPC4'][i], 'corr_PC6']
        df_full.loc[i, 'laadPC'] = PC6_temp.iloc[0]
    
    if pd.isna(df_full['losPC6'][i]) and pd.isna(df_full['losPC4'][i]) == False and df_full['losPC4'][i] > 999:
        PC6_temp = df_PC4_lu.loc[df_PC4_lu['PC4'] == df_full['losPC4'][i], 'corr_PC6']
        df_full.loc[i, 'losPC'] = PC6_temp.iloc[0]

# keep only the rows for which a PC6 is available (original or found), and save
# dataframe 
df_full['laadPC'] = df_full['laadPC'].replace("", pd.NA)
df_full['laadPC'] = df_full['laadPC'].replace('nan', pd.NA)
df_full = df_full.loc[~pd.isna(df_full['laadPC'])]
df_full['losPC'] = df_full['losPC'].replace("", pd.NA)
df_full['losPC'] = df_full['losPC'].replace('nan', pd.NA)
df_full = df_full.loc[~pd.isna(df_full['losPC'])]
df_full.to_csv(subfolder/f"{stad}_bestPC.csv",sep=';', index=False)

# %% 
print(f"Done! All three documents are saved: {stad}_bestPC.csv, {filename6} and {filename4}")

