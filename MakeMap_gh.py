# -*- coding: utf-8 -*-

"""
Created on Sun Jan 12 14:49:45 2025
Updated on Fri Mar 23 20:29:-- 2025

This script is used to plot the municipalities VESDI data on a folium map. 

"""

import requests
import folium
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from datetime import datetime
import math
import numpy as np
from pathlib import Path
import shutil

version = 'v1'

# %% 1. User input

start_to_process = 0 # row data file from which to start processing. Usefull in case processing needs to be done in batches.
end_to_process = 'all' # use "all" or an int value
stad = '' # name of the city, used for creating new directories and naming files
deelrittenbestand = '' # filename of deelrittenbestand csv in which a column with the 'best available PC'  is available
postcode_lu = '' # filename of the lookup file linking PC6 to GPS-coordinates

url = 'http://localhost:8080/ors/v2/directions/driving-hgv/gpx' # URL open route services local server
headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    'Content-Type': 'application/json; charset=utf-8'
}

# %% 2. Make folder to store results and save this script in the folder, for future reference

current_time = datetime.now().strftime("%Y%m%d_%H%M") # used in filenames of created files
subfolder = Path(stad)/"results"/current_time
subfolder.mkdir(parents=True, exist_ok = True) # create subfolder in which resulting docs are saved

huidig_script = Path(__file__)  
shutil.copy(huidig_script, subfolder / huidig_script.name)

# %% 3. Read required documents

# deelrittenbestand
deelritten_df = pd.read_csv(f"{stad}/Postcode lookup/{deelrittenbestand}", sep=';', on_bad_lines='skip')
deelritten_df.columns = deelritten_df.columns.str.strip()

if end_to_process == "all":
    end_to_process = len(deelritten_df)

# lookup file PC6 - coordinates
pc6_lu_df = pd.read_csv(f"{stad}/Postcode lookup/{postcode_lu}", sep=',', on_bad_lines='skip')
pc6_lu_df.columns = pc6_lu_df.columns.str.strip()

# Merge laadPC6 with corresponding lat lon (start)
deelritten_df = deelritten_df.merge(pc6_lu_df[['PC6', 'Latitude', 'Longitude']], left_on='laadPC', right_on='PC6', how='left')
deelritten_df = deelritten_df.rename(columns={'Latitude': 'start_lat', 'Longitude': 'start_lon'})

# Merge losPC6 with corresponding lat/lon (end)
deelritten_df = deelritten_df.merge(pc6_lu_df[['PC6', 'Latitude', 'Longitude']], left_on='losPC', right_on='PC6', how='left')
deelritten_df = deelritten_df.rename(columns={'Latitude': 'end_lat', 'Longitude': 'end_lon'})

# select relevant columns
df = pd.DataFrame({
    'start_lat': deelritten_df['start_lat'][start_to_process:end_to_process].tolist(),
    'start_lon': deelritten_df['start_lon'][start_to_process:end_to_process].tolist(),
    'end_lat': deelritten_df['end_lat'][start_to_process:end_to_process].tolist(),
    'end_lon': deelritten_df['end_lon'][start_to_process:end_to_process].tolist(),
    'aantalDeelritten': deelritten_df['aantalDeelritten'][start_to_process:end_to_process].tolist(),
    'logKlasse': deelritten_df['stadslogistieke_klasse_code'][start_to_process:end_to_process].tolist(),
    'euronorm': deelritten_df['euronormKlasse'][start_to_process:end_to_process].tolist()
})

df = df.dropna(subset=['start_lat', 'start_lon', 'end_lat', 'end_lon'])

# %% 4. Function definitions

def get_route(start_coord, end_coord):
    
    start_coord = [float(start_coord[0]), float(start_coord[1])]  # [Latitude, Longitude]
    end_coord = [float(end_coord[0]), float(end_coord[1])]  # [Latitude, Longitude]
    
    body = {
        "coordinates": [
            [start_coord[1], start_coord[0]],  # [Longitude, Latitude]
            [end_coord[1], end_coord[0]]
        ]
    }
    response = requests.post(url, json=body, headers=headers)
    return response

def adjust_coordinates(coords1, coords2): 
# When there is two-way traffic on a road where lanes in opposite directions 
# are not physically separated, the segments for both directions are plotted 
# on top of each other, making the lower segment invisible. This function 
# slightly shifts the position of a segment either laterally and/or vertically,
# while maintaining the angle, to prevent the stacking of segments.

    lat_start, lon_start = coords1
    lat_end, lon_end = coords2
    L1 = abs(lat_end - lat_start)
    L2 = abs(lon_end - lon_start)
    Y0 = 0.000015 # distance by which segmnets are shifted
    
    if lat_end == lat_start:
        # horizontal segment - only has to be shifted in lat
        if lon_end > lon_start:
            lat_start = lat_start - Y0
            lat_end = lat_end - Y0
        elif lon_end < lon_start:
            lat_start = lat_start + Y0
            lat_end = lat_end + Y0
            
    elif lon_end == lon_start:
        # vertical segment - only has to be shifted in lon
        if lat_end > lat_start:
            lon_start = lon_start + Y0
            lon_end = lon_end + Y0
        elif lat_end < lat_start:
            lon_start = lon_start - Y0
            lon_end = lon_end - Y0   
    
    else:
        # segment at differnt angle - lat and not need to be shifted
        alpha = math.atan(L2/L1)
        Y1 = math.sin(alpha)*Y0
        Y2 = math.cos(alpha)*Y0
    
        if lat_end > lat_start and lon_end > lon_start:
            lat_start = lat_start - Y1
            lat_end = lat_end - Y1
            lon_start = lon_start + Y2
            lon_end = lon_end + Y2
            
        elif lat_end > lat_start and lon_end < lon_start:
            lat_start = lat_start + Y1
            lat_end = lat_end + Y1
            lon_start = lon_start + Y2
            lon_end = lon_end + Y2  
        
        elif lat_end < lat_start and lon_end < lon_start:
            lat_start = lat_start + Y1
            lat_end = lat_end + Y1
            lon_start = lon_start - Y2
            lon_end = lon_end - Y2
        
        elif lat_end < lat_start and lon_end > lon_start:
            lat_start = lat_start - Y1
            lat_end = lat_end - Y1
            lon_start = lon_start - Y2
            lon_end = lon_end - Y2
        else:
            print(f"Splitting two way traffic went wrong between these coordinates: {coords1}, {coords2} ")
        
    new_coords1 = (round(lat_start,6), round(lon_start,6))
    new_coords2 = (round(lat_end, 6), round(lon_end,6))

    return new_coords1, new_coords2

def get_color(frequency):
    # gets color for the segment on the map. For clarity buckets are used,
    # instead of a continous color spectrum
    
    if frequency <= bu[1]:
        red_value = 246
        green_value = 248
        blue_value = 174
    elif bu[1] < frequency <= bu[2]:
        red_value = 237
        green_value = 205
        blue_value = 135
    elif bu[2] < frequency <= bu[3]:
        red_value = 225
        green_value = 174
        blue_value = 54
    elif bu[3] < frequency <= bu[4]:
        red_value = 254
        green_value = 128
        blue_value = 48
    elif bu[4] < frequency <= bu[5]:
        red_value = 241
        green_value = 93
        blue_value =  89
    elif bu[5] < frequency <= bu[6]:
        red_value = 217
        green_value = 81
        blue_value =  95
    else: 
        red_value = 176
        green_value = 38
        blue_value =  51
    
    return f"rgb({red_value}, {green_value}, {blue_value})"

def calculate_euclidean_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

# %% 5. Loop through DataFrame to determine routes and frequency of found segments
print('1. Determine routes')

segment_data = defaultdict(lambda:{'frequency':0, 'e6':0, 'e5':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '90':0}) 

# get route segmnets for all entries
for index, row in df.iterrows():
    print(f"\rWorking on route {index + 1} out of {end_to_process}", end="", flush=True)
    start_coord = [row['start_lat'], row['start_lon']]
    end_coord = [row['end_lat'], row['end_lon']]

    response = get_route(start_coord, end_coord)

    if response.status_code == 200:
        root = ET.fromstring(response.text)
        lat_lon_list = []

        for rtept in root.findall('.//{https://raw.githubusercontent.com/GIScience/openrouteservice-schema/main/gpx/v2/ors-gpx.xsd}rtept'):
            lat = float(rtept.get('lat'))
            lon = float(rtept.get('lon'))
            lat_lon_list.append([lat, lon])
        
        add_e5 = 0
        add_e6 = 0
        if row['euronorm'] =='6':
            add_e6 = row['aantalDeelritten']
        elif row['euronorm'] =='0-5':
            add_e5 = row['aantalDeelritten']
                
        LK = str(row['logKlasse'])

        for i in range(1, len(lat_lon_list)):
            segment = (tuple(lat_lon_list[i - 1]), tuple(lat_lon_list[i]))                
            segment_data[segment]['frequency'] += row['aantalDeelritten']
            segment_data[segment]['e5'] += add_e5
            segment_data[segment]['e6'] += add_e6
            segment_data[segment][LK] += row['aantalDeelritten']
            
    else:
        print(f"Fout bij API-aanroep voor route {index}, {start_coord} - {end_coord}: {response.status_code} - {response.reason}")
        
print()

df_seg1 = pd.DataFrame(segment_data.items(), columns=['coordinates', 'data'])

# Split data column in separate columns, delete data column afterwards and 
# replace logistics code by logistics genre
df_seg1[['frequency', 'e6', 'e5', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '90']] = pd.DataFrame(df_seg1['data'].to_list(), index=df_seg1.index)
df_seg1 = df_seg1.drop(columns='data')
df_seg1.columns = ['coordinates','frequency', 'e6', 'e5', 'Afval (bedrijven)', 'Afval consumenten','Bouw (gebouwen, overig)',' Bouw (infrastructuur)','Bouw (service)','Bouw (specialisten)','Diensten en service','Facilitair','Groothandel','Horeca (food / non-food)','Industrie','Post en pakketten','Retail (food)','Retail (non-food regulier)','Retail (non-food specialisten)','Tweemans leveringen','Vers-thuisbezorging','***Lege_rit***']

# determine angle of all segments
df_seg1['helling'] = df_seg1['coordinates'].apply(
    lambda coords: round((coords[1][1] - coords[0][1]) / (coords[1][0] - coords[0][0]), 3)
    if (coords[1][0] - coords[0][0]) != 0 else np.nan
)

# save the dataframe
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = subfolder/f"routes{version}_{stad}_{current_time}_{start_to_process}-{end_to_process}.csv"
df_seg1.to_csv(csv_filename, index=False)

# %% 6. Adjust coordinates of overlapping segments
print("2. Adjust coordinates of matching segments")

# Adjust segments where two-way traffic-segments are stacked
matching_indexes = []
reversed_coords_dict = {}

for i, row in df_seg1.iterrows():
    coords1, coords2 = row['coordinates']  
    reversed_key = (coords2, coords1)
    
    if reversed_key in reversed_coords_dict:
        other_i = reversed_coords_dict[reversed_key]
        matching_indexes.append((i, other_i))
    
    reversed_coords_dict[(coords1, coords2)] = i

df_matches = pd.DataFrame(matching_indexes, columns=['index_1', 'index_2'])

for idx, match in df_matches.iterrows():
    index_1 = match['index_1']
    index_2 = match['index_2']
    
    coords1, coords2 = df_seg1.loc[index_1, 'coordinates']  
    coords3, coords4 = df_seg1.loc[index_2, 'coordinates']  
    
    adjusted_coords1, adjusted_coords2 = adjust_coordinates(coords1, coords2)
    adjusted_coords3, adjusted_coords4 = adjust_coordinates(coords3, coords4)
    
    df_seg1.at[index_1, 'coordinates'] = adjusted_coords1, adjusted_coords2  
    df_seg1.at[index_2, 'coordinates'] = adjusted_coords3, adjusted_coords4  

print(f"{len(df_matches)} locations where two way traffic was presented in segments on top of each other - adjusted ({round(2*100*((len(df_matches)/len(df_seg1))))}% of all segments)")

# Adjust segments that are partly overlapping and in same direction 
start_coord_dict = {}
end_coord_dict = {}
tellen = 0 

last_columns = ['frequency', 'e6', 'e5', 'Afval (bedrijven)', 'Afval consumenten','Bouw (gebouwen, overig)',' Bouw (infrastructuur)','Bouw (service)','Bouw (specialisten)','Diensten en service','Facilitair','Groothandel','Horeca (food / non-food)','Industrie','Post en pakketten','Retail (food)','Retail (non-food regulier)','Retail (non-food specialisten)','Tweemans leveringen','Vers-thuisbezorging','***Lege_rit***']

for i in range(len(df_seg1)):
    row1 = df_seg1.iloc[i]
    coords = row1['coordinates']
    lat1, lon1 = coords[0]  # Start
    lat2, lon2 = coords[1]  # End
    
    if (lat1, lon1) not in start_coord_dict:
        start_coord_dict[(lat1, lon1)] = []
    start_coord_dict[(lat1, lon1)].append(i)
    
    if (lat2, lon2) not in end_coord_dict:
        end_coord_dict[(lat2, lon2)] = []
    end_coord_dict[(lat2, lon2)].append(i)

   
tellen = 0    

for i in range(len(df_seg1)):
    # adjust segments with equal start-coordinates
    exitloopS = False
    row1 = df_seg1.iloc[i]
    lat1_1, lon1_1 = row1['coordinates'][0]  # Start of segment 1
    lat2_1, lon2_1 = row1['coordinates'][1]  # End of segment 1
    helling1 = row1['helling']
    
    matching_start_indices = start_coord_dict.get((lat1_1, lon1_1), [])

    for j in matching_start_indices:
        if i != j:  # Not compare are row with itself
            row2 = df_seg1.iloc[j]
            
            lat1_2, lon1_2 = row2['coordinates'][0]
            lat2_2, lon2_2 = row2['coordinates'][1]
            
            if abs(row2['helling'] - helling1)<0.1: 
               distance_1 = calculate_euclidean_distance((lat1_1, lon1_1), (lat2_1, lon2_1))
               distance_2 = calculate_euclidean_distance((lat1_2, lon1_2), (lat2_2, lon2_2))
               
               if distance_1 > distance_2:
                   df_seg1.at[i, 'coordinates'] = [df_seg1['coordinates'][j][1], df_seg1['coordinates'][i][1]]  
                   df_seg1.loc[j, last_columns] += df_seg1.loc[i, last_columns]
                   tellen +=1
                   
                   start_coord_dict[(lat1_1, lon1_1)].remove(i)
                   if (lat2_2, lon2_2) not in start_coord_dict:
                       start_coord_dict[(lat2_2, lon2_2)] = []
                   start_coord_dict[(lat2_2, lon2_2)].append(i)

                   lat1_1 = lat2_2
                   lon1_1 = lon2_2
                   
                   break
               
               elif distance_1 == distance_2:
                   df_seg1.loc[j, last_columns] += df_seg1.loc[i, last_columns]
                   df_seg1.loc[i,:] = 0
                   end_coord_dict[(lat2_1, lon2_1)].remove(i)
                   start_coord_dict[(lat1_1, lon1_1)].remove(i)   
                   
                   exitloopS = True
                   break  
                 
               else:                   
                   df_seg1.at[j, 'coordinates'] = [df_seg1['coordinates'][i][1], df_seg1['coordinates'][j][1]]  
                   df_seg1.loc[i, last_columns] += df_seg1.loc[j, last_columns]
                   tellen +=1
                   
                   start_coord_dict[(lat1_2, lon1_2)].remove(j)               
                   if (lat2_1, lon2_1) not in start_coord_dict:
                       start_coord_dict[(lat2_1, lon2_1)] = []
                   start_coord_dict[(lat2_1, lon2_1)].append(j)
                   
                   break
    if exitloopS == True:
        continue

    # adjust segments with equal end-coordinates

    matching_end_indices = end_coord_dict.get((lat2_1, lon2_1), [])

    for j in matching_end_indices:
        if i != j:  # don't compare a row with itself
            row2 = df_seg1.iloc[j]
            
            lat1_2, lon1_2 = row2['coordinates'][0]
            lat2_2, lon2_2 = row2['coordinates'][1]
            
            if abs(row2['helling'] - helling1)<0.1: 
               distance_1 = calculate_euclidean_distance((lat1_1, lon1_1), (lat2_1, lon2_1))
               distance_2 = calculate_euclidean_distance((lat1_2, lon1_2), (lat2_2, lon2_2))
               
               if distance_1 > distance_2:
                   df_seg1.at[i, 'coordinates'] = [df_seg1['coordinates'][i][0], df_seg1['coordinates'][j][0]]  
                   df_seg1.loc[j, last_columns] += df_seg1.loc[i, last_columns]

                   tellen +=1
                   
                   end_coord_dict[(lat2_1, lon2_1)].remove(i)
                   if (lat1_2, lon1_2) not in end_coord_dict:
                       end_coord_dict[(lat1_2, lon1_2)] = []
                   end_coord_dict[(lat1_2, lon1_2)].append(i)
                  
                   lat2_1 = lat1_2
                   lon2_1 = lon1_2
                   
                   break
               
               elif distance_1 == distance_2:
                   df_seg1.loc[j, last_columns] += df_seg1.loc[i, last_columns]
                   df_seg1.loc[i,:] = 0
                   end_coord_dict[(lat2_1, lon2_1)].remove(i)
                   start_coord_dict[(lat1_1, lon1_1)].remove(i)   
                                      
                   break
               
               else:                   
                   df_seg1.at[j, 'coordinates'] = [df_seg1['coordinates'][j][0], df_seg1['coordinates'][i][0]]  
                   df_seg1.loc[i, last_columns] += df_seg1.loc[j, last_columns]
                   tellen +=1
                   
                   end_coord_dict[(lat2_2, lon2_2)].remove(j)               
                   if (lat1_1, lon1_1) not in end_coord_dict:
                       end_coord_dict[(lat1_1, lon1_1)] = []
                   end_coord_dict[(lat1_1, lon1_1)].append(j)
                   
                   break

print(f"{tellen} locations where segments in the same direction were overlapping - adjusted ({round(100*((tellen/len(df_seg1))))}% of all segments)")

df_seg1 = df_seg1[df_seg1['coordinates'] != 0]

# save the dataframe
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = subfolder/f"routes_adjusted{version}_{stad}_{current_time}_{start_to_process}-{end_to_process}.csv"
df_seg1.to_csv(csv_filename, index=False)

# %% 7. Make map
print('3. Make map')

# Make map object, centering at avg lat-lon coordinates
lat_avg = np.mean(np.array([coord[0][0] for coord in df_seg1['coordinates'][1:1000]]))
lon_avg = np.mean(np.array([coord[0][1] for coord in df_seg1['coordinates'][1:1000]]))
m = folium.Map(location=[lat_avg, lon_avg], zoom_start=11, tiles='CartoDB positron')

# Determine color bins
max_frequency = int(df_seg1['frequency'].max())
min_frequency = int(df_seg1['frequency'].min())

no_buckets = 7 
step = (max_frequency-min_frequency)/no_buckets

bu = [
      min_frequency,
      min_frequency + step,
      min_frequency + (2*step),
      min_frequency + (3*step),
      min_frequency + (4*step),
      min_frequency + (5*step),
      min_frequency + (6*step),
      max_frequency
      ]

if max_frequency > 15000 and min_frequency < 1000:
    # to use values that are rounded up/down when possible, so legend is easier
    bu = [
          min_frequency,
          5000,
          5000 + (1*step),
          5000 +  (2*step),
          5000 +  (3*step),
          5000 +  (4*step),
          5000 +  (5*step),
          max_frequency
          ]
    
    bu = [int(round(bu[i], -3)) if i > 1 and i != len(bu)-1 else bu[i] for i in range(len(bu))]

# display segmenst on map
for i, row in df_seg1.iterrows():
    coords1, coords2 = row['coordinates'] 
    start_lat, start_lon = coords1
    end_lat, end_lon = coords2
    frequency = round(row['frequency'])
    euro5perc = round((row['e5']/frequency)*100)
    euro6perc = round((row['e6']/frequency)*100)
    
    SLC = row.iloc[-18:]
    SLC = pd.to_numeric(SLC, errors='coerce')
    no_SLC = SLC[SLC>0]
    
    # get the top 3/2/1 logistics-genres per segment for in pop-up
    # text depending on number of distinct logistics-genres on segment:
    if len(no_SLC) > 2:
        top_columns = no_SLC.nlargest(3)
        top_col_list = [(col, top_columns[col]) for col in top_columns.index]
        SLK1 = top_col_list[0][0]
        freq1 = round(((top_col_list[0][1])/frequency)*100)
        SLK2 = top_col_list[1][0]
        freq2 = round(((top_col_list[1][1])/frequency)*100)
        SLK3 = top_col_list[2][0]
        freq3 = round(((top_col_list[2][1])/frequency)*100)
        popup_text = f"{frequency} deelritten <br><br><b> Euronorm voertuig</b><br> {euro5perc}% euro 0-5 <br>{euro6perc}% euro 6 <br> <br> <b> Stadslogistieke klasse met meeste aantal deelritten </b> <br> {freq1}% {SLK1}<br> {freq2}% {SLK2}<br> {freq3}% {SLK3}"
        
    elif len(no_SLC) ==2:
         top_columns = no_SLC.nlargest(2)
         top_col_list = [(col, top_columns[col]) for col in top_columns.index]
         SLK1 = top_col_list[0][0]
         freq1 = round(((top_col_list[0][1])/frequency)*100)
         SLK2 = top_col_list[1][0]
         freq2 = round(((top_col_list[1][1])/frequency)*100)
         popup_text = f"{frequency} deelritten <br><br><b> Euronorm voertuig</b><br> {euro5perc}% euro 0-5 <br>{euro6perc}% euro 6 <br> <br> <b> Stadslogistieke klasse met meeste aantal deelritten </b> <br> {freq1}% {SLK1}<br> {freq2}% {SLK2}"
    
    elif len(no_SLC) ==1:
        top_columns = no_SLC.nlargest(1)
        top_col_list = [(col, top_columns[col]) for col in top_columns.index]
        SLK1 = top_col_list[0][0]
        freq1 = round(((top_col_list[0][1])/frequency)*100)
        popup_text = f"{frequency} deelritten <br><br> <b> Euronorm voertuig</b><br> {euro5perc}% euro 0-5 <br>{euro6perc}% euro 6 <br><br><b> Stadslogistieke klasse met meeste aantal deelritten </b> <br> {freq1}% {SLK1}"
        
    else: 
        popup_text = f"{frequency} deelritten <br> <br> <b> Euronorm voertuig</b> <br>{euro5perc}% euro 0-5 <br>{euro6perc}% euro 6"
        
    popup = folium.Popup(popup_text, max_width=300, min_width=200)  # Pas max_width en min_width aan voor grotere popups

    # Add segment Polyline on map
    folium.PolyLine(
        locations = [(start_lat, start_lon), (end_lat, end_lon)],
        color = get_color(frequency),
        weight = 3,
        opacity = 1.0,
        tooltip = f"{frequency} deelritten",  # Tooltip met frequentie
        popup = popup
    ).add_to(m)
    
# Make legend, get colors for the legend and add legend to map
legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 200px; height: 180px; 
                 background-color: white; border:2px solid black; z-index:9999;
                 font-size: 14px; padding: 10px;">
        <b>Aantal deelritten</b><br>
'''
color_scale = [
     (get_color(max_frequency), f"{int(bu[6])+1} - {max_frequency}"), 
     (get_color(bu[5]+1), f"{int(bu[5])+1} - {int(bu[6])}"),  
     (get_color(bu[4]+1), f"{int(bu[4])+1} - {int(bu[5])}"), 
     (get_color(bu[3]+1), f"{int(bu[3])+1} - {int(bu[4])}"),  
     (get_color(bu[2]+1), f"{int(bu[2])+1} - {int(bu[3])}"),  
     (get_color(bu[1]+1), f"{int(bu[1])+1} - {int(bu[2])}"),  
     (get_color(min_frequency), f"{min_frequency} - {int(bu[1])}"), 
 ]

for color, label in color_scale:
    legend_html += f'<div><i style="background-color:{color}; width: 20px; height: 17px; float: left; margin-right: 5px;"></i>{label}</div>'

legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# save map
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = subfolder/f"routes{version}_{stad}_{current_time}_{start_to_process}-{end_to_process}.html"
m.save(filename, embed=True)

#%%
print("Done!")
