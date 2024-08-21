#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


START_YEAR = 2020
#%%

# Define the directory path for the years of interest
data_dir_path = Path("/Users/tesatesa/devaus/datasets/CRNS0101-05-202403110550/CRNS0101-05-202403110550")

# Create a dictionary to hold the list of stations for each year
stations_per_year = {}

# Iterate over the years and the files in the respective directories
for year in range(START_YEAR, 2025):  # Assuming the user wants to check from 2006 as mentioned in readme
    year_dir_path = data_dir_path / str(year)
    if year_dir_path.is_dir():
        # List all files in the directory for the year
        files = os.listdir(year_dir_path)
        
        # Extract station names from the files
        stations = set()
        for file_name in files:
            # Filename format is CRNS0101-05-YYYY-AK_Aleknagik_1_NNE.txt, we need the part after YYYY-
            parts = file_name.split("-")
            if len(parts) > 3:
                station_name = parts[3].split(".")[0]  # Get the part before '.txt'
                stations.add(station_name)
        
        # Update the dictionary with the list of stations for the year
        stations_per_year[year] = stations

# Now find the intersection of all sets to determine which stations are in all files
common_stations = set.intersection(*stations_per_year.values())
common_stations

# %%
len(common_stations)
# 94

#%%
all_stations = set.union(*stations_per_year.values())
# %%
len(all_stations)

#%% for each station find the number of files it is present in, and the years it was and wasn't
# Assuming 'stations_per_year' is your existing dictionary with years as keys and sets of stations as values

# Initialize dictionaries to hold presence and absence data
station_presence = {station: [] for station in all_stations}
station_absence = {station: [] for station in all_stations}

# Populate the presence and absence dictionaries
for year in range(START_YEAR, 2025):
    for station in all_stations:
        if station in stations_per_year.get(year, set()):
            station_presence[station].append(year)
        else:
            station_absence[station].append(year)

# Now, 'station_presence' will have years the station was present and '
station_absence

# get stations that are present in all years, absence is empty
empty = [k for k, v in station_absence.items() if not v]
# %%
len(empty)
# 94
#%%

headers_path = './headers.txt'

# Read the headers
with open(headers_path, 'r') as file:
    file.readline()
    headers = file.readline().strip().split()

# Define the column formats with their respective missing/invalid data indicators
column_formats = {
    'WBANNO': (-9999, int),
    'UTC_DATE': (-9999, int),
    'UTC_TIME': (-9999, int),
    'LST_DATE': (-9999, int),
    'LST_TIME': (-9999, int),
    'CRX_VN': ('-9999', str),
    'LONGITUDE': (-999.0, float),
    'LATITUDE': (-999.0, float),
    'AIR_TEMPERATURE': (-9999.0, float),
    'PRECIPITATION': (-9999.0, float),
    'SOLAR_RADIATION': (-99999.0, float),
    'SR_FLAG': (-99, int),
    'SURFACE_TEMPERATURE': (-9999.0, float),
    'ST_TYPE': ('-99', str),
    'ST_FLAG': (-99, int),
    'RELATIVE_HUMIDITY': (-9999.0, float),
    'RH_FLAG': (-99, int),
    'SOIL_MOISTURE_5': (-99.0, float),
    'SOIL_TEMPERATURE_5': (-9999.0, float),
    'WETNESS': (-9999, int),
    'WET_FLAG': (-99, int),
    'WIND_1_5': (-99.0, float),
    'WIND_FLAG': (-99, int)
}
def load_and_prepare_data(file_path, station_suffix):
    # Load the data with specified headers
    df = pd.read_csv(file_path, delim_whitespace=True, names=headers, comment='#')
    
    # Create a timestamp from UTC_DATE and UTC_TIME and set it as the index
    df['TIMESTAMP'] = pd.to_datetime(df['UTC_DATE'].astype(str) + df['UTC_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    df.set_index('TIMESTAMP', inplace=True)
    
    # Define columns to retain based on dynamic content
    dynamic_columns = ['AIR_TEMPERATURE', 'PRECIPITATION', 'SOLAR_RADIATION', 'SURFACE_TEMPERATURE', 'RELATIVE_HUMIDITY', 'SOIL_MOISTURE_5', 'WETNESS', 'WIND_1_5']

    # Initialize a new DataFrame to hold processed columns
    processed_df = pd.DataFrame(index=df.index)

    # Process each column according to column_formats
    for column in dynamic_columns:
        # Construct the full column name with the station suffix
        full_column_name = f"{column}___{station_suffix}"
        
        # Check if the column is present in the data (considering dynamic naming with station_suffix)
        if column in headers:
            # Handle missing values based on column_formats
            missing_value, dtype = column_formats[column]
            processed_column = df[column].replace(missing_value, np.nan).astype(float)
            
            # Rename and add to the processed DataFrame
            processed_df[full_column_name] = processed_column

    return processed_df

# Define the directory path where the data is stored
data_dir_path = Path("./")

# Define the range of years of interest
year_range = range(START_YEAR, 2025)

# Initialize an empty DataFrame for the mega table
mega_table = pd.DataFrame()
# Initialize a list to hold DataFrames for each year
year_data_list = []

from tqdm import tqdm
# Iterate over each year in the range
for year in tqdm(year_range):
    year_data = []  # To hold data DataFrames for the current year
    year_dir_path = data_dir_path / str(year)
    
    # Ensure the directory for the year exists
    if not year_dir_path.is_dir():
        continue

    # Iterate over files in the year directory
    for file_name in os.listdir(year_dir_path):
        station_name = file_name.split('-')[-1].rsplit('.', 1)[0]  # Extract station name from file name
        station_suffix = station_name  # Optional: create a suffix for column names

        # Check if the station is in the common stations set
        if station_name in common_stations:
            file_path = year_dir_path / file_name
            station_data = load_and_prepare_data(file_path, station_suffix)
            year_data.append(station_data)

    # Merge all station data for the year and append to the list
    if year_data:
        year_data_df = pd.concat(year_data, axis=1)
        year_data_df['year'] = year
        year_data_list.append(year_data_df)

# Concatenate all years' data at once to form the mega table
if year_data_list:
    mega_table = pd.concat(year_data_list, axis=0)
    mega_table.set_index(['year', mega_table.index], inplace=True)

#%%
mega_table = mega_table.copy()
#%%
mega_table.shape
#%%
mega_table.columns

#%%
# Now, missing_data_percentages contains the missing data percentages for each station's variables
def print_sorted_missing_data_percentages(df):
    """
    Print the percentage of missing (NaN) data for each column in the DataFrame,
    sorted by the percentage of missing data from highest to lowest.

    Parameters:
    - df: pandas DataFrame
    """
    # Calculate the percentage of missing data for each column and sort
    missing_percentages = df.isna().mean() * 100
    sorted_missing_percentages = missing_percentages.sort_values(ascending=False)
    
    # Print the sorted results
    print("Missing Data Percentages by Column (Sorted):")
    for column, percentage in sorted_missing_percentages.items():
        print(f"{column}: {percentage:.2f}%")

# Assuming 'mega_table' is your DataFrame prepared earlier
print_sorted_missing_data_percentages(mega_table)


# %%
def remove_columns_with_missing_data(df, threshold=10.0):
    """
    Remove columns from a DataFrame that have a higher percentage of missing (NaN) data than the specified threshold.

    Parameters:
    - df: pandas DataFrame
    - threshold: float, percentage threshold for removing columns with missing data
    """
    # Calculate the percentage of missing data for each column
    missing_percentages = df.isna().mean() * 100
    
    # Identify columns with missing data above the threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    
    print('shape before', df.shape)
    # Drop these columns from the DataFrame
    df_dropped = df.drop(columns=columns_to_drop)

    print('shape after ', df_dropped.shape)
    
    # Return the DataFrame with columns removed
    return df_dropped

# Remove columns from 'mega_table' with more than 10% missing data
mega_table_cleaned = remove_columns_with_missing_data(mega_table, threshold=10.0)
mega_table_cleaned.head()
#%%
mega_table_cleaned_interpolated = mega_table_cleaned.interpolate(method='linear', limit_direction='both')
mega_table_cleaned_interpolated.index = mega_table_cleaned_interpolated.index.get_level_values(1)
# %%

num_cols = 20

# Setting the figure size and layout
fig, axs = plt.subplots(num_cols, 1, figsize=(10, 20), sharex=True)

# Make subplots for each variable
idx = 0
for i, header in enumerate(mega_table_cleaned_interpolated.columns):
    axs[i].plot(mega_table_cleaned_interpolated.index, mega_table_cleaned_interpolated[header], label=header)
    axs[i].set_ylabel(header)
    axs[i].legend(loc="upper right")
    idx += 1
    if idx >= num_cols:
        break

# Set common labels
plt.xlabel('Index')
plt.tight_layout()  # Adjust layout to not overlap
plt.show()
# %%
mega_table_cleaned_interpolated.to_parquet(f'{START_YEAR}_to_2024.parquet')

# %%
# plot correlation matrix of all 1k variables for last year of data
import pandas as pd
data = pd.read_parquet(f'2020_to_2024.parquet')
data.shape
#mega_table_cleaned_interpolated = pd.read_parquet(f'{START_YEAR}_to_2024.parquet')
#%%
import seaborn as sns
last_year = mega_table_cleaned_interpolated[mega_table_cleaned_interpolated.index.year == 2024]
corr = last_year.corr()
sns.heatmap(corr)
# %%
