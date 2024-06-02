# Module to read data from files and directories
import numpy as np
import pandas as pd

def read_billionaires_data(
    folder_directory: str='../../Data/billionaire_data/forbes/',
    only_years: list[str]=None,
    only_regions: list[str]=None,
    self_made: bool=None,
    raw: bool=False,
    year_and_month_int: bool=False
) -> pd.DataFrame:
    if only_years is None:
        file_path = folder_directory + 'all_billionaires_1997_2023.csv'
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()
        for year in only_years:
            file_path = folder_directory + f'billionaires_{year}.csv'
            df = pd.concat([df, pd.read_csv(file_path)])

    df['year'] = pd.to_datetime(df['year'], format='%Y')
    if raw:
        # Warn that the filters do not work with raw data
        print('Warning: Filters do not work with raw data, returning raw data.')
        return df
    
    # Preprocessing
    # If the citizenship is not available, find for the same full_name a row where the citizenship is available
    df['country_of_citizenship'] = df['country_of_citizenship'].fillna(df.groupby('full_name')['country_of_citizenship'].transform('first'))
    
    # Net worth is denoted with a number and then the letter B, convert it to a number
    df['net_worth'] = df['net_worth'].str.replace('B', '').astype(float)

    countries_by_region = {
        'North America': 
            ['United States', 'Canada'],
        'Europe': 
            ['Germany', 'United Kingdom', 'Ireland', 'Cyprus', 'Czech Republic', 'Czechia', 'Denmark', 'Austria',
            'Belgium', 'Spain', 'France', 'Greece', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal', 
            'Sweden', 'Switzerland', 'Liechtenstein', 'Lithuania', 'Monaco', 'Estonia', 'Finland', 'Slovakia', 
            'Romania', 'Hungary', 'Bulgaria', 'Guernsey', 'Iceland'],
        'China': 
            ['China', 'Hong Kong', 'Macau', 'Macao'],
        'East Asia': 
            ['Thailand', 'Malaysia', 'Singapore', 'Taiwan', 'Philippines', 'Indonesia', 'South Korea', 'Japan',
            'Australia', 'Vietnam', 'New Zealand'],
        'India': 
            ['India'],
        'Central Eurasia': 
            ['Russia', 'Kazakhstan', 'Ukraine', 'Armenia', 'Georgia'],
        'South America': 
            ['Brazil', 'Chile', 'Argentina', 'Peru', 'Venezuela', 'Colombia', 'Uruguay', 'Guatemala',
            'Panama', 'Barbados', 'Belize', 'Mexico'],
        'Middle East': 
            ['Turkey', 'Egypt', 'Israel', 'Saudi Arabia', 'United Arab Emirates', 'Kuwait', 'Qatar', 'Oman',
            'Lebanon'],
    }

    countries_by_sub_region = {
        'U.S.':
            ['United States'],
        'Canada':
            ['Canada'],
        'Germany':
            ['Germany'],
        'British Islands':
            ['United Kingdom', 'Ireland'],
        'Scandinavia':
            ['Denmark', 'Norway', 'Sweden', 'Finland'],
        'France':
            ['France', 'Monaco'],
        'Alps':
            ['Switzerland', 'Liechtenstein', 'Austria'],
        'Italy':
            ['Italy'],
        'China': 
            ['China', 'Hong Kong'],
        'Southeast Asia':
            ['Thailand', 'Malaysia', 'Singapore'],
        'Asian Islands':
            ['Taiwan', 'Philippines', 'Indonesia'],
        'South Korea':
            ['South Korea'],
        'Japan':
            ['Japan'],
        'Australia':
            ['Australia'],
        'India': 
            ['India'],
        'Russia':
            ['Russia'],
        'Brazil':
            ['Brazil'],
        'Israel + Turkey':
            ['Israel', 'Turkey']
    }
    # Function to find the region for a given country
    def find_region(country):
        for region, countries in countries_by_region.items():
            if country in countries:
                return region
        return "Rest of World"
    
    # Function to find the sub-region for a given country
    def find_sub_region(country):
        for region, countries in countries_by_sub_region.items():
            if country in countries:
                return region
        return "Not a sub-region"
    
    df['region'] = df['country_of_citizenship'].apply(find_region)
    df['sub_region'] = df['country_of_citizenship'].apply(find_sub_region)

    # Only retain the columns we need
    if year_and_month_int:
        df['year_int'] = df['year'].dt.year
        df['month_int'] = df['year'].dt.month
        df = df[['year_int', 'month_int', 'rank', 'net_worth', 'full_name', 'self_made', 'country_of_citizenship', 'region', 'sub_region']]
    else:
        df = df[['year', 'rank', 'net_worth', 'full_name', 'self_made', 'country_of_citizenship', 'region', 'sub_region']]

    df['log_net_worth'] = np.log(df['net_worth'])

    if only_regions is not None:
        df = df[df['region'].isin(only_regions)]
    
    if self_made is not None:
        df = df[df['self_made'] == self_made]

    return df

def read_bloomberg_data(
    folder_directory: str='../../Data/billionaire_data/bloomberg/500_richest_people_2021.csv',
    only_regions: list[str]=None,
    raw: bool=False
) -> pd.DataFrame:
    df = pd.read_csv(folder_directory, sep=';')

    if raw:
        # Warn that the filters do not work with raw data
        print('Warning: Filters do not work with raw data, returning raw data.')
        return df

    # Preprocessing
    # net worth is written as $X.XXB, convert it to a number
    # Drop rows where Total Net Worth is nan
    # Drop rows where Total Net Worth is not of this format
    df = df.dropna(subset=['Total Net Worth'])

    df['net_worth'] = df['Total Net Worth'].str.replace('$', '').str.replace('B', '').astype(float)
    df['rank'] = df['Rank']
    df['full_name'] = df['Name']

    countries_by_region = {
        'North America': 
            ['United States', 'Canada'],
        'Europe': 
            ['Germany', 'United Kingdom', 'Ireland', 'Cyprus', 'Czech Republic', 'Czechia', 'Denmark', 'Austria',
            'Belgium', 'Spain', 'France', 'Greece', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal', 
            'Sweden', 'Switzerland', 'Liechtenstein', 'Lithuania', 'Monaco', 'Estonia', 'Finland', 'Slovakia', 
            'Romania', 'Hungary', 'Bulgaria', 'Guernsey', 'Iceland'],
        'China': 
            ['China', 'Hong Kong', 'Macau', 'Macao'],
        'East Asia': 
            ['Thailand', 'Malaysia', 'Singapore', 'Taiwan', 'Philippines', 'Indonesia', 'South Korea', 'Japan',
            'Australia', 'Vietnam', 'New Zealand'],
        'India': 
            ['India'],
        'Central Eurasia': 
            ['Russia', 'Kazakhstan', 'Ukraine', 'Armenia', 'Georgia'],
        'South America': 
            ['Brazil', 'Chile', 'Argentina', 'Peru', 'Venezuela', 'Colombia', 'Uruguay', 'Guatemala',
            'Panama', 'Barbados', 'Belize', 'Mexico'],
        'Middle East': 
            ['Turkey', 'Egypt', 'Israel', 'Saudi Arabia', 'United Arab Emirates', 'Kuwait', 'Qatar', 'Oman',
            'Lebanon'],
    }

    # Function to find the region for a given country
    def find_region(country):
        for region, countries in countries_by_region.items():
            if country in countries:
                return region
        return "Rest of World"
    
    df['region'] = df['Country'].apply(find_region)
    df['country'] = df['Country']

    # Only retain the columns we need
    df = df[['rank', 'net_worth', 'full_name', 'country', 'region']]

    if only_regions is not None:
        df = df[df['region'].isin(only_regions)]

    df['log_net_worth'] = np.log(df['net_worth'])

    return df

def read_population_data(
    folder_directory: str='../../Data/population/World_Population_Live_Dataset.csv',
    raw: bool=False
) -> pd.DataFrame:
    df = pd.read_csv(folder_directory)
    
    if raw:
        return df

    # Set the 'CCA3' column as the index for easier manipulation
    df.set_index('Name', inplace=True)

    # Select the population columns
    population_columns = ['2022', '2020', '2015', '2010', '2000', '1990', '1980', '1970']

    # Ensure the columns are treated as strings before interpolation
    df[population_columns] = df[population_columns].astype(float)

    # Reindex the DataFrame to include all years from 1970 to 2022
    all_years = list(map(str, range(1970, 2024)))
    df_population = df[population_columns].T  # Transpose for easier interpolation
    df_population = df_population.reindex(all_years)  # Add all years
    df_population = df_population.T  # Transpose back

    # Interpolate the missing values
    df_population = df_population.interpolate(method='linear', axis=1)

    df_population = 1000 * df_population  # Convert to millions

    # Add a World row
    df_population.loc['World'] = df_population.sum()

    return df_population

def read_gdp_data(
    folder_directory: str='../../Data/gdp_data/GDP_per_capita_data.csv',
    raw: bool=False
) -> pd.DataFrame:
    df = pd.read_csv(folder_directory)

    if raw:
        return df

    df.set_index('Country Code', inplace=True)
    # drop columns that are not needed
    df.drop(['Country Name', 'Indicator Name', 'Indicator Code', 'Unnamed: 68'], axis=1, inplace=True)
    df = df.T

    # linearly interpolate missing values
    df = df.interpolate(method='linear', axis=0)

    return df

def read_stock_market_data(
    folder_directory_1: str='../../Data/stock_market/MSCI_World_historical.csv',
    folder_directory_2: str='../../Data/stock_market/SPX_historical.csv',
    raw: bool=False
) -> pd.DataFrame:
    df_1 = pd.read_csv('../../Data/stock_market/MSCI_World_historical.csv')
    df_2 = pd.read_csv('../../Data/stock_market/SPX_historical.csv')

    if raw:
        return df_1, df_2

    df_1['Date'] = pd.to_datetime(df_1['Date'])
    df_2['Date'] = pd.to_datetime(df_2['Date'])
    df_1.set_index('Date', inplace=True)
    df_2.set_index('Date', inplace=True)
    df_1 = df_1.resample('ME').last()
    df_2 = df_2.resample('ME').last()

    df = pd.merge(df_1, df_2, left_index=True, right_index=True, how='inner', suffixes=('_MSCI', '_SPX'))

    df = df[['Adj Close_MSCI', 'Adj Close_SPX']]
    df.columns = ['Adj_Close_MSCI', 'Adj_Close_SPX']

    return df

def read_iso_codes(
    folder_directory: str='../../Data/regions/iso_codes_mapping.csv',
    raw: bool=False
) -> pd.DataFrame:
    df = pd.read_csv(folder_directory)

    if raw:
        return df

    df = df.set_index('name')

    df = df[['alpha-2', 'alpha-3']]
    df.columns = ['ISO2', 'ISO3']

    return df

def read_panel_data(
    folder_directory: str='../../Data/panel_data/panel_data.csv',
    raw: bool=False
) -> pd.DataFrame:
    df = pd.read_csv(folder_directory)

    if raw:
        return df

    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df.set_index(['year', 'country'], inplace=True)

    return df