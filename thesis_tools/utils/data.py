# Module to read data from files and directories
import numpy as np
import pandas as pd

def read_billionaires_data(
    folder_directory: str='../Data/billionaire_data/forbes/',
    only_years: list[str]=None,
    only_regions: list[str]=None,
    self_made: bool=None,
    raw: bool=False
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

    # Function to find the region for a given country
    def find_region(country):
        for region, countries in countries_by_region.items():
            if country in countries:
                return region
        return "Rest of World"
    
    df['region'] = df['country_of_citizenship'].apply(find_region)

    # Only retain the columns we need
    df = df[['year', 'rank', 'net_worth', 'full_name', 'self_made', 'country_of_citizenship', 'region']]

    df['log_net_worth'] = np.log(df['net_worth'])

    if only_regions is not None:
        df = df[df['region'].isin(only_regions)]
    
    if self_made is not None:
        df = df[df['self_made'] == self_made]

    return df