# Module to read data from files and directories

import pandas as pd

def read_billionaires_data(
    folder_directory: str='../Data/billionaire_data/',
    only_years: List[str]=None,
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
        return df
    
    # Preprocessing
    
    # Net worth is denoted with a number and then the letter B, convert it to a number
    df['net_worth'] = df['net_worth'].str.replace('B', '').astype(float)

    # Cluster nationalities into regions
    'Thailand', 'Lebanon', 'India', 'Germany', 'Colombia',
    'Switzerland', 'Brazil', 'Hong Kong', 'France', 'Norway', 'Japan',
    'United States', 'Saudi Arabia', 'United Kingdom', 'Sweden',
    'Mexico', 'Italy', 'Spain', 'Kuwait', 'Venezuela', 'Greece',
    'South Africa', 'Canada', 'Israel', 'Turkey', 'Malaysia', 'Taiwan',
    'Australia', 'Ireland', 'United Arab Emirates', 'Denmark',
    'Russia', 'Argentina', 'South Korea', 'Chile', 'Philippines',
    'Portugal', 'Belgium', 'Egypt', 'Singapore', 'Netherlands',
    'China', 'Austria', nan, 'Ukraine', 'Indonesia', 'Kazakhstan',
    'Poland', 'Monaco', 'Czech Republic', 'New Zealand', 'Cyprus',
    'Iceland', 'Oman', 'Romania', 'Nigeria', 'Belize', 'Finland',
    'Pakistan', 'Georgia', 'Morocco', 'Peru', 'St. Kitts and Nevis',
    'Swaziland', 'Angola', 'Guernsey', 'Vietnam', 'Nepal', 'Algeria',
    'Macau', 'Uganda', 'Tanzania', 'Lithuania', 'Liechtenstein',
    'Guatemala', 'Qatar', 'Slovakia', 'Zimbabwe', 'Hungary', 'Czechia',
    'Eswatini (Swaziland)', 'Macao', 'Armenia', 'Bulgaria', 'Barbados',
    'Uruguay', 'Estonia', 'Bangladesh', 'Panama'

    df['region'] = np.nan

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
    
    df['Region'] = df['Country'].apply(find_region)

    return df