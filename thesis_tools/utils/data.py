# Module to read data from files and directories
import numpy as np
import pandas as pd
import copy

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
    # Drop rows where net worth is less than 1
    df = df[df['net_worth'] >= 1]

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

def read_stock_market_data() -> pd.DataFrame:
    # CAC40 data
    df_CAC40 = pd.read_csv('../../Data/stock_market/CAC40_historical.csv')
    df_CAC40['Date'] = pd.to_datetime(df_CAC40['Date'])
    df_CAC40.set_index('Date', inplace=True)
    df_CAC40 = df_CAC40.resample('ME').last()
    df_CAC40['CAC40'] = df_CAC40['Adj Close']
    df_CAC40 = df_CAC40[['CAC40']]

    # DAX data
    df_DAX = pd.read_csv('../../Data/stock_market/DAX_historical.csv')
    df_DAX['Date'] = pd.to_datetime(df_DAX['Date'])
    df_DAX.set_index('Date', inplace=True)
    df_DAX = df_DAX.resample('ME').last()
    df_DAX['DAX'] = df_DAX['Adj Close']
    df_DAX = df_DAX[['DAX']]

    # FTSE100 data
    df_FTSE100 = pd.read_csv('../../Data/stock_market/FTSE100_historical.csv')
    df_FTSE100['Date'] = pd.to_datetime(df_FTSE100['Date'], format='%m/%d/%y')
    df_FTSE100.set_index('Date', inplace=True)
    df_FTSE100 = df_FTSE100.resample('ME').last()
    df_FTSE100['FTSE100'] = df_FTSE100[' Close']
    df_FTSE100 = df_FTSE100[['FTSE100']]

    # MOEX data
    df_MOEX = pd.read_csv('../../Data/stock_market/MOEX_historical.csv')
    df_MOEX['Date'] = pd.to_datetime(df_MOEX['Date'])
    df_MOEX.set_index('Date', inplace=True)
    df_MOEX = df_MOEX.resample('ME').last()
    df_MOEX['MOEX'] = df_MOEX['Adj Close']
    df_MOEX = df_MOEX[['MOEX']]

    # MSCI data
    df_MSCI = pd.read_csv('../../Data/stock_market/MSCI_World_historical.csv')
    df_MSCI['Date'] = pd.to_datetime(df_MSCI['Date'])
    df_MSCI.set_index('Date', inplace=True)
    df_MSCI = df_MSCI.resample('ME').last()
    df_MSCI['MSCI'] = df_MSCI['Adj Close']
    df_MSCI = df_MSCI[['MSCI']]

    # NIFTY data
    df_NIFTY = pd.read_csv('../../Data/stock_market/NIFTY_historical.csv')
    df_NIFTY['Date'] = pd.to_datetime(df_NIFTY['Date'])
    df_NIFTY.set_index('Date', inplace=True)
    df_NIFTY = df_NIFTY.resample('ME').last()
    df_NIFTY['NIFTY'] = df_NIFTY['Adj Close']
    df_NIFTY = df_NIFTY[['NIFTY']]

    # OMX40 data
    df_OMX40 = pd.read_csv('../../Data/stock_market/OMX40_historical.csv')
    df_OMX40['Date'] = pd.to_datetime(df_OMX40['Date'])
    df_OMX40.set_index('Date', inplace=True)
    df_OMX40 = df_OMX40.resample('ME').last()
    df_OMX40['OMX40'] = df_OMX40['Close/Last']
    df_OMX40 = df_OMX40[['OMX40']]

    # SPX data
    df_SPX = pd.read_csv('../../Data/stock_market/SPX_historical.csv')
    df_SPX['Date'] = pd.to_datetime(df_SPX['Date'])
    df_SPX.set_index('Date', inplace=True)
    df_SPX = df_SPX.resample('ME').last()
    df_SPX['SPX'] = df_SPX['Adj Close']
    df_SPX = df_SPX[['SPX']]

    # SSE data
    df_SSE = pd.read_csv('../../Data/stock_market/SSE_historical.csv')
    df_SSE['Date'] = pd.to_datetime(df_SSE['Date'])
    df_SSE.set_index('Date', inplace=True)
    df_SSE = df_SSE.resample('ME').last()
    df_SSE['SSE'] = df_SSE['Adj Close']
    df_SSE = df_SSE[['SSE']]

    # Merge all the dataframes
    df_list = [df_CAC40, df_DAX, df_FTSE100, df_MOEX, df_MSCI, df_NIFTY, df_OMX40, df_SPX, df_SSE]
    df_merged = pd.concat(df_list, axis=1)

    return df_merged

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
    AGGREGATE_TYPE: str='sub_region',
    observations_threshold: int=10
) -> pd.DataFrame:
    # Assert that AGGREGATE_TYPE is either 'sub_region', 'region' or 'country_of_citizenship'
    assert AGGREGATE_TYPE in ['sub_region', 'region', 'country_of_citizenship']

    billionaire_df = read_billionaires_data(year_and_month_int=True)
    # drop region = 'Rest of World'
    billionaire_df = billionaire_df[billionaire_df['region'] != 'Rest of World']
    # drop sub_region = '"Not a sub-region"
    billionaire_df = billionaire_df[billionaire_df['sub_region'] != 'Not a sub-region']

    iso_codes = read_iso_codes()
    df = pd.merge(billionaire_df, iso_codes, left_on='country_of_citizenship', right_on='name', how='left')

    # use year int and month int to get the year and month and make a date column that sets as date the last day of the month
    df['date'] = pd.to_datetime(df['year_int'].astype(str) + '-' + df['month_int'].astype(str) + '-01') + pd.offsets.MonthEnd(0)

    stock_market_data = read_stock_market_data()
    stock_indices = stock_market_data.columns.tolist()

    # merge stock market data on dates
    df = pd.merge(df, stock_market_data, left_on='date', right_on='Date', how='left')

    population_data = read_population_data()
    population_data.reset_index(inplace=True)
    population_data = pd.merge(population_data, iso_codes, left_on='Name', right_on='name', how='left')
    population_data.set_index('ISO3', inplace=True)
    population_data.drop(columns=['Name', 'ISO2'], inplace=True)
    population_data.columns = population_data.columns.astype(int)

    # add population data
    # for each row, find ISO 3 and year, and find the corresponding population
    df['country_population'] = df.apply(lambda x: population_data.loc[x['ISO3'], x['year_int']], axis=1)

    gdp_data = read_gdp_data()
    gdp_data['TWN'] = 20000 # average over the interesting period
    gdp_data['GGY'] = 30000 # average over the interesting period
    gdp_data['PRK'] = gdp_data['KOR'] # North Korea is not in the data, so we use South Korea's GDP - there seems to be some confusion in the codes
    # TODO: find data on Taiwan, Guernsey

    # find gdp per capita based on year int and ISO3
    df['gdp_per_capita'] = df.apply(lambda x: gdp_data.loc[str(x['year_int']), x['ISO3']], axis=1)

    # ISO_3 codes by region
    ISO_3_by_region = {}
    for region, countries in countries_by_region.items():
        # the country name is in the iso_codes dataframe index
        ISO_3_by_region[region] = iso_codes.loc[countries, 'ISO3'].values

    # ISO_3 codes by sub region
    ISO_3_by_sub_region = {}
    for sub_region, countries in countries_by_sub_region.items():
        # the country name is in the iso_codes dataframe index
        ISO_3_by_sub_region[sub_region] = iso_codes.loc[countries, 'ISO3'].values

    # make a population by region dataframe
    population_by_region = {}
    for region, ISO_3s in ISO_3_by_region.items():
        population_by_year = {}
        for year in population_data.columns:
            population_by_year[year] = population_data.loc[ISO_3s, year].sum()
        population_by_region[region] = population_by_year
    population_by_region = pd.DataFrame(population_by_region)

    # make a population by sub region dataframe
    population_by_sub_region = {}
    for sub_region, ISO_3s in ISO_3_by_sub_region.items():
        population_by_year = {}
        for year in population_data.columns:
            population_by_year[year] = population_data.loc[ISO_3s, year].sum()
        population_by_sub_region[sub_region] = population_by_year
    population_by_sub_region = pd.DataFrame(population_by_sub_region)

    # make a weighted gdp per capita by region dataframe
    gdp_per_capita_by_region = {}
    for region, ISO_3s in ISO_3_by_region.items():
        gdp_per_capita_by_year = {}
        for year in population_by_region.index:
            year = int(year)
            divisor = population_by_region.loc[year, region]
            sum = (gdp_data.loc[str(year), ISO_3s] * population_data.loc[ISO_3s, year]).sum()
            gdp_per_capita_by_year[year] = sum / divisor 
        gdp_per_capita_by_region[region] = gdp_per_capita_by_year
    gdp_per_capita_by_region = pd.DataFrame(gdp_per_capita_by_region)

    # make a weighted gdp per capita by sub region dataframe
    gdp_per_capita_by_sub_region = {}
    for sub_region, ISO_3s in ISO_3_by_sub_region.items():
        gdp_per_capita_by_year = {}
        for year in population_by_sub_region.index:
            year = int(year)
            divisor = population_by_sub_region.loc[year, sub_region]
            sum = (gdp_data.loc[str(year), ISO_3s] * population_data.loc[ISO_3s, year]).sum()
            gdp_per_capita_by_year[year] = sum / divisor 
        gdp_per_capita_by_sub_region[sub_region] = gdp_per_capita_by_year
    gdp_per_capita_by_sub_region = pd.DataFrame(gdp_per_capita_by_sub_region)

    df['region_gdp_per_capita'] = df.apply(lambda x: gdp_per_capita_by_region.loc[x['year_int'], x['region']], axis=1)
    df['sub_region_gdp_per_capita'] = df.apply(lambda x: gdp_per_capita_by_sub_region.loc[x['year_int'], x['sub_region']], axis=1)
    panel_df = copy.deepcopy(df)

    panel_df = panel_df[['year_int', 'country_of_citizenship', 'region', 'sub_region', 'gdp_per_capita', 'region_gdp_per_capita', 'sub_region_gdp_per_capita'] + stock_indices + ['net_worth']]

    data = {}
    if AGGREGATE_TYPE == 'sub_region':
        for sub_region in panel_df['sub_region'].unique():
            for year in panel_df[panel_df['sub_region']==sub_region]['year_int'].unique():
                data[(sub_region, year)] = panel_df[(panel_df['sub_region'] == sub_region) & (panel_df['year_int'] == year)]['net_worth']
        panel_df = panel_df[['sub_region', 'year_int', 'sub_region_gdp_per_capita'] + stock_indices]
    elif AGGREGATE_TYPE == 'region':
        for region in panel_df['region'].unique():
            for year in panel_df[panel_df['region']==region]['year_int'].unique():
                data[(region, year)] = panel_df[(panel_df['region'] == region) & (panel_df['year_int'] == year)]['net_worth']
        panel_df = panel_df[['region', 'year_int', 'region_gdp_per_capita'] + stock_indices]
    elif AGGREGATE_TYPE == 'country_of_citizenship':
        for country in panel_df['country_of_citizenship'].unique():
            for year in panel_df[panel_df['country_of_citizenship']==country]['year_int'].unique():
                data[(country, year)] = panel_df[(panel_df['country_of_citizenship'] == country) & (panel_df['year_int'] == year)]['net_worth']
        panel_df = panel_df[['country_of_citizenship', 'year_int', 'gdp_per_capita'] + stock_indices]
    else:
        raise ValueError('AGGREGATE_TYPE must be either "sub_region", "region" or "country_of_citizenship"')

    panel_df = panel_df.drop_duplicates()

    panel_df['net_worth'] = panel_df.apply(lambda x: data[(x[AGGREGATE_TYPE], x['year_int'])].tolist(), axis=1)

    panel_df['N_net_worth'] = panel_df['net_worth'].apply(len)

    # remove rows with less than 10 observations
    panel_df = panel_df[panel_df['N_net_worth'] >= observations_threshold]

    for group in panel_df[AGGREGATE_TYPE].unique():
        while True:
            min_year = panel_df[panel_df[AGGREGATE_TYPE] == group]['year_int'].min()
            max_year = panel_df[panel_df[AGGREGATE_TYPE] == group]['year_int'].max()
            years = panel_df[panel_df[AGGREGATE_TYPE] == group]['year_int'].unique()
            theo_years = range(min_year, max_year+1)
            if set(theo_years) == set(years):
                break
            # remove the row with year_int == min_year and panel_df[AGGREGATE_TYPE] == group
            panel_df = panel_df[~((panel_df['year_int'] == min_year) & (panel_df[AGGREGATE_TYPE] == group))]

    panel_df.columns = ['group', 'year', 'gdp_pc'] + stock_indices + ['net_worth', 'N_net_worth']
    panel_df.sort_values(by='year', inplace=True)

    grouped_df = panel_df.groupby('group')

    # Define a function to calculate the log change
    def calculate_log_change(group):
        group['log_change_gdp_pc'] = np.log(group['gdp_pc']).diff()
        for stock_index in stock_indices:
            group[f'log_change_{stock_index}'] = np.log(group[stock_index]).diff()
        return group

    # Apply the function to each group
    result = grouped_df.apply(calculate_log_change, include_groups=False)

    df = result.reset_index()
    df.drop(columns='level_1', inplace=True)

    # Add a constant column
    df['constant'] = 1

    # Exclude the year 2023 TODO: complete the data for 2023, may not be possible
    df = df[df['year'] != 2023]

    return df
