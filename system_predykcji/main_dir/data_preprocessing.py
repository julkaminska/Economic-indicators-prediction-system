import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

infl_file = 'data/raw_data/inflacja.csv'
unempl_file1 = 'data/raw_data/stopa_bezrobocia_msc.csv'
unempl_file2 = 'data/raw_data/stopa_bezrobocia_kw.csv'
gdp_file = 'data/raw_data/pkb.csv'
final_file = 'processed_data/indicators.csv'
months = ['styczen', 'luty', 'marzec', 'kwiecien', 'maj', 'czerwiec',
              'lipiec', 'sierpien', 'wrzesien', 'pazdziernik', 'listopad', 'grudzien']

months_mapping = {
        1: 'styczen',
        2: 'luty',
        3: 'marzec',
        4: 'kwiecien',
        5: 'maj',
        6: 'czerwiec',
        7: 'lipiec',
        8: 'sierpien',
        9: 'wrzesien',
        10: 'pazdziernik',
        11: 'listopad',
        12: 'grudzien'
    }

months_eng_mapping = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }

quarter_to_month_mapping = {
        '1 kwartal': 'kwiecień',
        '2 kwartal': 'lipiec',
        '3 kwartal': 'październik',
        '4 kwartal': 'styczen'
    }
def map_number_to_month_eng(month_num):
    return months_eng_mapping[month_num]
def map_month_to_number(month):
    month_num = -1
    for num, name in months_mapping.items():
        if name == month:
            month_num = num
            break
    month_num = int(float(month_num)) if isinstance(month_num, float) else int(month_num)
    return month_num

def map_number_to_month(month_num):
    return months_mapping[month_num]

def refactor_monthly_unemployed(file):
    df = pd.read_csv(file, nrows=1, sep=';', encoding='ascii', decimal = ',')
    new_columns = []
    for col in df.columns:
        parts = col.split(';')
        if len(parts) == 4:
            month, indicator, year, unit = parts
            new_columns.append((year, month, indicator))
        else:
            new_columns.append((None, None, col))
    df.columns = pd.MultiIndex.from_tuples(new_columns, names=['year', 'month', 'indicator'])
    df = df.melt(ignore_index=False, value_name='unemployment').reset_index()
    df = df[['year', 'month', 'unemployment']]
    df['month'] = df['month'].apply(map_month_to_number)
    df['unemployment'] = pd.to_numeric(df['unemployment'], errors='coerce')
    df = df.drop([0,1])
    df['datetime'] = pd.to_datetime(df['year'].astype(str) + df['month'].apply(map_number_to_month_eng), format='%Y%B')
    df = df.set_index('datetime')
    df.dropna()
    return df


def refactor_quarterly_unemployed(file):
    df = pd.read_csv(file, nrows=1, sep=';', encoding='utf-8', decimal = ',')

    new_columns = []
    for col in df.columns:
        parts = col.split(';')
        if len(parts) == 3:
            quarter, year, unit = parts
            month = map_month_to_number(quarter_to_month_mapping[quarter])
            new_columns.append((year, month))
        else:
            new_columns.append((None, col))

    df.columns = pd.MultiIndex.from_tuples(new_columns, names=['year', 'month'])
    df = df.melt(ignore_index=False, value_name='unemployment').reset_index(drop=True)
    df = df.drop([0, 1])
    new_rows = []

    for year in df['year'].unique():
        for month in (map_month_to_number(month1) for month1 in months):
            if not ((df['year'] == year) & (df['month'] == month)).any():
                new_rows.append({'year': year, 'month': month, 'unemployment': None})

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df.sort_values(by=['year', 'month'], inplace=True)
    df['unemployment'] = pd.to_numeric(df['unemployment'], errors='coerce')
    df = df.reset_index(drop=True)
    df['unemployment'] = df['unemployment'].interpolate(limit_area = 'outside', limit_direction='backward').round(1)
    df['unemployment'] = df['unemployment'].interpolate(method = 'cubic').round(1)
    df['datetime'] = pd.to_datetime(df['year'].astype(str) + df['month'].apply(map_number_to_month_eng), format='%Y%B')
    df = df.set_index('datetime')
    return df

def refactor_inflation(file):
    df = pd.read_csv(file, sep=';', encoding='Windows-1252', decimal = ',')
    df = df[df['Sposób prezentacji'] == "Analogiczny miesi¹c poprzedniego roku = 100"]
    df = df.drop(columns=['Nazwa zmiennej', 'Jednostka terytorialna', 'Flaga', 'Sposób prezentacji'])
    df.rename(columns={'Rok' : 'year', 'Miesi¹c': 'month', 'Wartoœæ': 'inflation'}, inplace=True)
    df['inflation'] = df['inflation'] - 100
    df['year'] = df['year'].astype(str)
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df['year'].astype(str) + df['month'].apply(map_number_to_month_eng), format='%Y%B')
    df = df.set_index('datetime')
    # print(df.head())
    return df

def refactor_gdp(file):
    df = pd.read_csv(file, sep = ';', encoding = 'utf-8', decimal = ',')
    df = df.drop(df.columns[list(range(42)) + list(range(44, 51))], axis = 1) #axis = 1 oznacza kolumny
    df['datetime'] = df['opis_okres'].str.replace(' ', '', regex=False)
    df['datetime'] = pd.PeriodIndex(df['datetime'], freq='Q').to_timestamp()
    df[['year', 'quarter']] = df['opis_okres'].str.split(' ', n=1, expand=True)
    df = df.drop(columns = ['opis_okres'])
    df.rename(columns={'wartosc': 'gdp'}, inplace=True)
    df = df[['datetime', 'year', 'quarter', 'gdp']]
    df = df.set_index('datetime')
    return df

def merge_dfs(*dfs):
    final_df = dfs[0]

    for df in dfs[1:]:
        cols_to_use = df.columns.difference(final_df.columns)
        final_df = pd.merge(final_df, df[cols_to_use], left_index=True, right_index=True, how='outer')
        final_df.update(df)

    final_df['year'] = final_df.index.year
    final_df['quarter'] = final_df.index.quarter
    final_df['month'] = final_df.index.month
    final_df.loc[final_df['gdp'].isnull(), 'quarter'] = np.nan
    final_df.loc[final_df['gdp'].isnull() == False, 'month'] = np.nan

    final_df = final_df[['year', 'quarter', 'month', 'gdp', 'inflation', 'unemployment']]
    final_df.sort_index(inplace=True)
    # print(final_df)
    return final_df

def save_csv(df, filename):
    df.to_csv(filename, index=True)

unempl1 = refactor_monthly_unemployed(unempl_file1)
unempl2 = refactor_quarterly_unemployed(unempl_file2)
infl = refactor_inflation(infl_file)
gdp = refactor_gdp(gdp_file)
save_csv(merge_dfs(gdp, unempl1, unempl2, infl), final_file)
