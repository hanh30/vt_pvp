import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates
import seaborn as sns
from datetime import datetime, date
import re
import glob
from scipy.stats import ttest_ind, wilcoxon

pd.options.mode.chained_assignment = None  # default='warn'


step_filter = [
    "['start', 'stop']",
    "['xs', 'start', 'stop']",
    "['start', 'reverse', 'stop']",
    "['xs', 'start', 'reverse', 'stop']"
]

# parts
def read_part(path):
    df_part = pd.read_excel(path)
    df_part = df_part.rename(columns={'KKS':'equipment_code'})
    df_part.columns = [c.lower() for c in df_part.columns]
    return df_part

# sootblowing
def read_sb(path):
    df_list = []
    for p in glob.glob(path):
        if 'csv' in p.lower():
            df_list.append(pd.read_csv(p, header=None))
    df = pd.concat(df_list).reset_index(drop=True)
    return df

def preprocessing(df, df_part):
    # drop unnecessary columns
    cols_drop = [1,2,4,5,7,8,10]
    df = df.drop(columns=cols_drop)

    # create date and time columns
    df = df.rename(columns={0:'datetime', 3:'id'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['month'] = pd.to_datetime(df['date'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)).dt.date
    df['hour'] = df['datetime'].dt.hour

    df['equipment_code'] = df[9].str.strip('MS$|MR$')

    df = df.merge(df_part[['equipment_code', 'abbr', 'zone']], how='left', on='equipment_code')

    # create status columns
    df['status'] = np.where(
        df[9].str.contains('MS$'),
        'start',
        np.where(
            df[9].str.contains('MR$'),
            'reverse',
            np.where(
                (df[9]==df['equipment_code']) & (df[11].str.contains('MV01')),
                'xs',
                np.where(
                    (df[9]==df['equipment_code']) & (df[11].str.contains('MV02')),
                    'stop',
                    'N/A'
                )
            )
        )
    )

    df = df.sort_values(by=['group', 'equipment_code', 'datetime'])

    df['status_next'] = df.groupby('equipment_code')['status'].shift(-1)
    df['status_previous'] = df.groupby('equipment_code')['status'].shift(1)
    # df['status_previous_2'] = df.groupby('equipment_code')['status'].shift(2)
    df['datetime_next'] = df.groupby('equipment_code')['datetime'].shift(-1)
    df['delta_datetime_next'] = (df['datetime_next'] - df['datetime']).dt.seconds/60
    df['datetime_previous'] = df.groupby('equipment_code')['datetime'].shift(1)
    df['delta_datetime_previous'] = (df['datetime'] - df['datetime_previous']).dt.seconds/60

    # equipment cycle
    '''
        - An equipment cycle begin when 'xs'
            | or 'start' after a 'stop' | or 'start' after a long time (>= 30 mins)
            | or 'stop' after a 'stop' | or 'stop' after a long time (>= 30 mins)
        - An equipment cycle end when 'stop' after whatever status other than 'stop'
    '''
    df['cycle'] = np.where(
        ((df['status'] == 'xs')) |
        ((df['status'] == 'start') & (df['status_previous'].isin(['stop', np.NaN]))) |
        ((df['status'] == 'start') & (df['delta_datetime_previous'] >= 30)) |
        ((df['status'] == 'stop') & (df['status_previous'] == 'stop')) |
        ((df['status'] == 'stop') & (df['delta_datetime_previous'] >= 30)),
        'begin',
        np.where(
            (df['status'] == 'stop') & (df['status_previous'] != 'stop'),
            'end',
            np.NaN
        )
    )

    df['cycle_begin'] = np.where(df['cycle'] == 'begin', 1, 0)
    df['cycle_gr'] = df['cycle_begin'].cumsum()

    return df

def create_df_cycle(df, full_gr=False):
    if full_gr is True:
      df_gr = df.groupby('cycle_gr').agg(
          equipment_code = ('equipment_code', 'min'),
          cycle_full_gr = ('cycle_full_gr', 'min'),
          step = ('status', list),
          xs = ('datetime', 'min'),
          stop = ('datetime', 'max')
      ).reset_index()
    elif full_gr is False:
      df_gr = df.groupby('cycle_gr').agg(
          equipment_code = ('equipment_code', 'min'),
          # cycle_full_gr = ('cycle_full_gr', 'min'),
          step = ('status', list),
          xs = ('datetime', 'min'),
          stop = ('datetime', 'max')
      ).reset_index()

    df_gr_2 = df[df['status'] != 'xs'].groupby('cycle_gr').agg(
        start = ('datetime', 'min')
    ).reset_index()

    df_gr = df_gr.merge(df_gr_2, how='left', on='cycle_gr')
    df_gr['step_str'] = df_gr['step'].astype(str)
    if full_gr is True:
      df_gr = df_gr[['equipment_code', 'cycle_gr', 'cycle_full_gr', 'step', 'step_str', 'xs', 'start', 'stop']]
    elif full_gr is False:
      df_gr = df_gr[['equipment_code', 'cycle_gr', 'step', 'step_str', 'xs', 'start', 'stop']]

    df_gr['start'] = np.where(
        df_gr['step'].str.contains('xs'),
        df_gr['start'],
        df_gr['xs']
    )
    df_gr['xs'] = np.where(
        df_gr['step_str'].str.contains('xs'),
        df_gr['xs'],
        np.NaN
    )

    df_gr['time_xs_m'] = (df_gr['start'] - df_gr['xs']).dt.seconds/60
    df_gr['time_sb_m'] = (df_gr['stop'] - df_gr['start']).dt.seconds/60

    return df_gr


def add_cycle(df):
    # full cycle
    '''
        A full cycle begin from the beginning of the equipment cycle, and if the beginning time is >= 30 mins from the previous step
    '''
    df_full = df[['group', 'datetime', 'id', 'cycle_begin']].sort_values(by=['group', 'datetime'])
    df_full['datetime_previous'] = df_full.groupby('group')['datetime'].shift(1)
    df_full['delta_datetime_previous'] = (df_full['datetime'] - df_full['datetime_previous']).dt.seconds/60

    df_full['cycle_full'] = np.where(
        (df_full['cycle_begin'] == 1) & ((df_full['delta_datetime_previous'] >= 30) | (df_full['delta_datetime_previous'].isnull())), 1, 0
    )
    df_full['cycle_full_gr'] = df_full['cycle_full'].cumsum()
    df = df.merge(df_full[['id', 'cycle_full', 'cycle_full_gr']], how='left', on='id')

    # zone cycle
    '''
        A zone cycle begin when in a full cycle, the first equipment in the zone begin
    '''
    df_zone_begin = df.loc[df[df['cycle']=='begin'].groupby(['group', 'cycle_full_gr', 'zone'])['datetime'].idxmin()][['id']]
    df_zone_begin['cycle_zone'] = 1

    df = df.merge(df_zone_begin, how='left', on='id').sort_values(by=['group', 'cycle_full_gr', 'zone', 'datetime'])
    df['cycle_zone'] = df['cycle_zone'].fillna(0)
    df['cycle_zone_gr'] = df['cycle_zone'].cumsum()

    return df



