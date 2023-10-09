from torchvision.datasets.utils import download_url
from datetime import datetime, timedelta,date
import xarray as xr
import pandas as pd
import os
import gzip
import requests
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
import numpy as np

def import_train(scrap_date):
    os.makedirs('data/compressed', exist_ok=True)
    os.makedirs('data/uncompressed', exist_ok=True)
    with open('data/URLs.csv', 'r') as file:
        lines = file.readlines()
    url_list = []
    for url in lines:
        for date in scrap_date:
            if date+'000000' in url:
                url_list.append(url)
    for url in url_list:
        root = 'data/compressed'
        filename = url.split('_')[1] + '_'+url.split('_')[3][1:-6]+'.nc.gz'
        if filename in os.listdir(root):
            continue
        else:
            download_url(url, root, filename)

#format: YYYYMMDD000000
def automated_preprocessing(scrap_date: list, sep = False):
    os.makedirs('data/compressed', exist_ok=True)
    os.makedirs('data/uncompressed', exist_ok=True)
    os.makedirs('data/DSCOVR_L2/faraday', exist_ok=True)
    os.makedirs('data/DSCOVR_L1/faraday', exist_ok=True)
    os.makedirs('data/DSCOVR_L2/magnetometer', exist_ok=True)
    os.makedirs('data/DSCOVR_L1/magnetometer', exist_ok=True)
    with open('data/URLs.csv', 'r') as file:
        lines = file.readlines()
    url_list = []
    for url in lines:
        for date in scrap_date:
            if date+'000000' in url:
                url_list.append(url)
    for url in url_list:
        root = 'data/compressed'
        filename = url.split('_')[1] + '_'+url.split('_')[3][1:-6]+'.nc.gz'
        if filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L2/faraday') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L1/faraday') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L1/magnetometer') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L2/magnetometer'):
            continue
        elif filename[:-3] in os.listdir('data/uncompressed'):
            output_file = os.path.join('data/uncompressed',filename)[:-3]
            pass
        else:
            download_url(url, root, filename)
            file = os.path.join(root,filename)
            output_file = os.path.join('data/uncompressed',filename)[:-3]
            with gzip.open(file, 'rb') as compressed_file:
                    with open(output_file, 'wb') as decompressed_file:
                        decompressed_file.write(compressed_file.read())
            os.remove(file)
        if 'fc1' in filename:
            dataset = xr.open_dataset(output_file)

            df = dataset.to_dataframe()

            dataset.close()

            important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']

            faraday_cup = df[important_variables]

            faraday_cup = faraday_cup.resample('1min').mean()

            faraday_cup.to_csv(f'data/DSCOVR_L1/faraday/{filename[:-6]}.csv')

            os.remove(output_file)

        elif 'f1m' in filename:
            dataset = xr.open_dataset(output_file)

            df = dataset.to_dataframe()

            dataset.close()

            important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']

            faraday_cup = df[important_variables]

            faraday_cup.to_csv(f'data/DSCOVR_L2/faraday/{filename[:-6]}.csv')
            
            os.remove(output_file)
        elif 'm1m' in filename:
            dataset = xr.open_dataset(output_file)

            df = dataset.to_dataframe()

            dataset.close()

            important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

            magnetometer = df[important_variables]

            magnetometer.to_csv(f'data/DSCOVR_L2/magnetometer/{filename[:-6]}.csv')
            
            os.remove(output_file)
        else:
            dataset = xr.open_dataset(output_file)

            df = dataset.to_dataframe()

            dataset.close()

            important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

            magnetometer = df[important_variables]

            magnetometer = magnetometer.resample('1min').mean()

            magnetometer.to_csv(f'data/DSCOVR_L1/magnetometer/{filename[:-6]}.csv')
            os.remove(output_file)

    start_time =scrap_date[0]
    end_time = scrap_date[-1]

    level_1, level_2 = from_csv(start_time, end_time, sep)

    dst, kp = import_targets(scrap_date)

    return level_1, level_2,dst, kp

def from_csv(start_time, end_time, sep = False):
    fc1_list = []
    for file in os.listdir('data/DSCOVR_L1/faraday'):
        file = os.path.join('data/DSCOVR_L1/faraday', file)
        data = pd.read_csv(file, index_col=0)
        fc1_list.append(data)
    
    mg1_list = []
    for file in os.listdir('data/DSCOVR_L1/magnetometer'):
        file = os.path.join('data/DSCOVR_L1/magnetometer', file)
        data = pd.read_csv(file, index_col=0)
        mg1_list.append(data)
    
    f1m_list = []
    for file in os.listdir('data/DSCOVR_L2/faraday'):
        file = os.path.join('data/DSCOVR_L2/faraday', file)
        data = pd.read_csv(file, index_col=0)
        f1m_list.append(data)
    
    m1m_list = []
    for file in os.listdir('data/DSCOVR_L2/magnetometer'):
        file = os.path.join('data/DSCOVR_L2/magnetometer', file)
        data = pd.read_csv(file, index_col=0)
        m1m_list.append(data)

    fc1 = pd.concat(fc1_list)
    mg1 = pd.concat(mg1_list)
    f1m = pd.concat(f1m_list)
    m1m = pd.concat(m1m_list)
    fc1 = fc1[~fc1.index.duplicated(keep='first')]
    mg1 = mg1[~mg1.index.duplicated(keep='first')]
    f1m = f1m[~f1m.index.duplicated(keep='first')]
    m1m = m1m[~m1m.index.duplicated(keep='first')]
    fc1.index = pd.to_datetime(fc1.index)
    mg1.index = pd.to_datetime(mg1.index)
    f1m.index = pd.to_datetime(f1m.index)
    m1m.index = pd.to_datetime(m1m.index)
    start_time_ = f'{start_time[:4]}-{start_time[4:6]}-{start_time[-2:]} 00:00:00'
    end_time_ = f'{end_time[:4]}-{end_time[4:6]}-{end_time[-2:]} 23:59:00' 
    freq = '1T'
    full_time_index = pd.date_range(start=start_time_, end=end_time_, freq=freq)
    fc1 = fc1.reindex(full_time_index).interpolate(method = 'linear')
    mg1 = mg1.reindex(full_time_index).interpolate(method = 'linear')
    f1m = f1m.reindex(full_time_index).interpolate(method = 'linear')
    m1m = m1m.reindex(full_time_index).interpolate(method = 'linear')
    if sep:
        level_1 = (fc1, mg1)
        level_2 = (f1m, m1m)
    else:
        level_1 = pd.concat([fc1, mg1], axis =1)
        level_2 = pd.concat([f1m, m1m], axis =1)
        
        level_1.index = pd.to_datetime(level_1.index)
        level_2.index = pd.to_datetime(level_2.index)
    
    return level_1, level_2

def gzip_to_nc():
    #defining raw data dataframes
    fc1 = []
    mg1 = []
    f1m = []
    m1m = []

    ##uncompress and save
    for file in os.listdir('data/compressed'):
        output_file = os.path.join('data/uncompressed/',file[:-3]) 
        file = os.path.join('data/compressed/',file)
        if output_file in os.listdir('data/uncompressed'):
            continue
        if 'fc1' in file:
            fc1.append(output_file)
        elif 'f1m' in file:
            f1m.append(output_file)
        elif 'm1m' in file:
            m1m.append(output_file)
        else:
            mg1.append(output_file)
        with gzip.open(file, 'rb') as compressed_file:
            with open(output_file, 'wb') as decompressed_file:
                decompressed_file.write(compressed_file.read())
    return fc1, mg1, f1m, m1m

def l1_faraday_preprocess(dataframes):
    
    data_list = []
    
    #cleaning data

    for nc_file in dataframes: 

        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']

        faraday_cup = df[important_variables]

        faraday_cup = faraday_cup.resample('1min').mean()

        faraday_cup.to_csv(f'data/DSCOVR_L1/faraday/{nc_file[-3]}.csv')

        data_list.append(faraday_cup)
    return pd.concat(data_list)

def l1_magnet_preprocess(dataframes):
    
    data_list = []

    for nc_file in dataframes:
        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

        magnetometer = df[important_variables]

        magnetometer = magnetometer.resample('1min').mean()

        magnetometer.to_csv(f'data/DSCOVR_L1/magnetometer/{nc_file[-3]}.csv')
        data_list.append(magnetometer)
    return pd.concat(data_list)

def l2_faraday_preprocess(dataframes):
    
    data_list = []
    
    #cleaning data

    for nc_file in dataframes: 

        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']

        faraday_cup = df[important_variables]
        faraday_cup.to_csv(f'data/DSCOVR_L2/faraday/{nc_file[-3]}.csv')
        data_list.append(faraday_cup)
    return pd.concat(data_list)

def l2_magnet_preprocess(dataframes):
    
    data_list = []

    for nc_file in dataframes:
        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

        magnetometer = df[important_variables]

        magnetometer.to_csv(f'data/DSCOVR_L2/magnetometer/{nc_file[-3]}.csv')
        
        data_list.append(magnetometer)
    return pd.concat(data_list)

def preprocessing():
    fc1, mg1, f1m, m1m = gzip_to_nc()
    l1_faraday = l1_faraday_preprocess(fc1)
    l1_magnetometer = l1_magnet_preprocess(mg1)
    l2_faraday = l2_faraday_preprocess(f1m)
    l2_magnetometer = l2_magnet_preprocess(m1m)
    return pd.concat([l1_faraday, l1_magnetometer], axis =1), pd.concat([l2_faraday, l2_magnetometer], axis =1)

def import_Dst(months = [str(date.today()).replace('-', '')[:6]]):
    os.makedirs('data/Dst_index', exist_ok = True)
    for month in months:
        if month+'.csv' in os.listdir('data/Dst_index'):
            continue
        # Define the URL from the kyoto Dst dataset
        if int(str(month)[:4])==int(date.today().year):
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{month}/index.html'
        elif 2017<=int(str(month)[:4])<=int(date.today().year)-1:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{month}/index.html'
        else:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_final/{month}/index.html'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.text

            soup = BeautifulSoup(data, 'html.parser')
            data = soup.find('pre', class_='data')
            with open('data/Dst_index/'+ url.split('/')[-2]+'.csv', 'w') as file:
                file.write('\n'.join(data.text.replace('\n\n', '\n').replace('\n ','\n').split('\n')[7:39]).replace('-', ' -').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        else:
            print('Unable to access the site')


def interval_time(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return date_list
#format: month: YYYYMM day: D for < 10 and DD for > 10.

def day_Dst(interval_time):
    data_list = []
    for day in interval_time:
        today_dst = pd.read_csv(f'data/Dst_index/{day[:6]}.csv',index_col = 0, header = None).T[int(day[6:])]
        for i,k in enumerate(today_dst):
            if isinstance(k, str): 
                today_dst[i+1] = float(today_dst[i+1])
            if np.abs(today_dst[i+1])>500:
                today_dst[i+1] = np.nan
        
        data_list.append(today_dst)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'Dst'
    return series
def day_Kp(interval_time):
    data_list = []
    kp = pd.read_csv(f'data/Kp_index/data.csv',index_col = 0, header = None).T
    for day in interval_time:
            try:
                today_kp = kp[day][0:8]
            except IndexError:
                continue
            for i,k in enumerate(today_kp):
                if isinstance(k, str): 
                    if np.abs(float(today_kp[i+1][0]))>9:
                        today_kp[i+1] = np.nan
                if isinstance(k, (int, float)):
                    if np.abs(today_kp[i+1])>9:
                        today_kp[i+1] = np.nan
            
            data_list.append(today_kp)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'Kp'
    return series

def import_targets(interval_time):
    kp = day_Kp(interval_time)
    dst = day_Dst(interval_time)
    return dst, kp