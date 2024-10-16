# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:47:19 2023

@author: MarqueLiu
"""
import requests
import pandas as pd
from Point2Raster import point2raster

def is_coordinate_in_china(lat,lon):
    ##to judge whether a point from Edgar file is located inside China
    url = ''#set your BaiduMap service URL, you can have a survice URL at https://lbsyun.baidu.com/faq/api?title=webapi/guide/webservice-geocoding-abroad
    
    try:
        result = requests.get(url=url)
        result_json = result.json()
        code=result_json['address']['country_code'].upper()
    except:
        code=''
    if code=='CN':
        return True
    else:
        return  False


for year in ['Energy','coal','ONG']:
    EdgarPath=r''#download edgar files and save them into excel,you can download at https://edgar.jrc.ec.europa.eu/emissions_data_and_maps
    df=pd.read_excel(EdgarPath,sheet_name=str(year))
    print('finish read')
    size=len(df)
    dfRes=pd.DataFrame(columns=['lon','lat',year])
    for i in range(size):
        lat=df.loc[i,'lat'];lon=df.loc[i,'lon'];emi=df.loc[i,'emi']
        if is_coordinate_in_china(lat,lon):
            new_row={'lon':lon,'lat':lat,year:emi}
            dfRes.append(new_row, ignore_index=True)
    print(year)
    resultPath=r''#set your result path
    point2raster(dfRes,resultPath,year)