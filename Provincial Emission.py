# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:20:44 2023

@author: MarqueLiu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:47:19 2023

@author: MarqueLiu
"""
'''aggregate gridded emissions into provincial level'''


import pandas as pd
from shapely.geometry import shape, Point
import fiona
import pyproj
from osgeo import gdal
import requests
import json

url_Baidu='https://api.map.baidu.com/reverse_geocoding/v3/'
AK=''#set your BaiduMap survice AK


#调用百度地图API
def get_loc_info(location):
    params={'location':location,
            'ak':AK,
            'output':'json',
            }
    try:
        res=requests.get(url_Baidu,params)
        jd=json.loads(res.text)
    except:
        jd={}
    if 'result' in jd.keys():
        try:
            province=jd['result']['addressComponent']['province']
        except:
            province={}    
    else:
        province={}
    return province

# 打开栅格文件


for year in range(2011,2021,1):
    input_raster_path = ''#please input the path of the targeted raster file 
    dataset = gdal.Open(input_raster_path)
    
    df=pd.DataFrame(columns=['value','province'])
    
    if dataset is None:
        print("无法打开栅格文件")
    else:
        # 获取投影坐标系和地理坐标系
        band=dataset.GetRasterBand(1)
        data=band.ReadAsArray()
        nondata=band.GetNoDataValue()
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        
        if projection and geotransform:
            source_proj = pyproj.Proj(init="epsg:2380")  # 投影坐标系
            target_proj = pyproj.Proj(init="epsg:4326")  # 地理坐标系
            
            # 获取栅格数据的宽度和高度
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            
            for y in range(height):
                for x in range(width):
                    value=data[y,x]
                    if value==nondata:
                        continue
                    # 获取栅格元的投影坐标
                    x_proj = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                    y_proj = geotransform[3] + x * geotransform[4] + y * geotransform[5]
                    
                    # 转换为地理坐标（WGS 1984）
                    x_wgs84, y_wgs84= pyproj.transform(source_proj,target_proj,x_proj, y_proj)
                    province=get_loc_info(str(y_wgs84)+','+str(x_wgs84))
                    new_row={'value':value,'province':province}
                    df = df.append(new_row, ignore_index=True)
                   
            # 关闭数据集
            dataset = None
        else:
            print("栅格文件没有投影信息或地理变换信息")
    pathOut=''#please set the path you want to save results
    df.to_excel(pathOut,sheet_name='sheet1')


