# -*- coding: utf-8 -*-
"""
Created on Mon May 14 09:25:19 2018

@author: Wizza
"""

import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
#import simplejson    #getting JSON in simplified format
import urllib        #for url stuff
#import gmaps       #for using google maps to visulalize places on maps
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
from haversine import haversine # for calculating haversine distance
import math          #for basic maths operations
import seaborn as sns #for making plots
import matplotlib.pyplot as plt # for plotting
import os  # for os commands
from scipy.misc import imread, imresize, imsave  # for plots 
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_notebook, show
from IPython.display import HTML
from matplotlib.pyplot import *
from matplotlib import cm
from matplotlib import animation
import io
import base64

address='https://www.kaggle.com/maheshdadhich/strength-of-visualization-python-visuals-tutorial/code'

#导入和合并数据
s=time.time()
train_fr_1=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\trip duration\fastest_routes_train_part_1.csv')
train_fr_2=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\trip duration\fastest_routes_train_part_2.csv')
train_fr=pd.concat([train_fr_1,train_fr_2])
train_fr_new=train_fr[['id','total_distance','total_travel_time','number_of_steps']]
train_df=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\trip duration\train.csv')
train=pd.merge(train_df,train_fr_new,on='id',how='left')
train_df=train.copy()
end=time.time()
print('time taken by above cell is {}'.format((end-s)))
train_df.head()

#检查ID是否唯一
start=time.time()
train_data=train_df.copy()
print('number of columns and rows are {} and {} respectively.'.format(train_data.shape[1],train_data.shape[0]))
if np.unique(train_data['id']).shape[0]==train_data['id'].shape[0]:
    print('ids are unique')
print('number of nulls - {}'.format(train_data.isnull().sum().sum()))
end=time.time()
print('time taken by above cell is {}.'.format(end-start))

#预测变量行程时间可视化
start=time.time()
sns.set(style='white',palette='muted',color_codes=True)
f,axes=plt.subplots(1,1,figsize=(11,7),sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_df['trip_duration'].values+1),
             axlabel='log(trip_duration)',label='log(trip_duration)',bins=50,color='r')
plt.setp(axes,yticks=[])
plt.tight_layout()
end=time.time()
print('time taken by above cell is {}.'.format((end-start)))
plt.show()
#结论：trip_duration普遍分布在e^4-1~e^8-1之间

#经纬度分析
start=time.time()
sns.set(style='white',palette='deep',color_codes=True)
f,axes=plt.subplots(2,2,figsize=(10,10))
sns.despine(left=True)
sns.distplot(train_df['pickup_latitude'].values,label='pickup_latitude',color='m',bins=100,ax=axes[0,0])
sns.distplot(train_df['pickup_longitude'].values,label='pickup_longitide',color='m',bins=100,ax=axes[0,1])
sns.distplot(train_df['dropoff_latitude'].values,label='dropoff_latitude',color='m',bins=100,ax=axes[1,0])
sns.distplot(train_df['dropoff_longitude'].values,label='dropoff_longitide',color='m',bins=100,ax=axes[1,1])
plt.setp(axes,yticks=[])
plt.tight_layout()
end=time.time()
print('time taken by above cell is {}.'.format((end-start)))
plt.show()
#可视化结论：上车和下车纬度（latitude）集中于40~42之间，经度（longitude）集中于-76~-73之间

#缩小经纬度范围
df=train_df.loc[(train_df.pickup_latitude>40.6)&(train_df.pickup_latitude<40.9)]
df=df.loc[(df.dropoff_latitude>40.6)&(df.dropoff_latitude<40.9)]
df=df.loc[(df.dropoff_longitude>-74.05)&(df.dropoff_longitude<-73.7)]
df=df.loc[(df.pickup_longitude>-74.05)&(df.pickup_longitude<-73.7)]
train_data_new=df.copy()
sns.set(style='white',palette='muted',color_codes=True)
f,axes=plt.subplots(2,2,figsize=(12,12))
sns.despine(left=True)
sns.distplot(train_data_new['pickup_latitude'].values,label='pickup_latitude',color='m',bins=100,ax=axes[0,0])
sns.distplot(train_data_new['pickup_longitude'].values,label='pickup_longitide',color='g',bins=100,ax=axes[0,1])
sns.distplot(train_data_new['dropoff_latitude'].values,label='dropoff_latitude',color='m',bins=100,ax=axes[1,0])
sns.distplot(train_data_new['dropoff_longitude'].values,label='dropoff_longitide',color='g',bins=100,ax=axes[1,1])
plt.setp(axes,yticks=[])
plt.tight_layout()
print(df.shape[0],train_data.shape[0])
#结论：latitude：40.6~40.9,longitude:-74.05~-73.7

temp=train_data.copy()
train_data['pickup_datetime']=pd.to_datetime(train_data.pickup_datetime)
train_data.loc[:,'pick_date']=train_data['pickup_datetime'].dt.date
train_data.head()

ts_v1=pd.DataFrame(train_data.loc[train_data['vendor_id']==1].groupby('pick_date')['trip_duration'].mean())
ts_v1.reset_index(inplace=True)
ts_v2=pd.DataFrame(train_data.loc[train_data['vendor_id']==2].groupby('pick_date')['trip_duration'].mean())
ts_v2.reset_index(inplace=True)

from bokeh.palettes import Spectral4
from bokeh.plotting import figure,output_notebook,show

output_notebook()
p=figure(plot_width=800, plot_height=250, x_axis_type="datetime")
p.title.text = 'Click on legend entries to hide the corresponding lines'

for data, name, color in zip([ts_v1, ts_v2], ["vendor 1", "vendor 2"], Spectral4):
    df = data
    p.line(df['pick_date'], df['trip_duration'], line_width=2, color=color, alpha=0.8, legend=name)

p.legend.location = "top_left"
p.legend.click_policy="hide"
show(p)

#显示地图坐标
rgb=np.zeros((3000,3500,3),dtype=np.uint8)
train_data_new['pick_lat_new']=list(map(int,(train_data_new['pickup_latitude']-40.6)*10000))
train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))
train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))
train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))

summary_plot=pd.DataFrame(train_data_new.groupby(['pick_lat_new','pick_lon_new'])['id'].count())
summary_plot.reset_index(inplace=True)
summary_plot.head(120)
lat_list=summary_plot['pick_lat_new'].unique()

for i in lat_list:
    lon_list=summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()
    unit=summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()
    for j in lon_list:
        a=unit[lon_list.index(j)]
        if(a//50)>0:
            rgb[i][j][0]=255
            rgb[i,j,1]=0
            rgb[i,j,2]=255
        elif(a//10)>0:
            rgb[i,j,0]=0
            rgb[i,j,1]=255
            rgb[i,j,2]=0
        else:
            rgb[i,j,0]=255
            rgb[i,j,1]=0
            rgb[i,j,2]=0
fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(14,20))
ax.imshow(rgb,cmap='hot')
ax.set_axis_off()

rgb1=np.zeros((3000,3500,3),dtype=np.uint8)
for k in range(len(summary_plot['id'])):
    a=summary_plot['id']
    i=summary_plot['pick_lat_new'][k]
    j=summary_plot['pick_lon_new'][k]
    a1=summary_plot['id'][k]
    if(a1//50)>0:
        rgb1[i][j][0]=255
        rgb1[i,j,1]=0
        rgb1[i,j,2]=255
    elif(a1//10)>0:
        rgb1[i,j,0]=0
        rgb1[i,j,1]=255
        rgb1[i,j,2]=0
    else:
        rgb1[i,j,0]=255
        rgb1[i,j,1]=0
        rgb1[i,j,2]=0

fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(14,20))
ax.imshow(rgb1,cmap='hot')
ax.set_axis_off()
#finds：红色点表明<10个的trips作为起点，绿色表明10~50个的trips作为起点，黄色表明>50的trips作为起点

#特征提取
def haversine_(lat1,lng1,lat2,lng2):
    #该函数用来计算两个经纬度之间的距离
    lat1,lng1,lat2,lng2=map(np.radians,(lat1,lng1,lat2,lng2))
    AVG_EARTH_RADIUS=6371 #KM
    lat=lat2-lat1
    lng=lng2-lng1
    d=np.sin(lat*0.5)**2+np.cos(lat1)*np.cos(lat2)*np.sin(lng*0.5)**2
    h=2*AVG_EARTH_RADIUS*np.arcsin(np.sqrt(d))
    return(h)

def manhattan_distance_pd(lat1,lng1,lat2,lng2):
    #该函数用来计算曼哈顿距离
    a=haversine_(lat1,lng1,lat1,lng2)
    b=haversine_(lat1,lng1,lat2,lng1)
    return a+b

def bearing_array(lat1,lng1,lat2,lng2):
    #该函数用于计算方位
    lng_delta_rad=np.radians(lng2-lng1)
    lat1,lng1,lat2,lng2=map(np.radians,(lat1,lng1,lat2,lng2))
    y=np.sin(lng_delta_rad)*np.cos(lat2)
    x=np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y,x))

train_data=temp.copy()
train_data['pickup_datetime']=pd.to_datetime(train_data.pickup_datetime)
train_data.loc[:,'pick_month']=train_data['pickup_datetime'].dt.month
train_data.loc[:,'pick_hour']=train_data['pickup_datetime'].dt.hour
train_data.loc[:,'week_of_year']=train_data['pickup_datetime'].dt.weekofyear
train_data.loc[:,'day_of_year']=train_data['pickup_datetime'].dt.dayofyear
train_data.loc[:,'day_of_week']=train_data['pickup_datetime'].dt.dayofweek
train_data.loc[:,'hvsine_pick_drop']=haversine(train_data['pickup_latitude'].values
              ,train_data['pickup_longitude'].values,train_data['pickoff_latitude'].values,
              train_data['pickoff_longitude'].values)
train_data.loc[:,'manhtn_pick_drop']=manhattan_distance_pd(train_data['pickup_latitude'].values
              ,train_data['pickup_longitude'].values,train_data['pickoff_latitude'].values,
              train_data['pickoff_longitude'].values)
train_data.loc[:,'bearing']=bearing_array(train_data['pickup_latitude'].values
              ,train_data['pickup_longitude'].values,train_data['pickoff_latitude'].values,
              train_data['pickoff_longitude'].values)

