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
from matplotlib.animation import FuncAnimation
import io
import base64

address='https://www.kaggle.com/maheshdadhich/strength-of-visualization-python-visuals-tutorial/code'

#导入和合并数据
s=time.time()
train_fr_1=pd.read_csv(r'C:\Users\miya\Documents\GitHub\trip_duration\fastest_routes_train_part_1.csv')
train_fr_2=pd.read_csv(r'C:\Users\miya\Documents\GitHub\trip_duration\fastest_routes_train_part_2.csv')
train_fr=pd.concat([train_fr_1,train_fr_2])
train_fr_new=train_fr[['id','total_distance','total_travel_time','number_of_steps']]
train_df=pd.read_csv(r'C:\Users\miya\Documents\GitHub\trip_duration\train.csv')
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
train_data.loc[:,'hvsine_pick_drop']=haversine_(train_data['pickup_latitude'].values
              ,train_data['pickup_longitude'].values,train_data['dropoff_latitude'].values,
              train_data['dropoff_longitude'].values)
train_data.loc[:,'manhtn_pick_drop']=manhattan_distance_pd(train_data['pickup_latitude'].values
              ,train_data['pickup_longitude'].values,train_data['dropoff_latitude'].values,
              train_data['dropoff_longitude'].values)
train_data.loc[:,'bearing']=bearing_array(train_data['pickup_latitude'].values
              ,train_data['pickup_longitude'].values,train_data['dropoff_latitude'].values,
              train_data['dropoff_longitude'].values)

def color(hour):
    return hour*10

def Animation(hour,temp):
    train_data_new=temp.loc[temp['hour']==hour]
    rgb1=np.zeros((3000,3500,3),dtype=np.uint8)
    train_data_new['pick_lat_new']=list(map(int,(train_data_new['pickup_latitude']-40.6)*10000))
    train_data_new['drop_lat_new']=list(map(int,(train_data_new['dropoff_latitude']-40.6)*10000))
    train_data_new['pick_lon_new']=list(map(int,(train_data_new['pickup_longitude']-(-74.05))*10000))
    train_data_new['drop_lon_new']=list(map(int,(train_data_new['dropoff_longitude']-(-74.05))*10000))
    summary_plot=pd.DataFrame(train_data_new.groupby(['pick_lat_new','pick_lon_new'])['id'].count())
    
    summary_plot.reset_index(inplace=True)
    summary_plot.head(120)
    
    for k in range(len(summary_plot['id'])):
        i=summary_plot['pick_lat_new'][k]
        j=summary_plot['pick_lon_new'][k]
        a1=summary_plot['id'][k]
        if(a1//50)>0:
            rgb1[i][j][0]=255-color(hour)
            rgb1[i,j,1]=255-color(hour)
            rgb1[i,j,2]=0+color(hour)
        elif(a1//10)>0:
            rgb1[i,j,0]=0+color(hour)
            rgb1[i,j,1]=255-color(hour)
            rgb1[i,j,2]=0+color(hour)
        else:
            rgb1[i,j,0]=255-color(hour)
            rgb1[i,j,1]=0+color(hour)
            rgb1[i,j,2]=0+color(hour)
    
    return rgb1

images_list=[]
train_data_new['pickup_datetime']=pd.to_datetime(train_data_new['pickup_datetime'])
train_data_new.loc[:,'hour']=train_data_new['pickup_datetime'].dt.hour

for i in list(range(0,24)):
    im=Animation(i,train_data_new)
    images_list.append(im)

#画出上车地点随小时的变化动图
def build_gif(imgs=images_list,show_gif=False,save_gif=True,title=''):
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    ax.set_axis_off()
    hr_range=(list(range(0,24)))
    
    def show_im(pairs):
        ax.clear()
        ax.set_title('absolute traffic-hour'+str(int(pairs[0]))+':00')
        ax.imshow(pairs[1])
        ax.set_axis_off()
    
    pairs=list(zip(hr_range,imgs))
    im_ani=animation.FuncAnimation(fig,show_im,pairs,interval=500,repeat_delay=0,blit=False)
    plt.cla()
    if save_gif:
        im_ani.save('animation.html',writer='imagemagick')
    if show_gif:
        plt.show()
    return

build_gif()

filename = 'animation.html'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
#findings：上午2点到6点出行较少，上午7点到下午4点出行为中等，下午5点到凌晨1点出行较多

#一周的每一天与总行程时间的关系
summary_wdays_avg_duration=pd.DataFrame(train_data.groupby(['vendor_id','day_of_week'])['trip_duration'].mean())
summary_wdays_avg_duration.reset_index(inplace=True)
summary_wdays_avg_duration['unit']=1
sns.set(style='white',palette='muted',color_codes=True)
sns.set_context('poster')
sns.tsplot(data=summary_wdays_avg_duration,time='day_of_week',unit='unit',condition='vendor_id',
           value='trip_duration')
sns.despine(bottom=False)
#findings
#一周的每一天，vendor 1比vendor2的trip时间长，介于1~250之间

#利用琴型图（violinplot）分析乘客数与trip时间的关系
sns.set(style='white',palette='pastel',color_codes=True)
sns.set_context('poster')
train_data2=train_data.copy()
train_data2['trip_duration']=np.log(train_data['trip_duration'])
sns.violinplot(x='passenger_count',y='trip_duration',hue='vendor_id',data=train_data2,inner='quart',split=True,
               palette={1:'g',2:'r'})
sns.despine(left=True)
#findings
#当乘客数是0时，vendor1和2都会有trip时间为负值，应去除
#乘客数为0的解释是，乘客提前叫了一辆出租车，在等待
#vendor1在人数是2和3的，有长的trip时间
#人数为7，8，9的trip duration非常少

#箱型图猜测一周每一天与形成时间的关系
sns.set(style='ticks')
sns.boxplot(x='day_of_week',y='trip_duration',hue='vendor_id',data=train_data,palette='PRGn')
plt.ylim(0,6000)
sns.despine(trim=True,offset=10)
#findings
#周1，2，3，4比其他天的trip duration 长

#line-plots分析一周每一天按小时的trip duration变化
summary_hour_duration=pd.DataFrame(train_data.groupby(['day_of_week','pick_hour'])['trip_duration'].mean())
summary_hour_duration.reset_index(inplace=True)
summary_hour_duration['unit']=1
sns.set(style='white',palette='muted',color_codes=False)
sns.tsplot(time='pick_hour',unit='unit',condition='day_of_week',value='trip_duration',data=summary_hour_duration)
sns.despine(bottom=False)
#findings
#在am5：00~15：00时，周六和周日的trip duration明显比工作日少
#周六的深夜 trip dutation比其他的时间多

#集群分析（cluster）

def assign_cluster(df,k):
    df_pick=df[['pickup_longitude','pickup_latitude']]
    df_drop=df[['dropoff_longitude','dropoff_latitude']]
    init=np.array([[ -73.98737616,   40.72981533],
       [-121.93328857,   37.38933945],
       [ -73.78423222,   40.64711269],
       [ -73.9546417 ,   40.77377538],
       [ -66.84140269,   36.64537175],
       [ -73.87040541,   40.77016484],
       [ -73.97316185,   40.75814346],
       [ -73.98861094,   40.7527791 ],
       [ -72.80966949,   51.88108444],
       [ -76.99779701,   38.47370625],
       [ -73.96975298,   40.69089596],
       [ -74.00816622,   40.71414939],
       [ -66.97216034,   44.37194443],
       [ -61.33552933,   37.85105133],
       [ -73.98001393,   40.7783577 ],
       [ -72.00626526,   43.20296402],
       [ -73.07618713,   35.03469086],
       [ -73.95759366,   40.80316361],
       [ -79.20167796,   41.04752096],
       [ -74.00106031,   40.73867723]])
    k_means_pick=KMeans(n_clusters=k,init=init,n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick=k_means_pick.labels_
    df['label_pick']=clust_pick.tolist()
    df['label_drop']=k_means_pick.predict(df_drop)
    return df,k_means_pick

train_cl,k_means=assign_cluster(train_data,20)
centroid_pickups=pd.DataFrame(k_means.cluster_centers_,columns=['centroid_pick_long','centroid_pick_lat'])
centroid_dropoff=pd.DataFrame(k_means.cluster_centers_,columns=['centroid_drop_long','centroid_drop_lat'])
centroid_pickups['label_pick']=centroid_pickups.index
centroid_dropoff['label_drop']=centroid_dropoff.index
train_cl=pd.merge(train_cl,centroid_pickups,how='left',on='label_pick')
train_cl=pd.merge(train_cl,centroid_dropoff,how='left',on='label_drop')

#根据集群，计算上下车的距离和方位
train_cl.loc[:,'hvsine_pick_cent_p']=haversine_(train_cl['pickup_latitude'].values,train_cl['pickup_longitude'].values,
            train_cl['centroid_pick_lat'].values,train_cl['centroid_pick_long'].values)#计算上车地点只集群点的距离
train_cl.loc[:,'hvsine_drop_cent_d']=haversine_(train_cl['dropoff_latitude'].values,train_cl['dropoff_longitude'].values,
            train_cl['centroid_drop_lat'].values,train_cl['centroid_drop_long'].values)#计算下车地点至集群点的距离
train_cl.loc[:,'hvsine_cent_p_cent_d']=haversine_(train_cl['centroid_pick_lat'].values,train_cl['centroid_pick_long'].values,
            train_cl['centroid_drop_lat'].values,train_cl['centroid_drop_long'].values)#计算上车集群点和下车集群点之间的距离

train_cl.loc[:,'manhtn_pick_cent_p']=manhattan_distance_pd(train_cl['pickup_latitude'].values,train_cl['pickup_longitude'].values,
            train_cl['centroid_pick_lat'].values,train_cl['centroid_pick_long'].values)#计算上车地点只集群点的距离
train_cl.loc[:,'manhtn_drop_cent_d']=manhattan_distance_pd(train_cl['dropoff_latitude'].values,train_cl['dropoff_longitude'].values,
            train_cl['centroid_drop_lat'].values,train_cl['centroid_drop_long'].values)#计算下车地点至集群点的距离
train_cl.loc[:,'manhtn_cent_p_cent_d']=manhattan_distance_pd(train_cl['centroid_pick_lat'].values,train_cl['centroid_pick_long'].values,
            train_cl['centroid_drop_lat'].values,train_cl['centroid_drop_long'].values)#计算上车集群点和下车集群点之间的距离

train_cl.loc[:,'bearing_pick_cent_p']=bearing_array(train_cl['pickup_latitude'].values,train_cl['pickup_longitude'].values,
            train_cl['centroid_pick_lat'].values,train_cl['centroid_pick_long'].values)#计算上车地点只集群点的方位
train_cl.loc[:,'bearing_drop_cent_d']=bearing_array(train_cl['dropoff_latitude'].values,train_cl['dropoff_longitude'].values,
            train_cl['centroid_drop_lat'].values,train_cl['centroid_drop_long'].values)#计算下车地点至集群点的方位
train_cl.loc[:,'bearing_cent_p_cent_d']=bearing_array(train_cl['centroid_pick_lat'].values,train_cl['centroid_pick_long'].values,
            train_cl['centroid_drop_lat'].values,train_cl['centroid_drop_long'].values)#计算上车集群点和下车集群点之间的方位

train_cl['speed_hvsn']=train_cl.hvsine_pick_drop/train_cl.total_travel_time
train_cl['speed_manhtn']=train_cl.manhtn_pick_drop/train_cl.total_travel_time
train_cl.head()

def cluster_summary(sum_df):
    #该函数求每一个集群点的trip平均时间和最多的起始点行程次数
    summary_avg_time=pd.DataFrame(sum_df.groupby('label_pick')['trip_duration'].mean())
    summary_avg_time.reset_index(inplace=True)
    summary_pref_clus=pd.DataFrame(sum_df.groupby(['label_pick','label_drop'])['id'].count())
    summary_pref_clus.reset_index(inplace=True)
    summary_pref_clus=summary_pref_clus.loc[summary_pref_clus.groupby('label_pick')['id'].idxmax()]
    summary=pd.merge(summary_avg_time,summary_pref_clus,how='left',on='label_pick')
    summary=summary.rename(columns={'trip_duration':'avg_triptime'})
    return summary

import folium
def show_fmaps(train_data,path=1):
    full_data=train_data
    summary_full_data=pd.DataFrame(full_data.groupby('label_pick')['id'].count())
    summary_full_data.reset_index(inplace=True)
    summary_full_data=summary_full_data.loc[summary_full_data['id']>70000]
    map_1=folium.Map(location=[40.7679,-73.9821],zoom_start=10,tiles='Stamen Toner')
    new_df=train_data.loc[train_data['label_pick'].isin(summary_full_data.label_pick.tolist())].sample(50)
    new_df.reset_index(inplace=True,drop=True)
    for i in range(new_df.shape[0]):
#        pick_long=new_df.loc[i,'pickup_longitude']
#        pick_lat=new_df.loc[i,'pickup_latitude']
#        dest_long=new_df.loc[i,'dropoff_longitude']
#        dest_lat=new_df.loc[i,'dropoff_latitude']
        pick_long = new_df.loc[new_df.index ==i]['pickup_longitude'].values[0]
        pick_lat = new_df.loc[new_df.index ==i]['pickup_latitude'].values[0]
        dest_long = new_df.loc[new_df.index ==i]['dropoff_longitude'].values[0]
        dest_lat = new_df.loc[new_df.index ==i]['dropoff_latitude'].values[0]
        folium.Marker([pick_lat,pick_long]).add_to(map_1)
        folium.Marker([dest_lat,dest_long]).add_to(map_1)
    return map_1

osm=show_fmaps(train_data,path=1)
osm.save('map.html')
    
def clusters_map(clus_data,full_data,tile='OpenStreetMap',sig=0,zoom=12,circle=0,radius_=30):
    map_1=folium.Map(location=[40.7679,-73.9821],zoom_start=zoom,tiles=tile)
    summary_full_data=pd.DataFrame(full_data.groupby('label_pick')['id'].count())
    summary_full_data.reset_index(inplace=True)
    if sig==1:
        summary_full_data=summary_full_data.loc[summary_full_data['id']>70000]
    sig_cluster=summary_full_data['label_pick'].tolist()
    clus_summary=cluster_summary(full_data)
    for i in sig_cluster:
        pick_long=clus_data.loc[clus_data.index==i]['centroid_pick_long'].values[0]
        pick_lat=clus_data.loc[clus_data.index==i]['centroid_pick_lat'].values[0]
        clus_no=clus_data.loc[clus_data.index==i]['label_pick'].values[0]
        most_visited_clus=clus_summary.loc[clus_summary['label_pick']==i]['label_drop'].values[0]
        avg_triptime=clus_summary.loc[clus_summary['label_pick']==i]['avg_triptime'].values[0]
        pop='cluster='+str(clus_no)+'&most visited cluster='+str(most_visited_clus)+'&avg triptime from this cluster='+str(avg_triptime)
        if circle==1:
            folium.CircleMarker(location=[pick_lat,pick_long],radius=radius_,
                                color='#F08080',
                                fill_color='#3186cc',popup=pop).add_to(map_1)
        folium.Marker([pick_lat,pick_long],popup=pop).add_to(map_1)
    return map_1

clus_map=clusters_map(centroid_pickups,train_cl,sig=0,zoom=3.2,circle=1,tile='Stamen Terrain')
clus_map.save('clus_map.html')

clus_map_sig=clusters_map(centroid_pickups,train_cl,sig=1,circle=1)
clus_map_sig.save('clus_map_sig.html')        

from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(train_data.sample(1200)[['vendor_id','day_of_week','passenger_count',
                     'pick_month','label_pick','pick_hour']],'vendor_id',colormap='rainbow')
plt.show()
        
#从测试数据中提取特征

fastest_routes_test=pd.read_csv(r'C:\Users\miya\Documents\GitHub\trip_duration\fastest_routes_test.csv')
test=pd.read_csv(r'C:\Users\miya\Documents\GitHub\trip_duration\test.csv')

test_sc=fastest_routes_test[['id','total_distance','total_travel_time','number_of_steps']]
test_new=pd.merge(test,test_sc,on='id',how='left')

test_new['pickup_datetime']=pd.to_datetime(test_new['pickup_datetime'])
test_new['pick_hour']=test_new['pickup_datetime'].dt.hour
test_new['pick_month']=test_new['pickup_datetime'].dt.month
test_new['day_of_week']=test_new['pickup_datetime'].dt.dayofweek
test_new['day_of_year']=test_new['pickup_datetime'].dt.dayofyear
test_new['week_of_year']=test_new['pickup_datetime'].dt.weekofyear

test_new['hvsine_pick_drop']=haversine_(test_new['pickup_latitude'].values,test_new['pickup_longitude'].values,
        test_new['dropoff_latitude'].values,test_new['dropoff_longitude'].values)
test_new['manhtn_pick_drop']=manhattan_distance_pd(test_new['pickup_latitude'].values,test_new['pickup_longitude'].values,
        test_new['dropoff_latitude'].values,test_new['dropoff_longitude'].values)
test_new['bearing']=bearing_array(test_new['pickup_latitude'].values,test_new['pickup_longitude'].values,
        test_new['dropoff_latitude'].values,test_new['dropoff_longitude'].values)

test_new['label_pick']=k_means.predict(test_new[['pickup_latitude','pickup_longitude']])
test_new['label_drop']=k_means.predict(test_new[['dropoff_latitude','dropoff_longitude']])
test_cl=pd.merge(test_new,centroid_pickups,how='left',on='label_pick')
test_cl=pd.merge(test_cl,centroid_dropoff,how='left',on='label_drop')

test_cl['hvsine_pick_cent_p']=haversine_(test_cl['pickup_latitude'].values,test_cl['pickup_longitude'].values,
       test_cl['centroid_pick_lat'].values,test_cl['centroid_pick_long'].values)
test_cl['hvsine_drop_cent_d']=haversine_(test_cl['dropoff_latitude'].values,test_cl['dropoff_longitude'].values,
       test_cl['centroid_drop_lat'].values,test_cl['centroid_drop_long'].values)
test_cl['hvsine_cent_p_cent_d']=haversine_(test_cl['centroid_pick_lat'].values,test_cl['centroid_pick_long'].values,
       test_cl['centroid_drop_lat'].values,test_cl['centroid_drop_long'].values)

test_cl['manhtn_pick_cent_p']=manhattan_distance_pd(test_cl['pickup_latitude'].values,test_cl['pickup_longitude'].values,
       test_cl['centroid_pick_lat'].values,test_cl['centroid_pick_long'].values)
test_cl['manhtn_drop_cent_d']=manhattan_distance_pd(test_cl['dropoff_latitude'].values,test_cl['dropoff_longitude'].values,
       test_cl['centroid_drop_lat'].values,test_cl['centroid_drop_long'].values)
test_cl['manhtn_cent_p_cent_d']=manhattan_distance_pd(test_cl['centroid_pick_lat'].values,test_cl['centroid_pick_long'].values,
       test_cl['centroid_drop_lat'].values,test_cl['centroid_drop_long'].values)

test_cl['bearing_pick_cent_p']=bearing_array(test_cl['pickup_latitude'].values,test_cl['pickup_longitude'].values,
       test_cl['centroid_pick_lat'].values,test_cl['centroid_pick_long'].values)
test_cl['bearing_drop_cent_d']=bearing_array(test_cl['dropoff_latitude'].values,test_cl['dropoff_longitude'].values,
       test_cl['centroid_drop_lat'].values,test_cl['centroid_drop_long'].values)
test_cl['bearing_cent_p_cent_d']=bearing_array(test_cl['centroid_pick_lat'].values,test_cl['centroid_pick_long'].values,
       test_cl['centroid_drop_lat'].values,test_cl['centroid_drop_long'].values)

test_cl['speed_hvsn']=test_cl['hvsine_pick_drop']/test_cl['total_travel_time']
test_cl['speed_manhtn']=test_cl['manhtn_pick_drop']/test_cl['total_travel_time']

#建立XGB模型
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import warnings

train=train_cl
test=test_cl
coords=np.vstack((train[['pickup_latitude','pickup_longitude']].values,
                 train[['dropoff_latitude','dropoff_longitude']].values,
                 test[['pickup_latitude','pickup_longitude']].values,
                 test[['dropoff_latitude','dropoff_longitude']].values))

pca=PCA().fit(coords)#主成份分析
train['pickup_pca0']=pca.transform(train[['pickup_latitude','pickup_longitude']])[:,0]#特征1的成分
train['pickup_pca1']=pca.transform(train[['pickup_latitude','pickup_longitude']])[:,1]#特征2的成分
train['dropoff_pca0']=pca.transform(train[['dropoff_latitude','dropoff_longitude']])[:,0]
train['dropoff_pca1']=pca.transform(train[['dropoff_latitude','dropoff_longitude']])[:,1]
test['pickup_pca0']=pca.transform(test[['pickup_latitude','pickup_longitude']])[:,0]#特征1的成分
test['pickup_pca1']=pca.transform(test[['pickup_latitude','pickup_longitude']])[:,1]#特征2的成分
test['dropoff_pca0']=pca.transform(test[['dropoff_latitude','dropoff_longitude']])[:,0]
test['dropoff_pca1']=pca.transform(test[['dropoff_latitude','dropoff_longitude']])[:,1]

train['store_and_fwd_flag_int']=np.where(train['store_and_fwd_flag']=='N',0,1)
test['store_and_fwd_flag_int']=np.where(test['store_and_fwd_flag']=='N',0,1)

feature_names=list(train.columns)
print('different features in train and test are {}'.format(np.setdiff1d(train.columns,test.columns)))

do_not_use_for_training=['pickup_datetime','id','dropoff_datetime','store_and_fwd_flag','trip_duration']
feature_names=[f for f in train.columns if f not in do_not_use_for_training]
print('we will be using following features for training {}'.format(feature_names))
print('')
print('total number of features are {}'.format(len(feature_names)))

y=np.log(train['trip_duration'].values+1)

#对train进行训练数据和测试数据分割
xtr,xv,ytr,yv=train_test_split(train[feature_names].values,y,test_size=0.2,random_state=1987)
dtrain=xgb.DMatrix(xtr,label=ytr)
dvalid=xgb.DMatrix(xv,label=yv)
dtest=xgb.DMatrix(test[feature_names].values)
watch=[(dtrain,'train'),(dvalid,'valide')]
xgb_pars={'min_child_weight':50,'eta':0.3,'colsample_bytree':0.3,'max_depth':10,'subsample':0.8,
          'lambda':1.,'nthread':-1,'booster':'gbtree','silent':1,'eval_metric':'rmse','objective':'reg:linear'}
model=xgb.train(xgb_pars,dtrain,15,watch,early_stopping_rounds=2,maximize=False,verbose_eval=1)
print('modeling RMSLE %.5f'% model.best_score)

#考虑天气对trip_duration的影响
weather=pd.read_csv(r'C:\Users\miya\Documents\GitHub\trip_duration\weather_data_nyc_centralpark_2016.csv')

from ggplot import *
weather['date']=pd.to_datetime(weather.date)
weather['day_of_year']=weather['date'].dt.dayofyear
p=ggplot(aes(x='date'),data=weather)+geom_line(aes(y='minimum temperature',colour='blue'))+geom_line(aes(y='maximum temperature',colour='red'))
p+geom_point(aes(y='minimum temperature',colour='blue'))
#findings
#二月份的最小温度达到了零下，发现trip_duration比其他时间多


train_plot=train[['pickup_datetime','trip_duration']]
train_plot['pickup_date']=train_plot['pickup_datetime'].dt.date
train_plot.drop('pickup_datetime',axis=1,inplace=True)
train_plot['trip_duration']=np.log(train_plot['trip_duration'])
train_plot_grouped=train_plot.groupby('pickup_date')['trip_duration'].mean()
train_plot_grouped=pd.DataFrame(train_plot_grouped)
train_plot_grouped.reset_index(inplace=True)
train_plot_grouped.rename(columns={'trip_duration':'trip_duration_log'},inplace=True)
train_plot_grouped['pickup_date']=pd.to_datetime(train_plot_grouped.pickup_date)
p1=ggplot(aes(x='pickup_date'),data=train_plot_grouped)+geom_line(aes(y='trip_duration_log',colour='blue'))
p1+geom_point(aes(y='trip_duration_log',colour='blue'))

weather['precipitation'].unique()
weather['precipitation']=np.where(weather['precipitation']=='T',0,weather['precipitation'])
weather['precipitation']=list(map(float,weather['precipitation']))
weather['snow fall']=np.where(weather['snow fall']=='T',0,weather['snow fall'])
weather['snow fall']=list(map(float,weather['snow fall']))
weather['snow depth']=np.where(weather['snow depth']=='T',0,weather['snow depth'])
weather['snow depth']=list(map(float,weather['snow depth']))

import plotly.graph_objs as go
import plotly
import plotly.plotly as py

random_x=weather['date'].values
random_y0=weather['precipitation']
random_y1=weather['snow fall']
random_y2=weather['snow depth']

trace0=go.Scatter(x=random_x,y=random_y0,mode='markers',name='precipitation')
trace1=go.Scatter(x=random_x,y=random_y1,mode='markers',name='snow fall')
trace2=go.Scatter(x=random_x,y=random_y2,mode='markers',name='snow depth')
data=[trace0,trace1,trace2]

plotly.offline.iplot(data, filename='scatter-mode')
p1=ggplot(aes(x='pickup_date'),data=train_plot_grouped)+geom_line(aes(y='trip_duration_log',colour='blue'))

def freq_turn(step_dir):
    #功能是获得step_dir的每一个方向的个数
    from collections import Counter
    step_dir_new=step_dir.split('|')
    a_list=Counter(step_dir_new).most_common()
    path={}
    for i in range(len(a_list)):
        path.update({a_list[i]})
    a=0
    b=0
    c=0
    if 'straigth' in (path.keys()):
        a=path['straigth']
    if 'left' in (path.keys()):
        b=path['left']
    if 'right' in (path.keys()):
        c=path['right']
    return a,b,c

train_fr['straigth']=0
train_fr['left']=0
train_fr['right']=0

train_fr['straigth'],train_fr['left'],train_fr['right']=zip(*train_fr['step_direction'].map(freq_turn))
train_fr_new=train_fr[['id','straigth','left','right']]
train=pd.merge(train,train_fr_new,on='id',how='left')

train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
train['date']=train['pickup_datetime'].dt.date
train=pd.merge(train,weather[['date','minimum temperature', 'precipitation', 'snow fall', 'snow depth']],
               on='date',how='left')
train.loc[:,'hvsine_pick_cent_d'] = haversine_(train['pickup_latitude'].values, 
         train['pickup_longitude'].values, train['centroid_drop_lat'].values, train['centroid_drop_long'].values)
train.loc[:,'hvsine_drop_cent_p'] = haversine_(train['dropoff_latitude'].values,
         train['dropoff_longitude'].values, train['centroid_pick_lat'].values, train['centroid_pick_long'].values)

test.loc[:,'hvsine_pick_cent_d'] = haversine_(test['pickup_latitude'].values, test['pickup_longitude'].values, 
        test['centroid_drop_lat'].values, test['centroid_drop_long'].values)
test.loc[:,'hvsine_drop_cent_p'] = haversine_(test['dropoff_latitude'].values, test['dropoff_longitude'].values, 
        test['centroid_pick_lat'].values, test['centroid_pick_long'].values)

#分析新特性的影响
temp=train[['hvsine_drop_cent_p','hvsine_pick_cent_d','hvsine_drop_cent_d','hvsine_pick_cent_p','hvsine_pick_drop',
            'hvsine_cent_p_cent_d','total_distance']]
temp.total_distance.dropna(inplace=True)
print('total_number of nulls:'.format(temp.total_distance.isnull().sum()))
temp['distance_pick_cp_cd_drop']=temp['hvsine_pick_cent_p']+temp['hvsine_cent_p_cent_d']+temp['hvsine_drop_cent_d']
temp['distance_pick_cd_drop']=temp['hvsine_pick_cent_d']+temp['hvsine_drop_cent_d']
temp['distance_pick_cp_drop']=temp['hvsine_pick_cent_p']+temp['hvsine_drop_cent_p']
temp['total_distance']=np.floor(temp['total_distance']/1000)
temp['distance_pick_cp_drop']=np.floor(temp['distance_pick_cp_drop'])
temp['distance_pick_cd_drop']=np.floor(temp['distance_pick_cd_drop'])
temp['distance_pick_cp_cd_drop']=np.floor(temp['distance_pick_cp_cd_drop'])
temp1=temp.copy()
temp=temp1.sample(100000)
aggregation={'distance_pick_cp_cd_drop':'count','distance_pick_cp_drop':'count','distance_pick_cd_drop':'count'}
temp2=pd.DataFrame(temp.groupby('total_distance').agg(aggregation))
x_plot=np.linspace(0,temp.total_distance.max(),temp.shape[0])
temp2.rename(columns={'total_distance':'count'},inplace=True)
temp2.reset_index(inplace=True)
temp2.total_distance=list(map(int,temp2.total_distance))#map后生成map对象，若显示结果，需要在生成map时使用List
x_plot=temp.total_distance.unique()
a=np.histogram(temp[['total_distance']].values,list(range(0,95)))
N=temp.shape[0]
data=[]
trace1=go.Scatter(x=a[1],y=a[0],mode='lines',fill='tozeroy',line={'color': 'black', 'width': 2},name='Total_distance_OSRM')
data.append(trace1)

for kernel in ['distance_pick_cp_cd_drop','distance_pick_cd_drop','distance_pick_cp_drop']:
    trace2=go.Scatter(x=a[1],y=np.histogram(temp[[kernel]].values,range(0,95))[0],mode='lines',line=dict(width=2,dash='dash'),name=kernel)
    data.append(trace2)
layout=go.Layout(annotations=[dict(x=6,y=0.38,showarrow=False,text='N={0} points'.format(N))],
                              xaxis=dict(zeroline=False),hovermode='closest')
fig=go.Figure(data=data,layout=layout)
plotly.offline.iplot(fig)

sns.set(style='white')
temp3=train.copy()
corr=temp3.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f,ax=plt.subplots(figsize=(15,13))
cmap=sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(corr,mask=mask,cmap=cmap,vmax=3,center=0,square=True,linewidths=5,cbar_kws=dict(shrink=5))

test_fr=fastest_routes_test.copy()
test_fr['straigth']=0
test_fr['left']=0
test_fr['right']=0
test_fr['straigth'],test_fr['left'],test_fr['right']=zip(*test_fr['step_direction'].map(freq_turn))
test_fr_new=test_fr[['id','straigth','left','right']]
test=pd.merge(test,test_fr_new,on='id',how='left')
print(test.columns.shape[0])

test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])
test['date']=test['pickup_datetime'].dt.date
test['date']=pd.to_datetime(test['date'])
test=pd.merge(test,weather[['date','minimum temperature','precipitation','snow fall','snow depth']],
              on='date',how='left')

yvalid=model.predict(dvalid)
ytest=model.predict(dtest)

do_not_use_for_training=['pickup_datetime','id','dropoff_datetime','store_and_fwd_flag','trip_duration','date']
feature_names=[f for f in train.columns if f not in do_not_use_for_training]
xtr1,xv1,ytr1,yv1=train_test_split(train[feature_names].values,y,test_size=0.2,random_state=1987)
dtrain1=xgb.DMatrix(xtr1,label=ytr1)
dvalid1=xgb.DMatrix(xv1,label=yv1)
dtest1=xgb.DMatrix(test[feature_names].values)
watch=[(dtrain1,'train'),(dvalid1,'valide')]
xgb_pars={'min_child_weight':50,'eta':0.3,'colsample_bytree':0.3,'max_depth':10,'subsample':0.8,
          'lambda':1.,'nthread':-1,'booster':'gbtree','silent':1,'eval_metric':'rmse','objective':'reg:linear'}
model1=xgb.train(xgb_pars,dtrain1,15,watch,early_stopping_rounds=2,maximize=False,verbose_eval=1)
print('modeling1 RMSLE %.5f'% model1.best_score)

fig,ax=plt.subplots(nrows=2,sharex=True,sharey=True)
sns.distplot(yvalid,ax=ax[0],color='blue',label='Validation')
sns.distplot(ytest,ax=ax[1],color='green',label='Test')
plt.show()

feature_importance_dict=model1.get_fscore()
fs=['f%i'% i for i in range(len(feature_names))]
f1=pd.DataFrame({'f':list(feature_importance_dict.keys()),
                 'importance':list(feature_importance_dict.values())})
f2=pd.DataFrame({'f':fs,'feature_name':feature_names})

feature_importance=pd.merge(f2,f1,on='f',how='left')
feature_importance=feature_importance.fillna(0)
feature_importance[['feature_name','importance']].sort_values(by='importance',ascending=False)

for index,row in feature_importance.head(5).iterrows():
    print(row)