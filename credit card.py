# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:59:52 2018

@author: miya
"""

import pandas as pd

train_log_address=r'C:\Users\miya\Documents\GitHub\credit card\train_log.csv'
train_agg_address=r'C:\Users\miya\Documents\GitHub\credit card\train_agg.csv'
train_flg_address=r'C:\Users\miya\Documents\GitHub\credit card\train_flg.csv'

train_log=pd.read_csv(train_log_address)
train_log.head()
train_log=train_log['USRID\tEVT_LBL\tOCC_TIM\tTCH_TYP'].str.split('\t',expand=True)
train_log.columns=['USRID\tEVT_LBL\tOCC_TIM\tTCH_TYP'][0].split('\t')
train_log['EVT_LBL_0']=train_log['EVT_LBL'].str.split('-',expand=True)[0]
train_log['EVT_LBL_1']=train_log['EVT_LBL'].str.split('-',expand=True)[1]
train_log['EVT_LBL_2']=train_log['EVT_LBL'].str.split('-',expand=True)[2]
train_log['USRID']=train_log['USRID'].astype('int')
train_log['TCH_TYP']=train_log['TCH_TYP'].astype('int')

train_agg=pd.read_csv(train_agg_address)
train_agg_column=train_agg.columns.values
train_agg=train_agg[train_agg_column[0]].str.split('\t',expand=True)
train_agg.columns=train_agg_column[0].split('\t')
train_agg['USRID']=train_agg['USRID'].astype('int')
train_agg=train_agg.sort_values(by='USRID')
train_log_grouped=train_log.groupby(['USRID','EVT_LBL']).count()
train_log_dealed=train_log_grouped.groupby('USRID')['OCC_TIM'].agg(['max','mean'])
train_log_dealed=train_log_dealed.reset_index()
train_log_dealed_EL0=train_log.groupby(['USRID','EVT_LBL_0'])['EVT_LBL'].count()
train_log_dealed_EL0=train_log_dealed_EL0.reset_index()
train_log_dealed_EL0=train_log_dealed_EL0.groupby('USRID').count()
train_log_dealed_EL0=train_log_dealed_EL0.reset_index()
train_log_dealed_EL0.columns=['USRID','EVT_LBL_0_TYPE_C','EVT_LBL_0_TYPE_CC']
train_log_dealed_EL1=train_log.groupby(['USRID','EVT_LBL_1'])['EVT_LBL'].count()
train_log_dealed_EL1=train_log_dealed_EL1.reset_index()
train_log_dealed_EL1=train_log_dealed_EL1.groupby('USRID').count()
train_log_dealed_EL1=train_log_dealed_EL1.reset_index()
train_log_dealed_EL1.columns=['USRID','EVT_LBL_1_TYPE_C','EVT_LBL_1_TYPE_CC']
train_log_dealed_EL2=train_log.groupby(['USRID','EVT_LBL_2'])['EVT_LBL'].count()
train_log_dealed_EL2=train_log_dealed_EL2.reset_index()
train_log_dealed_EL2=train_log_dealed_EL2.groupby('USRID').count()
train_log_dealed_EL2=train_log_dealed_EL2.reset_index()
train_log_dealed_EL2.columns=['USRID','EVT_LBL_2_TYPE_C','EVT_LBL_2_TYPE_CC']
train_log_dealed.columns=['USRID','EVT_LBL_MAX','EVT_LBL_MEAN']
train_log_dealed=pd.merge(train_log_dealed,train_log_dealed_EL0[['USRID','EVT_LBL_0_TYPE_C']],how='left',on='USRID')
train_log_dealed=pd.merge(train_log_dealed,train_log_dealed_EL1[['USRID','EVT_LBL_1_TYPE_C']],how='left',on='USRID')
train_log_dealed=pd.merge(train_log_dealed,train_log_dealed_EL2[['USRID','EVT_LBL_2_TYPE_C']],how='left',on='USRID')


train_log_tch_typ=train_log.groupby('USRID')['TCH_TYP'].count()
train_log_tch_typ=train_log_tch_typ.reset_index()
train_log_tch_typ_0=train_log.loc[train_log['TCH_TYP']==0].groupby(['USRID','TCH_TYP'])['EVT_LBL'].count()
train_log_tch_typ_0=train_log_tch_typ_0.reset_index()
train_log_tch_typ_0.columns=['USRID','TCH_TYP','TCH_TYP_0_COUNT']
train_log_tch_typ_2=train_log.loc[train_log['TCH_TYP']==2].groupby(['USRID','TCH_TYP'])['EVT_LBL'].count()
train_log_tch_typ_2=train_log_tch_typ_2.reset_index()
train_log_tch_typ_2.columns=['USRID','TCH_TYP','TCH_TYP_2_COUNT']
train_log_tch_typ=pd.merge(train_log_tch_typ,train_log_tch_typ_0[['USRID','TCH_TYP_0_COUNT']],how='left',on='USRID')
train_log_tch_typ=pd.merge(train_log_tch_typ,train_log_tch_typ_2[['USRID','TCH_TYP_2_COUNT']],how='left',on='USRID')
train_log_tch_typ=train_log_tch_typ.fillna(0)

train_flg=pd.read_csv(train_flg_address)
train_flg_column=train_flg.columns.values
train_flg=train_flg[train_flg_column[0]].str.split('\t',expand=True)
train_flg.columns=train_flg_column[0].split('\t')
train_flg['USRID']=train_flg['USRID'].astype('int')
train_flg=train_flg.sort_values(by='USRID')

train_agg=pd.merge(train_agg,train_flg,how='left',on='USRID')
train_agg=pd.merge(train_agg,train_log_dealed,how='left',on='USRID')
train_agg=pd.merge(train_agg,train_log_tch_typ,how='left',on='USRID')
train_agg=train_agg.fillna(0)
