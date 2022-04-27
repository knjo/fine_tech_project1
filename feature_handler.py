# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:41:40 2021

@author: kevin
"""
import pandas as pd
import numpy as np
import math

## building fall class
def triple_barrier(df, period, ub, lb,label):
    ## 取超出邊界的資料，記錄第一筆為邊界收益，沒超出的話取時間內最後一筆
    def record_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]
    price = df.收盤價

    p = price.rolling(period+1).apply(record_price, raw=True).shift(-period)

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1
    signal.loc[p < lb] = -1
    #ret = pd.DataFrame({'triple_barrier_profit':p, 'triple_barrier_signal':signal})
    df[label] = signal

def drop_estimate(df, day , upper , lower , label):
    i=0
    list_ = []
    while i < len(df):
        try :
            UP = df['收盤價'][i+1:i+day+1].tolist() 
            U =(min(UP)-float(df['開盤價'][i+1]))/float(df['開盤價'][i+1]+0.0001)
            L =(max(LP)-float(df['開盤價'][i+1]))/float(df['開盤價'][i+1]+0.0001) 
            
            if ( U < -upper ) :
                if (L > -lower ) and (LP.index(max(LP)) < UP.index(min(UP))):
                    list_.append(0)
                else:
                    list_.append(1)                   
            else :
                list_.append(0)
        except Exception as e:
            print(e, i )
            list_.append(0)
        i+=1
    df[label] = list_
    
## building rise class
def rise_estimate(df, day , upper , lower ,label):
    i=0
    list_ = []
    while i < len(df):
        try :
            UP = df['最高價'][i+1:i+day+1].tolist()
            LP = df['最低價'][i+1:i+day+1].tolist()
            U =(max(UP)-float(df['開盤價'][i+1]))/float(df['開盤價'][i+1]+0.0001)
            L =(min(LP)-float(df['開盤價'][i+1]))/float(df['開盤價'][i+1]+0.0001)
            
            if U > upper :
                if ((L < lower) and ( LP.index(min(LP)) < UP.index(max(UP)))) :
                    list_.append(0)
                else:
                    list_.append(1)
            else :
                list_.append(0)
        except Exception as e:
            print(e, i )
            list_.append(0)
        i+=1
    df[label] = list_

# featur-------------------------------------
def merge_yesterday(xt,yt,xte,yte,ytp,ytep,t):
   
    xt["label"]=yt
    xt["預測結果"]=ytp
    
    
    xte["label"]=yte
    xte["預測結果"]=ytep
    
    ALL=xt.append(xte)

    ALL["昨天預測結果"]=ALL["預測結果"].shift(1)
    ALL["前天預測結果"]=ALL["預測結果"].shift(2)
    ALL["大前天預測結果"]=ALL["預測結果"].shift(3)
    ALL["昨天是否預測正確"]=(ALL["label"].shift(1)==ALL["昨天預測結果"]).astype(int)
    ALL["前天是否預測正確"]=ALL["昨天是否預測正確"].shift(1)
    ALL["大前天是否預測正確"]=ALL["昨天是否預測正確"].shift(2)
    ALL=ALL.dropna(subset=["昨天預測結果", "前天預測結果","大前天預測結果"])
    ALL["date"]=t["date"]
    ALL["證券代號"]=t["證券代號"]
    return ALL


def 加入產概(model,產概):
    model=model.merge(產概, how='left', left_on='證券代號', right_on='代號').drop(columns=["代號"])
    return model

def shift_price(df, day , upper ):
    i=0
    label_rise = "pre" + str(day) + "d" + str(upper)
    label_drop = "pre" + str(day) + "p" + str(upper)
    list_d = []
    list_p = []
    while i < (len(df)):
        try :        
            D =((df['最低價'][i-day])-float(df['收盤價'][i]))/float(df['收盤價'][i])
            if -D > upper :
                list_d.append(1)
            else :
                list_d.append(0)                
        except Exception as e:
            if i > day:
                print(e , i)
            list_d.append(0)
            
        try : 
            U =((df['最高價'][i-day])-float(df['收盤價'][i]))/float(df['收盤價'][i])
            if U > upper :
                list_p.append(1)
            else :
                list_p.append(0)                
        except Exception as e:
            if i > day:
                print(e , i)
            list_p.append(0)
        i+=1
    df[label_drop] = list_d
    df[label_rise] = list_p

def 前高(df,test_):
    i=0
    LP = 0 
    day = 0
    while i<len(df):         
        try:
            if (i < 60):
                test_.append(0)
            else:
                P = max(df['最高價'][i-60:i])
                if (P > LP) :
                    test_.append(P)
                    LP = P
                    day = i
                elif (day < (i-250)) :
                    test_.append(P)
                    LP = P
                    day = i
                else :
                    test_.append(LP)
        except Exception as e:
            print(e , i)
            test_.append(0)
        i+=1


def 波段前高(df,test_):
    i=0
    while i<len(df):         
        try:
            if (i < 60):
                test_.append(0)
            else:
                P = max(df['最高價'][i-60:i])
                test_.append(P)
        except Exception as e:
            print(e , i)
            test_.append(0)
        i+=1

def MA_index(df,day,label):
    return df[label].rolling(day).apply(np.mean)
        
def MA_cross(df, short, long,word=''):
    i=0
    list_=[]   
    short_label = 'MA' + str(short)
    long_label = 'MA' + str(long)
    while i <len(df):
        try:
            if(df[short_label][i]>df[long_label][i]):
                if(df[short_label][i-1]<df[long_label][i-1]):
                    list_.append(1)
                    if i+1< len(df):
                        list_.append(0.5)
                        i += 1
                else:
                    list_.append(0)
            else:
                if(df[short_label][i-1]>df[long_label][i-1]):
                    list_.append(-1)
                    if i+1< len(df):
                        list_.append(-0.5)
                        i += 1
                else:
                    list_.append(0)
        except Exception as e:
            print(e)
            list_.append(0)
        i+=1
    label = word +'MA_cross_'+ str(short)+ '_'+ str(long)
    df[label] = list_
    
def pre交叉型態(df):
    i=0
    list_=[]   
    while i <len(df):
        try:
            if(df['pre_5_ratio'][i]>df['pre_15_ratio'][i]):
                if(df['pre_5_ratio'][i-1]<df['pre_15_ratio'][i-1]):
                    list_.append(1)
                    if i< len(df):
                        list_.append(0.5)
                        i += 1
                else:
                    list_.append(0)
            else:
                if(df['pre_5_ratio'][i-1]>df['pre_15_ratio'][i-1]):
                    list_.append(-1)
                    if i< len(df):
                        list_.append(-0.5)
                        i += 1
                else:
                    list_.append(0)
        except Exception as e:
            print(e)
            list_.append(0)
        i+=1
    df['pre交叉型態'] = list_

def KD(df):
    i=0
    list_K = []
    list_D = []
    while i < len(df):
        try:
            RSV=(float(df['收盤價'][i])-min(df['最低價'][i-9:i]))/((max(df['最高價'][i-9:i])-min(df['最低價'][i-9:i]))+0.01)*100
            K = 2/3*list_K[i-1] + 1/3*RSV
            
            if K > 99 :
                K = 99
            elif K < 1 :
                K = 1

            list_K.append(K)
            list_D.append(2/3*list_D[i-1]+1/3*list_K[i-1])
        except Exception as e :
            list_K.append(50)
            list_D.append(50)
            if i > 9 :
                print(e)
                print (i,list_K[i-1])            
        i+=1
    df['K'] = list_K
    df['D'] = list_D    
    
#黃金交叉=1 死亡交叉=2 沒有交叉=3
def KD交叉型態(df):
    i=0
    list_ = []   
    while i<len(df):
        try:
            if(df['K'][i]>df['D'][i]):
                if(df['K'][i-1]<df['D'][i-1]):
                    list_.append(1)
                else:
                    list_.append(0)
            else:
                if(df['K'][i-1]>df['D'][i-1]):
                    list_.append(-1)
                else:
                    list_.append(0)
        except Exception as e:
            list_.append(0)
            if i >0 :                
                print(e)
        i+=1
    df['KD交叉型態'] = list_
    
# keep the index for the later three days
def KD交叉型態2(df):
    i=0
    list_ = []   
    while i<len(df):
        try:
            if(df['K'][i]>df['D'][i]):
                if(df['K'][i-1]<df['D'][i-1]):
                    list_.append(1)
                    if i+2< len(df):
                        list_.append(0.5)
                        list_.append(0.25)
                        i+=2
                else:
                    list_.append(0)
            else:
                if(df['K'][i-1]>df['D'][i-1]):
                    list_.append(-1)
                    if i+2< len(df):
                        list_.append(-0.5)
                        list_.append(-0.25)
                        i+=2
                else:
                    list_.append(0)
        except Exception as e:
            if i >0 :                
                print(e)
            list_.append(0)
        i+=1
    df['KD交叉型態2'] = list_[0:len(df['date'])]

def 漲跌(df):
    i=0
    list_ = []
    while i < len(df):
        try:
            list_.append(round((float(df['收盤價'][i])-float(df['收盤價'][i-1]))/(float(df['收盤價'][i-1])+0.001)*100,2))
        except Exception as e:
            list_.append(0)
            if i >0 :                
                print(e)
        i+=1
    df['漲跌'] = list_

#均線值是昨日的均線值
def RSI(price, period):
    up_move = np.zeros(len(price))
    down_move = np.zeros(len(price))
    
    for x in range(1,len(price)):
        

        if price[x] > price[x-1]:
            up_move[x] = price[x] - price[x-1]

        if price[x] < price[x-1]:
            down_move[x] = abs(price[x] - price[x-1])  

    ## 計算移動上升與下降，並計算RS 跟 RSI
    average_up = np.zeros(len(price))
    average_down = np.zeros(len(price))
    
    average_up[period] = up_move[1:period+1].mean()
    average_down[period] = down_move[1:period+1].mean()
    
    RS = np.zeros(len(price))
    RSI = np.zeros(len(price))
    
    RS[14] = average_up[period] / average_down[period]
    RSI[14] = 100 - (100/(1+RS[period]))
    
    ## 更新移動上升與下降，並計算RS 跟 RSI
    for x in range(period+1, len(price)):
        average_up[x] = (average_up[x-1]*(period-1)+up_move[x])/period
        average_down[x] = (average_down[x-1]*(period-1)+down_move[x])/period
        RS[x] = average_up[x] / average_down[x]
        RSI[x] = 100 - (100/(1+RS[x]))
    return RSI


        
def KD_indicator (data , day, word = '') : 
    high = data['High']
    low = data['Low']
    close = data['Close']

    RSV = [''] * len(data)
    RSV_days = day
    K_value = [''] * len(data)
    K_value[RSV_days-1] = 50
    D_value = [''] * len(data)
    D_value[RSV_days-1] = 50
    KD_cross = [''] * len(data)
    KD_over = [''] * len(data)
    KD_over_3days = [''] * len(data)


    for  i in range(RSV_days-1, len(data)):
        diff = max(high[i-RSV_days+1:i+1]) - min(low[i-RSV_days+1:i+1])
        RSV[i] = (close[i] - min(low[i-RSV_days+1:i+1])) / diff * 100
        if i >= RSV_days:
            if diff != 0 :
                K_value[i] = 2.0/3.0 * K_value[i-1] + 1.0/3.0 * RSV[i]
                D_value[i] = 2.0/3.0 * D_value[i-1] + 1.0/3.0 * K_value[i]
            else :
                K_value[i] = K_value[i-1]
                D_value[i] = D_value[i-1] 

            if K_value[i-1] < D_value[i-1] and K_value[i] > D_value[i]:
                KD_cross[i] = 1 #golden cross
            elif K_value[i-1] > D_value[i-1] and K_value[i] < D_value[i]:
                KD_cross[i] = -1 #death cross
            else:
                KD_cross[i] = 0

            if K_value[i] >= 80:
                KD_over[i] = 1 #overbought
            elif K_value[i] <= 20:
                KD_over[i] = -1 #oversold
            else:
                KD_over[i] = 0

        if i > RSV_days + 1:
            if KD_over[i] == 1 and KD_over[i-1] == 1 and KD_over[i-2] == 1:
                KD_over_3days[i] = 1
            elif KD_over[i] == -1 and KD_over[i-1] == -1 and KD_over[i-2] == -1:
                KD_over_3days[i] = -1
            else:
                KD_over_3days[i] = 0

    overBS = word+'overBuyOrSold(80/20)_' + str(day)
    cross = word+'cross_' + str(day)
    over_3days =  word+'over_3days_' + str(day)

    data['K'] = K_value
    data['D'] = D_value
    data[overBS] = KD_over
    data[cross] = KD_cross
    data[over_3days] = KD_over_3days      


def MACD(df , day , word='') :
    # 取12、26天的指數權重平均
    k = df.Close.ewm(span=12, adjust=False, min_periods=12).mean()
    d = df.Close.ewm(span=26, adjust=False, min_periods=26).mean()
    MACD_cross = [''] * len(df)
    macd = k - d
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=day).mean()
    macd_histogram = macd - macd_signal
    # Add all of our new values for the MACD to the dataframe
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD hist'] = macd_histogram

    for i in range(1,len(df)):
        if  macd_histogram.iloc[i-1] < 0 and  macd_histogram.iloc[i] > 0 :
            MACD_cross[i] = 1
        elif  macd_histogram.iloc[i-1] > 0 and macd_histogram.iloc[i] < 0  :
            MACD_cross[i] = -1
        else:
            MACD_cross[i] = 0
    label = word + 'MACD_cross_' + str(day)
    df[label] = MACD_cross

def Slope(df , feature , label):    
    def compute_gradient(s):
        if s[0] == 0 : 
            return  0
        return  round(s[1]/s[0] , 3)
    df[feature] = df[label].rolling(2).apply(compute_gradient,raw=True)

def RSI鈍化(df):
    i = 0 
    list_=[]
    while i<len(df):
        try:
            if(df['RSI'][i]>75):
                if df['收盤價'][i] > df['MA5'][i] :
                    list_.append(1)
                else:
                    list_.append(0)
            elif(df['RSI'][i]<20):
                if df['收盤價'][i] < df['MA5'][i] :
                    list_.append(-1)
                else:
                    list_.append(0) 
            else:
                list_.append(0)

        except Exception as e:
            print(e)
            list_.append (0)
        i+=1
    df['RSI5鈍化'] = list_

def intercept(df , feature , label1 , label2):
    list_ = []
    i = 0
    while i<len(df):
      
        try:
            K = df[label1][i]*df[label2][i]
            list_.append(K)
        except Exception as e:
            print(e)
            list_.append(0)
            
        i+=1
    df[feature] = list_

def 紅棒(df , feature , feature2):
    list_longbar = []
    list_antenna = []
    i = 0
    while i<len(df):
        try:
            if (df['漲跌'][i] > 1):
                K  = (df['收盤價'][i]-df['開盤價'][i]) /(df['最高價'][i]- df['開盤價'][i]+0.01)
                if K > 0 :
                    list_antenna.append(1-K)
                    list_longbar.append(K)
                else :
                    list_antenna.append(0)
                    list_longbar.append(0)
            else :
                list_antenna.append(0)
                list_longbar.append(0)
        except Exception as e:
            print(e)
            list_antenna.append(0)
            list_longbar.append(0)
        i+=1
    df[feature] = list_longbar
    df[feature2] = list_antenna


def 黑棒(df , feature , feature2):
    list_longbar = []
    list_antenna = []
    i = 0
    while i<len(df):
        try:
            if (df['漲跌'][i] < -1):
                K  = (df['收盤價'][i]-df['開盤價'][i]) / (df['最低價'][i]-df['開盤價'][i]+0.01) 
                if K > 0 :
                    list_antenna.append(1-K)
                    list_longbar.append(K)
                else :
                    list_antenna.append(0)
                    list_longbar.append(0)
            else :
                list_antenna.append(0)
                list_longbar.append(0)
        except Exception as e:
            print(e)
            list_antenna.append(0)
            list_longbar.append(0)
        i+=1
    df[feature] = list_longbar
    df[feature2] = list_antenna

def 避雷針(df , feature , feature2):
    list_U = []
    list_D = []
    i = 0
    while i<len(df['date']):
        try:
            K  = (df['最高價'][i]-df['收盤價'][i]) / df['開盤價'][i] * 100
            D  = (df['最低價'][i]-df['收盤價'][i]) / df['開盤價'][i] * 100
            list_U.append(K) 
            list_D.append(D)   
            # if (K > df['漲跌'][i]*3) and (K > 3):
            #     list_U.append(1)
            # else :
            #     list_U.append(0)  
              
                
            # if ( D < df['漲跌'][i]*3) and (D < -3) : 
            #     list_D.append(1)
            # else :
            #     list_D.append(0)
        except Exception as e:
            print(e)
            list_U.append(0)
            list_D.append(0)
        i+=1
    df[feature] = list_U
    df[feature2] = list_D

def 連續(df , feature ):
    list_U = []
    i = 0
    K = 0
    while i<len(df['date']):
        try:
            if ((df['實紅棒'][i] > 0) and (df['漲跌'][i] > 0)):
                if K < 0 :
                    K = 0
                else : 
                    K += 1
                list_U.append(K)
                
            elif ((df['實黑棒'][i] > 0 and (df['漲跌'][i] < 0))):
                if K > 0 :
                    K = 0
                else :
                    K -= 1
                list_U.append(K)
            else :
                if list_U[-1] == K:
                    list_U.append(K)                  
                else:
                    K = 0
                    list_U.append(K)
                
        except Exception as e:
            if i >= 1 :
                print(e)
            list_U.append(0)
        i+=1
    df[feature] = list_U

def distance(df , feature ,label1 , label2):
    list_ = []
    i = 0
    while i<len(df):
        
        try:
            K = round((df[label1][i] - df[label2][i])/(df[label1][i]+0.01),2)
            list_.append(K)
        except Exception as e:
            print(e)
            list_.append(0)
        i+=1
    df[feature] = list_

def Updown_state(df , feature ,label1 , label2):
    list_ = []
    i = 0
    while i<len(df):
        
        try:
            if df[label1][i] > (df[label2][i] * 1.02) :
                list_.append(1)
            elif df[label1][i] < (df[label2][i] * 0.97):
                list_.append(-1)
            else:
                list_.append(0)
        except Exception as e:
            print(e)
            list_.append(0)
        i+=1
    df[feature] = list_

def 均線糾結(df):
    i = 0 
    list_=[]
    list_2 = [] 
    while i<len(df):
        try:
            MA5 = df['MA5'][i]
            MA10 = df['MA10'][i]
            MA20 = df['MA20'][i]
            MA60 = df['MA60'][i]
            d = min(MA5,MA10,MA20) / max(MA5,MA10,MA20) 
            d2 = min(MA5,MA10,MA20,MA60) / max(MA5,MA10,MA20,MA60) 
            if(d >0.96) :
                list_.append(1)
            else:
                list_.append(0)
            if(d2 >0.95) :
                list_2.append(1)
            else:
                list_2.append(0)
                
        except Exception as e:
            if (max(MA5,MA10,MA20,MA60) !=0) :
                print(e)
            list_.append (0)
        i+=1
    df['均線糾結1'] = list_
    df['均線糾結2'] = list_2

def 三陽開泰 (df):
    i = 0
    list_ = []
    while i<len(df):
        try:
   
            P = max(df.MA5[i],df.MA10[i],df.MA20[i])
            if ((df['收盤價'][i] > P) & (df['收盤價'][i] > df['開盤價'][i]) ):
                if (max(list_[-10:-2])== 0):
                    list_.append(df['漲跌'][i])
                else:
                    list_.append(0)
            else :
                list_.append(0)
        except Exception as e:
            print(e , i)
            list_.append(0)
        i+=1
    df['三陽開泰'] = list_

def 突破前高(df):
    i = 0
    list_ = []
    while i<len(df):
        try:
            if((df['年前高'][i]!=0) & (df['收盤價'][i] > df['年前高'][i])):
                list_.append(1)
            else :
                list_.append(0)
        except Exception as e:
            print(e , i)
            list_.append(0)
            
        i+=1
    df['突破前高'] = list_

def 突破整理(df):
    i = 0
    list_ = []
    while i<len(df):
        try:
            if((max(df['均線糾結1'][i-10:i])==1) & (df['收盤價'][i] > df['季前高'][i])) :
                if (max(list_[-10:]) != 1):
                    list_.append(1)
                else :
                    list_.append(0)
            else :
                list_.append(0)
        except Exception as e:
            print(e , i)
            list_.append(0)
            
        i+=1
    df['突破前高'] = list_

def Ratio(df , feature , label1 , label2):
    list_ = []
    i = 0
    while i<len(df):      
        try:
            K = df[label1][i] / (df[label2][i]+0.001)
            list_.append(K)
        except Exception as e:
            print(e)
            list_.append(0)
            
        i+=1
    df[feature] = list_

def day_Volatility (df , feature) :
    list_ = []
    i = 0
    while i<len(df):     
        try:
            K = (df['最高價'][i]- df['最低價'][i])*100/(df['最低價'][i]+0.001)
            list_.append(K)
        except Exception as e:
            print(e)
            list_.append(0)
        i+=1
    df[feature] = list_
    
def Implied_Volatility (df , feature , day) :   
    df[feature] = df['日振幅'].rolling(day).apply(np.mean)

def 季均漲跌天數(df , day) :
    list_rise = []
    list_fall = []
    i = 0
    while i<len(df):
        if i < day :
            list_rise.append(0)
            list_fall.append(0)
            i+=1
            continue
        try:
            interval_day = df['連漲跌'][i-day:i+1]
            
            P = interval_day[interval_day > 0]
            D = interval_day[interval_day < 0]
            if len(P) == 0 :
                list_rise.append(0)
            else :
                list_rise.append(round(sum(P)/len(P)))
            if len(D) == 0 :
                list_fall.append(0)
            else :
                list_fall.append(round(sum(D)/len(D)))
        except Exception as e:
            print(e)
            list_rise.append(0)
            list_fall.append(0)
        i+=1
    df['均漲天數'] = list_rise
    df['均跌天數'] = list_fall

def 超漲跌(df) :
    list_ = []
    i = 0
    while i<len(df):        
        try:
            if df['連漲跌'][i] > df['均漲天數'][i] :
                list_.append(1)
            elif df['連漲跌'][i] < df['均跌天數'][i] :
                list_.append(-1)
            else :
                list_.append(0)
        except Exception as e:
            print(e)
            list_.append(0)
        i+=1
    df['超漲跌'] = list_

    

