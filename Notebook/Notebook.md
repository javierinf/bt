# BEAT challenge


```python
import pandas as pd
import numpy as np
import pickle
```

## Load Pickled requests.csv


```python
with open('request.pickle', 'rb') as handle:
    df_req = pickle.load(handle)
#df_req = df_req.sample(n=5000)

```

## Convert columns to datetime


```python
df_req['created_at'] = pd.to_datetime(df_req['created_at'])
df_req['cancelled_at'] = pd.to_datetime(df_req['cancelled_at'])
```


```python
df_req.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_request</th>
      <th>id_passenger</th>
      <th>id_city</th>
      <th>from_latitude</th>
      <th>from_longitude</th>
      <th>to_latitude</th>
      <th>to_longitude</th>
      <th>created_at</th>
      <th>cancelled_at</th>
      <th>timedout_at</th>
      <th>passenger_device</th>
      <th>passenger_payment_mean</th>
      <th>passenger_udid</th>
      <th>distance_estimate</th>
      <th>duration_estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3513924914</td>
      <td>2590587260</td>
      <td>1</td>
      <td>19.593987</td>
      <td>-99.039326</td>
      <td>19.578557</td>
      <td>-99.041541</td>
      <td>2019-09-19 04:07:56</td>
      <td>2019-09-19 04:08:01</td>
      <td>NaN</td>
      <td>PIXI5-6_4G/9008A</td>
      <td>cash</td>
      <td>2026460688</td>
      <td>3.47</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1594446343</td>
      <td>1501665959</td>
      <td>1</td>
      <td>19.303307</td>
      <td>-98.886432</td>
      <td>19.261320</td>
      <td>-98.878055</td>
      <td>2019-09-19 22:44:43</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>tissot_sprout/Mi A1</td>
      <td>cash</td>
      <td>4196047656</td>
      <td>6.81</td>
      <td>512.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2312709661</td>
      <td>1546836297</td>
      <td>1</td>
      <td>19.326330</td>
      <td>-99.121689</td>
      <td>19.310036</td>
      <td>-99.124487</td>
      <td>2019-09-19 23:00:37</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>deen_sprout/motorola one</td>
      <td>cash</td>
      <td>166030629</td>
      <td>2.19</td>
      <td>788.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1498373688</td>
      <td>2884063389</td>
      <td>1</td>
      <td>19.401755</td>
      <td>-99.175365</td>
      <td>19.400671</td>
      <td>-99.188342</td>
      <td>2019-09-20 16:31:24</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>HWNXT/HUAWEI NXT-L09</td>
      <td>cash</td>
      <td>510294940</td>
      <td>2.93</td>
      <td>750.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197696634</td>
      <td>2893722658</td>
      <td>1</td>
      <td>19.392388</td>
      <td>-99.057337</td>
      <td>19.370834</td>
      <td>-99.005069</td>
      <td>2019-09-20 23:26:56</td>
      <td>2019-09-20 23:29:31</td>
      <td>NaN</td>
      <td>dream2lte/SM-G955F</td>
      <td>cash</td>
      <td>3552408027</td>
      <td>9.53</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## One-hot encode payment method


```python

df_req = pd.get_dummies(df_req, columns=['passenger_payment_mean'])

```


```python
df_req.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_request</th>
      <th>id_passenger</th>
      <th>id_city</th>
      <th>from_latitude</th>
      <th>from_longitude</th>
      <th>to_latitude</th>
      <th>to_longitude</th>
      <th>created_at</th>
      <th>cancelled_at</th>
      <th>timedout_at</th>
      <th>passenger_device</th>
      <th>passenger_udid</th>
      <th>distance_estimate</th>
      <th>duration_estimate</th>
      <th>passenger_payment_mean_cash</th>
      <th>passenger_payment_mean_cc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3513924914</td>
      <td>2590587260</td>
      <td>1</td>
      <td>19.593987</td>
      <td>-99.039326</td>
      <td>19.578557</td>
      <td>-99.041541</td>
      <td>2019-09-19 04:07:56</td>
      <td>2019-09-19 04:08:01</td>
      <td>NaN</td>
      <td>PIXI5-6_4G/9008A</td>
      <td>2026460688</td>
      <td>3.47</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1594446343</td>
      <td>1501665959</td>
      <td>1</td>
      <td>19.303307</td>
      <td>-98.886432</td>
      <td>19.261320</td>
      <td>-98.878055</td>
      <td>2019-09-19 22:44:43</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>tissot_sprout/Mi A1</td>
      <td>4196047656</td>
      <td>6.81</td>
      <td>512.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2312709661</td>
      <td>1546836297</td>
      <td>1</td>
      <td>19.326330</td>
      <td>-99.121689</td>
      <td>19.310036</td>
      <td>-99.124487</td>
      <td>2019-09-19 23:00:37</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>deen_sprout/motorola one</td>
      <td>166030629</td>
      <td>2.19</td>
      <td>788.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1498373688</td>
      <td>2884063389</td>
      <td>1</td>
      <td>19.401755</td>
      <td>-99.175365</td>
      <td>19.400671</td>
      <td>-99.188342</td>
      <td>2019-09-20 16:31:24</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>HWNXT/HUAWEI NXT-L09</td>
      <td>510294940</td>
      <td>2.93</td>
      <td>750.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197696634</td>
      <td>2893722658</td>
      <td>1</td>
      <td>19.392388</td>
      <td>-99.057337</td>
      <td>19.370834</td>
      <td>-99.005069</td>
      <td>2019-09-20 23:26:56</td>
      <td>2019-09-20 23:29:31</td>
      <td>NaN</td>
      <td>dream2lte/SM-G955F</td>
      <td>3552408027</td>
      <td>9.53</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Create was_canceled column

This column will be 1 if the request was canceled


```python
df_req['was_canceled'] = df_req['cancelled_at'].isnull()
df_req['row'] = 1 #this is a helper column
```

## Created time_to_cancel column
As the difference from created_at and cancelled_at. Nulls were filled with -999 just to be able to use plotting tools


```python
df_req['time_to_cancel']= df_req['cancelled_at'] - df_req['created_at']
df_req['time_to_cancel'] = df_req['time_to_cancel'].fillna(pd.Timedelta(seconds=0))
df_req['time_to_cancel'].describe()
```




    count                     10425144
    mean     0 days 00:00:23.326642106
    std      0 days 00:16:19.898338541
    min                0 days 00:00:00
    25%                0 days 00:00:00
    50%                0 days 00:00:00
    75%                0 days 00:00:05
    max               33 days 00:12:42
    Name: time_to_cancel, dtype: object



## Histogram 

We could see a peak from 1 to 5 seconds. This is in line with the challange instructions


```python
import matplotlib.pyplot as plt



mask = (df_req["time_to_cancel"] > pd.Timedelta(seconds=0)) & (df_req["time_to_cancel"] < pd.Timedelta(seconds=30))
df = df_req[mask]

df.set_index('time_to_cancel', inplace=True)
df.resample('1S').size().plot.bar()

```




    <AxesSubplot:xlabel='time_to_cancel'>




![png](output_16_1.png)


### Now we will see what if we filter CC requests only


```python
df[df['passenger_payment_mean_cc']==1].resample('1S').size().plot.bar()
```




    <AxesSubplot:xlabel='time_to_cancel'>




![png](output_18_1.png)


There is a spike on 2 and 3 seconds, but compared to the average, this delta is small. 


```python
#df_req['count_canceled_by_user'] = df_req.groupby('id_passenger')['was_canceled'].transform(np.sum)
#df_req['count_request_by_user'] = df_req.groupby('id_passenger')['row'].transform(np.sum)
```

## Group by user and drop some columns


```python
df_req_gp = df_req.groupby('id_passenger').sum()
df_req_gp = df_req_gp.drop(columns=['from_latitude', 'from_longitude','to_latitude','to_longitude'])
```

### Create a cancel_rate column


```python
df_req_gp = df_req_gp[df_req_gp['was_canceled'].notnull()]
df_req_gp['cancel_rate']=df_req_gp['was_canceled']/df_req_gp['row']
df_req_gp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_request</th>
      <th>id_city</th>
      <th>passenger_udid</th>
      <th>distance_estimate</th>
      <th>duration_estimate</th>
      <th>passenger_payment_mean_cash</th>
      <th>passenger_payment_mean_cc</th>
      <th>was_canceled</th>
      <th>row</th>
      <th>cancel_rate</th>
    </tr>
    <tr>
      <th>id_passenger</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4421</th>
      <td>4009186975</td>
      <td>1</td>
      <td>211686013</td>
      <td>1.97</td>
      <td>298.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17500</th>
      <td>16735817555</td>
      <td>9</td>
      <td>708257610</td>
      <td>73.87</td>
      <td>2690.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>9</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>21566</th>
      <td>25916832734</td>
      <td>14</td>
      <td>39344800208</td>
      <td>64.74</td>
      <td>1558.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>14</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>26581</th>
      <td>5567630726</td>
      <td>9</td>
      <td>4296822432</td>
      <td>15.90</td>
      <td>2439.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>34374</th>
      <td>323163638</td>
      <td>1</td>
      <td>1607201580</td>
      <td>3.43</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Plot cancel_rate with some dimensions

### Plot for all users


```python
mask = df_req_gp['passenger_payment_mean_cc']>0
cc = df_req_gp[mask]

mask3 = df_req_gp['passenger_payment_mean_cash']>0
cashs = df_req_gp[mask3]
mask4 = cashs['passenger_payment_mean_cc'] == 0
cash = cashs[mask4]
```


```python
import seaborn as sns
sns.distplot(cc["cancel_rate"], bins = 20,hist = False,label = 'CC')
sns.distplot(cash["cancel_rate"], bins = 20,hist = False ,label = 'CASH ONLY')



```




    <AxesSubplot:xlabel='cancel_rate'>




![png](output_28_1.png)


### Plot for more than 4 usages


```python
mask = df_req_gp['passenger_payment_mean_cc']>4
cc = df_req_gp[mask]

mask3 = df_req_gp['passenger_payment_mean_cash']>4
cashs = df_req_gp[mask3]
mask4 = cashs['passenger_payment_mean_cc'] == 0
cash = cashs[mask4]
sns.distplot(cc["cancel_rate"], bins = 20,hist = False,label = 'CC')
sns.distplot(cash["cancel_rate"], bins = 20,hist = False ,label = 'CASH ONLY')
```




    <AxesSubplot:xlabel='cancel_rate'>




![png](output_30_1.png)


### Plor for more than 20 usages


```python
mask = df_req_gp['passenger_payment_mean_cc']>20
cc = df_req_gp[mask]

mask3 = df_req_gp['passenger_payment_mean_cash']>20
cashs = df_req_gp[mask3]
mask4 = cashs['passenger_payment_mean_cc'] == 0
cash = cashs[mask4]
sns.distplot(cc["cancel_rate"], bins = 20,hist = False,label = 'CC')
sns.distplot(cash["cancel_rate"], bins = 20,hist = False ,label = 'CASH ONLY')
```




    <AxesSubplot:xlabel='cancel_rate'>




![png](output_32_1.png)


As we can see there is a significan amount of users with cancelation_rate = 1, even in more than 20 usages.
It's clear that this suspucious spike is smaller for frequent users and it decreeses faster for credit card users.

In the last plot, it seems that the cash only group is the sum of two totally diffent distributions, maybe fraudsters and legit users.

This two different groups is not so evident on credit card group, but in lower usage there is a suspicious amount of cancel_rate = 1 on four uses. We could not asume that credit card users are totally legit, but we could be less severe.

## How to setup a threeshold

First we will filter users with a cancel_rate equals 1 and analyze how many requests had they made


```python
mask = df_req_gp['cancel_rate']== 1
fil = df_req_gp[mask]
fil['row'].describe()
```




    count    256751.000000
    mean          3.692753
    std           5.566465
    min           1.000000
    25%           1.000000
    50%           2.000000
    75%           4.000000
    max         225.000000
    Name: row, dtype: float64



We could see that 256k users from 750k, had no non-canceled requests. Most of them made no more than 4 requests, but in some cases they made more than 200.

Based on this, we should get a threeshold number of requests from when we decide that a user is making a fraudulent request. 
Ideally we should have ground truth values, but as is not provided, I found that 10 gives us around a 2% of fraud and is a reasonable number for a user to 'learn' how to use the service


```python
mask = df_req_gp['cancel_rate']== 1
fil2 = df_req_gp[mask]
mask2 = fil2['row'] > 10
fil = fil2[mask2]
fil['row'].describe()
```




    count    16719.000000
    mean        19.381004
    std         12.163098
    min         11.000000
    25%         12.000000
    50%         15.000000
    75%         21.000000
    max        225.000000
    Name: row, dtype: float64



### Threeshold conclusions

* We will label all users with more than 10 requests and a cancelation_rate of 1, as fraudsters

* Please note that users with more than 10 and some requests accepted, will not be labeled, for example if some user one the 15th attempt, actually pays the trip, is not a fraudster

* Payment method will not be used to select fraudsters, but will be a feature to predict among users

## Next steps for the next notebook

1) Select a way to split the dataset in order to test the model results

2) Select a way to measure the accuracy of the predictions

3) Build the corresponding features and train a model





```python

```
