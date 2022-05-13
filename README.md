# Container forcast using raindom forest model

In this project, we aims the build a prediction model for container throughput in Thailand port. 

We first load all the data collected throughout year 2001-2021. This included inbound and outbound container throughput. As for the features, we use features as follows

- Consumer price index
- Export value
- Import value
- GDP constant
- Inflation rate
- Interest rate
- Manufacture product index
- Population
- Unemployment rate
- USD to THB conversion rate


```python
import numpy as np
import pandas as pd
```

# Loading label for model prediction


```python
label_df = pd.read_csv("container_throughput_label.csv")
label_df.head()

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
      <th>year</th>
      <th>month</th>
      <th>inbound</th>
      <th>outbound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2002</td>
      <td>January</td>
      <td>107493</td>
      <td>96214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002</td>
      <td>February</td>
      <td>97798</td>
      <td>97257</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>March</td>
      <td>111474</td>
      <td>112393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2002</td>
      <td>April</td>
      <td>101110</td>
      <td>107746</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>May</td>
      <td>112976</td>
      <td>119299</td>
    </tr>
  </tbody>
</table>
</div>



# Load features for model prediction


```python
exportval_df = pd.read_csv("export_value.csv")
gdp_df = pd.read_csv("GDP_constant.csv")
importval_df = pd.read_csv("import_value.csv")
inflate_df = pd.read_csv("inflation_%.csv")
interest_df = pd.read_csv("interest_rate.csv")
manu_df = pd.read_csv("manufac_prod_index.csv")
pop_df = pd.read_csv("population.csv")
unemp_df = pd.read_csv("unemployment.csv")
ex_df = pd.read_csv("usd_thb.csv")
cons_df = pd.read_csv("consumer_price_index.csv")
```


```python
exportval_df = exportval_df.drop('id', axis=1)

exportval_df = exportval_df[["year", "month", "export_value"]]
exportval_df['export_value'] = exportval_df['export_value'].map(lambda x: x.replace(',', ''))
exportval_df['export_value'] = pd.to_numeric(exportval_df['export_value'])
```


```python
gdp_df = gdp_df.drop('id', axis=1)

gdp_df = gdp_df[["year", "month", "GDP_constant"]]
gdp_df['GDP_constant'] = gdp_df['GDP_constant'].map(lambda x: x.replace(',', ''))
gdp_df['GDP_constant'] = pd.to_numeric(gdp_df['GDP_constant'])
```


```python
importval_df = importval_df.drop('id', axis=1)

importval_df = importval_df[["year", "month", "import_value"]]
importval_df['import_value'] = importval_df['import_value'].map(lambda x: x.replace(',', ''))
importval_df['import_value'] = pd.to_numeric(importval_df['import_value'])
```


```python
inflate_df = inflate_df.drop('id', axis=1)

inflate_df = inflate_df[["year", "month", "inflation_percentage_change"]]
inflate_df['inflation_percentage_change'] = inflate_df['inflation_percentage_change'].map(lambda x: x.replace('%', ''))
inflate_df['inflation_percentage_change'] = pd.to_numeric(inflate_df['inflation_percentage_change'])
```


```python
interest_df = interest_df.drop('id', axis=1)

interest_df = interest_df[["year", "month", "interest_rate"]]
interest_df['interest_rate'] = pd.to_numeric(interest_df['interest_rate'])
```


```python
manu_df = manu_df.drop('id', axis=1)

manu_df = manu_df[["year", "month", "manufac_prod_index"]]
manu_df['manufac_prod_index'] = pd.to_numeric(manu_df['manufac_prod_index'])
```


```python
pop_df = pop_df.drop('id', axis=1)

pop_df = pop_df[["year", "month", "population"]]
pop_df['population'] = pop_df['population'].map(lambda x: x.replace(',', ''))
pop_df['population'] = pd.to_numeric(pop_df['population'])
```


```python
unemp_df = unemp_df.drop('id', axis=1)

unemp_df = unemp_df[["year", "month", "unemployment_rate"]]
unemp_df = unemp_df.replace(' n.a. ', np.nan)
unemp_df['unemployment_rate'] = pd.to_numeric(unemp_df['unemployment_rate'])
```


```python
ex_df = ex_df.drop('id', axis=1)

ex_df = ex_df[["year", "month", "exchange_rate"]]
ex_df = ex_df.replace(' n.a. ', np.nan)
ex_df['exchange_rate'] = pd.to_numeric(ex_df['exchange_rate'])
```


```python
cons_df = cons_df.drop('id', axis=1)

cons_df = cons_df[["year", "month", "consumer_price_index"]]
cons_df = cons_df.replace(' n.a. ', np.nan)
cons_df['consumer_price_index'] = pd.to_numeric(cons_df['consumer_price_index'])
```

# Merge all Dataframe into one


```python
months = ['January', 'February', 'March', 'April', 'May', 
'June', 'July', 'August', 'September', 'October', 'November', 'December']

f_df = pd.DataFrame(data={'year': np.array([[year]*12 for year in range(2001,2022)]).flatten(),
                        'month': months*21 })

```


```python
f_df = pd.merge(left=f_df, right=exportval_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=gdp_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=importval_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=inflate_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=interest_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=manu_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=pop_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=unemp_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=ex_df, how='left', on=['year','month'])
f_df = pd.merge(left=f_df, right=cons_df, how='left', on=['year','month'])
```


```python
f_df.head()
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
      <th>year</th>
      <th>month</th>
      <th>export_value</th>
      <th>GDP_constant</th>
      <th>import_value</th>
      <th>inflation_percentage_change</th>
      <th>interest_rate</th>
      <th>manufac_prod_index</th>
      <th>population</th>
      <th>unemployment_rate</th>
      <th>exchange_rate</th>
      <th>consumer_price_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>279973.0</td>
      <td>459359.0</td>
      <td>255061.0</td>
      <td>0.72</td>
      <td>2.5</td>
      <td>52.47</td>
      <td>62308887.0</td>
      <td>5.73</td>
      <td>43.12</td>
      <td>68.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>279973.0</td>
      <td>459359.0</td>
      <td>255061.0</td>
      <td>0.44</td>
      <td>2.0</td>
      <td>53.02</td>
      <td>62308887.0</td>
      <td>4.25</td>
      <td>42.64</td>
      <td>69.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>279973.0</td>
      <td>459359.0</td>
      <td>255061.0</td>
      <td>0.00</td>
      <td>2.0</td>
      <td>52.18</td>
      <td>62308887.0</td>
      <td>4.04</td>
      <td>43.90</td>
      <td>69.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>283056.0</td>
      <td>442241.0</td>
      <td>255379.0</td>
      <td>0.72</td>
      <td>2.0</td>
      <td>51.29</td>
      <td>62308887.0</td>
      <td>4.06</td>
      <td>45.46</td>
      <td>69.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>283056.0</td>
      <td>442241.0</td>
      <td>255379.0</td>
      <td>0.29</td>
      <td>2.0</td>
      <td>52.51</td>
      <td>62308887.0</td>
      <td>4.24</td>
      <td>45.48</td>
      <td>69.8</td>
    </tr>
  </tbody>
</table>
</div>

