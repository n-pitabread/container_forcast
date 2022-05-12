```python
import numpy as np
import pandas as pd
```

# Loading label for model


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



# Load export vaule feature


```python
exportval_df = pd.read_csv("export_value.csv")
```


```python
exportval_df = exportval_df.drop('id', axis=1)

exportval_df = exportval_df[["year", "month", "export_value"]]
exportval_df['export_value'] = exportval_df['export_value'].map(lambda x: x.replace(',', ''))
exportval_df['export_value'] = pd.to_numeric(exportval_df['export_value'])
exportval_df.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>279973</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>279973</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>279973</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>283056</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>283056</td>
    </tr>
  </tbody>
</table>
</div>



# Load GDP constant feature


```python
gdp_df = pd.read_csv("GDP_constant.csv")
```


```python
gdp_df = gdp_df.drop('id', axis=1)

gdp_df = gdp_df[["year", "month", "GDP_constant"]]
gdp_df['GDP_constant'] = gdp_df['GDP_constant'].map(lambda x: x.replace(',', ''))
gdp_df['GDP_constant'] = pd.to_numeric(gdp_df['GDP_constant'])
gdp_df.head()
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
      <th>GDP_constant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>459359</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>459359</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>459359</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>442241</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>442241</td>
    </tr>
  </tbody>
</table>
</div>



# Load Import value feature


```python
importval_df = pd.read_csv("import_value.csv")
```


```python
importval_df = importval_df.drop('id', axis=1)

importval_df = importval_df[["year", "month", "import_value"]]
importval_df['import_value'] = importval_df['import_value'].map(lambda x: x.replace(',', ''))
importval_df['import_value'] = pd.to_numeric(importval_df['import_value'])
importval_df.head()
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
      <th>import_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>255061</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>255061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>255061</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>255379</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>255379</td>
    </tr>
  </tbody>
</table>
</div>



# Load Inflation percentage feature


```python
inflate_df = pd.read_csv("inflation_%.csv")
```


```python
inflate_df = inflate_df.drop('id', axis=1)

inflate_df = inflate_df[["year", "month", "inflation_percentage_change"]]
inflate_df['inflation_percentage_change'] = inflate_df['inflation_percentage_change'].map(lambda x: x.replace('%', ''))
inflate_df['inflation_percentage_change'] = pd.to_numeric(inflate_df['inflation_percentage_change'])
inflate_df.head()
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
      <th>inflation_percentage_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>0.29</td>
    </tr>
  </tbody>
</table>
</div>



# Load Interest rate feature


```python
interest_df = pd.read_csv("interest_rate.csv")
```


```python
interest_df = interest_df.drop('id', axis=1)

interest_df = interest_df[["year", "month", "interest_rate"]]
interest_df['interest_rate'] = pd.to_numeric(interest_df['interest_rate'])
interest_df.head()
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
      <th>interest_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



# Load Manufacture Production feature


```python
manu_df = pd.read_csv("manufac_prod_index.csv")
```


```python
manu_df = manu_df.drop('id', axis=1)

manu_df = manu_df[["year", "month", "manufac_prod_index"]]
manu_df['manufac_prod_index'] = pd.to_numeric(manu_df['manufac_prod_index'])
manu_df.head()
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
      <th>manufac_prod_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>52.47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>53.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>52.18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>51.29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>52.51</td>
    </tr>
  </tbody>
</table>
</div>



# Load Population feature


```python
pop_df = pd.read_csv("population.csv")
```


```python
pop_df = pop_df.drop('id', axis=1)

pop_df = pop_df[["year", "month", "population"]]
pop_df['population'] = pop_df['population'].map(lambda x: x.replace(',', ''))
pop_df['population'] = pd.to_numeric(pop_df['population'])
pop_df.head()
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
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>62308887</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>62308887</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>62308887</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>62308887</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>62308887</td>
    </tr>
  </tbody>
</table>
</div>



# Load Unemployment feature


```python
unemp_df = pd.read_csv("unemployment.csv")
```


```python
unemp_df = unemp_df.drop('id', axis=1)

unemp_df = unemp_df[["year", "month", "unemployment_rate"]]
unemp_df = unemp_df.replace(' n.a. ', np.nan)
unemp_df['unemployment_rate'] = pd.to_numeric(unemp_df['unemployment_rate'])
unemp_df.head()
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
      <th>unemployment_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>5.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>4.04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>4.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>4.24</td>
    </tr>
  </tbody>
</table>
</div>



# Load Exchange rate feature


```python
ex_df = pd.read_csv("usd_thb.csv")
```


```python
ex_df = ex_df.drop('id', axis=1)

ex_df = ex_df[["year", "month", "exchange_rate"]]
ex_df = ex_df.replace(' n.a. ', np.nan)
ex_df['exchange_rate'] = pd.to_numeric(ex_df['exchange_rate'])
ex_df.head()
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
      <th>exchange_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>43.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>42.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>43.90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>45.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>45.48</td>
    </tr>
  </tbody>
</table>
</div>



# Load Consumer price feature


```python
cons_df = pd.read_csv("consumer_price_index.csv")
```


```python
cons_df = cons_df.drop('id', axis=1)

cons_df = cons_df[["year", "month", "consumer_price_index"]]
cons_df = cons_df.replace(' n.a. ', np.nan)
cons_df['consumer_price_index'] = pd.to_numeric(cons_df['consumer_price_index'])
cons_df.head()
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
      <th>consumer_price_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>January</td>
      <td>68.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>February</td>
      <td>69.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>March</td>
      <td>69.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>April</td>
      <td>69.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>May</td>
      <td>69.8</td>
    </tr>
  </tbody>
</table>
</div>



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




```python

```
