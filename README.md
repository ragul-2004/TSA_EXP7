<H1 ALIGN =CENTER> Ex.No: 07 -- AUTO REGRESSIVE MODEL... </H1> 

### Date: 

### AIM :

To Implement an Auto Regressive Model using Python.

### ALGORITHM :

### Step 1 :

Import necessary libraries.

### Step 2 :

Read the CSV file into a DataFrame.

### Step 3 :

Perform Augmented Dickey-Fuller test.

### Step 4 :

Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags.

### Step 5 :

Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF).

### Step 6 :

Make predictions using the AR model.Compare the predictions with the test data.

### Step 7 :

Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM :

#### Import necessary libraries :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```

#### Read the CSV file into a DataFrame :

```python
data = pd.read_csv("/content/Temperature.csv")  
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
```

#### Perform Augmented Dickey-Fuller test :

```python
result = adfuller(data['temp']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

#### Split the data into training and testing sets :

```python
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]
```

#### Fit an AutoRegressive (AR) model with 13 lags :

```python
lag_order = 13
model = AutoReg(train_data['temp'], lags=lag_order)
model_fit = model.fit()
```

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

```python
plot_acf(data['temp'])
plt.title('Autocorrelation Function (ACF)')
plt.show()
plot_pacf(data['temp'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

#### Make predictions using the AR model :

```python
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
```

#### Compare the predictions with the test data :

```python
mse = mean_squared_error(test_data['temp'], predictions)
print('Mean Squared Error (MSE):', mse)
```

#### Plot the test data and predictions :

```python
plt.plot(test_data.index, test_data['temp'], label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.show()
```

### OUTPUT :

#### GIVEN DATA :

![img1](https://github.com/anto-richard/TSA_EXP7/assets/93427534/9e7920fd-da3a-42fd-96b7-64b76755d14e)

#### Augmented Dickey-Fuller test :

![img2](https://github.com/anto-richard/TSA_EXP7/assets/93427534/b9e5e07f-d75e-4a64-b401-9a5c6a236405)

#### PACF - ACF :

![img3](https://github.com/anto-richard/TSA_EXP7/assets/93427534/ece00793-67b8-4d96-be8d-7dc468d3a490)

![img4](https://github.com/anto-richard/TSA_EXP7/assets/93427534/fa4ced00-d07b-47f0-a21e-e6d0eee08d00)

#### Mean Squared Error :

![img5](https://github.com/anto-richard/TSA_EXP7/assets/93427534/3675a8fb-e804-4523-a9ae-9c623a11a33a)

#### PREDICTION :

![img6](https://github.com/anto-richard/TSA_EXP7/assets/93427534/6b3c0412-ce98-480e-b8a6-6a184923a725)

### RESULT :

Thus, we have successfully implemented the auto regression function using python.

