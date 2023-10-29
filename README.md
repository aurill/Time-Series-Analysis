# ***Time-Series-Analysis***
This GitHub repository contains the code and documentation for a comprehensive time series analysis of the Craigslist Vehicles dataset. The project explores temporal patterns, identifies seasonal trends, and analyzes demand-supply dynamics in the used vehicle market.

## **Week 4 Project Description**:

This repository contains a time series model focused on examining the temporal behavior of median vehicle prices over time, identifying seasonal patterns, and visualizing rolling statistics in the Craigslist Vechine Dataset.

## **Overview**

Time series analysis is a branch of statistics and data analysis that focuses on studying and modeling data points collected or recorded at equally spaced time intervals. It is particularly useful for understanding and making predictions about data that exhibits temporal dependencies or patterns. It is used for non-stationary data—things that are constantly fluctuating over time or are affected by time. Industries like finance, retail, and economics frequently use time series analysis because currency and sales are always changing. There are different types of data that describe how and when that time data was recorded. For example: Time series data is data that is recorded over consistent intervals of time, Cross-sectional data consists of several variables recorded at the same time, and Pooled data is a combination of both time series data and cross-sectional data. 

## **Step 1 - Data Preparation**

The necessary Python libraries and packages were installed and the dataset was uploaded. The Kaggle API was also installed to download the dataset easily. 
This involved unzipping the large file

```python
# Installing the Kaggle packages and other Python libraries. 
!pip install -q kaggle

from google.colab import files
files.upload()

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
```

```python
# Creating a directory named .kaggle in the home directory 
!mkdir -p ~/.kaggle

# Securely storing the Kaggle API credentials.
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Dowloading the Craigslist vehicle dataset
!kaggle datasets download -d mbaabuharun/craigslist-vehicles

# Unzipping the dataset.
!unzip craigslist-vehicles.zip
```

## **Step 2 - Data Loading and Initial Exploration**

This step involved loading the dataset into a Pandas data frame, visualizing the first 5 rows of the dataset, and obtaining the summary statistics about the data set. 

```python

# Loading the data set in a pandas data frame called df. 
df = pd.read_csv('craigslist_vehicles.csv')

# Visualizing the first 5 rows of the dataset
df.head()

# Obtaining the summary statistics about the data set
df.describe()

```

## **Step 3 - Data Cleaning**

This step was done to ensure that the model was prepared for Time Series Analysis. This involved dropping null columns like 'country', and handling missing values in the dataset. 

For the numerical columns ('year', 'odometer', 'lat', 'long'), the missing values in the columns were imputed with the median. 
For the categorial columns, the missing values in the columns were imputed with the mode (_Most frequent value_). Additionally, the year column was converted from a float to an integer.

```python

# Dropping null Column named county.
df.drop('county', axis=1, inplace=True)

# Handling missing values by imputing with the median for the numerical columns
numerical_cols = ['year', 'odometer', 'lat', 'long']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Handling missing values by imputing with mode for the categorial columns
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'VIN',
                    'drive', 'size', 'type', 'paint_color', 'image_url', 'description','posting_date', 'removal_date']
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0], axis=0)

# Converting year from float type to integer type
df['year'] = df['year'].astype(int)

```

## **Step 4 - Time Series Data Preparation**

This step involved converting the 'posting date' and 'removal date' columns to datetime objects. The following code below was used to achieve this. 

```python

# Converting posting date and removal date to datetime objects.
df['posting_date'] = pd.to_datetime(df['posting_date'], utc = True)
df['removal_date'] = pd.to_datetime(df['removal_date'], utc = True)

```

## **Step 5 - Time Series Analysis**

In this step, the data was grouped by 'posting date' and calculating the median price for each date. This created a time series of median prices. Also, a line plot was used to visualize the median price trends over time. 
There was a gradual decrease in the median price trends over the month specified. This could mean a variety of things, spanning from reasons of seasonal variation to supply and demand dynamics or economic factors. 

The code below shows how we achieved this.

```python

# Grouping by 'posting_date' and calculating the median price for each date
price_trend = df.groupby(pd.Grouper(key='posting_date', freq='D'))['price'].median()

# Creating a line plot to visualize the median price trends over time
plt.figure(figsize=(12, 6))
plt.plot(price_trend.index, price_trend.values, color='royalblue', linewidth=2)
plt.title('Median Price Trends Over Time')
plt.xlabel('Posting Date')
plt.ylabel('Median Price')
plt.grid(True)

plt.show()

```

## **Step 6 - Time Series Decomposition**

The time series was decomposed into its components: Trend, Seasonal, and residual using the Seasonal- Trend decomposition using the LOSS(STL) method. These decompositions were plotted to visualize the trends and seasonal patterns in the data. 
There was a decrease in the price over time in the dataset. This is consistent with the reasons outlined in the gradual decrease in the median price. Whereas it speaks to the trend decomposition, we noticed a decrease. This specifies that there was a downward movement in the data over time which could mean that there might be a decline in the variable, reduced demand, or other aspects affecting the market for vehicles. For seasonal decomposition, there were spikes and falls in the data. This could mean that seasonality was less stable in the past but as time progressed, it had gotten more predictable and more consistent. This is the reason for the smoother patterns that we noticed. As for residuals, we noticed that the change follows closely with the changes observed in the seasonal decomposition. This suggests that the initial time series model may not have adequately captured the underlying patterns and structure in the data. This can be due to model simplicity, inadequately accounting for seasonality or other factors as there is not a constant trend that centers around the 0 mark. 

```python
# Decomposing the time series into trend, seasonal, and residual components
stl = STL(price_trend, seasonal=13)  # Seasonal period (e.g., 13 for weekly data)
result = stl.fit()

# Plotting the decomposed components
result.plot()
plt.show()

```

## **Step 7- Rolling Median Analysis**

A 7-day rolling median of the median prices was also calculated. A plot was used to visualize the original median price and the 7-day rolling median. 
This helps in smoothing out short-term fluctuations by highlighting longer-term trends. Additionally, a plot was created to visualize the original median price. 
The long-term trend showed that the 7-day median observed a less volatile but gradual drop in the median prices.

The code used to achieve this is shown below: 

```python

# Calculating the rolling median and rolling median absolute deviation
rolling_median = price_trend.rolling(window=7).median()  # 7-day rolling median

# Plotting the rolling median
plt.figure(figsize=(12, 6))
plt.plot(price_trend.index, price_trend.values, color='royalblue', label='Median Price')
plt.plot(rolling_median.index, rolling_median.values, color='red', label='7-Day Rolling Median')
plt.title('Median Price Trends and Rolling Median')
plt.xlabel('Posting Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


```

**Conclusion**

In this comprehensive time series analysis, we examined the temporal behavior of median vehicle prices in the Craigslist Vehicles dataset. This analysis aimed to identify temporal patterns, seasonal trends, and fluctuations in the used vehicle market, ultimately providing valuable insights into the dynamics of this market.

The analysis consisted of several key steps:

Data Preparation: We began by preparing the data, installing the necessary Python libraries, and uploading the dataset. After downloading and unzipping the data, we were ready to proceed.

Data Loading and Initial Exploration: We loaded the dataset into a Pandas DataFrame and conducted an initial exploration. This included visualizing the first five rows of the dataset and obtaining summary statistics to gain an overview of the data.

Data Cleaning: To ensure the dataset was ready for time series analysis, we performed data cleaning. This involved dropping irrelevant columns and handling missing values by imputing them with appropriate measures. We also converted the 'year' column to an integer type for consistency.

Time Series Data Preparation: Crucially, we converted the 'posting_date' and 'removal_date' columns to datetime objects. This conversion enabled us to work with time series data effectively.

Time Series Analysis: The heart of the project involved analyzing the time series of median vehicle prices. We grouped the data by 'posting_date' and calculated the median price for each date. Visualizations of the median price trends over time revealed interesting insights into the market's dynamics.

Time Series Decomposition: We used the Seasonal-Trend decomposition using LOESS (STL) method to decompose the time series into its components: trend, seasonal, and residual. The decomposition helped us better understand the underlying patterns and fluctuations in the data.

Rolling Median Analysis: To further analyze the data, we calculated the 7-day rolling median of the median prices. This rolling median allowed us to identify longer-term trends by smoothing out short-term fluctuations. We visualized the original median prices alongside the rolling median to highlight these trends.

In summary, this analysis provides a comprehensive view of the Craigslist Vehicles dataset. It uncovers valuable insights into the temporal patterns of median vehicle prices, offering a deeper understanding of the used vehicle market's dynamics. The project serves as a foundation for future analyses and decision-making in the domain of automotive sales and market trends.

The insights gained from this analysis can be leveraged for a variety of applications, including market forecasting, pricing strategies, and informed decision-making within the used vehicle industry.

This project showcases the power of time series analysis in uncovering patterns within data that evolve over time. It also demonstrates the importance of data preprocessing, visualization, and statistical techniques in extracting meaningful insights from real-world datasets.






