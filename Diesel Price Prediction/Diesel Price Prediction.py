########################################
# 
# This script imports previous diesel price data and extrapolates ten future 
# years of diesel prices.
#
# It takes input of one .csv file in the form:
# _________________________
# |Column 1   |   Column 2|
# |Year       |   Price   |
#
# 'Year' must be numbered from -n to -1 where n is the number of years of historic 
# price data used in the prediction. 
# 'Price' must be numeric - int or float is recomended.
#
# It out puts a .csv named 'Diesel Price Prediction.csv' which contains predictions
# for diesel prices over the next several years. By default it predicts 10 years
# this can be alterded by changing the value of 'duration' in line 33.
#
##########################################


import numpy as np
import pandas as pd


#Set name of target CSV and read to dataframe
target_csv = "Diesel Price Data.csv"
df = pd.read_csv(target_csv)

#Find line of best fit
m, b = np.polyfit(df['Year'], df['Price'], 1)

#Extrapolate future price data based on line of best fit
duration = 10
price = [[x,m*x+b] for x in range(duration)]

#Convert extrapolated price form list to dataframe and export to .csv file
prediction_df = pd.DataFrame(price, columns=['Year', 'Price'])
prediction_df.to_csv('Diesel Price Prediction.csv', index=False)

#This section is not required to produce the price prediction, but can be used as a
# 'sanity check' by graphing the historic price data and line of best fit.

'''
import matplotlib.pyplot as plt

#Prints eqation for line of best fit
print(f"Equation of the line: Y = {m:.2f}X + {b:.2f}")

#Produces a scatter plot of historic data and line of best fit
plt.scatter(df['Year'], df['Price'], color='blue', label='Data Points')
plt.plot(df['Year'], m * df['Year'] + b, color='red', label='Line of Best Fit')

#Displays the above plot
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.show()
'''