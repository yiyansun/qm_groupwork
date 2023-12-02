import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial

import statsmodels.api as sms

# 数据读入部分
data = pd.read_csv(r'C:\Users\15827\git_demo\qm_groupwork\data\datasource.csv')

column_1 = 'percent_green'
column_2 = 'meanpercent_homes_with_good_access'

column_3 = 'patient_ratio'

x = data[column_2].values.reshape(-1, 1)
y = data[column_3].values

''' 
# easier version

# 绘制散点图
plt.scatter(x, y, label='Scatter Plot')

# 计算相关性
correlation = np.corrcoef(x.flatten(), y)[0, 1]
print(f'Correlation: {correlation}')

# 使用线性回归拟合
model = LinearRegression()
model.fit(x, y)
fit_line_y = model.predict(x)

# 绘制拟合线
plt.plot(x, fit_line_y, color='red', label='Fit Line')

plt.xlabel(column_2)
plt.ylabel(column_3)
plt.title(f'Corr = {correlation}')
plt.legend()
plt.show()
'''

# Huanfa version

# Use the next line to set figure height and width (experiment to check the scale):
figure_width, figure_height = 7,7

# These lines extract the y-values and the x-values from the data:
x_values = x
y_values = y

# These lines perform the regression procedure:
X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
# and print a summary of the results:
print(regression_model_b.summary())
print() # blank line

# Now we store all the relevant values:
gradient = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared = regression_model_b.rsquared
MSE = regression_model_b.mse_resid
pvalue = regression_model_b.f_pvalue

# And print them:
print("gradient  = ", regression_model_b.params[1])
print("intercept = ", regression_model_b.params[0])
print("Rsquared  = ", regression_model_b.rsquared)
print("MSE       = ", regression_model_b.mse_resid)
print("pvalue    = ", regression_model_b.f_pvalue)

# This line creates the endpoints of the best-fit line:
x_lobf = [min(x_values),max(x_values)]
y_lobf = [x_lobf[0]*gradient + intercept,x_lobf[1]*gradient + intercept]

# This line creates the figure.
plt.figure(figsize=(figure_width,figure_height))

# The next lines create and save the plot:
plt.plot(x_values,y_values,'b.',x_lobf,y_lobf,'r--')