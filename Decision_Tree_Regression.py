# Decision Tree Regression tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 23SEP20

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)

# Train the Decision Tree Regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict new result
print("The predicted salary is $" + str(int(regressor.predict([[6.5]]))))


# Visualize the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
ax = plt.gca()
ax.set_facecolor('grey')
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Design Tree Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $US")
plt.show()
