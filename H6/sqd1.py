#!/usr/bin/python=
import sys
from csv import reader
from random import seed, randrange, sample
from math import sqrt
from sklearn import datasets, linear_model
import numpy as np

# Get the input parameters in this order: python <filename> <data.csv> learning_rate epochs
def getInputValues():
  if len(sys.argv) < 4:
    sys.exit("Insufficient parameters");
  filename = sys.argv[1]
  if ".csv" not in filename:
    sys.exit("Training Data file not a CSV file")
  learning_rate = sys.argv[2]
  # Learning rate should be between 0 and 0.1
  if float(learning_rate) == 0.0 or float(learning_rate) > 0.1:
    sys.exit("Learning rate value not in range (0.01, 0.1)")
  epochs = sys.argv[3]
  # Epoch should be between 1 and 100
  if int(epochs) > 100 or int(epochs) == 0:
    sys.exit("Epoch not in range 1-100")
  return filename, float(learning_rate), int(epochs)

# Read the CSV File and remove the header from the CSV file.
# Also, convert each string entry to a float entry.
def loadFile(filename):
  input_data = []
  with open(filename, 'r') as f:
    readStream = reader(f)
    for index, row in enumerate(readStream):
      if index != 0:
        input_data.append(map(float,row));
  return input_data

# Get the minimum and maximum value for each feature and then normalize using the min and max value
def normalizeData(input_data):
  min_max_vector = []
  for i in range(0, len(input_data[0])):
    feature_values = [x[i] for x in input_data]
    minimum = min(feature_values)
    maximum = max(feature_values)
    min_max_vector.append((minimum, maximum))

  for row in input_data:
      for i in range(0, len(row)):
          minimum = min_max_vector[i][0]
          maximum = min_max_vector[i][1]
          row[i] = (row[i] - minimum) / (maximum - minimum)

# Split the input data into training and testing based on 60-40 ratio.
def splitData(input_data):
    l = len(input_data)
    train_data = []
    test_data = []
    train_size = int(round(l * 60 / 100))
    indices = sample(range(l),train_size)
    for index, row in enumerate(input_data):
      if index in indices:
          train_data.append(row)
      else:
          test_data.append(row)
    return train_data, test_data

# Predict the value of Y when we have the X Values and coefficients.
def predict(row, coefficients):
  # Initialize predicted_y with B0
  predicted_y = coefficients[0]
  # Modify predicted_y = B0 + B1*X1 + B2*X2 + ... + BnXn
  for i in range(len(row)-1):
    predicted_y += coefficients[i + 1] * row[i]
  return predicted_y

# Get the coefficients using training_data.
def getCoefficients(train_data, learning_rate, epochs):
	coefficients = [0.0]*len(train_data)
    # Keep improving for epochs number of times.
	for epoch in range(epochs):
        # For each row in train_data
		for row in train_data:
            # Get the predicted output
			predicted_y = predict(row, coefficients)
            # Error is the difference between predicted and actual output.
			error = predicted_y - row[-1]
            # The intercept coefficient which is also called 'B0' is improved using the formnulae:
            # B0(t+1) = B0(t) - learning_rate * error(t)
			coefficients[0] = coefficients[0] - learning_rate * error
            # For the rest of the coefficients corresponding to each input value Xn,
            # Bn(t+1) = Bn(t) - learning_rate * error(t) * Xn(t)
			for i in range(len(row)-1):
				coefficients[i + 1] = coefficients[i + 1] - learning_rate * error * row[i]
	return coefficients

# Get the predicted_output using Stochastic Gradient Descent for Linear Regression.
def linear_regression_sgd(coefficients, test_data):
	predicted_output = []
    # Get the predicted output for each row in test-data using coefficients and input values.
	for row in test_data:
		predicted_y = predict(row, coefficients)
		predicted_output.append(predicted_y)
	return predicted_output

# Get root-mean-squared-error when we have the predicted and actual Y values.
def get_root_mean_square_error(predicted, actual):
    sum_error = 0.0;
    for i in range(0, len(predicted)):
        error = predicted[i] - actual[i]
        squared_error = error**2
        sum_error += squared_error
    mean_error = sum_error / float(len(predicted))
    root_mean_square_error = sqrt(mean_error)
    return root_mean_square_error


if __name__ == "__main__":
    filename, learning_rate, epochs = getInputValues()
    input_data = loadFile(filename)
    normalizeData(input_data)

    train_data, test_data = splitData(input_data)
    coefficients = getCoefficients(train_data, learning_rate, epochs)
    predicted_output = linear_regression_sgd(coefficients, test_data)
    actual_output = [row[-1] for row in test_data]
    error = get_root_mean_square_error(predicted_output, actual_output)
    print('Mean RMSE using SGD Linear Regression Model: %.3f' % error)

    # Create training Data
    train_data_X = []
    train_data_Y = []
    for row in train_data:
        train_data_X.append(row[:-1])
        train_data_Y.append(row[-1])

    # Create testing data
    test_data_X = []
    test_data_Y = []
    for row in test_data:
        test_data_X.append(row[:-1])
        test_data_Y.append(row[-1])

    # Create linear regression object
    regr = linear_model.LinearRegression()


    # Train the model using the training sets
    regr.fit(np.asarray(train_data_X), np.asarray(train_data_Y))
    print("Mean RMSE using Scikit Linear Regression: %.3f"
      % sqrt(np.mean((regr.predict(np.asarray(test_data_X)) - np.asarray(test_data_Y)) ** 2)))
