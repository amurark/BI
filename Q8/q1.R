require(fpp)
data(hsales)
plot(hsales, xlab = "Time", 
     ylab = "Sales", 
     main = "Monthly house sales in US (Jan/1973-Nov/1995)")
train = window(hsales, end = c(1993,12))
test = window(hsales, start = c(1994,1))

#Average
averageForecast = meanf(train, h = 23)$mean
accuracy(averageForecast, test)

#Seasonal Naive
sNaiveForecast = snaive(train, h = 23)$mean
accuracy(sNaiveForecast, test)

#Drift
driftForecast = rwf(train, drift = TRUE, h = 23)$mean
accuracy(driftForecast, test)
