#Discussed with Balaji R.

#1
plot(elecequip)
#Looking at the plot, there is nothing unusual about it.

#2
plot(stl(elecequip, s.window="periodic"))

#3
sadj <- seasadj(stl(elecequip, s.window="periodic"))
plot(sadj)

#4
#Variance stabilization not required

#5 - The data is not stationary as seen by the plot
Acf(sadj)

#6 - Yes the differencing made the data stationary
diff <- ndiffs(sadj)
data <- diff(sadj)
Acf(data)

#7 - p =3, d= 0, q = 1
fit <- auto.arima(data)
summary(fit)

#8 - Model 1 with 4,0,0 has the best AIC of 980.9 and the best AICc of 981.36 
m1 <- Arima(data, order=c(4,0,0))
m2 <- Arima(data, order=c(3,0,0))
m3 <- Arima(data, order=c(2,0,0))

#9 Yes the residuals behave like white-noise because the p-value is less than 0.5
Acf(residuals(m1))
Box.test(residuals(m1), lag=24, fitdf=4, type="Ljung")

#10
plot(forecast(m1))
