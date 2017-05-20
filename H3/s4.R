require(fpp)
data(ukcars)
plot(ukcars, ylab = "Production, thousands of cars")
stlFit <- stl(ukcars, s.window = "periodic")
plot(stlFit)
adjusted <- seasadj(stlFit)
plot(adjusted)

fcastHoltDamp = holt(adjusted, damped=TRUE, h = 8)
plot(ukcars, xlim = c(1977, 2008))
lines(fcastHoltDamp$mean + 
        stlFit$time.series[2:9,"seasonal"], 
      col = "red", lwd = 2)

dampHoltRMSE = sqrt(mean(((fcastHoltDamp$fitted + stlFit$time.series[,"seasonal"]) - ukcars)^2))
dampHoltRMSE

fcastHolt = holt(adjusted, h = 8)
plot(ukcars, xlim = c(1997, 2008))
lines(fcastHolt$mean + stlFit$time.series[2:9,"seasonal"], 
      col = "red", lwd = 2)

holtRMSE = sqrt(mean(((fcastHolt$fitted + stlFit$time.series[,"seasonal"]) - ukcars)^2))
holtRMSE


#A
damp_fit <- ets(adjusted, damped = TRUE)
fit <- ets(adjusted, damped = FALSE)
#The fit model seems a better one.

#B
rmse_fit <- sqrt(mean((fit$fitted - ukcars)^2))
rmse_fcastHolt <- sqrt(mean((fcastHolt$fitted - ukcars)^2))
#RMSE OF Sam's model is lesser than the fit model because of seasonal adjust and dampening

#C
fc1 <- forecast(fit, h = 8)
fc2 <- forecast(fcastHolt, h = 8)

plot(fc1)
plot(fc2)
#Looking at the plots the forecast for the forecast using seasonal adjust and dampped holt's method looks better