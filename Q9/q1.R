plot(elec)

# Stablizing variance
l = BoxCox.lambda(elec)
plot(BoxCox(elec, l))

# Stablize the mean
ndiffs(BoxCox(elec, l))
fit <- BoxCox(elec, l)
ns <- nsdiffs(fit)

data<- diff(fit, lag=frequency(fit), differences = ns)



# Check residuals
res <- residuals(auto.arima(data))
Acf(res)
Box.test(res, lag=10)
