require(fpp)
data(dj)
plot(dj)
acf(dj)
Box.test(dj, lag=10, fitdf=0, type="Lj")

diff_count <- ndiffs(dj)

diff_vec <- diff(dj, lag = frequency(dj), differences = diff_count)
plot(diff_vec)
acf(diff_vec)
Box.test(diff_vec, lag=10, fitdf=0, type="Lj")
diff_count2 <- ndiffs(diff_vec)
#From the results, Paul concluded that differencing the data to stabilize
#mean results in a stationary time-series. This is evident by the acf plot and Box test.