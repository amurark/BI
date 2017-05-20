require(fpp)
data(books)
hc <- books[,'Hardcover']
pb <- books[,'Paperback']

#A
holt_hc <- holt(hc, h = 4)
holt_pb <- holt(pb, h = 4)

#B
ses_hc = ses(hc, initial = "simple", h = 4)
ses_pb = ses(pb, initial = "simple", h = 4)

sse_holt_hc <- sum((holt_hc$fitted - hc)^2)
sse_holt_pb <- sum((holt_pb$fitted - pb)^2)

sse_ses_hc <- sum((ses_hc$fitted - hc)^2)
sse_ses_pb <- sum((ses_pb$fitted - pb)^2)


#C
fc_holt_hc <- forecast(holt_hc)
fc_holt_pb <- forecast(holt_pb)
plot(fc_holt_hc)
plot(fc_holt_pb)

fc_ses_hc <- forecast(ses_hc)
fc_ses_pb <- forecast(ses_pb)
plot(fc_ses_hc)
plot(fc_ses_pb)
# Looking at the SSE(sum of squared error) and the plot we can confidently
# say that Holt's linear model is better at forecasting than exponential 
# smoothing with SES.