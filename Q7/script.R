#require(fpp)
#data(package = "fma")
#data("hsales")
plot(diff(dj,1)) #Gives a graph where each sample at time, t is subtracted from the sample at time t-1.
fit <- stl(elecequip, s.window = 5)
plot(fit, main = "Electrical equipment manufacturing")




require(fpp)
#1
plot(plastics)
#The data follows an upward trend with an almost linear slope.

#2
x <- decompose(plastics)
plot(x)
#Looking closely at the trend, we can say that yes, it almost follows a linear slope. 
#Though towards the end, it stops following the linear trend.

#3
#The trend seems to be non linear in the beginning when the time period approaches 2 and towards the end after 5.

#4
seasAdj <- seasadj(x)
plot(seasAdj, main = "Seasonally Adjusted Data")

#5
new_plastics = plastics
new_plastics[30] = new_plastics[30] + 500
x1 <- decompose(new_plastics, type="multiplicative")
plot(x1)
seasAdj1 <- seasadj(x1)
plot(seasAdj1)
#The outlier changes the season and trend slightly. 
#The trend deviates a little more from the linear curve that it was following
#Also, there is a slight change in seasonality.

#6
new_plastics2 = plastics
new_plastics2[60] = new_plastics2[60] + 500
x11 <- decompose(new_plastics2, type="multiplicative")
plot(x11)
seasAdj11 <- seasadj(x11)
plot(seasAdj11)
#The outlier changes the season and trend slightly. But in this case the effect is lesser than an outlier in the middle 

#7
#First The following code snippet loads the plastics dataset
#Then it decomposes the dataset to see different properties of the data like trend, seasonal and random. 
#After this, seasadj is applied to remove seasonality from data
#Next rwf command is used to return forecasts and prediction intervals for a random walk with drift model applied to 24 values.
#Then this drift model is plotted with the y-axis limited between 500 and 2200
#Additional plots are superimposed in different colors showing the mean, upper and lower components of the drift model and these are compared with the original model






require(fpp)
data(books)
plot(books, main = "Data set books")
alpha = seq(0.01, 0.99, 0.01)
SSE = NA
for(i in seq_along(alpha)) {
  fcast = ses(books[,"Hardcover"], alpha = alpha[i], initial = "simple")
  SSE[i] = sum((books[,"Hardcover"] - fcast$fitted)^2)
}
plot(alpha, SSE, type = "l")
fcastPaperSimple = ses(books[,"Hardcover"], 
                       initial = "simple", 
                       h = 4)
fcastPaperSimple$model$par[1]
plot(fcastPaperSimple)

fcastPaperOpt = ses(books[,"Hardcover"], 
                    initial = "optimal", 
                    h = 4)
fcastPaperOpt$model$par[1]
plot(fcastPaperOpt)
as.numeric((fcastPaperOpt$mean - 
              fcastPaperSimple$mean)/fcastPaperSimple$mean) * 100
#In this case, the analyzed plots shows similar results for hardcover when compared to paperback. After exploring the data for different values of alpha, we realized that alpha ~ 0.3 gives the best accuracy.