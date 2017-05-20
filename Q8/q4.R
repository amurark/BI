require(fpp) 
plot(dole, xlab = "Time", ylab = "No. Unemployed",
     main = "Monthly Unemployed Benefit Usage, Australia (01/56-07/92)")
lambda = BoxCox.lambda(dole)
plot(BoxCox(dole, lambda), xlab = "Time", 
     ylab = paste("BoxCox(# people,", round(lambda, 2), ")"))

plot(usdeaths, xlab = "Time", ylab = "No. of Deaths",
     main = "Monthly Deaths in United States")
lambda = BoxCox.lambda(usdeaths)
plot(BoxCox(usdeaths, lambda), xlab = "Time", 
     ylab = paste("BoxCox(# people,", round(lambda, 2), ")"))

plot(bricksq, xlab = "Time", ylab = "No. of Bricks",
     main = "Quarterly production of bricks (in millions of units) at Portland, Australia")
lambda = BoxCox.lambda(bricksq)
plot(BoxCox(bricksq, lambda), xlab = "Time", 
     ylab = paste("BoxCox(# Bricks,", round(lambda, 2), ")"))

