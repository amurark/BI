sample_size <- 20
mean_likelihood <- 6
sd_likelihood <- 1.5
mean_prior <- 4
sd_prior <- 0.8
weight1 <- (sample_size*(sd_prior^2))/(sample_size*(sd_prior^2)+sd_likelihood^2)
weight2 <- (sd_likelihood^2)/(sample_size * (sd_prior^2) * (sd_likelihood^2))
mean_posterior <- weight1*mean_likelihood + weight2*mean_prior

x1 <- seq(6-10, 6+10, 0.01)
x2 <- seq(4-10, 4+10, 0.01)
x3 <- seq(mean_posterior-10, mean_posterior + 10, 0.01)

plot(x3, dnorm(x3, mean_posterior, sd = 0.31), col = "magenta", type="l",xlab = "X", ylab="Y")
lines(x2, dnorm(x2, mean_prior, sd_prior), col= "red")
lines(x1, dnorm(x1, mean_likelihood, sd_likelihood), col= "blue")
