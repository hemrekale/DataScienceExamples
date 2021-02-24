N = 10000
mu = 0

myrnorm <- function(n = 1)
{
  rnorm(n,mean = 0)
}

xt <- array(0, N)
error <- myrnorm(n = N)
xt[1] <- mu +  myrnorm()
error[1] <- myrnorm()
for(i in seq(2,N))
{
  xt[i] <-  mu - 1.01*xt[i - 1] + error[i] 
}

plot(seq(1,N), xt)


#lines(seq(1,1000), xt)
