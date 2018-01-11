# Written by Zafer Cavdar
# References: COMP 421 Textbook Introduction to Machine Learning, Ethem Alpaydin

library(MASS)
library(mixtools)
set.seed(521)
# mean parameters
class_means <- matrix(c(+2.5, +2.5,
                        -2.5, +2.5,
                        -2.5, -2.5,
                        +2.5, -2.5,
                         0.0,  0.0), 2, 5)
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +1.6,  0.0,  0.0, +1.6), c(2, 2, 5))
class_sizes <- c(50,50,50,50,100)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[4], mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[5], mu = class_means[,5], Sigma = class_covariances[,,5])
X <- rbind(points1, points2, points3, points4, points5)

H <- ncol(X)
N <- sum(class_sizes)
# plot data points generated
plot(X[,1], X[,2], type = "p", pch = 19, col = "black", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")

# 2- run k-means clustering for 2 times
K <- 5
centroids <- X[sample(1:N, K),]
D <- as.matrix(dist(rbind(centroids, X), method = "euclidean"))
D <- D[1:nrow(centroids), (nrow(centroids) + 1):(nrow(centroids) + nrow(X))]
assignments <<- sapply(1:ncol(D), function(i) {which.min(D[,i])})
for (k in 1:K) {
  centroids[k,] <- colMeans(X[assignments == k,])
} 
D <- as.matrix(dist(rbind(centroids, X), method = "euclidean"))
D <- D[1:nrow(centroids), (nrow(centroids) + 1):(nrow(centroids) + nrow(X))]
assignments <<- sapply(1:ncol(D), function(i) {which.min(D[,i])})
for (k in 1:K) {
  centroids[k,] <- colMeans(X[assignments == k,])
}
points(centroids, col="red", pch=19, type = "p")

# 3- calculate initial covariances and prior probs
prior <- sapply(1:K, function (c){
  sum(assignments == c)
}) / N
covariances <- t(sapply(1:K, function (c) {
  cov(X[assignments == c,])
}))
means <- centroids

get_covariances <- function (c) {
  return(matrix(covariances[c,],2,2))
}

for (it in 1:100) {
  Gic <- function (i, c) {
    prior[c] * ((det(get_covariances(c)))^(-0.5)) *
      exp((-0.5)* matrix((X[i,]- means[c,]), ncol=2) %*% t((X[i,]- means[c,]) %*% (solve(get_covariances(c)))))
  }
  
  gic <- matrix(nrow=N, ncol=5)
  for (i in 1:N) {
    for (c in 1:K) {
      gic[i,c] <- Gic(i,c)
    }
  }
  
  gicsum <- rep(0, N)
  for (i in 1:N) {
    gicsum[i] <-sum(sapply(1:K, function (c) {
      gic[i,c]
    }))
  }
  
  hic <- function (i,c) {
    gic[i,c] / gicsum[i]
  }
  
  # update priors
  for (c in 1:K) {
    sum <- 0
    for (i in 1:N) {
      sum <- sum + hic(i,c)
    }
    prior[c] <- sum / length(X)
  }
  
  # update means
  for (c in 1:K) {
    f_sum <- c(0,0)
    h_sum <- 0
    for (i in 1:N) {
      h <- hic(i,c)
      f_sum <- f_sum + h * X[i,]
      h_sum <- h_sum + h
    }
    means[c, ] <- f_sum / h_sum
  }
  
  # update covs
  for (c in 1:K) {
    h_sum <- 0
    s_sum <- matrix(c(0,0,0,0),2,2)
    for (i in 1:N) {
      h <- hic(i,c)
      s_sum <- s_sum + (X[i, ] - means[c,]) %*% t((X[i, ] - means[c,])) * h
      h_sum <- h_sum + h
    }
    covariances[c, ] <- s_sum / h_sum
  }
}

D <- as.matrix(dist(rbind(means, X), method = "euclidean"))
D <- D[1:nrow(means), (nrow(means) + 1):(nrow(means) + nrow(X))]
assignments <<- sapply(1:ncol(D), function(i) {which.min(D[,i])})

# plot data points generated
plot(X[assignments == 1, 1], X[assignments == 1, 2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(X[assignments == 2, 1], X[assignments == 2, 2], type = "p", pch = 19, col = "orange")
points(X[assignments == 3, 1], X[assignments == 3, 2], type = "p", pch = 19, col = "blue")
points(X[assignments == 4, 1], X[assignments == 4, 2], type = "p", pch = 19, col = "green")
points(X[assignments == 5, 1], X[assignments == 5, 2], type = "p", pch = 19, col = "purple")

for(c in 1:K){
  ellipse(class_means[,c], class_covariances[,,c], alpha = .05, npoints = class_sizes[c], newplot = FALSE, draw = TRUE, lty=2, lwd=2)
  ellipse(means[c,], matrix(covariances[c,], 2,2), alpha = .05, npoints = class_sizes[c], newplot = FALSE, draw = TRUE, lwd=2)
}
