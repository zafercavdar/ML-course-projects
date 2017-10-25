require(MASS)
set.seed(521)
# mean parameters
class_means <- matrix(c(0, 1.5,
                        -2.5, -3,
                        2.5, -3), 2,3, byrow = FALSE)
class_means_1 <- c(0, 1.5)
class_means_2 <- c(-2.5, -3)
class_means_3 <- c(2.5, 3)
# standard deviation parameters
class_deviations <- array(c(1, 0.2, 0.2, 3.2,
                            1.6, -0.8, -0.8, 1.0,
                            1.6, 0.8, 0.8, 1.0), c(2,2,3)) 
# sample sizes
class_sizes <- c(100, 100, 100)

# generate random samples
points_1 = mvrnorm(class_sizes[1], mu = class_means[,1], Sigma = class_deviations[,,1])
points_2 = mvrnorm(class_sizes[2], mu = class_means[,2], Sigma = class_deviations[,,2])
points_3 = mvrnorm(class_sizes[3], mu = class_means[,3], Sigma = class_deviations[,,3])
X <- rbind(points_1, points_2, points_3)
x1 <- X[,1]
x2 <- X[,2]

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), 
       rep(2, class_sizes[2]),
       rep(3, class_sizes[3]))

# plot data points generated
plot(points_1[,1], points_1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points_2[,1], points_2[,2], type = "p", pch = 19, col = "green")
points(points_3[,1], points_3[,2], type = "p", pch = 19, col = "blue")


# get number of classes and number of samples
K <- max(y)
N <- length(y)

# calculate sample means
sample_means_x1 <- sapply(X = 1:K, FUN = function(c) mean(x1[y == c]))
sample_means_x2 <- sapply(X = 1:K, FUN = function(c) mean(x2[y == c]))
sample_means <- matrix(rbind(sample_means_x1, sample_means_x2), nrow=2)

# calculate covariance matrix / sigma
variance_x1 <- sapply(X = 1:K, FUN = function(c) var(x1[y==c]))
variance_x2 <- sapply(X = 1:K, FUN = function(c) var(x2[y==c]))
covariance_x1_x2 <- sapply(X = 1:K, FUN = function(c) cov(x1[y==c], x2[y==c]))
sample_covariances <- array(c(variance_x1[1], covariance_x1_x2[1], covariance_x1_x2[1], variance_x2[1],
                              variance_x1[2], covariance_x1_x2[2], covariance_x1_x2[2], variance_x2[2],
                              variance_x1[3], covariance_x1_x2[3], covariance_x1_x2[3], variance_x2[3]
                              ), c(2,2,3))

# calculate prior probabilities
class_priors <- sapply(X = 1:K, FUN = function(c) mean(y == c))

# evaluate score functions
score <- function(data_point) {
  sapply(X = 1:K, FUN = function(c) - 0.5 * log(abs(sample_covariances[c]))
                                    - 0.5 * (t(matrix((data_point - sample_means[,c]))) %*% solve(sample_covariances[,,c]) %*% matrix((data_point - sample_means[,c])))
                                    + log(class_priors[c]))
}
#score_matrix_grid <- matrix(sapply(X = 1:length(x1_grid), FUN = function(n) score(matrix(c(x1_grid[n], x2_grid[n])))), ncol=K, byrow=TRUE)
score_matrix <- matrix(sapply(X = 1:N, FUN = function(n) score(X[n,])), ncol=K, byrow=TRUE)

# calculate confusion matrix
y_predicted <- sapply(X = 1:N, FUN = function(n){
  scores <- score_matrix[n,]
  return(which.max(scores))
})

y_truth <- y
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate discriminat function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

predict_class <- function(x1, x2){
  scores <- score(c(x1,x2))
  return(which.max(scores))
}
predicted_classes <- matrix(mapply(predict_class, x1_grid, x2_grid), nrow(x1_grid), ncol(x2_grid))

plot(x1_grid[predicted_classes == 1], x2_grid[predicted_classes == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.01), pch = 16,
       xlim = c(-6, +6),
       ylim = c(-6, +6),
       xlab = "x1", ylab = "x2", las = 1)
points(x1_grid[predicted_classes == 2], x2_grid[predicted_classes == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.01), pch = 16)
points(x1_grid[predicted_classes == 3], x2_grid[predicted_classes == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.01), pch = 16)
points(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red")
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
contour(x1_interval, x2_interval, predicted_classes, levels = c(1,2,3), add = TRUE, lwd = 1, drawlabels = FALSE)
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
