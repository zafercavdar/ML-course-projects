# Zafer Cavdar - COMP 421 Homework 4 - Nonparametric Regression

# read data into memory
data_set <- read.csv("hw04_data_set.csv")

# get x and y values
set.seed(521)
x_all <- data_set$x
y_all <- data_set$y
train_indices <- sample(length(x_all), 100)
x_train <- x_all[train_indices]
y_train <- y_all[train_indices]
x_test <- x_all[-train_indices]
y_test <- y_all[-train_indices]

# set bin width and borders
minimum_value <- floor(min(x_all)) - 2 
maximum_value <- ceiling(max(x_all)) + 2
bin_width <- 3
grid_interval <- 0.01
data_interval <- seq(from = minimum_value, to = maximum_value, by = grid_interval)

# Regressogram
left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)
g_head <- sapply(1:length(left_borders), function(i) {
    bin <- y_train[left_borders[i] < x_train & x_train <= right_borders[i]]
    return(mean(bin))
  }
)

get_bin_no <- function(v) {
  return(ceiling((v-minimum_value) / bin_width))
}

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(min(y_train), max(y_train)), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test,type = "p", pch = 19, col= "red")
legend(55,85, legend=c("training", "test"),
       col=c("blue", "red"), pch = 19, cex = 0.5, bty = "y")
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(g_head[b], g_head[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(g_head[b], g_head[b + 1]), lwd = 2, col = "black") 
  }
}

# Calculate RMSE for regressogram
distances <- sapply(1:length(y_test), function(i) {
    x_test_i <- x_test[i]
    bin <- get_bin_no(x_test_i)
    y_estimated_i <- g_head[bin]
    y_test_i <- y_test[i]
    diff <- y_test_i - y_estimated_i
    return(diff^2)
})
RMSE <- sqrt(sum(distances) / length(distances))
sprintf("Regressogram => RMSE is %s when h is %s", RMSE, bin_width)

# Running mean smoother
g_head <- sapply(data_interval, function(x) {
  y_train_bin <- y_train[(x - 0.5 * bin_width) < x_train & x_train <= (x + 0.5 * bin_width)]
  return(mean(y_train_bin))
})

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(min(y_train), max(y_train)), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test,type = "p", pch = 19, col= "red")
legend(55,85, legend=c("training", "test"),
       col=c("blue", "red"), pch = 19, cex = 0.5, bty = "y")
for (b in 1:length(data_interval)) {
  lines(c(data_interval[b], data_interval[b+1]), c(g_head[b], g_head[b]), lwd = 2, col = "black")
  if (b < length(data_interval)) {
    lines(c(data_interval[b+1], data_interval[b+1]), c(g_head[b], g_head[b + 1]), lwd = 2, col = "black") 
  }
}

# Calculate RMSE for running mean smoother
get_interval_no <- function(v) {
  return(ceiling((v-minimum_value) / grid_interval))
}

distances <- sapply(1:length(y_test), function(i) {
  x_test_i <- x_test[i]
  box <- get_interval_no(x_test_i)
  y_estimated_i <- g_head[box]
  y_test_i <- y_test[i]
  diff <- y_test_i - y_estimated_i
  return(diff^2)
})
RMSE <- sqrt(sum(distances) / length(distances))
sprintf("Running Mean Smoother => RMSE is %s when h is %s", RMSE, bin_width)

# Kernel Smoother
bin_width <- 1
gaussian_kernel = function(u) {
  (1 / sqrt((2 * pi))) * exp(-u^2 / 2)
}

g_head <- sapply(data_interval, function(x) {
  nominator <- sapply(1:length(x_train), function(i) {
    u <- (x - x_train[i]) / bin_width
    kernel <- gaussian_kernel(u)
    return(kernel*y_train[i])
  })
  denominator <- sapply(1:length(x_train), function(i) {
    u <- (x - x_train[i]) / bin_width
    kernel <- gaussian_kernel(u)
    return(kernel)
  })
  return(sum(nominator) / sum(denominator))
})

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(min(y_train), max(y_train)), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test,type = "p", pch = 19, col= "red")
legend(55,85, legend=c("training", "test"),
       col=c("blue", "red"), pch = 19, cex = 0.5, bty = "y")
for (b in 1:length(data_interval)) {
  lines(c(data_interval[b], data_interval[b+1]), c(g_head[b], g_head[b]), lwd = 2, col = "black")
  if (b < length(data_interval)) {
    lines(c(data_interval[b+1], data_interval[b+1]), c(g_head[b], g_head[b + 1]), lwd = 2, col = "black") 
  }
}

#for (i in 1:length(x_test)) {
#  lines(c(x_test[i], x_test[i]), c(y_test[i], g_head[get_interval_no(x_test[i])]), lwd = 2, col = "black")
#}

# Calculate RMSE for kernel smoother
distances <- sapply(1:length(y_test), function(i) {
  x_test_i <- x_test[i]
  box <- get_interval_no(x_test_i)
  y_estimated_i <- g_head[box]
  y_test_i <- y_test[i]
  diff <- y_test_i - y_estimated_i
  return(diff^2)
})
RMSE <- sqrt(sum(distances) / length(distances))
sprintf("Kernel Smoother => RMSE is %s when h is %s", RMSE, bin_width)
