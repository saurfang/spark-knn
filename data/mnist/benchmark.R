ns <- seq(2500, 10000, 2500)

spillTree_train <- c(7632.666666666666, 7165.333333333333, 7857.666666666666, 9299.333333333332)
spillTree_predict <- c(34317.0, 50162.666666666664, 62151.0, 74604.0)

bruteForce_train <- c(420.3333333333333, 714.6666666666666, 847.0, 876.6666666666666)
bruteForce_predict <- c(18028.666666666664, 47125.0, 94626.66666666666, 156086.66666666666)

library(ggplot2)
library(tidyr)
library(dplyr)

plotDF <- data.frame(ns, spillTree_train, spillTree_predict, bruteForce_train, bruteForce_predict) %>%
  gather(type, time, -ns) %>%
  separate(type, c("impl", "type")) %>%
  mutate(time = time / 1e3, impl = factor(impl))

plot <- ggplot(plotDF) +
  aes(ns, time, shape = impl, color = impl) +
  geom_line() +
  geom_point(size = 5) +
  facet_grid(type ~ .) +
  theme_bw() +
  labs(x = "number of data points", y = "wall-clock duration (seconds)", title = "MNIST Data on local[3]")

print(plot)
ggsave("data/mnist/benchmark.png", plot)