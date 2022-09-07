library(tidyverse)
library(tidymodels)
library(car)


# Para obtener resultados reproducibles
set.seed(912)

# Recuerdo del modelo de regresión logística ----------------------------------------
# Visualización de diferentes funciones de enlace
logistic <- function(x){exp(x)/(exp(x) + 1)}
probit <- function(x){pnorm(x)}
llc <- function(x){1 - exp(-exp(x))}

ggplot(data.frame(x = c(-10, 10)), aes(x = x)) + 
  stat_function(fun = logistic, col = 'red', lwd = 1) +
  stat_function(fun = probit, col = 'blue', lwd = 1) +
  stat_function(fun = llc, col = 'green', lwd = 1) +
  geom_hline(yintercept = c(0, 1), lty = 'dashed') +
  geom_hline(yintercept = 0.5, lty = 'dashed') +
  geom_vline(xintercept = 0, lty = 'dashed') +
  ylim(-0.1, 1.1) +
  labs(title = 'Diferentes funciones de enlace',
       subtitle = 'Rojo - logit, Azul - probit, Verde - llc',
       y = 'p(Y=1|x)')

