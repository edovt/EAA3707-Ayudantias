library(tidyverse)
library(tidymodels)
library(car)
library(GGally)
library(ggcorrplot)
library(ResourceSelection)


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

# Carga de datos --------------------------------------------------------------------
# Cargamos los datos
credit_full <- readr::read_table(
  file = here::here('Ayudantía 4', 'german.data'), 
  col_names = FALSE
)

# Pre-procesamiento de los datos
credit <- 
  credit_full %>% 
  dplyr::select(X2, X3, X5, X13, X15, X16, X21) %>% 
  dplyr::rename(duration = X2,
                credit_history = X3,
                credit_amount = X5,
                age = X13,
                housing = X15,
                n_credits = X16,
                risk = X21) %>%
  dplyr::mutate(risk = risk - 1) %>% # Parte nueva
  dplyr::mutate(housing = dplyr::recode(housing, 
                                        "A151" = 'rent',
                                        "A152" = 'own',
                                        "A153" = 'free'),
                risk = dplyr::recode(risk,
                                     `0` = 'good',
                                     `1` = 'bad')) %>% 
  dplyr::mutate(across(where(is.character), as_factor))

# Vemos cómo nos quedan
dplyr::glimpse(credit)

# Ejemplo práctico ------------------------------------------------------------------

## Linealidad del logito (visual) ---------------------------------------------------
### Edad
# Podemos contar según la edad, vemos que existe un n_i = 1
age_freqs <- credit %>% 
  dplyr::count(age, name = 'total')
age_freqs
any(age_freqs$total == 1)
sum(age_freqs$total == 1)

# Agrupamos por edad y calculamos las proporciones y logitos muestrales
prop_logit <- credit %>% 
  dplyr::group_by(age, risk, .drop = FALSE) %>% 
  dplyr::count() %>% 
  dplyr::filter(risk == 'bad') %>% 
  dplyr::left_join(age_freqs, by = 'age') %>% 
  dplyr::mutate(prop = n/total)

logitos <- car::logit(prop_logit$prop, adjust = 0.01)
prop_logit$logitos <- logitos

# Graficamos
## Proporciones
ggplot(prop_logit, mapping = aes(x = age, y = prop)) +
  geom_point(size = 6) +
  labs(title = 'Análisis visual proporciones muestrales',
       subtitle = 'Usando la variable Age',
       x = 'Edad', y = 'Proporción muestral')

## Logitos
ggplot(prop_logit, mapping = aes(x = age, y = logitos)) +
  geom_point(size = 6) +
  labs(title = 'Análisis visual linealidad del logito',
       subtitle = 'Usando la variable Age',
       x = 'Edad', y = 'Logito muestral')

### Monto del crédito
# Podemos contar según el monto del crédito, vemos que en este caso hay 847 n_i = 1
am_freqs <- credit %>% 
  dplyr::count(credit_amount, name = 'total')
am_freqs
sum(am_freqs$total == 1)

# Agrupamos los montos en intervalos
binned <- credit %>% 
  dplyr::mutate(amount_bin = cut(credit_amount, breaks = 15, labels = FALSE)) %>% 
  dplyr::select(credit_amount, amount_bin, risk)

amb_freqs <- binned %>% 
  dplyr::count(amount_bin, name = 'total')
amb_freqs

# Realizamos el conteo y calculamos proporciones y logitos
prop_logit <- binned %>% 
  dplyr::group_by(amount_bin, risk, .drop = FALSE) %>% 
  dplyr::count() %>% 
  dplyr::filter(risk == 'bad') %>% 
  dplyr::left_join(amb_freqs, by = 'amount_bin') %>% 
  dplyr::mutate(prop = n/total)

logitos <- car::logit(prop_logit$prop, adjust = 0.01)
prop_logit$logitos <- logitos

# Graficamos
## Proporciones
ggplot(prop_logit, mapping = aes(x = amount_bin, y = prop)) +
  geom_point(size = 6) +
  labs(title = 'Análisis visual proporciones muestrales',
       subtitle = 'Usando la variable Credit Amount',
       x = 'Monto del crédito ajustado', y = 'Proporción muestral')

## Logitos
ggplot(prop_logit, mapping = aes(x = amount_bin, y = logitos)) +
  geom_point(size = 6) +
  labs(title = 'Análisis visual linealidad del logito',
       subtitle = 'Usando la variable Credit Amount',
       x = 'Monto del crédito ajustado', y = 'Logito muestral')

## Multicolinealidad (visual) -------------------------------------------------------
## ggpairs
GGally::ggpairs(credit, columns = c('credit_amount', 'age', 'duration', 'n_credits'))

## ggcorrplot recibe la matriz de correlación
cor_mat <- credit %>% 
  dplyr::select(credit_amount, age, duration, n_credits) %>% 
  as.data.frame() %>% 
  cor()

ggcorrplot::ggcorrplot(cor_mat, lab = TRUE)

## Ajuste del modelo ----------------------------------------------------------------
# 1 - Dividimos los datos
split_info <- rsample::initial_split(
  data = credit,
  prop = 0.75,
  strata = risk
)

credit_train <- rsample::training(split_info)
credit_test <- rsample::testing(split_info)

# 2 - Especificamos el modelo
logreg_model <- 
  parsnip::logistic_reg() %>% 
  set_engine('glm') %>% 
  set_mode('classification')

# 3 - Creamos la receta
credit_recipe <- 
  recipes::recipe(risk ~ ., data = credit_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors())

# 4 - Juntamos todo en un workflow 
logreg_wf <-
  workflows::workflow() %>%
  add_model(logreg_model) %>%
  add_recipe(credit_recipe)

# 5 - Ajustamos el modelo y lo recuperamos
logreg_fit <- logreg_wf %>%
  parsnip::fit(data = credit_train) %>% 
  workflowsets::extract_fit_parsnip() %>% 
  .$fit

# Vemos un resumen del modelo ajustado
summary(logreg_fit)

## Test de linealidad del logito ----------------------------------------------------
### Visual
# Recuperamos residuos y valores ajustados
residuos <- residuals(logreg_fit)
ajustados <- fitted(logreg_fit)

aux <- tibble(res = residuos, ajust = ajustados)

ggplot(data = aux, mapping = aes(x = ajust, y = res)) +
  geom_point() +
  geom_hline(yintercept = 0, lty = 'dashed', col = 'red') +
  labs(title = 'Linealidad del logito', subtitle = 'Sin agrupar')

# Agrupamos nuevamente
aux2 <- aux %>% 
  dplyr::mutate(ajust_bin = cut(ajust, breaks = 30, labels = FALSE)) %>% 
  dplyr::group_by(ajust_bin) %>% 
  dplyr::summarise(res_medio = mean(res))

ggplot(data = aux2, mapping = aes(x = ajust_bin, y = res_medio)) +
  geom_point() +
  geom_hline(yintercept = 0, lty = 'dashed', col = 'red') +
  labs(title = 'Linealidad del logito', subtitle = 'Datos agrupados')

### Box-Tidwell
aux <- credit %>% 
  dplyr::mutate(risk = as.integer(risk) - 1)

car::boxTidwell(formula = risk ~ duration + credit_amount + age,
                data = aux)

## Multicolinealidad ----------------------------------------------------------------
# Calculamos los VIF
car::vif(logreg_fit)

# Tests de bondad de ajuste ---------------------------------------------------------
## Devianza
summary(logreg_fit)

## Test de Hosmer-Lemeshow
ResourceSelection::hoslem.test(logreg_fit$y, fitted(logreg_fit))
