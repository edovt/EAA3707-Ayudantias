library(tidyverse)
library(tidymodels)


# Para obtener resultados reproducibles
set.seed(912)

# El modelo de regresión logística --------------------------------------------------
## Función logística
logistic_fun <- function(x){exp(x)/(exp(x) + 1)}
ggplot(data.frame(x = c(-10, 10)), aes(x = x)) + 
  stat_function(fun = logistic_fun, col = 'red', lwd = 1) +
  geom_hline(yintercept = c(0, 1), lty = 'dashed') +
  ylim(-0.1, 1.1) +
  labs(title = 'Función logística')

## Regresión lineal
x <- seq(20, 100, length.out = 100) + runif(100, -5, 5)
y <- 3 + 4 * x + rnorm(100, sd = 50)
recta <- function(x){3 + 4*x}

ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) + 
  geom_point() + 
  stat_function(fun = recta, col = 'red', lwd = 2) +
  labs(title = 'Función lineal')

# Ajuste de un modelo de regresión logística ----------------------------------------
## Cargar datos
credit_full <- readr::read_table(
  file = 'Ayudantía 3/german.data', 
  col_names = FALSE
)

### Echamos un vistazo a los datos
dplyr::glimpse(credit_full)

### Selección y nombres
credit <- credit_full %>% 
  dplyr::select(X2, X3, X5, X13, X15, X16, X21) %>% 
  dplyr::rename(
    duration = X2,
    credit_history = X3,
    credit_amount = X5,
    age = X13,
    housing = X15,
    n_credits = X16,
    risk = X21) %>% 
  dplyr::mutate(housing = dplyr::recode(housing, 
                                        "A151" = 'rent',
                                        "A152" = 'own',
                                        "A153" = 'free'),
                risk = dplyr::recode(risk,
                                     `1` = 'good',
                                     `2` = 'bad')) %>% 
  dplyr::mutate(across(where(is.character), as_factor))

### Veamos nuevamente los datos pero ahora arreglados
dplyr::glimpse(credit)

## Análisis exploratorio
### Proporción riesgo
credit %>% 
  dplyr::count(risk) %>% 
  dplyr::mutate(prop = n/sum(n))

### Relación edad-riesgo
ggplot(data = credit, mapping = aes(x = risk, y = age, fill = risk)) +
  geom_boxplot() +
  labs(x = 'riesgo', y = 'edad',
       title = 'Relación edad y riesgo del cliente')

# Modelamiento ----------------------------------------------------------------------
## División de los datos
split_info <- rsample::initial_split(
  data = credit,
  prop = 0.75,
  strata = risk
)

credit_train <- rsample::training(split_info)
credit_test <- rsample::testing(split_info)

### Verificar proporciones
# Base de entrenamiento
credit_train %>% 
  dplyr::count(risk) %>% 
  dplyr::mutate(prop = n/sum(n))

# Base de testeo
credit_test %>% 
  dplyr::count(risk) %>% 
  dplyr::mutate(prop = n/sum(n))

## Especificación del modelo
logreg_model <- parsnip::logistic_reg() %>% 
  set_engine('glm') %>% 
  set_mode('classification')

logreg_model %>% 
  parsnip::translate()

## Preprocesamiento de los datos y feature engineering
credit_recipe <- 
  recipes::recipe(risk ~ ., data = credit_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors())

### Podemos ver los detalles de la receta
credit_recipe

## Ajuste del modelo
logreg_wf <- 
  workflows::workflow() %>% 
  add_model(logreg_model) %>% 
  add_recipe(credit_recipe)

logreg_fit <- logreg_wf %>%
  parsnip::fit(data = credit_train)

logreg_fit %>%
  extract_recipe(estimated = TRUE)

logreg_fit %>% 
  extract_fit_parsnip() %>% 
  broom::tidy()

### Podemos ver las predicciones del modelo y comparar con el real
credit_test_pred <- credit_test %>% 
  dplyr::select(risk) %>% 
  dplyr::mutate(predict(logreg_fit, new_data = credit_test)) %>% 
  dplyr::rename(risk_pred = .pred_class)

head(credit_test_pred)

### Podemos ver cuantos fueron categorizados como buenos y malos
credit_test_pred %>% 
  dplyr::count(risk_pred) %>% 
  dplyr::mutate(prop = n/sum(n))


## Evaluación del modelo
# last_fit ajusta y predice al mismo tiempo
final_logreg <- 
  logreg_wf %>% 
  tune::last_fit(split_info)

# Obtenemos algunas métricas con collect_metrics()
final_logreg %>% 
  collect_metrics()

# Modelamiento completo -------------------------------------------------------------

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

# 5 - Ajustamos el modelo
logreg_fit <- logreg_wf %>%
  parsnip::fit(data = credit_train)

# 6 - Evaluamos el modelo
logreg_metrics <- logreg_wf %>% 
  tune::last_fit(split_info) %>% 
  collect_metrics()

# Discusión: matriz de confusión ----------------------------------------------------

# yardstick recibe los valores reales y los predichos
credit_test_pred %>% 
  yardstick::conf_mat(truth = risk,
                      estimate = risk_pred)
