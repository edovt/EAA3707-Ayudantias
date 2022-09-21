library(tidyverse)
library(tidymodels)
library(patchwork)
library(kknn)
library(udunits2)


# Para obtener resultados reproducibles
set.seed(2211)

# Análisis exploratorio -------------------------------------------------------------
# Precio
ggplot(Sacramento, aes(x = price)) +
  geom_histogram(aes(y = ..density..), bins = 30, col = 'black', fill = 'salmon') +
  geom_density(col = 'blue', lwd = 2) +
  scale_x_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12)) +
  labs(title = 'Distribución del precio de las casas',
       subtitle = 'Condado de Sacramento, California',
       x = 'Precio (Dólares)', y = 'Densidad')

# Relación sqm y precio
ggplot(Sacramento, aes(x = sqm, y = price)) +
  geom_point() +
  theme(text = element_text(size = 12)) + 
  scale_y_continuous(labels = dollar_format()) +
  labs(title = 'Relación tamaño y precio',
       subtitle = 'Condado de Sacramento, California',
       x = 'Tamaño (Metros cuadrados)', y = 'Precio (Dólares)')

# Relación número habitaciones y precio
ggplot(Sacramento, aes(x = as.factor(beds), y = price, fill = as.factor(beds))) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme(text = element_text(size = 12)) +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = 'Relación número de piezas y precio de la casa',
       subtitle = 'Condado de Sacramento, California',
       x = 'Número de habitaciones', y = 'Precio (Dólares)',
       fill = 'Número de habitaciones')


# Ejemplo visual KNN ----------------------------------------------------------------
# Tomamos una muestra pequeña
small_sacramento <- dplyr::slice_sample(Sacramento, n = 30)

# Vemos la muestra y visualizamos lo que queremos predecir
small_plot <- ggplot(small_sacramento, aes(x = sqm, y = price)) +
  geom_point() +
  geom_vline(xintercept = 550, lty = 'dashed', col = 'red') +
  scale_y_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12)) +
  labs(title = 'Pequeña muestra de los datos junto a la nueva predicción',
       x = 'Tamaño (Metros cuadrados)', y = 'Precio (Dólares)')

small_plot

# Obtenemos los vecinos
vecinos <- small_sacramento %>% 
  dplyr::mutate(diff = abs(550 - sqm)) %>% 
  dplyr::arrange(diff) %>% 
  slice(1:7) %>% 
  dplyr::mutate(xend = rep(550, 7))

head(vecinos)

# Graficamos las distancias
vecinos_plot <- small_plot +
  geom_segment(data = vecinos,
               aes(x = sqm, xend = xend, y = price, yend = price),
               col = 'blue')
vecinos_plot

# Realizamos la predicción
prediccion <- mean(vecinos$price)

# Visualizamos
vecinos_plot +
  geom_point(aes(x = 550, y = prediccion), col = 'purple', size = 3)

# Implementación KNN ----------------------------------------------------------------
## Especificación del modelo
# 1 - División de los datos
split_info <- rsample::initial_split(
  data = Sacramento,
  prop = 0.75,
  strata = price
)

sacramento_train <- rsample::training(split_info)
sacramento_test  <- rsample::testing(split_info)

## Para seleccionar k, utilizaremos validación cruzada
sacr_cv <- rsample::vfold_cv(sacramento_train, v = 5, strata = price)

# 2 - Especificación del modelo
sacr_model <- parsnip::nearest_neighbor(weight_func = 'rectangular',
                                        neighbors = tune::tune()) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

# 3 - Especificación de la receta
sacr_recipe <- recipes::recipe(price ~ sqm, data = sacramento_train) %>% 
  step_normalize(all_predictors())

## Ajuste del modelo
sacr_wf <- 
  workflows::workflow() %>% 
  add_model(sacr_model) %>% 
  add_recipe(sacr_recipe)

sacr_wf

grid_k <- tibble(neighbors = seq(1, 200, by = 3))
sacr_results <- sacr_wf %>% 
  tune::tune_grid(resamples = sacr_cv, grid = grid_k) %>% 
  collect_metrics()

sacr_results

## Valor óptimo de k
# Nos fijamos solo en el RMSE
sacr_results_rmse <- sacr_results %>% 
  dplyr::filter(.metric == "rmse")

# Obtenemos el valor de K con el mínimo valor de RMSE
min_k <- sacr_results_rmse %>% 
  filter(mean == min(mean)) %>% 
  .$neighbors

# Graficamos
ggplot(sacr_results_rmse, aes(x = neighbors, y = mean)) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = min_k, lty = "dashed", col = "red", lwd = 1.5) +
  labs(title = "Evolución RMSE según el número de vecinos",
       x = "Vecinos (K)", y = "RMSE")

print(paste("El valor óptimo de k es", min_k))

# Usamos el valor óptimo de k
sacr_model <- parsnip::nearest_neighbor(weight_func = 'rectangular',
                                        neighbors = min_k) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

# Ajustamos y obtenemos las medidas de ajuste
sacr_fit <- workflow() %>% 
  add_model(sacr_model) %>% 
  add_recipe(sacr_recipe) %>% 
  fit(data = sacramento_train)

sacr_summary <- sacr_fit %>% 
  predict(sacramento_test) %>% 
  dplyr::bind_cols(sacramento_test) %>% 
  yardstick::metrics(truth = price, estimate = .pred) %>% 
  dplyr::filter(.metric == "rmse")

sacr_summary

## Recta ajustada
sacr_preds <- tibble(sqm = seq(from = 50, to = 1500, by = 10))

sacr_preds <- sacr_fit %>% 
  predict(sacr_preds) %>% 
  bind_cols(sacr_preds)

plot_final <- ggplot(sacramento_train, aes(x = sqm, y = price)) +
  geom_point(alpha = 0.4) +
  geom_line(data = sacr_preds, 
            mapping = aes(x = sqm, y = .pred), 
            color = "blue") +
  labs(title = paste0("Recta ajustada KNN - K = ", min_k),
       x = "Tamaño (metros cuadrados)", y = "Precio (Dólares)") +
  scale_y_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12))

plot_final

# Uso de pesos ----------------------------------------------------------------------
# Redefinimos el modelo
sacr_model <- parsnip::nearest_neighbor(weight_func = "gaussian",
                                        neighbors = min_k) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

# Ajustamos
sacr_fit <- workflow() %>% 
  add_model(sacr_model) %>% 
  add_recipe(sacr_recipe) %>% 
  fit(data = sacramento_train)

sacr_preds <- tibble(sqm = seq(from = 50, to = 1500, by = 10))

sacr_preds <- sacr_fit %>% 
  predict(sacr_preds) %>% 
  bind_cols(sacr_preds)

plot_final2 <- ggplot(sacramento_train, aes(x = sqm, y = price)) +
  geom_point(alpha = 0.4) +
  geom_line(data = sacr_preds, 
            mapping = aes(x = sqm, y = .pred), 
            color = "blue") +
  labs(title = paste0("Recta ajustada KNN - K = ", min_k),
       subtitle = "Usando ponderador gaussiano",
       x = "Tamaño (metros cuadrados)", y = "Precio (Dólares)") +
  scale_y_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12))

plot_final2 + plot_final

## Optimización con pesos
# 1 - División de los datos
split_info <- rsample::initial_split(
  data = Sacramento,
  prop = 0.75,
  strata = price
)

sacramento_train <- rsample::training(split_info)
sacramento_test  <- rsample::testing(split_info)

## Para seleccionar k, utilizaremos validación cruzada
sacr_cv <- rsample::vfold_cv(sacramento_train, v = 5, strata = price)

# 2 - Especificación del modelo (agregamos weight_func a tune)
sacr_model <- parsnip::nearest_neighbor(weight_func = tune::tune(),
                                        neighbors = tune::tune()) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

# 3 - Especificación de la receta
sacr_recipe <- recipes::recipe(price ~ sqm, data = sacramento_train) %>% 
  step_normalize(all_predictors())

# 4 - Ajustamos el modelo
sacr_wf <- 
  workflows::workflow() %>% 
  add_model(sacr_model) %>% 
  add_recipe(sacr_recipe)

grid_k_wfun <- expand.grid(
  neighbors = seq(1, 200, by = 3), 
  weight_func = c("rectangular", "inv", "gaussian", "triangular")
)

sacr_results <- sacr_wf %>% 
  tune::tune_grid(resamples = sacr_cv, grid = grid_k_wfun) %>% 
  collect_metrics()

## Evolución del RMSE
sacr_results %>% 
  dplyr::filter(.metric == "rmse") %>% 
  ggplot(aes(x = neighbors, y = mean, col = weight_func)) +
  geom_line() +
  geom_point() +
  labs(x = "Número de vecinos (K)", y = "RMSE",
       title = "Evolución del RMSE", col = "Función ponderadora")

## Obtenemos los parámetros con el mínimo RMSE
opt_par <- sacr_results %>% 
  dplyr::filter(.metric == "rmse") %>% 
  dplyr::filter(mean == min(mean))

opt_par

# Usamos los valores óptimos
sacr_model <- parsnip::nearest_neighbor(weight_func = "triangular",
                                        neighbors = 76) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

# Ajustamos y obtenemos las medidas de ajuste
sacr_fit <- workflow() %>% 
  add_model(sacr_model) %>% 
  add_recipe(sacr_recipe) %>% 
  fit(data = sacramento_train)

sacr_preds <- tibble(sqm = seq(from = 50, to = 1500, by = 10))

sacr_preds <- sacr_fit %>% 
  predict(sacr_preds) %>% 
  bind_cols(sacr_preds)

plot_final <- ggplot(sacramento_train, aes(x = sqm, y = price)) +
  geom_point(alpha = 0.4) +
  geom_line(data = sacr_preds, 
            mapping = aes(x = sqm, y = .pred), 
            color = "blue") +
  labs(title = paste0("Recta ajustada KNN - K = 76"),
       subtitle = "Ponderador triangular",
       x = "Tamaño (metros cuadrados)", y = "Precio (Dólares)") +
  scale_y_continuous(labels = dollar_format()) +
  theme(text = element_text(size = 12))

plot_final
