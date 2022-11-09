library(here)
library(beepr)
library(tidyverse)
library(tidymodels)
library(skimr)
library(corrplot)
library(rpart)


# Para obtener resultados reproducibles
set.seed(219)
here::i_am("Ayudantía 8/Ayudantía 8.R")

# Análisis Exploratorio -------------------------------------------------------------
# Cargamos los datos
mobile_prices_unclean <- readr::read_csv(here("Ayudantía 8", "mobile_price.csv"),
                                         show_col_types = FALSE)

# Damos un vistazo
skimr::skim(mobile_prices_unclean)

# Cambiamos a factor
mobile_prices <- mobile_prices_unclean %>% 
  dplyr::mutate(blue = factor(blue, levels=c(1, 0), labels=c("Yes", "No")),
                dual_sim = factor(dual_sim, levels=c(1, 0), labels=c("Yes", "No")),
                four_g = factor(four_g, levels=c(1, 0), labels=c("Yes", "No")),
                three_g = factor(three_g, levels=c(1, 0), labels=c("Yes", "No")),
                touch_screen = factor(touch_screen, levels=c(1, 0),
                                      labels=c("Yes", "No")),
                wifi = factor(wifi, levels=c(1, 0), labels=c("Yes", "No")),
                price_range = factor(price_range, levels=c(0, 1, 2, 3),
                                     labels=c("Low", "Medium", "High", "Very High")))

skimr::skim(mobile_prices)

# Correlación entre variables numéricas
mobile_prices %>% 
  select_if(is.numeric) %>% 
  cor() %>% 
  corrplot::corrplot()

# Boxplot
ggplot(mobile_prices,
       mapping = aes(x = price_range, y = clock_speed, fill = price_range)) +
  geom_boxplot() +
  labs(title = "Relación entre velocidad del procesador y rango de precio",
       x = "Rango de precio", y = "Velocidad del procesador")

ggplot(mobile_prices,
       mapping = aes(x = price_range, y = ram, fill = price_range)) +
  geom_boxplot() +
  labs(title = "Relación entre memoria RAM y rango de precio",
       x = "Rango de precio", y = "Memoria RAM")

# Árbol de Decisión -----------------------------------------------------------------
# 1. División de los datos
mobile_prices_split <- rsample::initial_split(
  data = mobile_prices,
  strata = price_range
)

mp_train <- rsample::training(mobile_prices_split)
mp_test <- rsample::testing(mobile_prices_split)
mp_cv <- rsample::mc_cv(mp_train, strata = price_range)

# 2. Especificación del modelo
mp_model <- parsnip::decision_tree(cost_complexity = tune(),
                                   tree_depth = tune(),
                                   min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# 3. Especificación de la receta
mp_recipe <- recipes::recipe(price_range ~ ., data = mp_train)

# 4. Modelo
mp_wf <- 
  workflows::workflow() %>% 
  add_model(mp_model) %>% 
  add_recipe(mp_recipe)

# Tuning ----------------------------------------------------------------------------
## Proponer valores iniciales
cost_complexity()
tree_depth()
min_n()

metrics <- yardstick::metric_set(accuracy)

# Grilla de partida
start_grid <- 
  extract_parameter_set_dials(mp_wf) %>% 
  grid_regular()
start_grid

mp_start <- mp_wf %>% 
  tune_grid(
    resamples = mp_cv,
    grid = start_grid,
    metrics = metrics
  ) ; beepr::beep(1)

autoplot(mp_start)
show_best(mp_start)

ctrl <- tune::control_bayes(verbose = TRUE)
mp_bayesopt <- 
  mp_wf %>% 
  tune_bayes(
    resamples = mp_cv,
    metrics = metrics,
    initial = mp_start,
    iter = 20,
    control = ctrl
  ) ; beepr::beep(1)

autoplot(mp_bayesopt, type = "performance")
autoplot(mp_bayesopt, type = "parameters")

select_best(mp_bayesopt)

# Obtenemos el fit final
mp_final_wf <- 
  mp_wf %>% 
  finalize_workflow(select_best(mp_bayesopt))
mp_final_wf

# Ajustamos y predecimos
mp_fit <- 
  mp_final_wf %>% 
  fit(mp_train)

mp_predictions <- 
  mp_fit %>% 
  predict(new_data = mp_test) %>% 
  dplyr::bind_cols(mp_test) %>% 
  dplyr::select(price_range, .pred_class)
head(mp_predictions)

conf_mat_tree <- conf_mat(
  data = mp_predictions,
  truth = price_range,
  estimate = .pred_class
)

autoplot(conf_mat_tree, "mosaic")
autoplot(conf_mat_tree, "heatmap")

# Bosque aleatorio ------------------------------------------------------------------
# 2. Especificación del modelo

mp_model_b <- parsnip::rand_forest(mtry = tune(),
                                   trees = tune(),
                                   min_n = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# 3. Especificación de la receta
mp_recipe_b <- recipes::recipe(price_range ~ ., data = mp_train)

# 4. Modelo
mp_wf_b <- 
  workflows::workflow() %>% 
  add_model(mp_model_b) %>% 
  add_recipe(mp_recipe_b)

# Tuning ----------------------------------------------------------------------------
## Proponer valores iniciales
mtry()
trees()
min_n()

finalize(mtry(), x = select(mp_train, -price_range))

# Grilla de partida
start_grid <- 
  parameters(
    finalize(mtry(), x = select(mp_train, -price_range)),
    trees(),
    min_n()
  ) %>% 
  grid_regular()
start_grid

metrics <- yardstick::metric_set(accuracy)
mp_start_b <- mp_wf_b %>% 
  tune_grid(
    resamples = mp_cv,
    grid = start_grid,
    metrics = metrics
  ) ; beepr::beep(1)

autoplot(mp_start_b)
show_best(mp_start_b)

ctrl <- tune::control_bayes(verbose = TRUE)
mp_bayesopt_b <- 
  mp_wf_b %>% 
  tune_bayes(
    resamples = mp_cv,
    metrics = metrics,
    initial = mp_start_b,
    param_info = parameters(finalize(mtry(), x = select(mp_train, -price_range)),
                            trees(),
                            min_n()),
    iter = 10,
    control = ctrl
  ) ; beepr::beep(1)

autoplot(mp_bayesopt_b, type = "performance")
autoplot(mp_bayesopt_b, type = "parameters")

select_best(mp_bayesopt_b)

# Obtenemos el fit final
mp_final_wf_b <- 
  mp_wf_b %>% 
  finalize_workflow(select_best(mp_bayesopt_b))
mp_final_wf_b

# Ajustamos y predecimos
mp_fit_b <- 
  mp_final_wf_b %>% 
  fit(mp_train)

mp_predictions_b <- 
  mp_fit_b %>% 
  predict(new_data = mp_test) %>% 
  dplyr::bind_cols(mp_test) %>% 
  dplyr::select(price_range, .pred_class)
head(mp_predictions)

conf_mat_tree_b <- conf_mat(
  data = mp_predictions_b,
  truth = price_range,
  estimate = .pred_class
)

autoplot(conf_mat_tree_b, "mosaic")
autoplot(conf_mat_tree_b, "heatmap")

# Vemos el valor del accuracy
acc_rf <- accuracy(
  data = mp_predictions_b,
  truth = price_range,
  estimate = .pred_class
)
acc_rf
acc_tree
