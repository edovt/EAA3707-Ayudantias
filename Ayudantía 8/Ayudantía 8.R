library(tidyverse)
library(tidymodels)
library(corrplot)
library(skimr)
library(here)

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
                                   min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# 3. Especificación de la receta
mp_recipe <- recipes::recipe(price_range ~ ., data = mp_train)

## Ver https://www.tmwr.org/pre-proc-table.html
  
# 4. Modelo
mp_wf <- 
  workflows::workflow() %>% 
  add_model(mp_model) %>% 
  add_recipe(mp_recipe)

# Tuning ----------------------------------------------------------------------------
cost_complexity()

bank_wf %>% extract_parameter_dials("cost")
mp_wf %>% extract_parameter_dials("cost_complexity")

# Cambio en los valores
c_par <- cost(range = c(-12, 5)) %>% grid_regular(levels = 10)
metrics <- yardstick::metric_set(accuracy)

# Grilla
bank_tune <-
  bank_wf %>%
  tune_grid(
    bankruptcy_cv,
    grid = c_par,
    metrics = metrics
  )

autoplot(bank_tune)

# Vemos el parámetro del mejor modelo
select_best(bank_tune, metric = "accuracy")

# Obtenemos el fit final
bank_f_wf <-
  bank_wf %>%
  finalize_workflow(select_best(bank_tune, metric = "accuracy"))
bank_f_wf

# Ajustamos y predecimos
bank_fit <-
  bank_f_wf %>%
  fit(bankruptcy_train)

bank_predictions <-
  bank_fit %>%
  predict(new_data = bankruptcy_test) %>%
  dplyr::bind_cols(bankruptcy_test) %>%
  dplyr::select(bankrupt, .pred_class)
head(bank_predictions, 10)

table(bank_predictions$.pred_class)

conf_mat(
  data = bank_predictions,
  truth = bankrupt,
  estimate = .pred_class
)


