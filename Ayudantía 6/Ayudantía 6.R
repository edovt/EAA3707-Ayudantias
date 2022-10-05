library(tidyverse)
library(tidymodels)
library(ggcorrplot)
library(LiblineaR)
library(kernlab)


# Para obtener resultados reproducibles
set.seed(219)


# Análisis exploratorio -------------------------------------------------------------
bankruptcy <- read_csv(here("Ayudantía 6", "bankruptcy.csv"), show_col_types = FALSE)
head(bankruptcy, 8)

bankruptcy <- bankruptcy %>% 
  dplyr::rename(bankrupt = `Bankrupt?`) %>% 
  dplyr::mutate(bankrupt = factor(bankrupt, levels = c("1", "0"),
                                  labels = c("Yes", "No")))
glimpse(bankruptcy)

table(bankruptcy$bankrupt)

bankruptcy <- bankruptcy[, c(1, 2, 5, 9, 14, 24, 29, 33, 48, 51, 56, 66, 72)]

dplyr::select(bankruptcy, -bankrupt) %>%
  as.data.frame() %>%
  cor() %>%
  ggcorrplot::ggcorrplot()

bankruptcy %>%
  tidyr::pivot_longer(!bankrupt, names_to = "features", values_to = "values") %>%
  ggplot(aes(x = bankrupt, y = log(values), fill = features)) +
  geom_boxplot() +
  facet_wrap(~features, scales = "free", ncol = 4) +
  scale_color_viridis_d(option = "plasma", end = .7) +
  theme(legend.position = "none")

# Modelo de clasificación -----------------------------------------------------------
# 1. División de los datos
bankruptcy_split <- rsample::initial_split(
  data = bankruptcy,
  strata = bankrupt
)

bankruptcy_train <- rsample::training(bankruptcy_split)
bankruptcy_test <- rsample::testing(bankruptcy_split)
bankruptcy_cv <- rsample::vfold_cv(bankruptcy_train, v = 5, strata = bankrupt)

# 2. Especificación del modelo
bank_model <- parsnip::svm_linear(cost = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 3. Especificación de la receta
bank_recipe <- recipes::recipe(bankrupt ~ ., data = bankruptcy_train) %>%
  step_normalize(all_predictors())

# 4. Modelo
bank_wf <-
  workflows::workflow() %>%
  add_model(bank_model) %>%
  add_recipe(bank_recipe)

# Tuning ----------------------------------------------------------------------------
cost()
bank_wf %>% extract_parameter_dials("cost")

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

# Modelo final ----------------------------------------------------------------------
set.seed(219)

# 1. División de los datos
bankruptcy_split <- rsample::initial_split(
  data = bankruptcy,
  strata = bankrupt
)

bankruptcy_train <- rsample::training(bankruptcy_split)
bankruptcy_test <- rsample::testing(bankruptcy_split)
bankruptcy_cv <- rsample::vfold_cv(bankruptcy_train, v = 5, strata = bankrupt)

# 2. Especificación del modelo
bank_model <-
  parsnip::svm_linear(cost = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 3. Especificación de la receta
bank_recipe <- 
  recipes::recipe(bankrupt ~ ., data = bankruptcy_train) %>%
  step_normalize(all_predictors())

# 4. Modelo
bank_wf <-
  workflows::workflow() %>%
  add_model(bank_model) %>%
  add_recipe(bank_recipe)

# Cambio en los valores
c_par <- cost(range = c(-12, 5)) %>% grid_regular(levels = 5)
metrics <- yardstick::metric_set(precision, recall)

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
select_best(bank_tune, metric = "recall")

# Obtenemos el fit final
bank_f_wf <-
  bank_wf %>%
  finalize_workflow(select_best(bank_tune, metric = "precision"))
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

conf_mat(
  data = bank_predictions,
  truth = bankrupt,
  estimate = .pred_class
)
