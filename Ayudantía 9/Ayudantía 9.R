library(here)
library(beepr)
library(tidyverse)
library(tidymodels)
library(patchwork)
library(skimr)
library(corrplot)

library(doParallel)
library(baguette)
library(rpart)
library(xgboost)


# Para obtener resultados reproducibles
set.seed(219)
here::i_am("Ayudantía 9/Ayudantía 9.R")

# OJO: vean bien el número de all_cores
all_cores <- parallel::detectCores(logical = FALSE)
all_cores
cl <- makePSOCKcluster(all_cores - 2)
registerDoParallel(cl)

# Análisis exploratorio -------------------------------------------------------------
# Cargamos los datos (notar na = "unknown")
banking_unclean <- readr::read_csv(here("Ayudantía 9", "banking.csv"), na = "unknown",
                                   show_col_types = FALSE)

# Damos un vistazo
skimr::skim(banking_unclean)

# Limpiamos la base
banking <- banking_unclean %>% 
  na.omit() %>% 
  dplyr::select(-pdays) %>% 
  dplyr::mutate(
    job = factor(job),
    marital = factor(marital),
    education = factor(education),
    default = factor(default, levels=c("yes", "no"), labels=c("Yes", "No")),
    housing = factor(housing, levels=c("yes", "no"), labels=c("Yes", "No")),
    loan = factor(loan, levels=c("yes", "no"), labels=c("Yes", "No")),
    contact = factor(contact),
    month = factor(month, 
                   levels=c("jan", "feb", "mar", "apr", "may", "jun", "jul",
                            "aug", "sep", "oct", "nov", "dec"),
                   ordered=TRUE),
    day_of_week = factor(day_of_week,
                         levels=c("mon", "tue", "wed", "thu", "fri"),
                         ordered=TRUE),
    poutcome = factor(poutcome),
    y = factor(y, levels=c("yes", "no"), labels=c("Yes", "No")))

skimr::skim(banking)

# Correlación entre variables
banking %>% 
  select_if(is.numeric) %>% 
  cor() %>% 
  corrplot::corrplot()

# Boxplot
p1 <- ggplot(banking, mapping = aes(x = y, y = age, fill = y)) +
  geom_boxplot() +
  labs(title = "Relación entre la edad y depósito a plazo",
       x = "Depósito a plazo", y = "Edad")

p2 <- ggplot(banking, mapping = aes(x = y, y = duration, fill = y)) +
  geom_boxplot() +
  labs(title = "Relación entre la duración de la llamada y depósito a plazo",
       x = "Depósito a plazo", y = "Edad")

p1 + p2

# Modelamiento ----------------------------------------------------------------------
# 1. División de los datos
banking_split <- rsample::initial_split(
  data = banking,
  strata = y
)

bnk_train <- rsample::training(banking_split)
bnk_test <- rsample::testing(banking_split)
bnk_cv <- rsample::vfold_cv(bnk_train, v = 5, strata = y)

# 2. Especificación de la receta
bnk_recipe <- recipes::recipe(y ~ ., data = bnk_train)

## Bagging --------------------------------------------------------------------------
# Especificación del modelo + workflow
## OJO: al parecer hay un error interno al usar tune() con class_cost
## ?bag_tree
## class_cost()

bnk_model_bag <-
  parsnip::bag_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()) %>%
  set_engine("rpart", times = 25) %>%
  set_mode("classification")
translate(bnk_model_bag)

bnk_wf_bag <- 
  workflows::workflow() %>% 
  add_model(bnk_model_bag) %>% 
  add_recipe(bnk_recipe)
bnk_wf_bag

# Tuning
metrics <- yardstick::metric_set(roc_auc)
parameters <- extract_parameter_set_dials(bnk_wf_bag)
parameters

start_grid <- 
  parameters %>% 
  grid_regular()

bnk_start_bag <- bnk_wf_bag %>% 
  tune_grid(
    resamples = bnk_cv,
    grid = start_grid,
    metrics = metrics,
  ) ; beepr::beep(1)

autoplot(bnk_start_bag)
show_best(bnk_start_bag)

bnk_bayes_bag <- bnk_wf_bag %>% 
  tune_bayes(
    resamples = bnk_cv,
    metrics = metrics,
    initial = bnk_start_bag,
    iter = 10
  ) ; beepr::beep(1)

autoplot(bnk_bayes_bag, type = "performance")
autoplot(bnk_bayes_bag, type = "parameters")

select_best(bnk_bayes_bag)

# Obtenemos el fit final
bnk_final_wf_bag <- bnk_wf_bag %>% 
  finalize_workflow(select_best(bnk_bayes_bag))
bnk_final_wf_bag

# Ajustamos y predecimos
bnk_bag_fit <- bnk_final_wf_bag %>% 
  fit(bnk_train)

bnk_bag_predictions <- 
  bnk_bag_fit %>% 
  predict(new_data = bnk_test) %>% 
  dplyr::bind_cols(bnk_test) %>% 
  dplyr::select(y, .pred_class)
head(bnk_bag_predictions)

# Matriz de confusión y recall
conf_mat_bag <- conf_mat(
  data = bnk_bag_predictions,
  truth = y,
  estimate = .pred_class
)

autoplot(conf_mat_bag, "heatmap")

bnk_bag_probs <- 
  bnk_bag_fit %>% 
  predict(new_data = bnk_test, type = "prob") %>% 
  dplyr::bind_cols(bnk_test) %>% 
  dplyr::select(y, .pred_Yes)

head(bnk_bag_probs)
roc_bag <- roc_curve(
  data = bnk_bag_probs,
  truth = y,
  estimate = .pred_Yes
)

plot_roc_bag <- autoplot(roc_bag) +
  labs(title = "Bagging")
plot_roc_bag

auc_bag <- roc_auc(
  data = bnk_bag_probs,
  truth = y,
  .pred_Yes
)

recall_bag <- recall(
  data = bnk_bag_predictions,
  truth = y,
  estimate = .pred_class
)

auc_bag
recall_bag

## Boosting -------------------------------------------------------------------------
## Es necesario usar variables dummy
bnk_recipe_boost <- recipes::recipe(y ~ ., data = bnk_train) %>% 
  step_dummy(all_nominal_predictors())

bnk_model_boost <-
  parsnip::boost_tree(
    tree_depth = tune(),
    min_n = tune(),
    mtry = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
translate(bnk_model_boost)

bnk_wf_boost <- 
  workflows::workflow() %>% 
  add_model(bnk_model_boost) %>% 
  add_recipe(bnk_recipe_boost)
bnk_wf_boost

# Tuning
metrics <- yardstick::metric_set(roc_auc)
parameters <- parameters(
  finalize(mtry(), bnk_train),
  min_n(),
  tree_depth(),
  learn_rate()
)
parameters

start_grid <- 
  parameters %>% 
  grid_regular()

bnk_start_boost <- bnk_wf_boost %>% 
  tune_grid(
    resamples = bnk_cv,
    grid = start_grid,
    metrics = metrics,
  ) ; beepr::beep(1)

autoplot(bnk_start_boost)
show_best(bnk_start_boost)

bnk_bayes_boost <- bnk_wf_boost %>% 
  tune_bayes(
    resamples = bnk_cv,
    metrics = metrics,
    initial = bnk_start_boost,
    param_info = parameters,
    iter = 10
  ) ; beepr::beep(1)

autoplot(bnk_bayes_boost, type = "performance")
autoplot(bnk_bayes_boost, type = "parameters")

select_best(bnk_bayes_boost)

# Obtenemos el fit final
bnk_final_wf_boost <- bnk_wf_boost %>% 
  finalize_workflow(select_best(bnk_bayes_boost))
bnk_final_wf_boost

# Ajustamos y predecimos
bnk_boost_fit <- bnk_final_wf_boost %>% 
  fit(bnk_train)

bnk_boost_predictions <- 
  bnk_boost_fit %>% 
  predict(new_data = bnk_test) %>% 
  dplyr::bind_cols(bnk_test) %>% 
  dplyr::select(y, .pred_class)
head(bnk_boost_predictions)

# Matriz de confusión y recall
conf_mat_boost <- conf_mat(
  data = bnk_boost_predictions,
  truth = y,
  estimate = .pred_class
)

autoplot(conf_mat_boost, "heatmap")

# Probabilidades para la curva ROC
bnk_boost_probs <- 
  bnk_boost_fit %>% 
  predict(new_data = bnk_test, type = "prob") %>% 
  dplyr::bind_cols(bnk_test) %>% 
  dplyr::select(y, .pred_Yes)
head(bnk_boost_probs)

roc_boost <- roc_curve(
  data = bnk_boost_probs,
  truth = y,
  estimate = .pred_Yes
)

plot_roc_boost <- autoplot(roc_boost) +
  labs(title = "Boosting")
plot_roc_boost

auc_boost <- roc_auc(
  data = bnk_boost_probs,
  truth = y,
  .pred_Yes
)

recall_boost <- recall(
  data = bnk_boost_predictions,
  truth = y,
  estimate = .pred_class
)
recall_boost

# Comparación -----------------------------------------------------------------------
plot_roc_bag + plot_roc_boost

cat(" AUC Bagging:  ", as.character(auc_bag[3]), "\n",
    "AUC Boosting: ", as.character(auc_boost[3]))

cat(" Recall Bagging:  ", as.character(recall_bag[3]), "\n",
    "Recall Boosting: ", as.character(recall_boost[3]))
