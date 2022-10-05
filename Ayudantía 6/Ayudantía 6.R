library(tidyverse)
library(tidymodels)
library(ggcorrplot)


# Para obtener resultados reproducibles
set.seed(219)


bankruptcy <- readr::read_csv("Ayudantía 6/data.csv")
glimpse(bankruptcy)

bankruptcy$`Bankrupt?` <- as.factor(bankruptcy$`Bankrupt?`)
bankruptcy <- dplyr::rename(bankruptcy, bankrupt = `Bankrupt?`)

## Variable respuesta
table(bankruptcy$bankrupt)

## Predictores
dplyr::select(bankruptcy, -bankrupt) %>% 
  as.data.frame() %>% 
  cor() %>% 
  ggcorrplot::ggcorrplot(tl.cex = 0)




set.seed(219)
# 1. División de los datos
bankruptcy_split <- rsample::initial_split(
  data = bankruptcy,
  strata = bankrupt
)

bankruptcy_train <- rsample::training(bankruptcy_split)
bankruptcy_test <- rsample::training(bankruptcy_split)

# 2. Especificación del modelo
bank_model <- 
  parsnip::svm_linear(cost = tune()) %>% 
  set_engine("LiblineaR") %>% 
  set_mode("classification")

# 3. Especificación de la receta
bank_recipe <- 
  recipes::recipe(bankrupt ~ ., data = bankruptcy_train) %>% 
  step_normalize(all_predictors())

bank_wf <- 
  workflows::workflow() %>% 
  add_model(bank_model) %>% 
  add_recipe(bank_recipe)

aux <- bank_wf %>% 
  extract_parameter_set_dials() %>% 
  update(cost = cost(c(-8, 1))) %>% 
  grid_regular(levels = 2)


