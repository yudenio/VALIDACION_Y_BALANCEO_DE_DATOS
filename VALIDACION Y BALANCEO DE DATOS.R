#----------------------------------#
#      CURSO: MACHINE LEARNING     #
#  VALIDACION Y BALANCEO DE DATOS  #    
#      Mg. Jesús Salinas Flores    # 
#      jsalinas@lamolina.edu.pe    #
#----------------------------------#

rm(list = ls())
graphics.off()
cat("\014")
options(scipen = 999)  # Eliminar la notación científica
options(digits = 3)    # El número de decimales

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

library(pacman)
p_load(caret, caTools, ggplot2, MLmetrics, yardstick, 
       dplyr, pROC, tidyr, performanceEstimation)       


#------------------------------------------------------------#
#  Predicción de fuga de clientes de una entidad financiera  #
#------------------------------------------------------------#
# Se tienen datos de un año del producto CTS (compensación por
# tiempo de servicios) de 5533 clientes en todas las agencias
# de una entidad financiera. 
#
# Variables predictoras:
# Tasa      	      Tasa de interés de la cuenta CTS
# Saldo_soles	      Monto de Saldo de la cuenta CTS, en Soles.
# Edad	            Edad del cliente en años
# EstadoCivil	      Estado Civil: Div.Sol.Viu = Divorciado, 
#                                               Soltero y Viudo y 
#                                 Cas.Conv = Casado, Conviviente
# Región	          Zona a la que pertenece el cliente: 
#                   NORTE.SUR, ORIENTE, CENTRO, LIMA_CALLAO
# CrossSell     	  Número de productos vigentes con el banco, 
#                   tanto pasivos o activos
# Ratio.Ant         Ratio Ant_Cts / Ant_Banco  
#     Ant_Banco	    Tiempo de antigüedad del cliente (en meses)
#     Ant_Cts	      Tiempo de antigüedad de la cuenta CTS (en meses)
#
# Variable dependiente:
# Fuga              0 = cliente no fugado, 1 = cliente fugado


# Lectura de datos --------------------------------------------
datos.cts <- read.table("fuga_desbalanceada_cts.csv",
                        sep = ";", dec = ",",
                        header = T, stringsAsFactors = T)

str(datos.cts)

datos.cts$Id <- NULL

# Convertir a factor y recodificar la variable Fuga
datos.cts$Fuga          <- as.factor(datos.cts$Fuga)
levels(datos.cts$Fuga)  <- c("No_Fuga", "Si_Fuga")

addmargins(table(datos.cts$Fuga))
round(prop.table(table(datos.cts$Fuga)) * 100, 2)

contrasts(datos.cts$Fuga)

# Para cambiar la categoría de referencia (v predictoras)
datos.cts$Fuga <- relevel(datos.cts$Fuga, ref = "No_Fuga") # ref = Negativo = 0
contrasts(datos.cts$Fuga)

# Selección de muestra de entrenamiento (80%) y de evaluación (20%) ----
library(caret)
set.seed(2024) 
index   <- createDataPartition(datos.cts$Fuga,   # Es el target 
                               p = 0.8, list = FALSE)

imbal_train    <- datos.cts[ index, ]  # 4427 
imbal_testing  <- datos.cts[-index, ]  # 1106 (no se balancea)

addmargins(table(datos.cts$Fuga))
round(prop.table(table(datos.cts$Fuga)) * 100, 2)

addmargins(table(imbal_train$Fuga))
round(prop.table(table(imbal_train$Fuga)) * 100, 2)

addmargins(table(imbal_testing$Fuga))
round(prop.table(table(imbal_testing$Fuga)) * 100, 2)

# 1. Sin balancear (imbal_train) ------------------------------
# 2. Undersampling (under_train) ------------------------------
set.seed(2024)
under_train <- downSample(x = imbal_train[, c(1:7)], # predictoras
                          y = imbal_train$Fuga,      # target
                          yname = "Fuga") # Nombre columna target

addmargins(table(under_train$Fuga))

# 3. OverSampling (over_train) --------------------------------
set.seed(2024)
over_train <- upSample(x = imbal_train[, c(1:7)],
                       y = imbal_train$Fuga,
                       yname = "Fuga")

addmargins(table(over_train$Fuga))

# 4. SMOTE (smote_train) --------------------------------------
library(performanceEstimation)        # Data Mining with R - Luis Torgo
set.seed(2024)
smote_train <- smote(Fuga ~ ., 
                     data = imbal_train, 
                     perc.over  = 2,    # SMOTE          (clase minoritaria)
                     perc.under = 1.5)  # Undersampling  (clase mayoritaria)                    

addmargins(table(imbal_train$Fuga))
addmargins(table(smote_train$Fuga))



# Entrenar el modelo con validación cruzada con v = 10 --------
# Usar como indicador el Accuracy

# Relación de modelos 
library(caret)   # Max Kuhn - library(tidymodels)
names(getModelInfo())

# Método de Validación Cruzada con k = 10 para todos los modelos  
ctrl <- trainControl(method = "cv", number = 10)

# 1. Modelo con los datos originales (desbalanceados) ---------
set.seed(2024)
modelo_imbal <- train(Fuga ~ ., 
                      data = imbal_train, 
                      method = "glm", family = "binomial", 
                      trControl = ctrl,
                      metric = "Accuracy")

modelo_imbal

modelo_imbal$resample

mean(modelo_imbal$resample$Accuracy)

modelo_imbal$finalModel

summary(modelo_imbal)

# 2. Modelo con los datos balanceados (undersampling) ---------
set.seed(2024)
modelo_under  <- train(Fuga ~ ., 
                       data = under_train, 
                       method = "glm", family = "binomial", 
                       trControl = ctrl, 
                       metric = "Accuracy")

modelo_under

modelo_under$resample

mean(modelo_under$resample$Accuracy)

# 3. Modelo con los datos balanceados (oversampling) ----------
set.seed(2024)
modelo_over    <- train(Fuga ~ ., 
                        data = over_train, 
                        method = "glm", family = "binomial", 
                        trControl = ctrl, 
                        metric = "Accuracy")

modelo_over

modelo_over$resample

mean(modelo_over$resample$Accuracy)

# 4. Modelo con los datos balanceados (SMOTE) -----------------
set.seed(2024)
modelo_smote      <- train(Fuga ~ ., 
                           data = smote_train, 
                           method = "glm", family = "binomial", 
                           trControl = ctrl, 
                           metric = "Accuracy")

modelo_smote

modelo_smote$resample

mean(modelo_smote$resample$Accuracy)

# Comparando la muestras para los cuatro modelos --------------
modelos  <- list(sin_balancear = modelo_imbal,
                      under    = modelo_under,
                      over     = modelo_over,
                      SMOTE    = modelo_smote)

comparacion_modelos <- resamples(modelos)
summary(comparacion_modelos)

dotplot(comparacion_modelos, metric = "Accuracy")

bwplot(comparacion_modelos, metric = "Accuracy")

# Predicción de los modelos en la data testing ----------------

# 1. Predicción del modelo_orig en la data testing ------------
proba.modelo_imbal <- predict(modelo_imbal,
                              newdata = imbal_testing, 
                              type = "prob")
head(proba.modelo_imbal)
proba.modelo_imbal <- proba.modelo_imbal[, 2]
head(proba.modelo_imbal)

# Clase se predice con un punto de corte (umbral) de 0.5
clase.modelo_imbal <- predict(modelo_imbal,
                              newdata = imbal_testing)
head(clase.modelo_imbal)

tabla1  <- table(Predicho = clase.modelo_imbal,
                 Real     = imbal_testing$Fuga)
                 
addmargins(tabla1)

result1 <- caret::confusionMatrix(clase.modelo_imbal,
                                  imbal_testing$Fuga,
                                  positive = "Si_Fuga")

result1
result1$byClass["Sensitivity"] 
result1$byClass["Specificity"] 
result1$overall["Accuracy"]
result1$byClass["Balanced Accuracy"]

# Curva ROC y AUC
library(caTools)
colAUC(proba.modelo_imbal,
       imbal_testing$Fuga,
       plotROC = TRUE) -> auc1
abline(0, 1, col = "red")
auc1

# Log-Loss
library(MLmetrics)
real <- as.numeric(imbal_testing$Fuga)  # El target
real <- ifelse(real == 2, 1, 0)
LogLoss(proba.modelo_imbal, real)  -> logloss1
logloss1


# 2. Predicción del modelo_under en la data testing -----------
proba.modelo_under <- predict(modelo_under,
                              newdata = imbal_testing, 
                              type = "prob")
head(proba.modelo_under)
proba.modelo_under <- proba.modelo_under[, 2]
head(proba.modelo_under)

clase.modelo_under <- predict(modelo_under,
                              newdata = imbal_testing )
head(clase.modelo_under)

tabla2  <- table(Predicho = clase.modelo_under,
                 Real     = imbal_testing$Fuga)

addmargins(tabla2)

result2 <- caret::confusionMatrix(clase.modelo_under,
                                  imbal_testing$Fuga,
                                  positive = "Si_Fuga")

result2
result2$byClass["Sensitivity"] 
result2$byClass["Specificity"] 
result2$overall["Accuracy"]
result2$byClass["Balanced Accuracy"]

# Curva ROC y AUC
library(caTools)
colAUC(proba.modelo_under,
       imbal_testing$Fuga,
       plotROC = TRUE) -> auc2
abline(0, 1, col = "red")
auc2

# Log-Loss
real <- as.numeric(imbal_testing$Fuga)
real <- ifelse(real == 2, 1, 0)
LogLoss(proba.modelo_under, real) -> logloss2
logloss2

# 3. Predicción del modelo_over en la data testing ------------
proba.modelo_over <- predict(modelo_over, 
                             newdata = imbal_testing, 
                             type = "prob")
head(proba.modelo_over)
proba.modelo_over <- proba.modelo_over[, 2]
head(proba.modelo_over)

clase.modelo_over <- predict(modelo_over,
                             newdata = imbal_testing )
head(clase.modelo_over)

tabla3  <- table(Predicho = clase.modelo_over,
                 Real     = imbal_testing$Fuga)

addmargins(tabla3)

result3 <- caret::confusionMatrix(clase.modelo_over,
                                  imbal_testing$Fuga,
                                  positive = "Si_Fuga")

result3
result3$byClass["Sensitivity"] 
result3$byClass["Specificity"] 
result3$overall["Accuracy"]
result3$byClass["Balanced Accuracy"]

# Curva ROC y AUC
library(caTools)
colAUC(proba.modelo_over,
       imbal_testing$Fuga,
       plotROC = TRUE) -> auc3
abline(0, 1, col = "red")
auc3

# Log-Loss
real <- as.numeric(imbal_testing$Fuga)
real <- ifelse(real == 2, 1, 0)
LogLoss(proba.modelo_over, real) -> logloss3
logloss3


# 4. Predicción del modelo_smote en la data testing -----------
proba.modelo_smote <- predict(modelo_smote, 
                              newdata = imbal_testing, 
                              type = "prob")
head(proba.modelo_smote)
proba.modelo_smote <- proba.modelo_smote[, 2]
head(proba.modelo_smote)

clase.modelo_smote <- predict(modelo_smote,
                              newdata = imbal_testing )
head(clase.modelo_smote)

tabla4  <- table(Predicho = clase.modelo_smote,
                 Real     = imbal_testing$Fuga)

addmargins(tabla4)

result4 <- caret::confusionMatrix(clase.modelo_smote,
                                  imbal_testing$Fuga,
                                  positive = "Si_Fuga")

result4
result4$byClass["Sensitivity"] 
result4$byClass["Specificity"] 
result4$overall["Accuracy"]
result4$byClass["Balanced Accuracy"]

# Curva ROC y AUC
library(caTools)
colAUC(proba.modelo_smote,
       imbal_testing$Fuga,
       plotROC = TRUE) -> auc4
abline(0, 1, col = "red")
auc4

# Log-Loss
real <- as.numeric(imbal_testing$Fuga)
real <- ifelse(real == 2, 1, 0)
LogLoss(proba.modelo_smote,real) -> logloss4
logloss4

# 5. Encontrando punto de corte óptimo (umbral), modelo_orig ----
proba.modelo_umbral1 <- predict(modelo_imbal,
                                newdata = imbal_testing, 
                                type = "prob")
proba.modelo_umbral1 <- proba.modelo_umbral1[, 2]
head(proba.modelo_umbral1)

# Área bajo la curva
library(pROC)
roc1 <- roc(imbal_testing$Fuga, proba.modelo_umbral1)
plot(roc1, legacy.axes = TRUE, print.thres = TRUE)
roc1$auc

# Punto de corte óptimo con mayor
# (sensibilidad + especificidad) usando pROC
# Si se usa best, por defecto max(sensibilidad + especificidad)
plot.roc(imbal_testing$Fuga, proba.modelo_umbral1, 
         xlab = "1 - Especificidad", 
         ylab = "Sensibilidad", 
         legacy.axes = TRUE,
         col = "blue",
         max.auc.polygon = TRUE,
         auc.polygon = TRUE,
         auc.polygon.col = "lightblue",
         print.auc = TRUE,
         print.thres = TRUE)

pROC::coords(roc1, "best",
             ret = c("threshold", "specificity", 
                     "sensitivity", "accuracy"))

umbral1 <- pROC::coords(roc1, "best")
umbral1

umbral1$threshold

# Prediciendo la clase con el punto de corte óptimo
clase.modelo_umbral1 <- ifelse(proba.modelo_umbral1 >= umbral1$threshold,
                              "Si_Fuga", "No_Fuga")
clase.modelo_umbral1 <- factor(clase.modelo_umbral1 ,
                               levels = c("No_Fuga", "Si_Fuga")) 
# Se deben especificar los levels, por defecto lo asigna 
# alfabéticamente

contrasts(clase.modelo_umbral1)

tabla5  <- table(Predicho = clase.modelo_umbral1,
                 Real     = imbal_testing$Fuga)

addmargins(tabla5)

result5 <- caret::confusionMatrix(clase.modelo_umbral1,
                                  imbal_testing$Fuga,
                                  positive = "Si_Fuga")

result5
result5$byClass["Sensitivity"] 
result5$byClass["Specificity"] 
result5$overall["Accuracy"]
result5$byClass["Balanced Accuracy"]

# Curva ROC y AUC
library(caTools)
colAUC(proba.modelo_umbral1,
       imbal_testing$Fuga,
       plotROC = TRUE) -> auc5
abline(0, 1, col = "red")
auc5

# Log-Loss
real <- as.numeric(imbal_testing$Fuga)
real <- ifelse(real == 2, 1, 0)
LogLoss(proba.modelo_umbral1, real) -> logloss5
logloss5

# Resumiendo los 5 modelos ------------------------------------
modelo        <- c("Sin Balancear","Undersampling",
                   "Oversampling","SMOTE",
                   "Sin balancear con umbral")

umbral <- c(0.5, 0.5,
            0.5, 0.5,
            umbral1$threshold)

sensibilidad  <- c(result1$byClass["Sensitivity"],
                   result2$byClass["Sensitivity"],
                   result3$byClass["Sensitivity"],
                   result4$byClass["Sensitivity"],
                   result5$byClass["Sensitivity"])

especificidad <- c(result1$byClass["Specificity"],
                   result2$byClass["Specificity"],
                   result3$byClass["Specificity"],
                   result4$byClass["Specificity"],
                   result5$byClass["Specificity"])

accuracy      <- c(result1$overall["Accuracy"],
                   result2$overall["Accuracy"],
                   result3$overall["Accuracy"],
                   result4$overall["Accuracy"],
                   result5$overall["Accuracy"])

accuracy_bal    <- c(result1$byClass["Balanced Accuracy"],
                     result2$byClass["Balanced Accuracy"],
                     result3$byClass["Balanced Accuracy"],
                     result4$byClass["Balanced Accuracy"],
                     result5$byClass["Balanced Accuracy"])

logloss       <- c(logloss1, logloss2, logloss3,
                   logloss4, logloss5)

auc           <- c(auc1, auc2, auc3, auc4, auc5)

comparacion <- data.frame(modelo,
                          umbral,
                          sensibilidad,
                          especificidad,
                          accuracy,
                          accuracy_bal,
                          auc,
                          logloss)

View(comparacion)

#________________________
# ________ \\|// ________
# ________( o o ) _______ 
# ___oo0____(_)____Ooo___
#       Ejercicio 1      #
# Añada a la comparación los modelos con umbral
# para undersampling, oversampling y SMOTE