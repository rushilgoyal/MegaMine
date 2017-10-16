#'Analyse a Dataset
#'
#'Takes in a Dataset and name of response variable
#'
#'Produces a comparison table across Data Mining algorithms
#'@param mydata Dataset, t Response_Variable
#'@return Comparison Table
#'@export

megaLearn <- function(mydata,t){

  require(caret)
  require(glmnet)
  require(dummies)
  require(devtools)
  require(MASS)
  require(gbm)
  require(e1071)
  require(rpart)
  require(kernlab)
  require(miscTools)

  ### Defining function that calculates accuracy, precision, recall and accuracy
  evaluate <- function(confusion_matrix){

    n = sum(confusion_matrix) # number of instances
    diag = diag(confusion_matrix) # number of correctly classified instances per class
    accuracy = sum(diag) / n

    nc = nrow(confusion_matrix) # number of classes
    rowsums = apply(confusion_matrix, 1, sum) # number of instances per class
    colsums = apply(confusion_matrix, 2, sum) # number of predictions per class
    precision = diag / colsums
    recall = diag / rowsums
    f1 = 2 * precision * recall / (precision + recall)
    data.frame(accuracy, precision, recall, f1)
  }

  ### Defining function that calculates model accuracy
  acc <- function(model, x,y ){
    predict <- predict(model, x)
    cm <- as.matrix(confusionMatrix(predict, y))
    accuracy <- evaluate(cm)$accuracy[1]
    return(accuracy)
  }

  ### Defining function that calculates r-square, mse
  r2 <- function(model,x,y){
    r2 <- rSquared(y, y - predict(model, x))
    return (r2[1])
  }

  mse <- function(model,x,y){
    mse <- mean((y - predict(model, x))^2)
    return (mse)
  }



  ##################################### DATA PREPARATION #########################################
  ################################################################################################

  mydata <- na.omit(mydata) # Removing missing values
  ### Setting the response variable as last column
  col_idx <- grep( t, names(mydata))
  mydata <- mydata[c((1:ncol(mydata))[-col_idx], col_idx)]

  a <- ncol(mydata)
  colnames(mydata)[a] <- "response_variable"

  mydata2 <- mydata ## Creating a separate copy of mydata to use its non-standardized version for weighted methods

  ##### STANDARDISING mydata ####

  # Scale all numeric columns in a data frame.
  # df is your data frame

  performScaling <- TRUE  # Turn it on/off for experimentation.

  if (performScaling) {

    # Loop over each column.
    for (colName in names(mydata)) {

      # Check if the column contains numeric data.
      if(class(mydata[,colName]) == 'integer' | class(mydata[,colName]) == 'numeric') {

        # Scale this column (scale() function applies z-scaling).
        mydata[,colName] <- scale(mydata[,colName])
      }
    }
  }


  # Separating the predictor and response variables
  response_column <- mydata[a]
  predictor_columns <- mydata[-a]

  # Separating the numeric variables
  nums <- sapply(predictor_columns, is.numeric)
  numeric <- predictor_columns[ , nums]

  # Separating the categorical variables
  cat <- sapply(predictor_columns, is.factor)
  categoric <- predictor_columns[,cat]
  categoric <- as.data.frame(categoric)
  categoric<-dummy.data.frame(categoric)

  # Binding the categorical dummy coded variables, numeric and response variables
  binded <- cbind(categoric,numeric,response_column)
  binded <- as.data.frame(binded)
  b <- ncol(binded)
  colnames(binded)[b] <- "response_variable"

  #################################  Create partitions - Training and testing for Binded data #####################
  ###################################################################################################################

  set.seed(167001326)
  library(caret)
  training_index <- createDataPartition(mydata$response_variable, p=0.80, list=FALSE)
  # select 20% of the data for testing
  test_set <- mydata[-training_index,]
  # use the remaining 80% of data to training and testing the models
  train_set <- mydata[training_index,]


  # split input and output variables for modeling process
  train_x <- train_set[,1:(a-1)]
  train_y <- train_set[,a]

  # Preparing test matrix and dataframe
  test <- as.data.frame(test_set)
  x_test <- test[,1:(a-1)]
  y_test <- test[,a]

  #### Preparing binded training set
  binded_train_set <- binded[training_index,]
  binded_test_set <- binded[-training_index,]
  binded_x <- binded_train_set[,1:(b-1)]
  binded_y <- binded_train_set[,b]
  # Preparing binded test matrix and dataframe
  binded_test <- as.data.frame(binded_test_set)
  binded_x_test <- binded_test[,1:(b-1)]
  binded_y_test <- binded_test[,b]
  binded_x_test <- as.matrix(binded_x_test)



  #########################MODELING ###

  check <- is.factor(mydata$response_variable)

  if(check == FALSE){
    ## Apply Regression

    ########### INITIALISING PREPARING RESULTS TABLE #####################

    df2 <- data.frame(matrix(ncol = 8, nrow = 6))
    rownames(df2) <- c("Training_R-Square", "Training_MSE", "Testing_R_Square", "Testing_MSE", "Cross_Validated_MSE", "Cross_Validated_R_Square")
    colnames(df2) <- c("Linear_Regression", "Random Forest", "Regression_Tree", "Ridge", "Lasso" , "Neural_Net", "PCA_Regression", "Weighted_Linear_Regression" )


    ################### REGRESSION MODELING PROCESS #############################################
    ################### LINEAR REGRESSION PROCESS #############################################

    lm_fit <- lm(response_variable ~ ., data=train_set)

    lm_training_r2 <- r2(lm_fit, train_x,train_y)
    lm_training_mse <- mse(lm_fit, train_x, train_y)
    lm_testing_r2 <- r2(lm_fit, x_test,y_test)
    lm_testing_mse <- mse(lm_fit, x_test, y_test)


    df2[1,1] <- lm_training_r2
    df2[2,1] <- lm_training_mse
    df2[3,1] <- lm_testing_r2
    df2[4,1] <- lm_testing_mse


    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- lm(response_variable ~ ., data = df[trg.idx, ])
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(mydata, 5)
    lm_cv_mse <- res$overall_mean_sqr_err
    df2[5,1] <- lm_cv_mse

    control = trainControl(method="repeatedcv", number=10, repeats=3)
    model = train(response_variable~., data=mydata, method="lm", preProcess="scale", trControl=control)
    lm_cv_r2 <- model$results[1,3]
    df2[6,1] <- lm_cv_r2


    ####################### Random Forest ################################################
    ######################################################################################
    library(randomForest)

    random_forest_fit <- randomForest(response_variable ~ ., data=train_set, ntree=20, importance = TRUE, mtry = 2)

    # Training and testing R-square and mse
    library(e1071)
    rf_training_r2 <- r2(random_forest_fit, train_x,train_y)
    rf_training_mse <- mse(random_forest_fit, train_x, train_y)
    rf_testing_r2 <- r2(random_forest_fit, x_test,y_test)
    rf_testing_mse <- mse(random_forest_fit, x_test, y_test)


    df2[1,2] <- rf_training_r2
    df2[2,2] <- rf_training_mse
    df2[3,2] <- rf_testing_r2
    df2[4,2] <- rf_testing_mse

    ### K-fold cross validation


    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- randomForest(response_variable ~ ., data = df[trg.idx, ], ntree=20, imporatnce = TRUE, mtry = 2)
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(mydata, 5)
    df2[5,2] <- rf_cv_mse <- res$overall_mean_sqr_err

    control = trainControl(method="repeatedcv", number=10, repeats=3)
    model = train(response_variable~., data=mydata, method="rf", preProcess="scale", trControl=control)
    rf_cv_r2 <- model$results[1,3]
    df2[6,2] <- rf_cv_r2


    ####################################### REGRESSION TREE ##################################

    library(rpart)
    library(MASS)
    library(randomForest) #random forests
    library(gbm) #gradient boosting
    library(caret) #tune hyper-parameters


    tree.pros = rpart(response_variable~., data=train_set)
    dt_training_r2 <- r2(tree.pros, train_x,train_y)
    dt_training_mse <- mse(tree.pros, train_x, train_y)
    dt_testing_r2 <- r2(tree.pros, x_test,y_test)
    dt_testing_mse <- mse(tree.pros, x_test, y_test)


    df2[1,3] <- dt_training_r2
    df2[2,3] <- dt_training_mse
    df2[3,3] <- dt_testing_r2
    df2[4,3] <- dt_testing_mse



    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- rpart(response_variable ~ ., data = df[trg.idx, ])
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(mydata, 5)
    dt_cv_mse <- res$overall_mean_sqr_err
    df2[5,3] <- dt_cv_mse

    control = trainControl(method="repeatedcv", number=10, repeats=3)
    model = train(response_variable~., data=mydata, method="rpart", preProcess="scale", trControl=control)
    dt_cv_r2 <- model$results[1,3]
    df2[6,3] <- dt_cv_r2



    ####################### Ridge Regression ################################################
    ############################################################################



    ridge_fit <- train(response_variable ~., data = train_set,
                       method='ridge',
                       lambda = 4,
                       preProcess=c('scale', 'center'))

    ridge_training_r2 <- r2(ridge_fit, train_x,train_y)
    ridge_training_mse <- mse(ridge_fit, train_x, train_y)
    ridge_testing_r2 <- r2(ridge_fit, x_test,y_test)
    ridge_testing_mse <- mse(ridge_fit, x_test, y_test)

    df2[1,4] <- ridge_training_r2
    df2[2,4] <- ridge_training_mse
    df2[3,4] <- ridge_testing_r2
    df2[4,4] <- ridge_testing_mse



    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- train(response_variable ~., data = df[trg.idx, ],
                   method='ridge',
                   lambda = 4,
                   preProcess=c('scale', 'center'))
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(mydata, 5)
    ridge_cv_mse <- res$overall_mean_sqr_err
    df2[5,4] <- ridge_cv_mse


    #### Cross Validation Ridge
    ctrl <- trainControl(method = "repeatedcv", repeats = 5)

    ridgeFit <- train(response_variable ~ ., data = mydata,method = 'ridge',preProc = c("center", "scale"),trControl = ctrl)


    ridge_fit_cv_r2 <- ridgeFit$results[1,3]
    df2[6,4] <- ridge_fit_cv_r2


    ####################### Lasso Regression ################################################
    ############################################################################



    lasso_fit <- train(response_variable ~., data = train_set,
                       method='lasso', preProcess=c('scale', 'center'))

    lasso_training_r2 <- r2(lasso_fit, train_x,train_y)
    lasso_training_mse <- mse(lasso_fit, train_x, train_y)
    lasso_testing_r2 <- r2(lasso_fit, x_test,y_test)
    lasso_testing_mse <- mse(lasso_fit, x_test, y_test)

    df2[1,5] <- lasso_training_r2
    df2[2,5] <- lasso_training_mse
    df2[3,5] <- lasso_testing_r2
    df2[4,5] <- lasso_testing_mse



    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- train(response_variable ~., data = df[trg.idx, ],
                   method='lasso',preProcess=c('scale', 'center'))
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(mydata, 5)
    lasso_cv_mse <- res$overall_mean_sqr_err
    df2[5,5] <- lasso_cv_mse


    #### Cross Validation Ridge
    ctrl <- trainControl(method = "repeatedcv", repeats = 5)

    lassoFit <- train(response_variable ~ ., data = mydata,method = 'lasso',preProc = c("center", "scale"),trControl = ctrl)


    lasso_fit_cv_r2 <- lassoFit$results[1,3]
    df2[6,5] <- lasso_fit_cv_r2


    ####################### Neural Network ################################################
    ############################################################################
    library(nnet)
    library(caret)
    library(devtools)
    ann_fit <- nnet(response_variable ~ ., data=train_set, size=6, decay = 0.1, maxit = 1000, linout = TRUE)

    ann_training_r2 <- r2(ann_fit, train_x,train_y)
    ann_training_mse <- mse(ann_fit, train_x, train_y)
    ann_testing_r2 <- r2(ann_fit, x_test,y_test)
    ann_testing_mse <- mse(ann_fit, x_test, y_test)


    df2[1,6] <- ann_training_r2
    df2[2,6] <- ann_training_mse
    df2[3,6] <- ann_testing_r2
    df2[4,6] <- ann_testing_mse



    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- nnet(response_variable ~ ., data = df[trg.idx, ], size=6, decay = 0.1, maxit = 1000, linout = TRUE)
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(mydata, 5)
    ann_cv_mse <- res$overall_mean_sqr_err
    df2[5,6] <- ann_cv_mse

    control = trainControl(method="repeatedcv", number=10, repeats=3)
    model = train(response_variable~., data=mydata, method="nnet", preProcess="scale", trControl=control)
    ann_cv_r2 <- model$results[1,3]
    df2[6,6] <- ann_cv_r2



    ############################## PCA - REGRESSION  #####################################################

    #To carry out a principal component analysis (PCA) on a multivariate data set, the first step is often to standardise
    #the variables under study using the “scale()” function (see above).
    standardised_train <- as.data.frame(scale(binded_train_set[1:(b-1)])) # standardise the variables
    pca_train <- prcomp(standardised_train)
    result <- summary(pca_train)


    vars <- result$sdev^2
    vars <- vars/sum(vars)
    cumulative_variance <- cumsum(vars)
    p <- which(cumulative_variance > 0.80)[1]
    p
    # We select number of components that explains 80% of variance



    #add a training set with principal components
    train.data <- data.frame(pca_train$x)
    #we are interested in first p PCAs
    train.data <- train.data[,1:p]
    train.data <- cbind(train.data, response_variable = binded_train_set$response_variable)

    # Principal Component Regression
    pca_lm <- lm(response_variable ~ ., data = train.data)
    pca_lm


    #transform test into PCA
    test.data <- predict(pca_train, newdata = binded_x_test)
    test.data <- as.data.frame(test.data)

    myvars <- names(train.data) %in% c("response_variable")
    new.train.data <- train.data[!myvars]

    pca_training_r2 <- r2(pca_lm, train.data,train_y)
    pca_training_mse <- mse(pca_lm, train.data, train_y)
    pca_testing_r2 <- r2(pca_lm, test.data,y_test)
    pca_testing_mse <- mse(pca_lm, test.data, y_test)


    df2[1,7] <- pca_training_r2
    df2[2,7] <- pca_training_mse
    df2[3,7] <- pca_testing_r2
    df2[4,7] <- pca_testing_mse



    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- lm(response_variable ~ ., data = df[trg.idx, ])
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(train.data, 5)
    pca_cv_mse <- res$overall_mean_sqr_err
    df2[5,7] <- pca_cv_mse

    control = trainControl(method="repeatedcv", number=10, repeats=3)
    model = train(response_variable~., data=train.data, method="lm", preProcess="scale", trControl=control)
    pca_cv_r2 <- model$results[1,3]
    df2[6,7] <- pca_cv_r2



    ##############################################WEIGHTED METHODS ##################################################################################################

    ################## PREPARING WEIGHTED DATASET ####################
    ##################################################################


    # Separating the predictor and response variables
    response_column <- mydata2[a]
    predictor_columns <- mydata2[-a]

    # Separating the numeric variables
    nums <- sapply(predictor_columns, is.numeric)
    numeric <- predictor_columns[ , nums]
    numeric <- as.data.frame(numeric)

    # Separating the categorical variables
    cat <- sapply(predictor_columns, is.factor)
    categoric <- predictor_columns[,cat]
    categoric <- as.data.frame(categoric)
    categoric<-dummy.data.frame(categoric)


    cat1 <- sapply(predictor_columns, is.factor)
    categoric1 <- predictor_columns[,cat1]
    head(categoric1)

    categoric1 <- cbind(categoric1, mydata2$response_variable)
    k <- ncol(categoric1)-1
    categorical_pvalue_vector <- vector(mode="numeric", length=k)

    colnames(categoric1)[k+1] <- "response_variable"

    ########################### CATEGORICAL WEIGHTS ############################
    ## ANOVA for categorical vs contionuous
    for (i in c(1:k) ) {
      categorical_pvalue_vector[i] = summary(aov(response_variable ~ categoric1[,i], data=categoric1))[[1]][["Pr(>F)"]][1]
    }

    categorical_pvalue_vector

    w1 = -log(categorical_pvalue_vector*length(categorical_pvalue_vector)/rank(categorical_pvalue_vector))
    # Make all negatives 0
    w1[which(w1<0)]=0


    l = 1
    for( m in 1:k){
      g <- nlevels(categoric1[,m])
      if (g ==2){
        for(j in l:(l+g-1))
        {categoric[j] <- categoric[j]*w1[m]}

        l <- l+g
      }
      else{
        for(j in l:(l+g-2))
        {categoric[j] <- categoric[j]*w1[m]}
        l <- l+g-1
      }
    }




    ##################### NUMERICAL WEIGHTS ###########################
    numeric1 <- predictor_columns[,nums]
    numeric1 <- cbind(numeric1, mydata2$response_variable)
    k <- ncol(numeric1)-1
    pvalue_vector <- vector(mode="numeric", length=k)

    colnames(numeric1)[k+1] <- "response_variable"

    ### Correlation_test for continuous vs continuous
    for (i in c(1:k) ) {
      pvalue_vector[i] <- cor.test(numeric1[,i] , numeric1[,(k+1)])$p.value
    }

    w2 = -log(pvalue_vector*length(pvalue_vector)/rank(pvalue_vector))
    w2[which(w2<0)]=0

    for(i in c(1:k))
    {
      numeric[,i] <- numeric[,i]*w2[i]
    }

    weighted_bind <- cbind(categoric, numeric, response_column)
    weighted_bind <- as.data.frame(weighted_bind)
    u <- ncol(weighted_bind)
    colnames(weighted_bind)[u] <- "response_variable"

    ### Splitting Weighted dataset into weighted train and weighted test
    weighted_train <- weighted_bind[training_index,]
    weighted_test <- weighted_bind[-training_index,]

    weighted_x <- weighted_train[,1:(u-1)]
    weighted_y <- weighted_train[,u]

    weighted_x_test <- weighted_test[,1:(u-1)]
    weighted_y_test <- weighted_test[,u]


    ##############################################WEIGHTED MODELING ########################################################################

    ### WEIGHTED REGRESSION #####


    weighted_lm_fit <- lm(response_variable ~ ., data=weighted_train)

    weighted_lm_training_r2 <- r2(weighted_lm_fit, weighted_x,weighted_y)
    weighted_lm_training_mse <- mse(weighted_lm_fit, weighted_x, weighted_y)
    weighted_lm_testing_r2 <- r2(weighted_lm_fit, weighted_x_test,weighted_y_test)
    weighted_lm_testing_mse <- mse(weighted_lm_fit, weighted_x_test, weighted_y_test)


    df2[1,8] <- weighted_lm_training_r2
    df2[2,8] <- weighted_lm_training_mse
    df2[3,8] <- weighted_lm_testing_r2
    df2[4,8] <- weighted_lm_testing_mse



    ### K-fold cross validation

    rdacb.kfold.crossval.reg <- function(df, nfolds) {
      fold <- sample(1:nfolds, nrow(df), replace = TRUE)
      mean.sqr.errs <- sapply(1:nfolds,
                              rdacb.kfold.cval.reg.iter,
                              df, fold)
      list("mean_sqr_errs"= mean.sqr.errs,
           "overall_mean_sqr_err" = mean(mean.sqr.errs),
           "std_dev_mean_sqr_err" = sd(mean.sqr.errs))
    }


    rdacb.kfold.cval.reg.iter <- function(k, df, fold) {
      trg.idx <- !fold %in% c(k)
      test.idx <-  fold %in% c(k)
      mod <- lm(response_variable ~ ., data = df[trg.idx, ])
      pred <- predict(mod, df[test.idx,])
      sqr.errs <- (pred - df[test.idx, "response_variable"])^2
      mean(sqr.errs)
    }

    res <- rdacb.kfold.crossval.reg(weighted_bind, 10)
    lm_cv_mse <- res$overall_mean_sqr_err
    df2[5,8] <- lm_cv_mse

    control = trainControl(method="repeatedcv", number=10, repeats=3)
    model = train(response_variable~., data=weighted_bind, method="lm",trControl=control)  ### We keep the scaling option off for this
    weighted_cv_r2 <- model$results[1,3]
    df2[6,8] <-  weighted_cv_r2

    return (df2)

  }

  else
  {

    # RUN CLASSIFICATION MODEL

    ########### INITIALISING PREPARING RESULTS TABLE #####################

    df <- data.frame(matrix(ncol = 11, nrow = 3))
    rownames(df) <- c("Training_Accuracy", "Test_Accuracy", "Cross_Validated_Accuracy")
    colnames(df) <- c("GLMNET", "Random Forest", "SVM", "Neural_Net", "Penalized_SVM" , "Naive_Bayes", "CART", "Adaboost", "PCA_LDA", "Weighted_GLMnet", "Weighted_SVM")

    ################### CLASSIFICATION MODELING PROCESS #############################################

    ########################## GLM NET ##############################################
    #################################################################################

    binded_x <- as.matrix(binded_x)
    binded_y <- as.matrix(binded_y)
    binded_y <- as.factor(binded_y)

    #glm_fit <- glmnet(x, y, family="binomial", alpha=0.8, lambda=0.001)   # use alpha = 0.8 for most cases # for lasso alpha = 1
    if (nlevels(binded$response_varaible == 2)){
      glm_fit <- glmnet(binded_x, binded_y, family="binomial", alpha=0.8, lambda=0.001)   # use alpha = 0.8 for most cases # for lasso alpha = 1
    }

    else{
      glm_fit <- glmnet(binded_x, binded_y, family="multinomial", alpha=0.8, lambda=0.001)   # use alpha = 0.8 for most cases # for lasso alpha = 1
    }

    summary(glm_fit)

    ### TRAINING ACCURACY
    glm_predict <- predict(glm_fit, type = 'class', binded_x) # The predict function in glmnet takes matrix input and not data frame
    glm_training_cm <- as.matrix(confusionMatrix(glm_predict, binded_y))
    glm_training_accuracy <- evaluate(glm_training_cm)$accuracy[1]

    df[1,1] <- glm_training_accuracy

    #### TESTING ACCURACY ##
    glm_predict <- predict(glm_fit, type = 'class', binded_x_test) # The predict function in glmnet takes matrix input and not data frame
    glm_testing_cm <- as.matrix(confusionMatrix(glm_predict, binded_y_test))
    glm_testing_acccuracy <- evaluate(glm_testing_cm)$accuracy[1]

    df[2,1] <- glm_testing_acccuracy



    #### CROSS VALIDATED ACCURACY ##
    classes <- mydata[, "response_variable"]
    q <- createDataPartition(classes, p = 0.8, list = FALSE)
    cs_data_train <- mydata[q, ]
    cs_data_test <- mydata[-q, ]
    glmnet_grid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                               lambda = seq(.01, .2, length = 20))
    glmnet_ctrl <- trainControl(method = "cv", number = 10)
    glmnet_fit <- train(response_variable ~ ., data = cs_data_train,
                        method = "glmnet",
                        preProcess = c("center", "scale"),
                        tuneGrid = glmnet_grid,
                        trControl = glmnet_ctrl)


    pred_classes <- predict(glmnet_fit, newdata = cs_data_test)
    table(pred_classes)

    cn <- as.matrix(confusionMatrix(pred_classes, cs_data_test$response_variable))
    glmnet_cv_accuracy <- evaluate(cn)$accuracy[1]

    df[3,1] <- glmnet_cv_accuracy


    ####################### Random Forest ################################################
    ######################################################################################
    library(randomForest)

    random_forest_fit <- randomForest(response_variable ~ ., data=train_set, ntree=20, imporatnce = TRUE, mtry = 2)

    # Training Accuracy
    library(e1071)
    random_forest_training_accuracy <- acc(random_forest_fit, train_x, train_y)
    df[1,2] <- random_forest_training_accuracy

    # Testing Accuracy
    x_test <- as.data.frame(x_test)
    random_forest_testing_accuracy <- acc(random_forest_fit, x_test, y_test)
    df[2,2] <-random_forest_testing_accuracy

    #### Cross validated accuracy
    # define training control
    train_control <- trainControl(method="repeatedcv", number=10, repeats = 3)
    metric <- "Accuracy"
    mtry <- sqrt(a-1)
    # fix the parameters of the algorithm
    tunegrid <- expand.grid(.mtry=mtry)
    # train the model
    random_forest_model <- train(response_variable~., data=mydata, metric=metric, trControl=train_control, method="rf", tuneGrid=tunegrid)
    random_forest_cv_accuracy <- random_forest_model$results[2][,1]
    df[3,2] <- random_forest_cv_accuracy



    ####################### Support Vector Machine ################################################
    ###############################################################################################
    library(kernlab)
    svm_fit <- svm(response_variable ~ .,  method="class", data=train_set)
    # Predicting response variable

    # Training Accuracy
    svm_training_accuracy <- acc(svm_fit, train_x, train_y)
    df[1,3] <- svm_training_accuracy
    # Testing Accuracy
    x_test <- as.data.frame(x_test)
    svm_testing_accuracy <- acc(svm_fit, x_test, y_test)
    df[2,3] <- svm_testing_accuracy
    #### Cross validated accuracy
    # define training control
    train_control <- trainControl(method="repeatedcv", number=10, repeats = 3)
    metric <- "Accuracy"
    # fix the parameters of the algorithm
    svm_model <- train(response_variable~., data=mydata, metric=metric, trControl=train_control, method="svmRadial")
    svm_model_cv_accuracy <- svm_model$results[1,3]
    df[3,3] <- svm_model_cv_accuracy



    ####################### Naive Bayes ################################################
    ####################################################################################
    library(e1071)
    naive_bayes_fit <- naiveBayes(response_variable ~ ., data=train_set, k= 5)

    # Testing Accuracy
    nb_training_accuracy <- acc(naive_bayes_fit, train_x, train_y)
    df[1,4] <- nb_training_accuracy

    # Testing Accuracy
    x_test <- as.data.frame(x_test)
    nb_testing_accuracy <- acc(naive_bayes_fit, x_test, y_test)
    df[2,4] <- nb_testing_accuracy

    #### Cross validated accuracy
    # define training control
    train_control <- trainControl(method="repeatedcv", number=10, repeats = 3)
    metric <- "Accuracy"
    # fix the parameters of the algorithm
    nb_model <- train(response_variable~., data=mydata, metric=metric, trControl=train_control, method="nb")
    nb_model_cv_accuracy <- nb_model$results[2,4]
    df[3,4] <- nb_model_cv_accuracy



    ####################### CART ################################################
    ##############################################################################

    library(rpart)
    classification_tree_fit = rpart(response_variable ~ . , data = train_set, method = "class" )

    ### Saving the decision tree
    png("classification_tree.png", width = 1200, height = 800)
    post(classification_tree_fit, file = "", title. = "Classification Tree",bp = 18)
    dev.off()

    # Training Accuracy
    cart_predict <- predict(classification_tree_fit, train_x, type = 'class')
    cart_training_cm <- as.matrix(confusionMatrix(cart_predict, train_y))
    cart_training_accuracy <- evaluate(cart_training_cm)$accuracy[1]
    df[1,5] <- cart_training_accuracy

    # Testing Accuracy
    x_test <- as.data.frame(x_test)
    cart_predict <- predict(classification_tree_fit, x_test, type = 'class')
    cart_testing_cm <- as.matrix(confusionMatrix(cart_predict, y_test))
    cart_testing_accuracy <- evaluate(cart_testing_cm)$accuracy[1]
    df[2,5] <- cart_testing_accuracy

    ##### Cross Validation

    train_control <- trainControl(method="repeatedcv", number=10, repeats = 3)
    cart_model<- train(response_variable~., data=mydata, trControl=train_control, method="rpart")
    cart_cv_accuracy <- cart_model$results[1,2]
    df[3,5] <- cart_cv_accuracy


    ####################### ANN ################################################
    ############################################################################
    library(nnet)
    library(caret)
    library(devtools)
    ann_fit <- nnet(response_variable ~ ., data=train_set, size=1, decay = 0.0001, maxit = 500)


    # Training Accuracy
    ann_predict <- predict(ann_fit, train_x, type = 'class')
    ann_training_cm <- as.matrix(confusionMatrix(ann_predict, train_y))
    ann_training_accuracy <- evaluate(ann_training_cm)$accuracy[1]
    df[1,6] <- ann_training_accuracy

    # Testing Accuracy
    x_test <- as.data.frame(x_test)
    ann_predict <- predict(ann_fit, x_test, type = 'class')
    ann_testing_cm <- as.matrix(confusionMatrix(ann_predict, y_test))
    ann_testing_accuracy <- evaluate(ann_testing_cm)$accuracy[1]
    df[2,6] <- ann_testing_accuracy

    ##### Cross Validation

    train_control <- trainControl(method="repeatedcv", number=10, repeats = 3)
    ann_model<- train(response_variable~., data=mydata, trControl=train_control, method="nnet")
    ann_cv_accuracy <- cart_model$results[1,2]
    df[3,6] <- ann_cv_accuracy




    #################### PENALIZED SVM #########################
    ############################################################


    n_level <- nlevels(train_set$response_variable)

    if(n_level == 2){

      labels <- train_set[ , a]
      labels <- as.numeric(as.character(labels))
      labels <- as.data.frame(labels)
      labels[labels == 0] <- -1

      test_labels <- test_set[ , a]
      test_labels <- as.numeric(as.character(test_labels))
      test_labels <- as.data.frame(test_labels)
      test_labels[test_labels == 0] <- -1

      library(penalizedSVM)
      Lambda.scad <- seq(0.01 ,0.05, 0.01)
      labels <- as.matrix(labels)
      penalized_svm_model <- svm.fs(binded_x, labels,fs.method = "scad", lambda1.set = Lambda.scad)
      test_labels <- as.matrix(test_labels)


      ### Training accuracy
      penalized_svm_training_pred <- predict(penalized_svm_model, binded_x,labels )
      penalized_svm_training_cm <- penalized_svm_training_pred$tab
      penalized_svm_training_accuracy <- evaluate(penalized_svm_training_cm)$accuracy[1]
      df[1,7] <- penalized_svm_training_accuracy


      ### Testing accuracy
      penalized_svm_testing_pred <- predict(penalized_svm_model, binded_x_test,test_labels )
      penalized_svm_testing_cm <- penalized_svm_testing_pred$tab
      penalized_svm_testing_accuracy <- evaluate(penalized_svm_testing_cm)$accuracy[1]
      df[2,7] <- penalized_svm_testing_accuracy

      #### Cross Validated accuracy
      cv_results <- vector()  # a vector to store cross validated results for label i
      rushil <- as.data.frame(cbind(binded_x, labels))
      C <- ncol(rushil)
      colnames(rushil)[C] <- "response_variable"
      folds <- createFolds(rushil$response_variable, k = 5, list = TRUE, returnTrain = TRUE)

      for(j in 1:5){
        goyal <- rushil[folds[[j]],]
        A <- goyal[,1:(C-1)]
        A <- as.matrix(A)
        B <- goyal[,C]
        B <- as.matrix(B)
        B <- as.factor(B)

        penalized_cv_svm_model <- svm.fs(A, B,fs.method = "scad", lambda1.set = Lambda.scad)

        shreya <- rushil[-folds[[j]],]
        I <- shreya[,1:(C-1)]
        I <- as.matrix(I)
        J <- shreya[,C]
        J <- as.matrix(J)
        J <- as.factor(J)

        penalized_cv_wah <- predict(penalized_cv_svm_model, I, J)
        penalized_cv_confusion_tab <- penalized_cv_wah$tab
        penalized_cv_accuracy <- evaluate(penalized_cv_confusion_tab)$accuracy[1]

        cv_results <- c(cv_results,penalized_cv_accuracy)
      }
      support_cv_accuracy <- mean(cv_results)
      df[3,7] <- support_cv_accuracy
    }


    ##################### The else condition of svm (if there are more than 2 levels) ######################
    else{
      labels <- train_set[ , a]
      labels <- as.data.frame(labels)
      labels<-dummy.data.frame(labels)
      labels[labels == 0] <- -1


      ### Separating all testing labels
      test_labels <- test_set[ , a]
      test_labels <- as.data.frame(test_labels)
      test_labels<-dummy.data.frame(test_labels)
      test_labels[test_labels == 0] <- -1


      penalized_training_results <- vector()
      penalized_testing_results <- vector()
      library(penalizedSVM)
      Lambda.scad <- seq(0.01 ,0.05, 0.01)

      c <- ncol(labels)
      for (i in 1:c){
        penalized_svm_model <- svm.fs(binded_x, labels[,i],fs.method = "scad", lambda1.set = Lambda.scad)

        ## Training Accuracy
        label_train <- as.matrix(labels[,1])
        training_wah <- predict(penalized_svm_model, binded_x,label_train)
        training_confusion_tab <- training_wah$tab
        penalized_training_accuracy <- evaluate(training_confusion_tab)$accuracy[1]
        penalized_training_results <- c(penalized_training_results,penalized_training_accuracy)

        ## Testing Accuracy
        label_test <- as.matrix(test_labels[,i])
        testing_wah <- predict(penalized_svm_model, binded_x_test,label_test )
        testing_confusion_tab <- testing_wah$tab
        penalized_testing_accuracy <- evaluate(testing_confusion_tab)$accuracy[1]
        penalized_testing_results <- c(penalized_testing_results,penalized_testing_accuracy)
      }

      penalized_model_training_accuracy <- mean(penalized_training_results, na.rm=TRUE)
      df[1,7] <- penalized_model_training_accuracy
      penalized_model_testing_accuracy <- mean(penalized_testing_results, na.rm=TRUE)
      df[2,7] <- penalized_model_testing_accuracy

      ########################## Penalized Cross Validation (More than 2 levels) #####################################

      penalized_cv_final_results <- vector()

      for (i in 1:c){  ## For every label column

        cv_results <- vector()  # a vector to store cross validated results for label i
        rushil <- as.data.frame(cbind(binded_x, labels[,i]))
        C <- ncol(rushil)
        colnames(rushil)[C] <- "response_variable"
        folds <- createFolds(rushil$response_variable, k = 5, list = TRUE, returnTrain = TRUE)

        for(j in 1:5){
          goyal <- rushil[folds[[j]],]
          A <- goyal[,1:(C-1)]
          A <- as.matrix(A)
          B <- goyal[,C]
          B <- as.matrix(B)
          B <- as.factor(B)

          penalized_cv_svm_model <- svm.fs(A, B,fs.method = "scad", lambda1.set = Lambda.scad)

          shreya <- rushil[-folds[[j]],]
          I <- shreya[,1:(C-1)]
          I <- as.matrix(I)
          J <- shreya[,C]
          J <- as.matrix(J)
          J <- as.factor(J)

          penalized_cv_wah <- predict(penalized_cv_svm_model, I, J)
          penalized_cv_confusion_tab <- penalized_cv_wah$tab
          penalized_cv_accuracy <- evaluate(penalized_cv_confusion_tab)$accuracy[1]

          cv_results <- c(cv_results,penalized_cv_accuracy)
        }
        support_cv_accuracy <- mean(cv_results)
        penalized_cv_final_results <- c(penalized_cv_final_results,support_cv_accuracy)
      }
      penalized_cv_final_mean <- mean(penalized_cv_final_results)
      df[3,7] <- penalized_cv_final_mean
    }





    ################## ADABOOST ############################# ############
    ######################################################################


    library(adabag)
    binded[1:(b-1)] <- lapply(binded[1:(b-1)], as.numeric)
    l <- length(binded[,1])
    sub <- sample(1:l,2*l/3)

    mfinal <- 10
    maxdepth <- 5
    control=rpart.control(cp=0.01)
    Vehicle.adaboost <- boosting(response_variable ~.,data=mydata[sub, ],mfinal=mfinal, coeflearn="Zhu",
                                 control=rpart.control(maxdepth=maxdepth))


    Vehicle.adaboost.pred_train <- predict.boosting(Vehicle.adaboost,newdata=mydata[sub, ])
    adaboost_training_cm <- Vehicle.adaboost.pred_train$confusion
    adaboost_training_cm <- as.matrix(adaboost_training_cm)
    adaboost_training_accuracy <- evaluate(adaboost_training_cm)$accuracy[1]
    df[1,8] <- adaboost_training_accuracy



    ###### Testing Accuracy ############
    Vehicle.adaboost.pred_test <- predict.boosting(Vehicle.adaboost,newdata=mydata[-sub, ])
    adaboost_testing_cm <- Vehicle.adaboost.pred_test$confusion
    adaboost_testing_cm <- as.matrix(adaboost_testing_cm)
    adaboost_testing_accuracy <- evaluate(adaboost_testing_cm)$accuracy[1]
    df[2,8] <- adaboost_testing_accuracy



    ##### CROSS VALIDATION ADABOOST
    adaboost_cv_fit <- boosting.cv(response_variable ~ ., data= mydata, v = 3, boos = TRUE, mfinal = 100,
                                   coeflearn = "Breiman", control)

    adaboost_cv_cm <- as.matrix(adaboost_cv_fit$confusion)
    adaboost_cv_accuracy <- evaluate(adaboost_cv_cm)$accuracy[1]
    df[3,8] <- adaboost_cv_accuracy





    ############################## LDA-PCA #####################################################

    #To carry out a principal component analysis (PCA) on a multivariate data set, the first step is often to standardise
    #the variables under study using the “scale()” function (see above).
    standardised_train <- as.data.frame(scale(binded[1:(b-1)])) # standardise the variables
    pca_train <- prcomp(standardised_train)
    result <- summary(pca_train)


    vars <- result$sdev^2
    vars <- vars/sum(vars)
    cumulative_variance <- cumsum(vars)
    p <- which(cumulative_variance > 0.80)[1]
    p

    #add a training set with principal components
    train.data <- data.frame(pca_train$x)
    #we are interested in first 5 PCAs
    train.data <- train.data[,1:p]
    train.data <- cbind(train.data, response_variable = binded$response_variable)

    #Linear Discrimiant Analysis
    library(MASS)
    model.lda <- lda(response_variable ~ ., data = train.data)
    model.lda


    ## TRAINING ACCURACY
    lda_train_prediction <- predict(model.lda, newdata = train.data)$class
    lda_pca_train_cm <- as.matrix(confusionMatrix(lda_train_prediction, train.data$response_variable))
    lda_pca_train_accuracy <- evaluate(lda_pca_train_cm)$accuracy[1]
    df[1,9] <- lda_pca_train_accuracy

    ## TESTING ACCURACY

    #transform test into PCA
    test.data <- predict(pca_train, newdata = binded_x_test)
    test.data <- as.data.frame(test.data)

    #select the first 21 components that explains 80% of the variance
    test.data <- test.data[,1:p]
    lda_test_prediction <- predict(model.lda, newdata = test.data)$class
    lda_pca_test_cm <- as.matrix(confusionMatrix(lda_test_prediction, binded_y_test))
    lda_pca_test_accuracy <- evaluate(lda_pca_test_cm)$accuracy[1]
    df[2,9] <- lda_pca_test_accuracy



    # CROSS VALIDATE THE MODEL
    # PERFORM THE LDA
    model.lda_loocv <- lda(response_variable ~ ., data = train.data, CV = TRUE)
    lda_pca_cv_cm <- as.matrix(table(model.lda_loocv$class,train.data$response_variable))
    lda_pca_accuracy <- evaluate(lda_pca_cv_cm)$accuracy[1]
    df[3,9] <- lda_pca_accuracy




    ############################## WEIGHTED METHODS ########################################################################

    ################## PREPARING WEIGHTED DATASET #################################################################

    # Separating the predictor and response variables
    response_column <- mydata2[a]
    predictor_columns <- mydata2[-a]

    # Separating the numeric variables
    nums <- sapply(predictor_columns, is.numeric)
    numeric <- predictor_columns[ , nums]
    numeric <- as.data.frame(numeric)

    # Separating the categorical variables
    cat <- sapply(predictor_columns, is.factor)
    categoric <- predictor_columns[,cat]
    categoric <- as.data.frame(categoric)
    categoric<-dummy.data.frame(categoric)


    cat1 <- sapply(predictor_columns, is.factor)
    categoric1 <- predictor_columns[,cat1]
    head(categoric1)

    categoric1 <- cbind(categoric1, mydata2$response_variable)
    k <- ncol(categoric1)-1
    categorical_pvalue_vector <- vector(mode="numeric", length=k)

    ########################### CATEGORICAL WEIGHTS ############################
    ## FISHER_test for categorical vs categorical
    for (i in c(1:k) ) {
      categorical_pvalue_vector[i] = fisher.test(table(categoric1[,i],categoric1[,(k+1)]))$p.v
    }

    w1 = -log(categorical_pvalue_vector*length(categorical_pvalue_vector)/rank(categorical_pvalue_vector))
    # Make all negatives 0
    w1[which(w1<0)]=0


    l = 1
    for( m in 1:k){
      g <- nlevels(categoric1[,m])
      if (g ==2){
        for(j in l:(l+g-1))
        {categoric[j] <- categoric[j]*w1[m]}

        l <- l+g
      }
      else{
        for(j in l:(l+g-2))
        {categoric[j] <- categoric[j]*w1[m]}
        l <- l+g-1
      }
    }


    ##################### NUMERICAL WEIGHTS ###########################
    numeric1 <- predictor_columns[,nums]
    numeric1 <- cbind(numeric1, mydata2$response_variable)
    k <- ncol(numeric1)-1
    pvalue_vector <- vector(mode="numeric", length=k)

    ### Anove_test for continuous vs categorical
    for (i in c(1:k) ) {
      p <- aov(numeric1[,i]~ numeric1[,(k+1)])
      pvalue_vector[i] = summary(p)[[1]][["Pr(>F)"]][1]
    }

    w2 = -log(pvalue_vector*length(pvalue_vector)/rank(pvalue_vector))
    # Make all negatives 0
    w2[which(w2<0)]=0

    for(i in 1:k)
    {
      numeric[,i] <- numeric[,i]*w2[i]
    }

    weighted_bind <- cbind(categoric, numeric, response_column)
    weighted_bind <- as.data.frame(weighted_bind)
    u <- ncol(weighted_bind)
    colnames(weighted_bind)[u] <- "response_variable"

    ### Splitting Weighted dataset into weighted train and weighted test
    weighted_train <- weighted_bind[training_index,]
    weighted_test <- weighted_bind[-training_index,]


    #####################################   WEIGHTED MODELING ########################################################################

    ###################################  WEIGHTED GLMNET #####################################################################
    weighted_x <- as.matrix(weighted_train[,1:(u-1)])
    weighted_y <- as.matrix(weighted_train[,u])

    weighted_x_test <- as.matrix(weighted_test[,1:(u-1)])
    weighted_y_test <- as.matrix(weighted_test[,u])

    weighted_y <- as.factor(weighted_y)
    if (nlevels(weighted_bind$response_variable == 2)){
      weighted_glm_fit <- glmnet(weighted_x, weighted_y, family="binomial", alpha=0.8, lambda=0.001, standardize = F)   # use alpha = 0.8 for most cases # for lasso alpha = 1
    }

    else{
      weighted_glm_fit <- glmnet(weighted_x, weighted_y, family="multinomial", alpha=0.8, lambda=0.001, standardize = F)
    }

    #### Training Accuracy #####
    weighted_glm_predict <- predict(weighted_glm_fit, type = 'class', weighted_x) # The predict function in glmnet takes matrix input and not data frame
    weighted_glm_training_cm <- as.matrix(confusionMatrix(weighted_glm_predict, weighted_y))
    weighted_glm_training_accuracy <- evaluate(weighted_glm_training_cm)$accuracy[1]
    df[1,10] <- weighted_glm_training_accuracy


    ### Testing Accuracy ##
    weighted_glm_test_pred <- predict(weighted_glm_fit, type = 'class', weighted_x_test)
    weighted_glm_testing_cm <- as.matrix(confusionMatrix(weighted_glm_test_pred, weighted_y_test))
    weighted_glm_testing_accuracy <- evaluate(weighted_glm_testing_cm)$accuracy[1]
    df[2,10] <- weighted_glm_testing_accuracy


    #### Cross Validation Accuracy

    classes <- weighted_bind[, "response_variable"]
    q <- createDataPartition(classes, p = 0.8, list = FALSE)
    weighted_cs_data_train <- weighted_bind[q, ]
    weighted_cs_data_test <- weighted_bind[-q, ]
    glmnet_grid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                               lambda = seq(.01, .2, length = 20))
    glmnet_weighted_ctrl <- trainControl(method = "cv", number = 10)
    glmnet_cv_weighted_fit <- train(response_variable ~ ., data = weighted_cs_data_train,
                                    method = "glmnet", tuneGrid = glmnet_grid,
                                    trControl = glmnet_weighted_ctrl)


    weighted_pred_classes <- predict(glmnet_cv_weighted_fit, newdata = weighted_cs_data_test)
    table(weighted_pred_classes)

    cn <- as.matrix(confusionMatrix(weighted_pred_classes, weighted_cs_data_test$response_variable))
    glmnet_weighted_cv_accuracy <- evaluate(cn)$accuracy[1]

    df[3,10] <- glmnet_weighted_cv_accuracy




    ##################################### WEIGHTED SVM ####################################################

    library(kernlab)

    weighted_svm_fit <- svm(response_variable ~ .,  method="class", data=weighted_train, standardize = FALSE , scale = F)
    # Predicting response variable

    # Training Accuracy
    weighted_svm_training_accuracy <- acc(weighted_svm_fit, weighted_x, weighted_y)
    df[1,11] <- weighted_svm_training_accuracy
    # Testing Accuracy
    weighted_svm_testing_accuracy <- acc(weighted_svm_fit, weighted_x_test, weighted_y_test)
    df[2,11] <- weighted_svm_testing_accuracy
    #### Cross validated accuracy
    # define training control
    train_control <- trainControl(method="repeatedcv", number=10, repeats = 3)
    metric <- "Accuracy"
    # fix the parameters of the algorithm
    weighted_cv_svm_model <- train(response_variable~., data=weighted_bind, metric=metric, trControl=train_control, method="svmRadial")
    weighted_svm_cv_accuracy <- weighted_cv_svm_model$results[1,3]
    df[3,11] <- weighted_svm_cv_accuracy

    return (df)
  }

}




























