def feature_select_wrapper():
    """
    lgm
    :param train
    :param test
    :return
    """
    # Part 1.
    print('feature_select_wrapper...')
    features = feature_names

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fse = pd.Series(0, index=features)
         
    for train_index, test_index in skf.split(X, y_split):

        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]

        model = lgb.LGBMRegressor(
                    objective = qwk_obj,
                    metrics = 'None',
                    learning_rate = 0.05,
                    max_depth = 5,
                    num_leaves = 10,
                    colsample_bytree=0.3,
                    reg_alpha = 0.7,
                    reg_lambda = 0.1,
                    n_estimators=700,
                    random_state=412,
                    extra_trees=True,
                    class_weight='balanced',
                    verbosity = - 1)

        predictor = model.fit(X_train_fold,
                              y_train_fold,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                              eval_metric=quadratic_weighted_kappa,
                              callbacks=callbacks)
        models.append(predictor)
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + a
        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)
        f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

#         cm = confusion_matrix(y_test_fold_int, predictions_fold, labels=[x for x in range(1,7)])

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                                       display_labels=[x for x in range(1,7)])
#         disp.plot()
#         plt.show()
        print(f'F1 score across fold: {f1_fold}')
        print(f'Cohen kappa score across fold: {kappa_fold}')

        fse += pd.Series(predictor.feature_importances_, features)  
    
    # Part 4.
    feature_select = fse.sort_values(ascending=False).index.tolist()[:13000]
    print('done')
    return feature_select
  
#-----------------------------------------------------------------------------------------------------------------------------

f1_scores = []
kappa_scores = []
models = []
predictions = []
callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75,first_metric_only=True)]
feature_select = feature_select_wrapper()

#-----------------------------------------------------------------------------------------------------------------------------

X = train_feats[feature_select].astype(np.float32).values
print('Features Select Number: ', len(feature_select))
