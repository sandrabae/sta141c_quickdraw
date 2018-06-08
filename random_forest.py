  %%time

  #Setting up a Random Forest Classifer
  clf_rf = RandomForestClassifier(n_jobs=-1, random_state=0)
  clf_rf.fit(X_train, y_train)
  print(clf_rf)
  y_pred_rf = clf_rf.predict(X_test)
  acc_rf = accuracy_score(y_test, y_pred_rf)
  print ('Random forest accuracy: ',acc_rf)

  #Defining parameters this time
  parameters = {'n_estimators': [10,20,40,60,80,100,120,140,160]}

  # Getting Wall Time
  clf_rf = RandomForestClassifier(n_jobs=-1, random_state=0)
  rf = GridSearchCV(clf_rf, parameters, n_jobs=-1)
  rf.fit(X_train, y_train)
  results = pd.DataFrame(rf.cv_results_)

  #Displaying Results
  results.sort_values('mean_test_score', ascending = False)

  #Plotting Results
  results.plot('param_n_estimators','mean_test_score');

  #Trying again with Random Forest with the suggested values from the documentation
  parameters = {'max_features' : ['auto', 15, 28, 50] } # 28 = sqrt(784), which is suggested in the documentation as a good value

  clf_rf = RandomForestClassifier(n_estimators= 100, n_jobs=-1, random_state=0)
  rf = GridSearchCV(clf_rf, parameters, n_jobs=-1)
  rf.fit(X_train, y_train)

  results = pd.DataFrame(rf.cv_results_)
  results.sort_values('mean_test_score', ascending = False)

  #Getting the Wall Time again
  %%time

  clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
  clf_rf.fit(X_train, y_train)
  y_pred_rf = clf_rf.predict(X_test)
  acc_rf = accuracy_score(y_test, y_pred_rf)
  print ('Random forest accuracy: ',acc_rf)
