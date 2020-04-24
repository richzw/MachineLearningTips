def grid_search_cv(x_train,y_train,gbrt,param_grid,cv=10,random_state=43,verbose=0):
    start = datetime.now()
    kinds = np.prod([len(i) for i in param_grid.values()])
    print('开始时间{}, 共计{}种'.format(start,kinds))
    
    grid_search = GridSearchCV(estimator=gbrt,param_grid=param_grid,cv=cv,verbose=verbose,return_train_score=True)
    grid_search.fit(x_train,y_train)
    
    end = datetime.now()
    seconds =(end - start).seconds
    print('grid_search_cv, 共计{}种，用时{}秒'.format(kinds,seconds))
    
    return grid_search.cv_results_,grid_search.best_params_
