function map2_maze3x3_run_all()
    theAgent=map2_agent();
    theEnvironment=maze3x3_environment();
    theExperimentFunc=@map2_maze3x3_experiment;
    
    togetherStruct.agent=theAgent;
    togetherStruct.environment=theEnvironment;
    togetherStruct.experiment=theExperimentFunc;
    
    runRLGlueMultiExperiment(togetherStruct);
    
end