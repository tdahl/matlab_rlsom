function map2_maze2_run_all()
    theAgent=map2_agent();
    theEnvironment=maze2_environment();
    theExperimentFunc=@map2_maze2_experiment;
    
    togetherStruct.agent=theAgent;
    togetherStruct.environment=theEnvironment;
    togetherStruct.experiment=theExperimentFunc;
    
    runRLGlueMultiExperiment(togetherStruct);
    
end