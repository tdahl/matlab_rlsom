function map_maze3x3_run_all()
    theAgent=map_agent();
    theEnvironment=maze3x3_environment();
    theExperimentFunc=@map_maze3x3_experiment;
    
    togetherStruct.agent=theAgent;
    togetherStruct.environment=theEnvironment;
    togetherStruct.experiment=theExperimentFunc;
    
    runRLGlueMultiExperiment(togetherStruct);
    
end