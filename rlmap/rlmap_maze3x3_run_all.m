function rlmap_maze3x3_run_all()
    theAgent=rlmap_agent();
    theEnvironment=maze3x3_environment();
    theExperimentFunc=@rlmap_maze3x3_experiment;
    
    togetherStruct.agent=theAgent;
    togetherStruct.environment=theEnvironment;
    togetherStruct.experiment=theExperimentFunc;
    
    runRLGlueMultiExperiment(togetherStruct);
    
end