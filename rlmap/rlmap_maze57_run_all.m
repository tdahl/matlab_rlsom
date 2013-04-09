function rlmap_maze57_run_all()
    theAgent=rlmap_agent();
    theEnvironment=maze57_environment();
    theExperimentFunc=@rlmap_experiment;
    
    togetherStruct.agent=theAgent;
    togetherStruct.environment=theEnvironment;
    togetherStruct.experiment=theExperimentFunc;
    
    runRLGlueMultiExperiment(togetherStruct);
    
end