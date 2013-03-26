function rlmap_maze2_run_all()
    theAgent=rlmap_agent();
    theEnvironment=maze2_environment();
    theExperimentFunc=@rlmap_maze2_experiment;
    
    togetherStruct.agent=theAgent;
    togetherStruct.environment=theEnvironment;
    togetherStruct.experiment=theExperimentFunc;
    
    runRLGlueMultiExperiment(togetherStruct);
    
end