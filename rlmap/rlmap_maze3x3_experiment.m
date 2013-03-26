function rlmap_maze3x3_experiment()
 
	RL_init();
    fprintf(1,'Starting offline demo\n----------------------------\n');
    fprintf(1,'Will alternate learning for 25 episodes, ');
    fprintf(1,'then freeze policy and evaluate for 10 episodes.\n\n');
	fprintf(1,'After Episode\tMean Return\tStandard Deviation\n');
    fprintf(1,'-------------------------------------------------------');
    fprintf(1,'------------------\n');
	offline_demo();

    fprintf(1,'\nNow we will save the agent''s learned value function to a file....\n');
	RL_agent_message('save_policy results.dat');

	fprintf(1,'\nCalling RL_cleanup and RL_init to clear the ');
    fprintf(1,'agent''s memory...\n');
	RL_cleanup();
	RL_init();

	fprintf(1,'Evaluating the agent''s default policy:\n\t\tMean Return\tStandardDeviation\n------------------------------------------------------\n');
	[theMean,theStdDev]=evaluate_agent();
	print_score(0,theMean,theStdDev);
	
	fprintf(1,'\nLoading up the value function we saved earlier.\n');
	RL_agent_message('load_policy results.dat');
    
	fprintf(1,'Evaluating the agent after loading the value function:\n\t\tMean Return\tStandardDeviation\n------------------------------------------------------\n');
	[theMean,theStdDev]=evaluate_agent();
	print_score(0,theMean,theStdDev);    
    
    fprintf(1,'Telling the environment to use random start state.\n');
    RL_env_message('set-random-start-state');
    RL_start();
    fprintf(1,'Telling the environment to print the current state ');
    fprintf(1,'to the screen\n');
    RL_env_message('print-state');

    fprintf(1,'Evaluating the agent a few times from a random start ');
    fprintf(1,'state:\n\t\tMean Return\tStandardDeviation\n');
    fprintf(1,'-------------------------------------------\n');
	[theMean,theStdDev]=evaluate_agent();
	print_score(0,theMean,theStdDev);

	RL_cleanup();
    disconnectGlue();
	fprintf(1,'\nProgram Complete.\n');

end


% /**
% * Tell the agent to stop learning, then execute n episodes with his
% * current policy.  Estimate the mean and variance of the return over
% * these episodes.
% * @return
% */
function[theMean,theStdDev]= evaluate_agent()
    n=10;
    sum=0;
    sum_of_squares=0;
 
    RL_agent_message('freeze learning');
    for i=1:n
        % We use a cutoff here in case the policy is bad and will never end
        % an episode
        RL_episode(5000);
        this_return=RL_return();
        sum=sum+this_return;
        sum_of_squares=sum_of_squares+this_return*this_return;
    end
 
    theMean=sum/n;
    variance = (sum_of_squares - n*theMean*theMean)/(n - 1.0);
    theStdDev=sqrt(variance);

    RL_agent_message('unfreeze learning');
end
 

%
% This function will freeze the agent's policy and test it after every 25
% episodes.
function offline_demo()

    statistics=[];
    
	[theMean,theStdDev]=evaluate_agent();
	print_score(0,theMean,theStdDev);
     
    statistics=[statistics; 0,theMean,theStdDev];
    
    for i=1:20
        for j=1:25
			RL_episode(0);
        end
    	[theMean,theStdDev]=evaluate_agent();
		print_score(i*25,theMean,theStdDev);
        statistics=[statistics; i*25,theMean,theStdDev];
    end
	
    errorbar(statistics(:,1),statistics(:,2),statistics(:,3))

end


function print_score(afterEpisodes, theMean, theStdDev)
    fprintf(1,'%d\t\t%.2f\t\t%.2f\n', afterEpisodes,theMean, theStdDev);
end



