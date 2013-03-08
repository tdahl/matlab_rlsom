function theAgent = map_agent()
    theAgent.agent_init = @map_agent_init;
    theAgent.agent_start = @map_agent_start;
    theAgent.agent_step = @map_agent_step;
    theAgent.agent_end = @map_agent_end;
    theAgent.agent_cleanup = @map_agent_cleanup;
    theAgent.agent_message = @map_agent_message;
end

function map_agent_init(taskSpecJavaString)
	global map_vars;
    
    theTaskSpec ...
        = org.rlcommunity.rlglue.codec.taskspec.TaskSpec( ...
        taskSpecJavaString);
    
    % Lots of assertions to make sure that we can handle this problem.
    assert (theTaskSpec.getNumDiscreteObsDims() == 1);
    assert (theTaskSpec.getNumContinuousObsDims() == 0);
    assert ( ...
        ~theTaskSpec.getDiscreteObservationRange(0).hasSpecialMinStatus());
    assert ( ...
        ~theTaskSpec.getDiscreteObservationRange(0).hasSpecialMaxStatus());
    assert (theTaskSpec.getNumDiscreteActionDims() == 1);
    assert (theTaskSpec.getNumContinuousActionDims() == 0);
    assert (~theTaskSpec.getDiscreteActionRange(0).hasSpecialMinStatus());
    assert (~theTaskSpec.getDiscreteActionRange(0).hasSpecialMaxStatus());

    map_vars.STM_LENGTH = 4;
    map_vars.STM_DECAY_RATE = 0.5;
    
    map_vars.numObservations ...
        = theTaskSpec.getDiscreteObservationRange(0).getMax() + 1;
    map_vars.numActions ...
        = theTaskSpec.getDiscreteActionRange(0).getMax() + 1;
    
    map_vars.map = rand(map_vars.numActions, map_vars.numObservations);
    map_vars.activations = zeros(map_vars.numActions, map_vars.numObservations);
    disp(map_vars.map);
    disp(map_vars.activations);
end    

%
% Get initial action
function theAction = map_agent_start(theObservation)
global map_vars;

    map_vars.step_count = 0;
    theAction = softmax_action(theObservation.getInt(0)); 
end

%
% Get an action for each step
function theAction = map_agent_step(~, theObservation)
global map_vars;

    map_vars.step_count = map_vars.step_count + 1;
    theAction = softmax_action(theObservation.getInt(0)); 
end

%
% Create an action by applying softmax to the observation-action values for
% the current observation
function theAction = softmax_action(newObsInt)
global map_vars;

    % find action vals for current observations 
    obs_act_vals = map_vars.map(:,newObsInt+1);
    
    % normalise actionvals
    obs_act_probs = softmax(obs_act_vals);
    
    % find softmax winner
    rand_prob = rand();
    prob = 0;
    for i = 1:size(obs_act_probs)
        %disp(i);
        prob = prob + obs_act_probs(i);
        %disp(prob);
        if rand_prob <= prob
            newActionInt = i;
            break;
        end
    end
    
    % discount activation values
    map_vars.activations = floor(map_vars.activations*map_vars.STM_DECAY_RATE);
    
    % add winner's activation value
    map_vars.activations(newActionInt, newObsInt+1) ...
        = map_vars.activations(newActionInt, newObsInt+1) ...
        + (1.0/map_vars.STM_DECAY_RATE)^map_vars.STM_LENGTH;
    disp(map_vars.activations);
    if(map_vars.step_count >= map_vars.STM_LENGTH)
        assert(sum(sum(map_vars.activations)) == 31);
        %surf(map_vars.activations);
    end
            
    theAction = org.rlcommunity.rlglue.codec.types.Action(1, 0, 0);
    theAction.setInt(0,newActionInt-1);
end

%
% The episode is over, do nothing.
function map_agent_end(~)
end

%
% Ignores all messages
function returnMessage = map_agent_message(theMessageJavaObject)
%Java strings are objects, and we want a Matlab string
    inMessage = char(theMessageJavaObject);

   	if strcmp(inMessage, 'freeze learning')
		returnMessage = 'message understood, policy frozen';
        return;
    end
    if strcmp(inMessage, 'unfreeze learning')
		returnMessage = 'message understood, policy unfrozen';
        return;
	end
	if strcmp(inMessage, 'freeze exploring')
		returnMessage = 'message understood, exploring frozen';
        return;
	end
	if strcmp(inMessage, 'unfreeze exploring')
		returnMessage = 'message understood, exploring unfrozen';
        return;
    end
    if strncmp(inMessage, 'save_policy',11)
		returnMessage = 'message understood, saving policy';
        return;
	end
	if strncmp(inMessage, 'load_policy', 11)
		returnMessage = 'message understood, loading policy';
        return;
    end
    
	returnMessage ...
        = 'MapAgent(Matlab) does not understand your message.';
end

function map_agent_cleanup()
end
