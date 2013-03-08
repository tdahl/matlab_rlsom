function theAgent = map2_agent()
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

    map_vars.STM_SIZE = 4;
    map_vars.STM_DECAY_RATE = 0.5;
    
    map_vars.MAP2_SIZE = 4;
    
    map_vars.numObservations ...
        = theTaskSpec.getDiscreteObservationRange(0).getMax() + 1;
    map_vars.numActions ...
        = theTaskSpec.getDiscreteActionRange(0).getMax() + 1;
    
    map_vars.map = rand(map_vars.numActions, map_vars.numObservations);
    map_vars.activations ...
        = zeros(map_vars.numActions, map_vars.numObservations);
    
    map_vars.map2 ...
        = zeros(map_vars.MAP2_SIZE, map_vars.MAP2_SIZE, ...
        map_vars.numActions, map_vars.numObservations);
    disp(size(map_vars.map2));
end    

%
% Get initial action
function theAction = map_agent_start(theObservation)
global map_vars;
    
    map_vars.step_count = 1;
    %fprintf(1,'step %d: o=%d\n', map_vars.step_count, ...
    %    theObservation.getInt(0));

    map_vars.lastObservation = theObservation;
    theAction = softmax_action(theObservation.getInt(0));
    map_vars.lastAction = theAction;
end

%
% Get an action for each step
function theAction = map_agent_step(theReward, theObservation)
global map_vars;

    map_vars.step_count = map_vars.step_count + 1;
    %fprintf(1,'step %d: o=%d a=%d r=%d o''=%d\n', map_vars.step_count, ...
    %    map_vars.lastObservation.getInt(0), ...
    %    map_vars.lastAction.getInt(0), theReward, ...
    %    theObservation.getInt(0));
    
    theAction = softmax_action(theObservation.getInt(0));
    map_vars.lastAction = theAction;

    if mod(map_vars.step_count, map_vars.STM_SIZE) == 0
        %fprintf(1,'update layer 1\n');
        map_update();
    end
    
    map_vars.lastObservation = theObservation;
 
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
    map_vars.activations ...
        = floor(map_vars.activations*map_vars.STM_DECAY_RATE);
    
    % add winner's activation value
    map_vars.activations(newActionInt, newObsInt+1) ...
        = map_vars.activations(newActionInt, newObsInt+1) ...
        + (1.0/map_vars.STM_DECAY_RATE)^map_vars.STM_SIZE;
    %disp(map_vars.activations);
    if(map_vars.step_count > map_vars.STM_SIZE)
        assert(sum(sum(map_vars.activations)) == 31);
        %surf(map_vars.activations);
    end
            
    theAction = org.rlcommunity.rlglue.codec.types.Action(1, 0, 0);
    theAction.setInt(0,newActionInt-1);
end

%
% Update the sequence encoding map
function map_update()
global map_vars;

    fprintf(1,'step %d\n',map_vars.step_count);
    % find matching node
    for row2 = 1:map_vars.MAP2_SIZE
        for col2 = 1:map_vars.MAP2_SIZE
            s = sum(sum(map_vars.map2(row2,col2,:,:)));
            if s == 0 || map_stm_match(map_vars.map2(row2,col2,:,:)) == 31
                fprintf(1,'using node %d %d: weight sum %d\n',row2,col2,s);
                return;
            end 
        end
    end
end

%
% The episode is over. Reached terminal state. Learn in all layers.
function map_agent_end(theReward)
global map_vars;

    map_vars.step_count = map_vars.step_count + 1;
    %fprintf(1,'step %d: o=%d a=%d r=%d\n', map_vars.step_count, ...
    %    map_vars.lastObservation.getInt(0), ...
    %    map_vars.lastAction.getInt(0), theReward);
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
        = 'Map2Agent(Matlab) does not understand your message.';
end

function map_agent_cleanup()
end
