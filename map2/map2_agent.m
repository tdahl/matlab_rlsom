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
    
    map_vars.MAP2_SIZE = 64;
    
    map_vars.numObservations ...
        = theTaskSpec.getDiscreteObservationRange(0).getMax() + 1;
    map_vars.numActions ...
        = theTaskSpec.getDiscreteActionRange(0).getMax() + 1;
    
    % Level 0 (bottom) map
    map_vars.map = rand(map_vars.numActions, map_vars.numObservations);
    map_vars.activations ...
        = zeros(map_vars.numActions, map_vars.numObservations);
    
    % Level 1 map
    map_vars.map2 ...
        = zeros(map_vars.numActions, map_vars.numObservations, ...
        map_vars.MAP2_SIZE, map_vars.MAP2_SIZE);
    map_vars.map2_counts ...
        = zeros(map_vars.MAP2_SIZE, map_vars.MAP2_SIZE);
end    

%
% Get initial action
function theAction = map_agent_start(theObservation)
global map_vars;
    
    map_vars.step_count = 0;
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
%     fprintf(1,'step %d: o=%d a=%d r=%d o''=%d\n', map_vars.step_count, ...
%         map_vars.lastObservation.getInt(0), ...
%         map_vars.lastAction.getInt(0), theReward, ...
%         theObservation.getInt(0));

    % select action
    theObservationInt = theObservation.getInt(0);
    theAction = softmax_action(theObservationInt);
    map_vars.lastAction = theAction;

    % discount activation values
    map_vars.activations ...
        = floor(map_vars.activations*map_vars.STM_DECAY_RATE);
    % add winner's activation value
    theActionInt = theAction.getInt(0);
    map_vars.activations(theActionInt+1, theObservationInt+1) ...
        = map_vars.activations(theActionInt+1, theObservationInt+1) ...
        + (1.0/map_vars.STM_DECAY_RATE)^(map_vars.STM_SIZE-1);
    %disp(map_vars.activations);
    if(map_vars.step_count > map_vars.STM_SIZE)
        assert(sum(sum(map_vars.activations)) ...
            == (1.0/map_vars.STM_DECAY_RATE)^(map_vars.STM_SIZE)-1);
        %surf(map_vars.activations);
    end
    
    % encode
    if mod(map_vars.step_count, map_vars.STM_SIZE) == 0
        %fprintf(1,'update layer 1\n');
        map_encode();
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
     
    theAction = org.rlcommunity.rlglue.codec.types.Action(1, 0, 0);
    theAction.setInt(0,newActionInt-1);
end

%
% Update the sequence encoding map
function map_encode()
global map_vars;
        
    % find identical or unused node
    for row = 1:map_vars.MAP2_SIZE
        for col = 1:map_vars.MAP2_SIZE
            s = sum(sum(map_vars.map2(:, :, row, col)));
            if s == 0
                %fprintf(1,'found unused node %d %d: weight sum %d\n', ...
                %    row, col, s);                
                % found unused node
                % change connection weights
                %disp('connections before encoding');
                %disp(map_vars.map2(:, :, row, col));
                map_vars.map2(:, :, row, col) ...
                    = map_vars.map2(:, :, row, col)+map_vars.activations;
                map_vars.map2_counts(row, col) ...
                    = map_vars.map2_counts(row, col)+1;
                %disp('connections after encoding');
                %disp(map_vars.map2(:, :, row, col));                
                return;
            end
            diff = map_vars.activations-map_vars.map2(:, :, row, col);
            if diff == 0
                % found identical node
                % weight change not required
                map_vars.map2_counts(row, col) ...
                    = map_vars.map2_counts(row, col)+1;
                %fprintf(1, ...
                %    'found identical node %d %d: weight sum %d\n', ...
                %    row, col, s);   
                return;
            end
        end
    end
    % Has run out of nodes.
    assert(false);
end

%
% The episode is over. Reached terminal state. Learn in all layers.
function map_agent_end(theReward)
global map_vars;

    map_vars.step_count = map_vars.step_count + 1;
    
    % encode reward node
    map_encode();
    for row = 1:map_vars.MAP2_SIZE
        for col = 1:map_vars.MAP2_SIZE
            s = sum(sum(map_vars.map2(:, :, row, col)));
            if s == 0
                disp('last node');
                %disp(
            end
        end
    end
    %fprintf(1,'step %d: o=%d a=%d r=%d\n', map_vars.step_count, ...
    %    map_vars.lastObservation.getInt(0), ...
    %    map_vars.lastAction.getInt(0), theReward);
    %disp(map_vars.map2_counts);
    surf(map_vars.map2_counts);

end

%
% Ignores all messages
function returnMessage = map_agent_message(theMessageJavaObject)
global map_vars;
%Java strings are objects, and we want a Matlab string
    inMessage = char(theMessageJavaObject);

   	if strcmp(inMessage,'freeze learning')
		map_vars.policyFrozen=true;
		returnMessage='message understood, policy frozen';
        return;
    end

    if strcmp(inMessage,'unfreeze learning')
		map_vars.policyFrozen=false;
		returnMessage='message understood, policy unfrozen';
        return;
	end
	if strcmp(inMessage,'freeze exploring')
		map_vars.exploringFrozen=true;
		returnMessage='message understood, exploring frozen';
        return;
	end
	if strcmp(inMessage,'unfreeze exploring')
		map_vars.exploringFrozen=false;
		returnMessage='message understood, exploring unfrozen';
        return;
    end
    if strncmp(inMessage,'save_policy',11)
        [commandString,remainder]=strtok(inMessage);
        fileName=strtok(remainder);
		fprintf(1,'Saving value function...');
        saveValueFunction(fileName);
		fprintf(1,'Saved.\n');
		returnMessage='message understood, saving policy';
        return;
	end
	if strncmp(inMessage,'load_policy',11)
        [commandString,remainder]=strtok(inMessage);
        fileName=strtok(remainder);
        loadValueFunction(fileName);
		fprintf(1,'Loaded.\n');
		returnMessage='message understood, loading policy';
        return;    
    end
    
	returnMessage ...
        = 'Map2Agent(Matlab) does not understand your message.';
end

function loadValueFunction(fileName)
global map_vars;

    loadedStruct=load(fileName,'-mat');    
    map_vars.map2_counts=loadedStruct.map2_counts;
end

function saveValueFunction(fileName)
global map_vars;

    theSaveCommand=sprintf('save %s -mat -struct ''map_vars'' ''map2_counts''',fileName);
    eval(theSaveCommand);
end

function map_agent_cleanup()
end
