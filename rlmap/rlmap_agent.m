function theAgent = rlmap_agent()
    theAgent.agent_init = @map_agent_init;
    theAgent.agent_start = @map_agent_start;
    theAgent.agent_step = @map_agent_step;
    theAgent.agent_end = @map_agent_end;
    theAgent.agent_cleanup = @map_agent_cleanup;
    theAgent.agent_message = @map_agent_message;
end

function map_agent_init(taskSpecJavaString)
	global rlmap_vars;
    
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

    rlmap_vars.STM_SIZE = 4;
    rlmap_vars.STM_DECAY_RATE = 0.5;
    rlmap_vars.REWARD_DISCOUNT_RATE = 0.9;
    rlmap_vars.MIN_DFR = -9999.9;
    rlmap_vars.EXPLORATION_RATE = 0.2;
    
    rlmap_vars.INPUT_SIZE = 3;
    rlmap_vars.MAP_SIZE = 8;
    rlmap_vars.MAP2_SIZE = 64;
    
    rlmap_vars.nodecount = 0;
    rlmap_vars.numObservations ...
        = theTaskSpec.getDiscreteObservationRange(0).getMax() + 1;
    rlmap_vars.numActions ...
        = theTaskSpec.getDiscreteActionRange(0).getMax() + 1;
    
    % Level 0 (bottom) map
    rlmap_vars.map_observations = zeros(rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP_SIZE);
    rlmap_vars.map_actions = zeros(rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP_SIZE);
    rlmap_vars.map_rewards = zeros(rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP_SIZE);
    rlmap_vars.activations = zeros(rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP_SIZE);
    rlmap_vars.map_counts = zeros(rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP_SIZE);
    rlmap_vars.disc_rewards = zeros(rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP_SIZE);
    
    % Level 1 map
    rlmap_vars.map2 = zeros(rlmap_vars.MAP_SIZE, rlmap_vars.MAP_SIZE, ...
        rlmap_vars.MAP2_SIZE, rlmap_vars.MAP2_SIZE);
    rlmap_vars.map2_counts = zeros(rlmap_vars.MAP2_SIZE, ...
        rlmap_vars.MAP2_SIZE);
    
    rlmap_vars.exploringFrozen = false;
end    

%
% Get initial action
function theAction = map_agent_start(theObservation)
global rlmap_vars;
    
    rlmap_vars.step_count = 0;
    rlmap_vars.lastObservation = theObservation;
    theAction = random_action();
    rlmap_vars.lastAction = theAction;
end

%
% Get an action for each step
function theAction = map_agent_step(theReward, theObservation)
global rlmap_vars;

    rlmap_vars.step_count = rlmap_vars.step_count + 1;
    
    %theObservationInt = theObservation.getInt(0);
    %lastObservationInt = rlmap_vars.lastObservation.getInt(0);
    %lastActionInt = rlmap_vars.lastAction.getInt(0);
    %fprintf(1, 'step %d: o=%d a=%d r=%d o''=%d\n', ...
    %    rlmap_vars.step_count, lastObservationInt, lastActionInt, ...
    %    theReward, theObservationInt);
    activate_and_decay(theReward);
    
    % encode
    if mod(rlmap_vars.step_count, rlmap_vars.STM_SIZE) == 0
        sequence_encode();
    end
    
    % select action
    theAction = select_action(theObservation);
    rlmap_vars.lastAction = theAction;
    rlmap_vars.lastObservation = theObservation;
 
end

%
% identify winning node and activate
% decay activation on other nodes
function activate_and_decay(theReward)
global rlmap_vars;

    lastObservationInt = rlmap_vars.lastObservation.getInt(0);
    lastActionInt = rlmap_vars.lastAction.getInt(0);
    
    % decay activation values
    rlmap_vars.activations ...
        = floor(rlmap_vars.activations*rlmap_vars.STM_DECAY_RATE);
    % find or construct winning node
    found = false;
    for row = 1:rlmap_vars.MAP_SIZE
        for col = 1:rlmap_vars.MAP_SIZE
            input_sum = 0.0;
            input_sum = input_sum+rlmap_vars.map_observations(row, col);
            input_sum = input_sum+rlmap_vars.map_actions(row, col);
            input_sum = input_sum+rlmap_vars.map_rewards(row, col);
            if input_sum == 0
                % found unused node
                %if theReward == 10.0
                %    fprintf(1, 'using new node %d %d\n', row, col);
                %end
                found = true;
                rlmap_vars.map_observations(row, col) = lastObservationInt;
                rlmap_vars.map_actions(row, col) = lastActionInt;
                rlmap_vars.map_rewards(row, col) = theReward;
                % add winner's activation value
                rlmap_vars.activations(row, col) ...
                    = (1.0/rlmap_vars.STM_DECAY_RATE) ...
                    ^((rlmap_vars.STM_SIZE)-1);
                rlmap_vars.map_counts(row, col) ...
                    = rlmap_vars.map_counts(row, col)+1;
                rlmap_vars.nodecount = rlmap_vars.nodecount+1;
            else
                %disp('map');
                %disp(rlmap_vars.map(:, row, col));
                %disp('input');
                %disp(input);
                diff = 0.0;
                diff = diff+abs(rlmap_vars.map_observations(row, col) ...
                    -lastObservationInt);
                diff = diff+abs(rlmap_vars.map_actions(row, col) ...
                    -lastActionInt);
                diff = diff+abs(rlmap_vars.map_rewards(row, col) ...
                    -theReward);
                %if theReward == 10
                %    disp('diff');
                %    disp(diff);
                %end
                %fprintf(1, 'trying node %d %d, difference %d\n', ...
                %    row, col, diff);                
                if diff == 0
                    % found identical node
                    %if theReward == 10
                    %    fprintf(1, 'reusing node %d %d\n', row, col);
                    %end
                    found = true;
                    % add winner's activation value
                    rlmap_vars.activations(row, col) ...
                        = rlmap_vars.activations(row, col) ...
                        +(1.0/rlmap_vars.STM_DECAY_RATE) ...
                        ^((rlmap_vars.STM_SIZE)-1);
                    rlmap_vars.map_counts(row, col) ...
                        = rlmap_vars.map_counts(row, col)+1;
                end
            end
            if found
                break
            end
        end
        if found
            break;
        end
        % Has run out of nodes.
        assert(~((row == rlmap_vars.MAP_SIZE) ...
            && (col==rlmap_vars.MAP_SIZE)));
    end
    
    if rlmap_vars.step_count > rlmap_vars.STM_SIZE
        assert(sum(sum(rlmap_vars.activations)) ...
            == (1.0/rlmap_vars.STM_DECAY_RATE)^(rlmap_vars.STM_SIZE)-1);
    end    
end

%
% Create an action by evaluating the local discounted rewards
function theAction = select_action(theObservation)
global rlmap_vars;

    % discount rewards
    rlmap_vars.disc_rewards ...
        = ones(rlmap_vars.MAP_SIZE, rlmap_vars.MAP_SIZE) ...
        *rlmap_vars.MIN_DFR;    
    for row2 = 1:rlmap_vars.MAP2_SIZE
        for col2 = 1:rlmap_vars.MAP2_SIZE
            connections = rlmap_vars.map2(:, :, row2, col2);
            [nzrows, nzcols] = find(connections>0);
            %if size(nzrows) ~= 0
            %    fprintf(1, 'map2 node %d %d non zero rows and columns\n', ...
            %        row2, col2);
            %    disp([nzrows, nzcols]);
            %end
            % for all non-zero connections
            for nzidx = 1:size(nzrows)
                nzrow = nzrows(nzidx);
                nzcol = nzcols(nzidx);
                weight = connections(nzrow, nzcol);
                sum_rewards = rlmap_vars.map_rewards(nzrow, nzcol);
                %fprintf(1, 'local reward %.4f\n', sum_rewards); 
                % consider all other non-zero connections
                for nzidx2 = 1:size(nzrows)
                    nzrow2 = nzrows(nzidx2);
                    nzcol2 = nzcols(nzidx2);
                    weight2 = connections(nzrow2, nzcol2);
                    if weight2 > weight
                        reward = ...
                            rlmap_vars.map_rewards(nzrow2,nzcol2);
                        dist = log2(weight2) - log2(weight);
                        disc_reward = reward ...
                            *rlmap_vars.REWARD_DISCOUNT_RATE^dist;
                        sum_rewards = sum_rewards+disc_reward;
                         %fprintf(1, 'weight %d (%d %d)', weight, nzrow, ...
                         %    nzcol);
                         %fprintf(1, ', higher %d (%d %d)', weight2, ...
                         %    nzrow2, nzcol2);
                         %fprintf(1, ', reward %.2f, dist %d', reward, dist)
                         %fprintf(1, ', disc.r. %.4f', disc_reward);
                         %fprintf(1, ', sum d.r. %.4f\n', sum_rewards);
                    end                   
                end
                if sum_rewards > rlmap_vars.disc_rewards(nzrow, nzcol)
                    rlmap_vars.disc_rewards(nzrow, nzcol) = sum_rewards;
                end
                %fprintf(1, 'discounted rewards %.4f\n', sum_rewards);
            end
        end
    end
    
    % observation matches
    explore_prob = rand();
    obs_match_mtx = rlmap_vars.map_observations ...
        == theObservation.getInt(0);
    if (~rlmap_vars.exploringFrozen) ...
            && (explore_prob < rlmap_vars.EXPLORATION_RATE)
        % time to explore
        newActionInt = randi(rlmap_vars.numActions);
        %fprintf(1, 'exploration time, random action %d\n', newActionInt); 
    elseif sum(sum(obs_match_mtx)) > 0
        % matching observations exist in the level 0 map 
        obs_match_act_mtx = rlmap_vars.map_actions.*obs_match_mtx;
        [r, c, obs_match_actions] = find(obs_match_act_mtx);
        obs_match_drew_mtx = rlmap_vars.disc_rewards.*obs_match_mtx;
        [~, ~, obs_match_disc_rewards] = find(obs_match_drew_mtx);
        %disp('obs');
        %disp(theObservation.getInt(0));
        %disp('observations');
        %disp(rlmap_vars.map_observations);
        %disp('obs match mtx');
        %disp(obs_match_mtx);
        %disp('actions');
        %disp(rlmap_vars.map_actions);
        %disp('obs matching action matrix');
        %disp(obs_match_act_mtx);
        %disp('obs matching actions');
        %disp(obs_match_actions);
        %disp('rewards');
        %disp(rlmap_vars.map_rewards);
        %disp('discounted rewards');
        %disp(rlmap_vars.disc_rewards);
        %disp('obs matching discounted rewards');
        %disp(obs_match_disc_rewards);

        % normalise discounted rewards
        obs_act_probs = softmax(obs_match_disc_rewards);
        %disp('softmax');
        %disp(obs_act_probs);
        % find softmax winner
        rand_prob = rand();
        %fprintf(1, 'rand prob %.4f\n', rand_prob);
        prob = 0.0;
        for action_idx = 1:size(obs_act_probs)
            %disp(action_idx);
            prob = prob+obs_act_probs(action_idx);
            %disp(prob);
            if rand_prob <= prob
                %fprintf('chose index %d\n', action_idx);
                break;
            end
        end
        newActionInt = obs_match_actions(action_idx);
        %fprintf(1,'action idx %d, action %d\n', action_idx, newActionInt);
    else
        % no records exist for the current observation
        newActionInt = randi(rlmap_vars.numActions);
        %fprintf(1, 'no records, random action %d\n', newActionInt); 
    end
    theAction = org.rlcommunity.rlglue.codec.types.Action(1, 0, 0);
    theAction.setInt(0,newActionInt);
end

%
% Create an action by applying softmax to the observation-action values for
% the current observation
function theAction = random_action()
global rlmap_vars;
     
    newActionInt = randi(rlmap_vars.numActions);
    theAction = org.rlcommunity.rlglue.codec.types.Action(1, 0, 0);
    theAction.setInt(0,newActionInt);
end

%
% Update the sequence encoding map
function sequence_encode()
global rlmap_vars;
        
    % find identical or unused node
    for row = 1:rlmap_vars.MAP2_SIZE
        for col = 1:rlmap_vars.MAP2_SIZE
            s = sum(sum(rlmap_vars.map2(:, :, row, col)));
            if s == 0                
                % found unused node
                % change connection weights
                rlmap_vars.map2(:, :, row, col) ...
                    = rlmap_vars.map2(:, :, row, col) ...
                    +rlmap_vars.activations;
                rlmap_vars.map2_counts(row, col) ...
                    = rlmap_vars.map2_counts(row, col)+1;               
                return;
            end
            diff = rlmap_vars.activations-rlmap_vars.map2(:, :, row, col);
            if diff == 0
                % found identical node
                % weight change not required
                rlmap_vars.map2_counts(row, col) ...
                    = rlmap_vars.map2_counts(row, col)+1;  
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
global rlmap_vars;

    rlmap_vars.step_count = rlmap_vars.step_count + 1;
    lastObservationInt = rlmap_vars.lastObservation.getInt(0);
    lastActionInt = rlmap_vars.lastAction.getInt(0);
    %fprintf(1, 'step %d: o=%d a=%d r=%d\n', rlmap_vars.step_count, ...
    %    lastObservationInt, lastActionInt, theReward);
    
    % record last o/a/r triplet activate winner and decay other nodes
    activate_and_decay(theReward);
    
    % encode reward node
    sequence_encode();
    for row = 1:rlmap_vars.MAP2_SIZE
        for col = 1:rlmap_vars.MAP2_SIZE
            s = sum(sum(rlmap_vars.map2(:, :, row, col)));
            if s == 0
                %disp('last node');
            end
        end
    end
    %fprintf(1, 'node count %d\n', rlmap_vars.nodecount);
    %disp(rlmap_vars.map);
    %surf(rlmap_vars.map_counts);
    %surf(rlmap_vars.map2_counts);
end

%
% Ignores all messages
function returnMessage = map_agent_message(theMessageJavaObject)
global rlmap_vars;
%Java strings are objects, and we want a Matlab string
    inMessage = char(theMessageJavaObject);

   	if strcmp(inMessage,'freeze learning')
		rlmap_vars.policyFrozen=true;
		returnMessage='message understood, policy frozen';
        return;
    end

    if strcmp(inMessage,'unfreeze learning')
		rlmap_vars.policyFrozen=false;
		returnMessage='message understood, policy unfrozen';
        return;
	end
	if strcmp(inMessage,'freeze exploring')
		rlmap_vars.exploringFrozen=true;
		returnMessage='message understood, exploring frozen';
        return;
	end
	if strcmp(inMessage,'unfreeze exploring')
		rlmap_vars.exploringFrozen=false;
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
        = 'RLMapAgent(Matlab) does not understand your message.';
end

function loadValueFunction(fileName)
global rlmap_vars;

    loadedStruct=load(fileName,'-mat');    
    rlmap_vars.map2_counts=loadedStruct.map2_counts;
end

function saveValueFunction(fileName)
global rlmap_vars;

    theSaveCommand=sprintf('save %s -mat -struct ''rlmap_vars'' ''map2_counts''',fileName);
    eval(theSaveCommand);
end

function map_agent_cleanup()
end
