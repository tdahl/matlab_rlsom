function theAgent=random_agent()
    theAgent.agent_init = @random_agent_init;
    theAgent.agent_start = @random_agent_start;
    theAgent.agent_step = @random_agent_step;
    theAgent.agent_end = @random_agent_end;
    theAgent.agent_cleanup = @random_agent_cleanup;
    theAgent.agent_message = @random_agent_message;
end

function random_agent_init(taskSpecJavaString)
	global random_vars;
    
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

    random_vars.numActions ...
        = theTaskSpec.getDiscreteActionRange(0).getMax() + 1;
end    

%
% Get a random action
function theAction = random_agent_start(~)

    theAction = random_action();
end

%
% Get a random action
function theAction=random_agent_step(~, ~)

    theAction = random_action();
end

%
% Create a random action
function theAction = random_action()
global random_vars;

    newActionInt = randi(random_vars.numActions)-1;

    theAction = org.rlcommunity.rlglue.codec.types.Action(1, 0, 0);
    theAction.setInt(0,newActionInt);
end

%
% The episode is over, do nothing.
function random_agent_end(~)
end

%
% Ignores all messages
function returnMessage = random_agent_message(theMessageJavaObject)
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
        = 'RandomAgent(Matlab) does not understand your message.';
end

function random_agent_cleanup()
end
