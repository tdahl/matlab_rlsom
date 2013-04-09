
function theEnvironment = maze57_environment()
%Assign members of the returning struct to be function pointers
	theEnvironment.env_init = @maze57_init;
	theEnvironment.env_start = @maze57_start;
	theEnvironment.env_step = @maze57_step;
	theEnvironment.env_cleanup = @maze57_cleanup;
	theEnvironment.env_message = @maze57_message;
end
 
%This is what will be called for env_init
function taskSpecString = maze57_init()
	global maze57_struct;
    
    %maze57_struct acts like 'this' would from Java.  It's where we store
    %what would be member variables if this was an object.
    maze57_struct.fixedStartState = true;
    maze57_struct.startRow = 2;
    maze57_struct.startCol = 2;
    
    maze57_struct.WORLD_FREE = 0;
    maze57_struct.WORLD_OBSTACLE = 1;
    maze57_struct.WORLD_GOAL = 2;
    
    maze57_struct.OBS_NORTH = 8;
    maze57_struct.OBS_EAST = 4;
    maze57_struct.OBS_SOUTH = 2;
    maze57_struct.OBS_WEST = 1;
    
    maze57_struct.MOVE_NORTH = 1;
    maze57_struct.MOVE_EAST = 2;
    maze57_struct.MOVE_SOUTH = 3;
    maze57_struct.MOVE_WEST = 4;

    maze57_struct.REWARD_GOAL = 10;
    maze57_struct.REWARD_STEP = -0.1;

    
    theWorld.map = [1,1,1,1,1,1,1,1,1,1,1,1,1;
                    1,0,0,0,0,0,0,0,0,0,0,0,1;
                    1,1,1,0,1,0,1,0,1,2,1,1,1;
                    1,1,1,1,1,1,1,1,1,1,1,1,1];
    
    numRows = size(theWorld.map,1);
    numCols = size(theWorld.map,2);
                    
    theWorld.agentRow = 1;
    theWorld.agentCol = 1;

    maze57_struct.theWorld = theWorld;
    
    theTaskSpecObject ...
        = org.rlcommunity.rlglue.codec.taskspec.TaskSpecVRLGLUE3();
    theTaskSpecObject.setEpisodic();
    %theTaskSpecObject.setDiscountFactor(1.0);

    observationRange ...
        = org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange(0, 15);
    %Specify that there will be an integer observation [0,108]
    %for the state
    theTaskSpecObject.addDiscreteObservation(observationRange);

    actionRange ...
        = org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange(0,3);
    %Specify that there will be an integer action [0,4]
    theTaskSpecObject.addDiscreteAction(actionRange);

    %Specify the reward range [-100,10]
    rewardRange = ...
        org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange(-100,10);

    theTaskSpecObject.setRewardRange(rewardRange);

    theTaskSpecObject.setExtra( ...
        'Maze57Environment(Matlab) by Torbjorn S. Dahl.');

    taskSpecString = theTaskSpecObject.toTaskSpec();
end

%This is what will be called for env_start
function theObservation = maze57_start()
	global maze57_struct;
    if maze57_struct.fixedStartState==true
        maze57_struct.theWorld.agentRow = maze57_struct.startRow;
        maze57_struct.theWorld.agentCol = maze57_struct.startCol;

        stateIsValid = ...
            set_agent_state(maze57_struct.startRow, ...
            maze57_struct.startCol);
        if ~stateIsValid
            set_random_state();
        end
    else
        set_random_state();
    end

    theObservation ...
        = org.rlcommunity.rlglue.codec.types.Observation(1, 0, 0);
    theObservation.setInt(0, getObservation());
end

%This is what will be called for env_step
function rewardObservation = maze57_step(thisAction)
    global maze57_struct;

%Make sure the action is valid
    assert (thisAction.getNumInts() ...
        == 1,'Expecting a 1-dimensional integer action.');
    assert (thisAction.getInt(0) ...
        > 0,'Action too small, should be in [1,5].');
    assert (thisAction.getInt(0) ...
        < 5,'Action too large, should be in [1,5].');
        
	updatePosition(thisAction.getInt(0));
    
    newObsInt = getObservation();
	newReward = getReward();
    isTerminalBoolean ...
        = check_terminal(maze57_struct.theWorld.agentRow, ...
        maze57_struct.theWorld.agentCol);

    if isTerminalBoolean
        isTerminalInt = 1;
    else
        isTerminalInt = 0;
    end

	theObservation = org.rlcommunity.rlglue.codec.types.Observation();
	theObservation.intArray = [newObsInt];

	rewardObservation = ...
        org.rlcommunity.rlglue.codec.types.Reward_observation_terminal( ...
        newReward, theObservation, isTerminalInt);
end


%This is what will be called for env_message
function returnMessage = maze57_message(theMessageJavaObject)
    global maze57_struct;

    %Java strings are objects, and we want a Matlab string
    inMessage = char(theMessageJavaObject);
    
    %Message Description
    %'set-random-start-state'
    %Action: Set flag to do random starting states (the default)
    if strcmp(inMessage, 'set-random-start-state')
        maze57_struct.fixedStartState = false;
        returnMessage = 'Message understood.  Using random start state.';
        return;
    end
    
    %Message Description
    %'set-start-state X Y'
    %Action: Set flag to do fixed starting states, (row=X, col=Y)
    %This will be 0-indexed, so we should add 1 to it
	if strncmp(inMessage,'set-start-state',15)
        [firstPart,Remainder] = strtok(inMessage); % disregarded
        [rowString,Remainder] = strtok(Remainder);
        colString = strtok(Remainder);

        maze57_struct.startRow = str2double(rowString)+1;
        maze57_struct.startCol = str2double(colString)+1;
        maze57_struct.fixedStartState = true;
        returnMessage = 'Message understood.  Using fixed start state.';
        return;
    end
    
    %Message Description
    %'print-state'
    %Action: Print the map and the current agent location
    if strcmp(inMessage,'print-state')
        printState();
        returnMessage = 'Message understood.  Printed the state.';
        return;
    end

    returnMessage ...
        = 'Maze57 Environment(Matlab) does not respond to that message.';

end

function maze57_cleanup()
	global maze57_struct;
	maze57_struct = rmfield(maze57_struct,'fixedStartState');
	maze57_struct = rmfield(maze57_struct,'startRow');
	maze57_struct = rmfield(maze57_struct,'startCol');

    maze57_struct = rmfield(maze57_struct,'WORLD_FREE');
	maze57_struct = rmfield(maze57_struct,'WORLD_OBSTACLE');
	maze57_struct = rmfield(maze57_struct,'WORLD_GOAL');
    
	maze57_struct = rmfield(maze57_struct,'OBS_NORTH');
	maze57_struct = rmfield(maze57_struct,'OBS_EAST');
	maze57_struct = rmfield(maze57_struct,'OBS_SOUTH');
	maze57_struct = rmfield(maze57_struct,'OBS_WEST');
    
	maze57_struct = rmfield(maze57_struct,'MOVE_NORTH');
	maze57_struct = rmfield(maze57_struct,'MOVE_EAST');
	maze57_struct = rmfield(maze57_struct,'MOVE_SOUTH');
	maze57_struct = rmfield(maze57_struct,'MOVE_WEST');

    maze57_struct = rmfield(maze57_struct,'REWARD_GOAL');
	maze57_struct = rmfield(maze57_struct,'REWARD_STEP');
        
    maze57_struct.theWorld = rmfield(maze57_struct.theWorld,'map');
    maze57_struct.theWorld = rmfield(maze57_struct.theWorld,'agentRow');
    maze57_struct.theWorld = rmfield(maze57_struct.theWorld,'agentCol');
	maze57_struct = rmfield(maze57_struct,'theWorld');

    clear maze57_struct;
end

%
%
%Utility functions below
%
%
function isTerminal = check_terminal(row,col)
global maze57_struct;

	if maze57_struct.theWorld.map(row, col) == maze57_struct.WORLD_GOAL
    	isTerminal = true;
    else
        isTerminal = false;
    end
end

%Checks if a row,col is valid (not a wall or out of bounds)
function isValid = check_valid(row,col)
global maze57_struct;

    numRows = size(maze57_struct.theWorld.map, 1);
    numCols = size(maze57_struct.theWorld.map, 2);

    isValid = false;

    if (row <= numRows) && (row >= 1) && (col <= numCols) && (col >= 1)
        if maze57_struct.theWorld.map(row, col) ...
                ~= maze57_struct.WORLD_OBSTACLE
            isValid = true;
        end
    end
end


%Sets state and returns true if valid, false if invalid or terminal 
function isValid = set_agent_state(row, col)
global maze57_struct;

    maze57_struct.theWorld.agentRow = row;
    maze57_struct.theWorld.agentCol = col;
	
	isValid = check_valid(row, col) && ~check_terminal(row, col);
end

%Put the agent in a random, valid, nonterminal state
function set_random_state()
global maze57_struct;
    
    numRows = size(maze57_struct.theWorld.map, 1);
    numCols = size(maze57_struct.theWorld.map, 2);

    startRow = randi(numRows);
    startCol = randi(numCols);

 	while ~set_agent_state(startRow, startCol)
        startRow = randi(numRows);
        startCol = randi(numCols);
    end
    
    maze57_struct.theWorld.agentRow = startRow;
    maze57_struct.theWorld.agentCol = startCol;

end

%Returns a number in [0,15]
function totalObservation = getObservation()
global maze57_struct;
    
    newRow = maze57_struct.theWorld.agentRow;
    newCol = maze57_struct.theWorld.agentCol;
 
    totalObservation = 1;
    if ~check_valid(newRow-1, newCol)
        totalObservation = totalObservation + maze57_struct.OBS_NORTH;
    end
    if ~check_valid(newRow, newCol+1)
        totalObservation = totalObservation + maze57_struct.OBS_EAST;
    end
    if ~check_valid(newRow+1, newCol)
        totalObservation = totalObservation + maze57_struct.OBS_SOUTH;
    end
    if ~check_valid(newRow, newCol-1)
        totalObservation = totalObservation + maze57_struct.OBS_WEST;
    end
end

%Updates the agent's position based on the action provided.
%When the move would result in hitting an obstacles, the agent doesn't move 
function updatePosition(theIntAction)
global maze57_struct;
    
    newRow = maze57_struct.theWorld.agentRow;
    newCol = maze57_struct.theWorld.agentCol;

    fprintf(1, 'updating position (%d %d) action %d\n', newCol, newRow, ...
        theIntAction);
	
	if theIntAction == maze57_struct.MOVE_NORTH
		newRow = newRow - 1;
	end
	if theIntAction == maze57_struct.MOVE_EAST
		newCol = newCol + 1;
	end
	if theIntAction == maze57_struct.MOVE_SOUTH
		newRow = newRow + 1;
	end
	if theIntAction == maze57_struct.MOVE_WEST
		newCol = newCol - 1;
    end
    
    fprintf(1, 'new pos (%d %d)\n', newCol, newRow);

	%Check if new position is out of bounds or inside an obstacle 
	if check_valid(newRow, newCol)
   		maze57_struct.theWorld.agentRow = newRow;
   		maze57_struct.theWorld.agentCol = newCol;
        fprintf(1, 'valid\n');
    else
        fprintf(1, 'not valid\n');	end
end

%Calculate the reward for the current state
function theReward = getReward()
global maze57_struct;

    agentRow = maze57_struct.theWorld.agentRow;
    agentCol = maze57_struct.theWorld.agentCol;

    if maze57_struct.theWorld.map(agentRow, agentCol) ...
            == maze57_struct.WORLD_GOAL
        theReward = maze57_struct.REWARD_GOAL;
        return;
    end
        
    theReward = maze57_struct.REWARD_STEP;
end


%These are 1-indexed, so decrement them when printing to make them 0-index
function printState()
global maze57_struct;

    numRows = size(maze57_struct.theWorld.map,1);
    numCols = size(maze57_struct.theWorld.map,2);

    agentRow = maze57_struct.theWorld.agentRow;
    agentCol = maze57_struct.theWorld.agentCol;

    fprintf(1, 'Agent is at: %d,%d\n', agentRow-1, agentCol-1);
	fprintf(1,'  Columns:0-%d\n',numCols);
	fprintf(1,'Col      ');
    
    for col = 1:numCols
		fprintf(1, '%d ', mod(col-1, 10));
    end

    for row = 1:numRows
        fprintf(1, '\nRow: %d   ', row-1);

        for col = 1:numCols
            if (agentRow == row) && (agentCol == col)
                fprintf(1, 'A ');
            else
                if maze57_struct.theWorld.map(row, col) ...
                        == maze57_struct.WORLD_GOAL
                    fprintf(1,'G ');
                end
                if maze57_struct.theWorld.map(row, col) ...
                        == maze57_struct.WORLD_OBSTACLE
                    fprintf(1,'* ');
                end
                if maze57_struct.theWorld.map(row, col) ...
                        == maze57_struct.WORLD_FREE
                    fprintf(1, '  ');
                end
            end
        end
    end
	fprintf(1, '\n');
end


