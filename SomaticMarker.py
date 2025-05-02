import numpy as np
import random
import logging

class SomaticMarker:
    
    def __init__ (self, up_threshold, down_threshold, decay, emotions:dict):
        #self._logger = logging.getLogger(__name__)
        #logging.basicConfig(filename='somaticMarker.log', level=logging.INFO)
        #self._logger.info('Started up_threshold[%d] down_threshold[%d] decay[%d]', up_threshold, down_threshold, decay)
        self._somaticMemory = {}
        self._up_threshold = up_threshold
        self._down_threshold = down_threshold
        self._decay = decay
        self._emotions = emotions
       

    def getSomaticAction(self, state):
        actionsOnState = self._getActionsOnState(state)

        if len(actionsOnState) > 0:
            maxAction = max(actionsOnState, key=actionsOnState.get)
            if actionsOnState[maxAction] > self._up_threshold:
                #self._logger.info('----> somatic action[%d] to state[%d]',maxAction,state)
                return maxAction
        
        return None

    def _getSomaticResponse(self, state):
        APs = []
        AMs = {}

        actionsOnState = self._getActionsOnState(state)

        for key in actionsOnState:
            if actionsOnState[key] < self._down_threshold:
                APs.append(key)
            else:
                AMs[key] = actionsOnState[key]

        return APs, AMs

    def _feel(self,originalState,action,destinationState, reward):
        feeling = 0.0
        ne = 0

        for emotion in self._emotions:
            f = self._emotions[emotion](originalState,action,destinationState, reward)
            #print("return emotion", emotion, f)
            if f != None and f != 0.0:
                feeling = feeling + f
                ne += 1
            
        if feeling != 0.0:
            feeling = feeling/ne

        return feeling
    
    def learning(self,originalState,action,destinationState, reward):
        feeling = self._feel(originalState,action,destinationState, reward)
        
        #self._logger.info('learning feeling[%f] - originalState[%d],action[%d],destinationState[%d], reward[%d]', feeling,originalState,action,destinationState, reward)

        if feeling != None and feeling != 0.0:
            if (originalState,action) in self._somaticMemory.keys():
                self._somaticMemory[(originalState,action)] = (1-self._decay)*self._somaticMemory[(originalState,action)] + self._decay*feeling
            else:
                self._somaticMemory[(originalState,action)] = feeling

    def getSomaticMemory(self):
        return self._somaticMemory
    
    def setSomaticMemory(self, sm:dict):
        self._somaticMemory = sm
    
    def _getActionsOnState(self, state)->dict:
        ret = {}
        for key in self._somaticMemory:
            if key[0] == state:
                ret[key[1]] = self._somaticMemory[key]

        return ret
    
    def chooseAction(self, possibleActions:dict, state):
        APs, AMs = self._getSomaticResponse(state)

        for a in APs:
            if (a in possibleActions):
                del possibleActions[a] 

        for a in AMs:
            if (a in possibleActions) and (AMs[a] != 0.0):
                possibleActions[a] = (possibleActions[a] + (AMs[a]*1))/2

        #self._logger.info('choosing action on[%d] from [%s]',state,possibleActions)

        maxAction = random.choice([k for k, v in possibleActions.items() if v == max(possibleActions.values())]) #getting random action from max values
        
        #self._logger.info('choose [%d]',maxAction)
        return maxAction

   
    