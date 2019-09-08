import numpy as np 

class discreteHMM:
	"""
    A class implementing a HMM with discrete observation values

    ...

    Attributes
    ----------
    states: int
    	The number of hidden states in the HMM
    transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
    observations : np.array(T)
       A sequence of observation values of length T
    probabilityMatrix : np.array((states,num_observations))
       The emission probability matrix, probabilityMatrix[i,j] is 
       the probability of observing observation j in hidden state i
    updateProbability : bool
    	Decide if the probabilityMatrix should be updated during fitting
    	or not

    Methods	
    -------
    fit_parameters(updateInitialState,updateProbability,maxIter)
      fit the model parameters using the EM algorithm for maxIter iterations

   	predict_path(num_samples):
			computes the most likely sequence of hidden states using the viterbi
			algorithm. The standard deviation of the posterior distribution of paths
			is calculated by sampling num_sample paths from the posterior distribution.

    """
  states = None
	transitionMatrix = None
	initialState = None
	observations = None
	probabilityMatrix = None
	updateProbability = True


	def __init__(self,observations,initalState,transitionMatrix,probabilityMatrix,updateProbability):
		"""
		Parameters
		----------
		observations : np.array(T)
			a numpy array containing the sequence of observations
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    probabilityMatrix : np.array((states,num_observations))
       The emission probability matrix, probabilityMatrix[i,j] is 
       the probability of observing observation j in hidden state i
    updateProbability : bool
    	Decide if the probabilityMatrix should be updated during fitting
    	or not
		"""
		self.states = transitionMatrix.shape[0]
		self.observations = observations
		self.transitionMatrix = transitionMatrix
		self.initialState = initialState
		self.probabilityMatrix = probabilityMatrix
		self.updateProbability = updateProbability


	def fit_parameters(self,updateInitialState,updateProbability,maxIter):
		""" Fits the parameters of the HMM

		If update initialState is true the initial state vector will be updated
		If updateProbability is true the emission probability vector will be updated
		The EM algorithm will be applied maxIter number of times or untill 
		a division by zero error is encountered.	

		Parameters
		----------
		updateInitialState : bool
			Weither the initial state vector will be updated
		updateProbability : bool
			Weither the emission probability matrix will be updated
		maxIter : int
			The maximum number of iterations to apply EM for

		 """
		bMatrix, transitionMatrix,initialState = self.__EM(self,self.transitionMatrix,
			self.observations,self.initialState,self.discreteEmissionProb,
			self.discreteEmissionUpdate,self.probabilityMatrix,maxIter,
			updateInitialState,updateProbability)
		self.probabilityMatrix = bMatrix
		self.transitionMatrix = transitionMatrix
		self.initialState = initialState

	def predict_path(self,num_samples):
		""" Computes the Viterbi path and "confidence"

		The Viterbi algorithm is used to compute the most likely path. 
		The standard deviation of the posterior distribution is calculated
		as this has been shown to correlate with the number of incorrectly
		predicted transitions.

		Parameters
		----------
		num_samples : int
			The number of samples to take from the posterior distribution
			when estimating its standard deviation

		Returns
		-------
		path : np.array((T))
			A array of length T with the predicted states
		std_post : float
      The estimated standard deviation on the posterior
      distribution of paths.
		 """

		path = veritebi(self,self.observations,self.transitionMatrix,
			self.probabilityMatrix,self.discreteEmissionProb,self.initialState)
		gamma,xi = self.forwardBackward(self,self.transitionMatrix,self.observations,
			self.initialState,self.discreteEmissionProb,self.probabilityMatrix)
		samples = self.generateSamples(self,num_samples,gamma,xi)
		std_post = np.mean(np.abs(np.std(samples,axis=0)))
		return path,std_post

	def generateSamples(self,num_samples,gamma,xi):
		""" Samples paths from the posterior distribution of states


		Parameters
		----------
		num_samples : int
			The number of samples to take from the posterior distribution

		gamma : np.array((T,Samples))
			The smoothed marginal distribution

		xi : np.array((T-1,Samples,Samples))
		  The smoothed two-sliced marginal

		Returns
		-------
		samples : np.array((num_samples,T))
			A array with the num_samples different sampled
			paths
		 """
	  samples = np.zeros((num_samples,gamma.shape[0]),dtype='int')
	  for sample in range(num_samples):
	    samples[sample,0] = 0
	    a = np.array([0,1,2,3])
	    for time in range(1,gamma.shape[0]):
        prob = xi[time-1,samples[sample,time-1],:]/np.sum(xi[time-1,samples[sample,time-1],:])
        samples[sample,time] = np.random.choice(a, p=prob)
	  return samples

	def discreteEmissionProb(self,state,observation,b):
		""" Returns the probability of emitting a observation in a state

		Parameters
		----------
		state : int
			The hidden state

		observation : int
			The observed observations

		b : np.array((state,num_observables))
		  The emission probability matrix

		Returns
		-------
		b[state,observation] : float
		  The probability of observing the observation in state state.
		 """
	  return b[state,observation]

	def discreteEmissionUpdate(self,state,gamma,observation,b):
	  """ updates the emission probabilities in state state

		Parameters
		----------
		state : int
			The hidden state

		gamma : np.array((T,Samples))
			The smoothed marginal distribution

		xi : np.array((T-1,Samples,Samples))
		  The smoothed two-sliced marginal

		b : np.array((state,num_observables))
		  The emission probability matrix

		Returns
		-------
		newStateEmission : np.array(observables)
		  The emission probability vector for state state
		 """

	  newStateEmission = np.zeros(b[state,:].shape)
	  for t,o in enumerate(observation):
	    newStateEmission[observation[t]] += gamma[t,state]
	  newStateEmission = newStateEmission/np.sum(gamma[:,state])
	  return newStateEmission
	  
	def forwardPass(self,transitionMatrix,observations,initialState,emissionProb,bMatrix):
	  """ Does a forward pass to compute the forward probability

		Parameters
		----------
		
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    observations : np.array(T)
			a numpy array containing the sequence of observations
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
    emissionProbability : function
    	The function that computes the emission probability for a given
    	state and observation
    bMatrix : np.array((states,num_observations))
       The emission probability matrix, bMatrix[i,j] is 
       the probability of observing observation j in hidden state i

		Returns
		-------
		f : np.array((T,states))
		  The forward probability
		 """
	  N = transitionMatrix.shape[0]
	  T = observations.shape[0]
	  f = np.zeros((T,N))
	  for t in range(T):
	    if t == 0:
	      for i in range(N):
	        f[0,i] = initialState[i]*emissionProb(i,observations[0],bMatrix)
	    else:
	      for i in range(N):
	        for j in range(N):
	          f[t,i] += f[t-1,j]*transitionMatrix[j,i]*emissionProb(j,observations[t],bMatrix)
	    f[t,:] = f[t,:]/np.sum(f[t,:])
	  return f
	  
	def backwardPass(self,transitionMatrix,observations,initialState,emissionProb,bMatrix):
	  """ Does a backwards pass to compute the backwards probability

		Parameters
		----------
		
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    observations : np.array(T)
			a numpy array containing the sequence of observations
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
    emissionProbability : function
    	The function that computes the emission probability for a given
    	state and observation
    bMatrix : np.array((states,num_observations))
       The emission probability matrix, bMatrix[i,j] is 
       the probability of observing observation j in hidden state i

		Returns
		-------
		b : np.array((T,states))
		  The backward probability
		 """
	  N = transitionMatrix.shape[0]
	  T = observations.shape[0]
	  b = np.zeros((T,N))
	  for t in range(T-1,-1,-1):
	    if t == T-1:
	      for i in range(N):
	        b[t,i] = 1
	    else:
	      for i in range(N):
	      	for j in range(N):
	          b[t,i] += transitionMatrix[i,j]*emissionProb(j,observations[t+1],bMatrix)*b[t+1,j]
	    b[t,:] = b[t,:]/np.sum(b[t,:])
	  return b
	  
	def forwardBackward(self,transitionMatrix,observations,initialState,emissionProb,bMatrix):
	  """ The forward backward algorithm

	  Computes the smoothed marginal distribution and the
	  two-sliced marginal distribution using the forwad-backwards 
	  algorithm

		Parameters
		----------
		
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    observations : np.array(T)
			a numpy array containing the sequence of observations
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
    emissionProbability : function
    	The function that computes the emission probability for a given
    	state and observation
    bMatrix : np.array((states,num_observations))
       The emission probability matrix, bMatrix[i,j] is 
       the probability of observing observation j in hidden state i

		Returns
		-------
		gamma : np.array((T,states))
		  The smoothed marginal distribution
		xi : np.array((T,states,states))
		  The two-sliced marginal distribution
		 """
	  N = transitionMatrix.shape[0]
	  T = observations.shape[0]
	  gamma = np.zeros((T,N))
	  xi = np.zeros((T-1,N,N))
	  f = forwardPass(transitionMatrix,observations,initialState,emissionProb,bMatrix)
	  b = backwardPass(transitionMatrix,observations,initialState,emissionProb,bMatrix)
	  for t in range(T-1):
	    for i in range(N):
	      for j in range(N):
	        xi[t,i,j] = f[t,i]*transitionMatrix[i,j]*emissionProb(j,observations[t+1],bMatrix)*b[t+1,j]
	    xi[t,:,:] = xi[t,:,:]/np.sum(xi[t,:,:])
	  for t in range(T):
	    for i in range(N):
	      gamma[t,i] = f[t,i]*b[t,i]
	    gamma[t,:] = gamma[t,:]/np.sum(gamma[t,:])    
	  return gamma,xi
	  
	def EMstep(self,transitionMatrix,observations,initialState,
	       emissionProb,updateRule,bMatrix,init,updateProbability):
	 """ The Expectation Maximization algorithm

	  Uses one iteration of the expectation maximization algorithm
	  to update the parameters of the hidden markov model. If any of 
	  the parameters evaluate to NaN the previos versions of the 
	  parameters are returned.

		Parameters
		----------
		
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    observations : np.array(T)
			a numpy array containing the sequence of observations
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
    emissionProbability : function
    	The function that computes the emission probability for a given
    	state and observation
    updateRule : function
    	The function that for a given state returns the updated vector of 
    	emission probabilities for that stae
    bMatrix : np.array((states,num_observations))
       The emission probability matrix, bMatrix[i,j] is 
       the probability of observing observation j in hidden state i
    init : bool
    	Weither the initial state probability should be updated
    updateProbability : bool
      Weither the emission probability matrix should be updated.

		Returns
		-------
		newbMat : np.array((states,observables))
		  the new emission probability matrix
		nTM : np.array((states,states))
		  the new transition matix
		newPI : np.array((states))
		  the new initial probability vector
		db : float
		  the size of the emission probability update
		dT : float
		  the size of the transition matrix update
		dP : float
		  the size of the initial probability update
		 """
	  
	  states = initialState.shape[0]
	  T = observations.shape[0]

	  gamma,xi = forwardBackward(transitionMatrix,observations,initialState,emissionProb,bMatrix)
	  N = transitionMatrix.shape[0]
	  #nTM = np.zeros(transitionMatrix.shape)
	  nTM = transitionMatrix.copy()
	  for n in range(N):
	    for m in range(N):
	      nTM[n,m] = np.sum(xi[:,n,m])/np.sum(xi[:,n,:]) 
	  
	  newbMat = np.empty(bMatrix.shape)
	  if init:
	    newPI = np.zeros(states)
	    newPI = gamma[0,:]
		else:
	    newPI = initialState
	  if updateProbability:
	  	for state in range(N):
	    	newbMat[state,:] = updateRule(state,gamma,observations,bMatrix)
	  else:
	  	newbMat = bMatrix

	  dP = np.sqrt(np.sum((initialState-newPI)**2))
	  db = np.sqrt(np.sum((bMatrix-newbMat)**2))
	  dT = np.sqrt(np.sum((transitionMatrix-nTM)**2))
	  
	  if np.isnan(newbMat).any() or np.isnan(nTM).any() or np.isnan(newPI).any():
	    return bMatrix,transitionMatrix,initialState,0,0,0
	  
	  return newbMat,nTM,newPI,db,dT,dP
	    
	def EM(self,transitionMatrix,observations,initialState,emissionProb,updateRule,
		bMatrix,maxIter,init,updateProbability):
		""" The expectation-maximization algorithm

		The expectation-maximization algorithm updates the parameters of the HMM
		for maxIter iterations or untill the update returns a paramer containins
		a NaN value in which case the previous version of the parameters are returned

		Parameters
		----------
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    observations : np.array(T)
			a numpy array containing the sequence of observations
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i
    emissionProbability : function
    	The function that computes the emission probability for a given
    	state and observation
    updateRule : function
    	The function that for a given state returns the updated vector of 
    	emission probabilities for that stae
    bMatrix : np.array((states,num_observations))
       The emission probability matrix, bMatrix[i,j] is 
       the probability of observing observation j in hidden state i
    maxIter : int
      The maximum number of iteratio

    init : bool
    	Weither the initial state probability should be updated
    updateProbability : bool
      Weither the emission probability matrix should be updated.

		Returns
		-------
		newbMat : np.array((states,observables))
		  the new emission probability matrix
		nTM : np.array((states,states))
		  the new transition matix
		newPI : np.array((states))
		  the new initial probability vector
		db : float
		  the size of the emission probability update
		dT : float
		  the size of the transition matrix update
		dP : float
		  the size of the initial probability update
		"""
	  steps = 0
	  while True:
	    bMatrix,transitionMatrix,initialState,db,dT,dP = EMstep(transitionMatrix,observations,initialState,
	   emissionProb,updateRule,bMatrix,init,updateProbability)
	    steps += 1
	    if steps==maxIter:
	      return bMatrix, transitionMatrix,initialState
	  return bMatrix, transitionMatrix,initialState

	def viterbi(self,observations,transitionMatrix,bMatrix,emissionProb,initialState):
		""" Computer the most probable path

		Uses the viterbi algorithm to compute the most probable path 

		Parameters
		----------

		observations : np.array(T)
			a numpy array containing the sequence of observations
		transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    bMatrix : np.array((states,num_observations))
       The emission probability matrix, bMatrix[i,j] is 
       the probability of observing observation j in hidden state i
    emissionProb : function
    	The function that computes the emission probability for a given
    	state and observation
		initialState : np.array(states)
       A one dimensional probability vector where initialState[i]
       is the probability of starting in state i

    Returns
    -------
    path : np.array(T)
    	The most probable sequence of hidden states
		"""
	  T = observations.shape[0]
	  N = transitionMatrix.shape[0]
	  path = np.empty(T)
	  v = np.zeros((T,N))
	  b = np.zeros((T,N))
	  path = np.ones(T)*(N-1)
	  
	  states = initialState.shape[0]
	  T = observations.shape[0]
	  
	  for t in range(T):
	    if t == 0:
	      for i in range(N):
	        v[t,i] = emissionProb(i,observations[t],bMatrix)*initialState[i]
	      else:
	        for i in range(N):
	          v[t,i] = np.max(emissionProb(i,observations[t],bMatrix)*v[t-1,:]*transitionMatrix[:,i])
	          b[t,i] = np.argmax(emissionProb(i,observations[t],bMatrix)*v[t-1,:]*transitionMatrix[:,i])
	  path[T-1] = np.argmax(v[T-1,:])
	  for t in range(T-2,-1,-1):
	    path[t] = b[t+1,int(path[t+1])]
	  return path
	  

