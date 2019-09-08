import numpy as np 

class gaussianHMM:
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    sigma : np.array(states,observables,observables)
      The covariance matrix

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
  mu = None
  sigma = None



  def __init__(self,observations,initalState,transitionMatrix,mu,sigma):
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    sigma : np.array(states,observables,observables)
      The covariance matrix
    probabilityMatrix : np.array((states,num_observations))
       The emission probability matrix, probabilityMatrix[i,j] is 
       the probability of observing observation j in hidden state i

    """
    self.states = transitionMatrix.shape[0]
    self.observations = observations
    self.transitionMatrix = transitionMatrix
    self.initialState = initialState
    self.mu = mu 
    self.sigma = sigma 



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
    mu,sigma, transitionMatrix,initialState,dM,dS,dT,dP = EM(self.transitionMatrix,
      self.observations,self.initialState,
      self.multivariateGaussian,self.multivariateGaussianMSStep,
      self.mu,self.sigma,maxIter,init=True)
  
    self.mu = mu 
    self.sigma = sigma 
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
    sigma = self.sigma
    mu = self.mu
    Q = np.empty((states,traces,traces))
    det = np.empty(states)
    for state in range(states):
      Q[state,:,:] = np.linalg.inv(sigma[state,:,:])
      det[state] = np.linalg.det(sigma[state,:,:])

    path = veritebi(self,self.observations,self.transitionMatrix,
      self.mu,self.sigma,self.multivariateGaussian,self.initialState)
    gamma,xi = self.forwardBackward(self,self.transitionMatrix,self.observations,
      self.initialState,self.multivariateGaussian,mu,det,Q)
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

  def multivariateGaussian(state,observation,mu,det,Q):
     """ Returns the probability of emitting a observation in a state

    Parameters
    ----------
    state : int
      The hidden state

    observation : np.array(observables)
      The observed observables

    mu : np.array((states,observables))
      The matrix of mean vectors

    det : np.array(states)
      The determinant of the covarianc matrix
      for all the states

    Q : np.array((states,observables,observables))
      The inverse of the covariance matrix for all the
      states

    Returns
    -------
    prob : float
      The probability of observing the observation in state state.
     """
    m = mu[state,:]
    d = det[state]
    q = Q[state,:,:]
    M = observation.shape[0]
    diff = np.matrix(observation-m)
    
    
    prob = np.exp(-0.5*(diff*q*diff.transpose())[0,0])/np.sqrt((2*np.pi)**M*d)
    return prob

  def multivariateGaussianMSStep(state,gamma,observation,mu,Sigma):
     """ updates the emission probabilities in state state

    Parameters
    ----------
    state : int
      The hidden state

    gamma : np.array((T,Samples))
      The smoothed marginal distribution

    mu : np.array((states,observables))
      The matrix of mean vectors

    sigma : np.array(states,states)
      The covariance matrix

    Returns
    -------
    muVec : np.array(observables)
      The new mean vector
    sigma : np.array(observables,observables)
      The new covariance matrix
     """
    gammaState = gamma[:,state]
    muVec = np.dot(gammaState,observation)/np.sum(gammaState)
    
    T,N = observation.shape
    sigma = np.zeros((N,N))
    for t in range(T):
      diff = observation[t,:]-muVec
      sigma += gammaState[t]*np.reshape(diff,(N,1))*np.reshape(diff,(1,N))
    sigma = sigma/np.sum(gammaState)
    return muVec,sigma

  def forwardPass(transitionMatrix,observations,initialState,emissionProb,mu,det,Q):
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    det : np.array(states)
      The determinant of the covarianc matrix
      for all the states

    Q : np.array((states,observables,observables))
      The inverse of the covariance matrix for all the
      states

    Returns
    -------
    f : np.array((T,states))
      The forward probability
     """
    N = transitionMatrix.shape[0]
    T,M = observations.shape
    f = np.zeros((T,N))
    for t in range(T):
      if t == 0:
        for i in range(N):
          f[0,i] = initialState[i]*emissionProb(i,observations[0,:],mu,det,Q)
      else:
        for i in range(N):
          for j in range(N):
            f[t,i] += f[t-1,j]*transitionMatrix[j,i]*emissionProb(j,observations[t,:],mu,det,Q)
      f[t,:] = f[t,:]/np.sum(f[t,:])
    return f
      
  def backwardPass(transitionMatrix,observations,initialState,emissionProb,mu,det,Q):
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    det : np.array(states)
      The determinant of the covarianc matrix
      for all the states

    Q : np.array((states,observables,observables))
      The inverse of the covariance matrix for all the
      states

    Returns
    -------
    b : np.array((T,states))
      The backward probability
     """
    N = transitionMatrix.shape[0]
    T,M = observations.shape
    b = np.zeros((T,N))
    for t in range(T-1,-1,-1):
      if t == T-1:
        for i in range(N):
          b[t,i] = 1
      else:
        for i in range(N):
          for j in range(N):
            b[t,i] += transitionMatrix[i,j]*emissionProb(j,observations[t+1,:],mu,det,Q)*b[t+1,j]
      b[t,:] = b[t,:]/np.sum(b[t,:])
    return b

  def forwardBackward(transitionMatrix,observations,initialState,emissionProb,mu,det,Q):
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    det : np.array(states)
      The determinant of the covarianc matrix
      for all the states

    Q : np.array((states,observables,observables))
      The inverse of the covariance matrix for all the
      states

    Returns
    -------
    gamma : np.array((T,states))
      The smoothed marginal distribution
    xi : np.array((T,states,states))
      The two-sliced marginal distribution
     """
    N = transitionMatrix.shape[0]
    T,M = observations.shape
    gamma = np.zeros((T,N))
    xi = np.zeros((T-1,N,N))
    f = forwardPass(transitionMatrix,observations,initialState,emissionProb,mu,det,Q)
    b = backwardPass(transitionMatrix,observations,initialState,emissionProb,mu,det,Q)
    for t in range(T-1):
      for i in range(N):
          for j in range(N):
            xi[t,i,j] = f[t,i]*transitionMatrix[i,j]*emissionProb(j,observations[t+1,:],mu,det,Q)*b[t+1,j]
      xi[t,:,:] = xi[t,:,:]/np.sum(xi[t,:,:])
    for t in range(T):
      for i in range(N):
        gamma[t,i] = f[t,i]*b[t,i]
      gamma[t,:] = gamma[t,:]/np.sum(gamma[t,:])    
    return gamma,xi
    
  def EMstep(transitionMatrix,observations,initialState,
       emissionProb,updateRule,mu,sigma,init=True):
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    sigma : np.array(states,observables,observables)
      The covariance matrix
    init : bool
      Weither the initial state probability should be updated
    

    Returns
    -------
    newMu : np.array((states,observables))
      the new mean value matrix
    newSigma : np.array((states,observables,observables))
      the new array of covariance matrixes
    nTM : np.array((states,states))
      the new transition matix
    newPI : np.array((states))
      the new initial probability vector
    dM : float
      the size of the mu vector update
    dS : float
      the size of the covariance update
    dT : float
      the size of the transition matrix update
    dP : float
      the size of the initial probability update
     """
    
    
    states = initialState.shape[0]
    T,traces = observations.shape

    Q = np.empty((states,traces,traces))
    det = np.empty(states)
    for state in range(states):
      Q[state,:,:] = np.linalg.inv(sigma[state,:,:])
      det[state] = np.linalg.det(sigma[state,:,:])
    
    gamma,xi = forwardBackward(transitionMatrix,observations,initialState,emissionProb,mu,det,Q)
    N = transitionMatrix.shape[0]
    nTM = transitionMatrix.copy()
    for n in range(N):
      for m in range(N):
        nTM[n,m] = np.sum(xi[:,n,m])/np.sum(xi[:,n,:]) 

    newMu = np.empty((states,traces))
    newSigma = np.empty((states,traces,traces))
    if init:
        newPI = gamma[0,:]
    else:
        newPI = initialState
    for state in range(N):
        newMu[state,:],newSigma[state,:,:] = updateRule(state,gamma,observations,mu,sigma)
    dP = np.sqrt(np.sum((initialState-newPI)**2))
    dM = np.sqrt(np.sum((mu-newMu)**2))
    dS = np.sqrt(np.sum((sigma-newSigma)**2))
    dT = np.sqrt(np.sum((transitionMatrix-nTM)**2))
    if np.isnan(newMu).any() or np.isnan(newSigma).any() or np.isnan(nTM).any() or np.isnan(newPI).any():
        return mu,sigma,transitionMatrix,initialState,0,0,0,0
    return newMu,newSigma,nTM,newPI,dM,dS,dT,dP
    
def EM(transitionMatrix,observations,initialState,emissionProb,updateRule,
  mu,sigma,maxIter,init=True):
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
    mu : np.array((states,observables))
      The matrix of mean vectors

    sigma : np.array(states,observables,observables)
      The covariance matrix
    maxIter : int
      The maximum number of iteratio

    init : bool
      Weither the initial state probability should be updated


    Returns
    -------
    newMu : np.array((states,observables))
      the new mean value matrix
    newSigma : np.array((states,observables,observables))
      the new array of covariance matrixes
    nTM : np.array((states,states))
      the new transition matix
    newPI : np.array((states))
      the new initial probability vector
    dM : float
      the size of the mu vector update
    dS : float
      the size of the covariance update
    dT : float
      the size of the transition matrix update
    dP : float
      the size of the initial probability update
    """
    steps = 0
    while True:
        mu,sigma,transitionMatrix,initialState,dM,dS,dT,dP = EMstep(transitionMatrix,observations,initialState,
       emissionProb,updateRule,mu,sigma,limit,back,init)
        steps += 1
        if steps == maxIter:
            return mu,sigma, transitionMatrix,initialState,dM,dS,dT,dP
    return mu,sigma, transitionMatrix,initialState,dM,dS,dT,dP
    
  
  
  def viterbi(self,observations,transitionMatrix,mu,sigma,emissionProb,initialState):
    """ Computer the most probable path

    Uses the viterbi algorithm to compute the most probable path 

    Parameters
    ----------

    observations : np.array(T)
      a numpy array containing the sequence of observations
    transitionMatrix : np.array((states,states))
       A probability matrix where transitionMatrix[i,j] is the
       probability of transitioning from state i to state j
    mu : np.array((states,observables))
      The matrix of mean vectors
    sigma : np.array(states,observables,observables)
      The covariance matrix
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
    T,M = observations.shape
    N = transitionMatrix.shape[0]
    path = np.empty(T)
    v = np.zeros((T,N))
    b = np.zeros((T,N))
    path = np.ones(T)*(N-1)
    
    states = initialState.shape[0]
    T,traces = observations.shape
    
    
    Q = np.empty((states,traces,traces))
    det = np.empty(states)
    for state in range(states):
        Q[state,:,:] = np.linalg.inv(sigma[state,:,:])
        det[state] = np.linalg.det(sigma[state,:,:])
    
    for t in range(T):
      if t == 0:
        for i in range(N):
          v[t,i] = emissionProb(i,observations[t],mu,det,Q)*initialState[i]
        else:
          for i in range(N):
            v[t,i] = np.max(emissionProb(i,observations[t],mu,det,Q)*v[t-1,:]*transitionMatrix[:,i])
            b[t,i] = np.argmax(emissionProb(i,observations[t],mu,det,Q)*v[t-1,:]*transitionMatrix[:,i])
    path[T-1] = np.argmax(v[T-1,:])
    for t in range(T-2,-1,-1):
      path[t] = b[t+1,int(path[t+1])]
    return path
    

