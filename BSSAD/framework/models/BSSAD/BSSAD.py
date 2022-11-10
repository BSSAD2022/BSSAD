import numpy as np
from tqdm import tqdm
import random
from ...preprocessing import ContinousSignal,DiscreteSignal
from filterpy.kalman import unscented_transform, JulierSigmaPoints
from numpy.random import multivariate_normal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
from scipy.linalg import cholesky
from ..base import BaseModel,DataExtractor,override
from . import nearestPD
from ...utils.metrics import bf_search
from scipy.spatial.distance import mahalanobis


class BSSAD(BaseModel,DataExtractor):
    '''
        Bayesian State Space Anomaly Detection (BSSAD)
        Parameters
        ----------
        signals : set of discrete or continuous signals
        input_range : the length of input sequence for encoder
        tau : time window
    ''' 
    def __init__(self, signals, tau, input_range):
        self.signals = signals
        self.tw = tau
        self.input_range = input_range
        
        self.targets = []
        self.sensors = []
        for signal in self.signals:
            if signal.isInput==True:
                if isinstance(signal, ContinousSignal):
                    self.sensors.append(signal.name)
                if isinstance(signal, DiscreteSignal):
                    self.sensors.extend(signal.get_onehot_feature_names())
            if signal.isOutput==True:
                self.targets.append(signal.name)
        
        'mean vector of hidden states'
        self.z_ekf = None
        self.z_ukf = None
        self.z_pf = None
        'covariance matrix of hidden states'
        self.P_ekf = None
        self.P_ukf = None
        self.P_pf = None
        'process noise covariance matrix'
        self.Q = None 
        'sensor measurement noise covariance matrix'
        self.R = None
        'neural networks for encoding to hidden state'
        self.h_nn = None
        'neural networks for state transition'
        self.f_nn = None
        'neural networks for decoding'
        self.h_inv_nn = None
        'Sigma points/Particles'
        self.sigmas = None
        
        self.transformed_particles_noisy = None
        
        self.x_particles_ukf = None
        
        self.loss_weights = [0.45,0.45,0.1]
    
    @override
    def extract_data(self, df_ori, freq=1, purpose='train', label=None):
        '''
        Extract training, testing & validatation sets from data frame   
        Ref - https://github.com/NSIBF/NSIBF
        Parameters
        ----------
        df : the Pandas DataFrame containing the data
        freq : the sampling frequency (default is 1)
        label : the name of the anomaly label column (defualt is None)
        
        Returns
        -------
        x : the output target variables, matrix of shape = [n_timesteps, n_targets]
        u : the output sensors, matrix of shape = [n_timesteps, input_range, n_features]
        y : if purpose = 'train', output target variables, matrix of shape = [n_timesteps, n_targets]
        z : if label is not None, anomaly labels, matrix of shape = [n_timesteps, tau]
        '''            
        df = df_ori.copy()
        x_feats, u_feats, y_feats, z_feats = [], [], [], []
        
        for entry in self.targets:
            for i in range(1,self.tw+1):
                if i < self.tw:
                    j = self.tw-i
                    df[entry+'-'+str(j)] = df[entry].shift(j)
                    x_feats.append(entry+'-'+str(j))
                else:
                    x_feats.append(entry)  
            if purpose == 'train':
                for i in range(1,self.tw+1):
                    df[entry+'+'+str(i)] = df[entry].shift(-i)
                    y_feats.append(entry+'+'+str(i))
            
        for entry in self.sensors:
            for i in range(1,self.input_range+1):
                if i < self.input_range:
                    j = self.input_range-i
                    df[entry+'-'+str(j)] = df[entry].shift(j)
                    u_feats.append(entry+'-'+str(j))
                else:
                    u_feats.append(entry)
                    
        if label is not None:
            for i in range(1,self.tw+1):
                if i < self.tw:
                    j = self.tw-i
                    df[label+'-'+str(j)] = df[label].shift(j)
                    z_feats.append(label+'-'+str(j))
                else:
                    z_feats.append(label)       
        
        df = df.dropna(subset=x_feats+u_feats+y_feats+z_feats)
        df = df.reset_index(drop=True)
        
        if freq > 1:
            df = df.iloc[::freq,:]
            df = df.reset_index(drop=True)
        
        x = df.loc[:,x_feats].values
        
        if len(u_feats) > 0:
            u = df.loc[:,u_feats].values
            u = np.reshape(u, (len(u),len(self.sensors),self.input_range) )
            u = np.transpose(u,(0,2,1))
        else:
            u = None
        
        if label is None:
            z = None
        else:
            z = df.loc[:,z_feats].values
        
        if purpose=='train':
            y = df.loc[:,y_feats].values
            return x, u, y, z
        elif purpose == 'predict':
            return x, u, None, z
        elif purpose=='AD':
            return x, u, None, z
    
    
    def score_samples(self, x, u, seed, n_particles = 20, filterType = "PF", resample_method="systematic",  reset_hidden_states=True):
        '''
        Calculate anomalies scores via Bayesian State Space Algorithm - Ensemble Kalman Filter (EnKF) and Particle Filter (PF)
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_timesteps, n_targets]
        u : the sensors, matrix of shape = [n_timesteps, input_range, n_features]
        n_particles : number of sigma_points/particles to sample from prior distribution (default is 20)
            possible values: 10, 20, 50 (for EnKF) & 500, 1000, 2000 (for PF)
        resample_method : resampling method to use for PF (default is systematic)
            possible values: "resample", "residual", "stratified", or "systematic"
        filterType : select filter type (EnKF, or PF) (default is PF)
        reset_hidden_states : Initialize filter with hidden states (default is True)
        
        Returns
        -------
        anomaly_scores : the anomaly scores from the second timestep, matrix of shape = [n_timesteps-1, ]
        '''
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if self.Q is None or self.R is None:
            print('please estimate noise before running this method!')
            return None
        
        if reset_hidden_states:
            self.z_ekf = self._encoding(x[0,:])
            self.P_ekf = np.diag([100.00001]*len(self.z_ekf))
            self.z_ukf = self._encoding(x[0,:])
            self.P_ukf = np.diag([0.00001]*len(self.z_ukf))
            self.z_pf = self._encoding(x[0,:])
            self.P_pf = np.diag([0.00001]*len(self.z_pf))
            
        if np.isscalar(self.Q):
            self.Q = np.eye(len(self.z_ekf)) * self.Q
         
        #Sigma points, size = number of sigma points, specify when passing in
        sp = multivariate_normal(mean = self.z_ekf, cov = self.P_ekf, size = n_particles)
        
        particles = multivariate_normal(mean = self.z_pf, cov = self.P_pf, size = n_particles)
        
        anomaly_scores = []
        #for t in range(1,len(x)):
            #print(t,'/',len(x))
        for t in tqdm(range(1, len(x))):
            u_t = u[t-1,:,:]
            x_t = x[t,:]
            
            if filterType == "PF":
                zmu, Pcov, particles_temp = self._UKFPF(x_t, u_t, n_particles, particles, resample_method)
                particles = particles_temp
                
            if filterType == "UKF-PF":
                zmu, Pcov, particles_temp = self._PF(x_t, u_t, n_particles, particles, resample_method)
                particles = particles_temp
                
            if filterType == "EnKF":
                zmu, Pcov = self._EnKF(x_t, u_t, sp, n_particles)
                
            if filterType == "UKF":
                zmu, Pcov = self._UKF(x_t, u_t)
            
                   
            inv_Pcov = np.linalg.pinv(Pcov)
            score = mahalanobis(x[t,:], zmu, inv_Pcov)
            anomaly_scores.append(score)
        
        return np.array(anomaly_scores)
    
  
    def estimate_noise(self, x, u, y):
        '''
        Estimate noise
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_timesteps, n_targets]
        u : the sensors, matrix of shape = [n_timesteps, input_range, n_features]
        y : the output data for targets, matrix of shape = [n_timesteps, n_targets]
        
        Returns
        -------
        self 
        Q : state space noise covariance matrix, matrix of shape = [z_dim, z_dim]
        R: measurement noise covariance matrix, matrix of shape = [n_targets, n_targets]
        '''   
        s = self.h_nn.predict(x)
        s_next_true = self.h_nn.predict(y)
        s_next_pred = self.f_nn.predict([s,u])
        self.Q = np.cov(np.transpose(s_next_pred-s_next_true))
        
        x_pred = self.h_inv_nn.predict(s)
        self.R = np.cov(np.transpose(x_pred-x))
        return self
    
    def _encoding(self, x):
        x = np.array([x]).astype(np.float)
        z = self.h_nn.predict(x)
        return z[0,:]
        
    def _state_transition_func(self, z, u):        
        U = np.array([u]*len(z))
        X = [z, U]
        z_next = self.f_nn.predict(X)
        return z_next 
    
    def _measurement_func(self, z):
        y = self.h_inv_nn.predict(z)
        return y
    
    def _sqrt_func(self, x):
        try:
            result = cholesky(x)
        except:
            result = np.linalg.cholesky(nearestPD(x))
        return result
    
    def auto_correlation(self, sigmas, musigmas, R):
        '''
        Calculating the autocorrelation   
        Parameters
        ----------
        sigmas : the sampled sigma points, matrix of shape = [n_particles, n_targets] 
        musigmas : the mean of sigma points of size, matrix of shape = [n_targets, ]
        R : Noise covariance for state measurements, matrix of shape = [n_targets, n_targets]
        
        Returns
        -------
        Pzz : covariance matrix of size, matrix of shape = [z_dim, z_dim]
        '''    
        Pzz = 0
        for sigma in sigmas:
            s = sigma - musigmas
            Pzz += np.outer(s, s)
        Pzz = Pzz / sigmas.shape[0] + R
        return Pzz

    def cross_correlation(self, Fsigmas, Hsigmas, muFsigmas, muHsigmas):
        '''
        Calculating the crosscorrelation   
        Parameters
        ----------
        Fsigmas : the sampled sigma points, matrix of shape = [n_particles, n_targets] 
        muFsigmas : the mean of sigma points of size, matrix of shape = [n_targets, ]
        Hsigmas : the sampled sigma points, matrix of shape = [n_particles, dim_z]
        muHsigmas : the mean of sigma points of size, matrix of shape = [dim_z, ]
        
        Returns
        -------
        Pxz : covariance matrix, , matrix of shape = [n_targets, n_targets]   
        '''  
        Pxz = 0
        for i in range(Hsigmas.shape[0]):
            Pxz += np.outer(np.subtract(Fsigmas[i], muFsigmas),np.subtract(Hsigmas[i], muHsigmas))
        Pxz /= (Hsigmas.shape[0] - 1)
        return Pxz
    
    def squared_error(self, x, y, sigma=100):
        '''
        RBF kernel, supporting masked values in the observation
        Parameters:
        -----------
        x : particles, matrix of shape = [n_particles, n_targets] 
        y : matrix of shape = [1, n_targets]

        Returns:
        -------
        distance : scalar
            Total similarity, using equation:

                d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))

            summed over all samples. Supports masked arrays.
        '''
        dx = (x - y) ** 2
        d = np.ma.sum(dx, axis=1)
        return np.exp(-d / (2.0 * sigma ** 2))
    
    def systematic_resample(self, weights):
        '''
        Systematic Resampling   
        Parameters
        ----------
        weights : particle filter weights, matrix of shape = [n_particles, ]
        
        Returns
        -------
        self : resampled indices, matrix of shape = [n_particles, ]    
        '''  
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform(0, 1)) / n
        return self.create_indices(positions, weights)
    
    
    def stratified_resample(self, weights):
        '''
        Stratified Resampling   
        Parameters
        ----------
        weights : particle filter weights, matrix of shape = [n_particles, ]
        
        Returns
        -------
        self : resampled indices, matrix of shape = [n_particles, ]    
        '''  
        n = len(weights)
        positions = (np.random.uniform(0, 1, n) + np.arange(n)) / n
        return self.create_indices(positions, weights)
    
    
    def residual_resample(self, weights):
        '''
        Residual Resampling   
        Parameters
        ----------
        weights : particle filter weights, matrix of shape = [n_particles, ]
        
        Returns
        -------
        self : resampled indices, matrix of shape = [n_particles, ]    
        '''  
        n = len(weights)
        indices = np.zeros(n, np.uint32)
        # take int(N*w) copies of each weight
        num_copies = (n * weights).astype(np.uint32)
        k = 0
        for i in range(n):
            for _ in range(num_copies[i]):  # make n copies
                indices[k] = i
                k += 1
        # use multinormial resample on the residual to fill up the rest.
        residual = weights - num_copies  # get fractional part
        residual /= np.sum(residual)
        cumsum = np.cumsum(residual)
        cumsum[-1] = 1
        indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
        return indices
    
    
    def create_indices(self, positions, weights):
        '''
        Residual Resampling   
        Parameters
        ----------
        positions, weights
        From systematic_resample/stratified_resmaple
        
        Returns
        -------
        indices : resampled indices, matrix of shape = [n_particles, ]    
        '''  
        n = len(weights)
        indices = np.zeros(n, np.uint32)
        cumsum = np.cumsum(weights)
        i, j = 0, 0
        while i < n:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
    
        return indices
      
    def multinomial_resample(self, weights):
        '''
        Multinomial Resampling   
        Parameters
        ----------
        weights : particle filter weights, matrix of shape = [n_particles, ]
        
        Returns
        -------
        self : resampled indices, matrix of shape = [n_particles, ]    
        '''
        return np.random.choice(np.arange(len(weights)), p=weights, size=len(weights))
    
    def resample(self, weights):
        '''
        resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html   
        Parameters
        ----------
        weights : particle filter weights, matrix of shape = [n_particles, ]
        
        Returns
        -------
        indices : resampled indices, matrix of shape = [n_particles, ]    
        '''
        n = len(weights)
        indices = []
        C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
        u0, j = np.random.random(), 0
        for u in [(u0 + i) / n for i in range(n)]:
            while u > C[j]:
                j += 1
            indices.append(j - 1)
        return indices
    
    def _EnKF(self, x_t, u_t, sigmas_ekf, n_particles):
        '''
        Ensemble Kalman Filter (EnKF) 
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_targets, ]
        u : the sensors, matrix of shape = [input_range, n_features]
        sigmas_ekf: sigma points, matrix of shape = [n_particles, z_dim]
        n_particles : number of sigma_points/particles to sample from prior distribution
        
        Returns
        -------
        z_mean : next state mean, matrix of shape = [n_targets, ]
        P_zz_ekf : next state covariance matrix, matrix of shape = [n_targets, n_targets] 
        '''   
        
        'Prediction step EnKF'       
        Fsigmas = self._state_transition_func(sigmas_ekf, u_t)
        
        'Update step EnKF'
        Hsigmas = self._measurement_func(Fsigmas)
        z_mean = np.mean(Hsigmas, axis=0)
        P_zz_ekf = self.auto_correlation(Hsigmas, z_mean, self.R)
        # P_xz_ekf = self.cross_correlation(Fsigmas, Hsigmas, x_mean, z_mean)
        P_xz_ekf = self.cross_correlation(Fsigmas, Hsigmas, self.z_ekf, z_mean)
        ekfK = np.dot(P_xz_ekf, np.linalg.inv(P_zz_ekf))  
        err_update = multivariate_normal(np.zeros(len(x_t)), self.R, n_particles)
        sigmas_temp = Fsigmas
        for i in range(n_particles):
            sigmas_temp[i] += np.dot(ekfK, x_t + err_update[i] - Hsigmas[i])
        x_temp = np.mean(sigmas_temp, axis=0)
        P_temp = self.P_ekf - np.dot(np.dot(ekfK, P_zz_ekf), ekfK.T)   
        sigmas_ekf = sigmas_temp
        self.z_ekf = x_temp
        self.P_ekf = P_temp

        return z_mean, P_zz_ekf
    
    def _UKF(self, x_t, u_t):
        '''
        Unscented Kalman Filter (UKF) 
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_targets, ]
        u : the sensors, matrix of shape = [input_range, n_features]
        
        Returns
        -------
        x_hat : next state mean, matrix of shape = [n_targets, ]
        Px_hat : next state covariance matrix, matrix of shape = [n_targets, n_targets] 
        ''' 
        
        'Prediction step UKF'
        points = JulierSigmaPoints(n=len(self.z_ukf),kappa=3-len(self.z_ukf),sqrt_method=self._sqrt_func)
        sigmas = points.sigma_points(self.z_ukf, self.P_ukf)
        sigmas_f = self._state_transition_func(sigmas,u_t)
        z_hat, P_hat = unscented_transform(sigmas_f,points.Wm,points.Wc,self.Q)
        
        'Update step UKF'
        sigmas_h = self._measurement_func(sigmas_f)
        x_hat, Px_hat = unscented_transform(sigmas_h,points.Wm,points.Wc,self.R)       
        Pxz = np.zeros((len(z_hat),len(x_hat)))
        for i in range(len(sigmas)):
            Pxz += points.Wc[i] * np.outer(sigmas_f[i]-z_hat,sigmas_h[i]-x_hat)
        
        try:
            K = np.dot(Pxz,np.linalg.inv(Px_hat))
        except:
            K = np.dot(Pxz,np.linalg.pinv(Px_hat))
        self.z_ukf = z_hat + np.dot(K,x_t-x_hat)
        self.P_ukf = P_hat - np.dot(K,Px_hat).dot(np.transpose(K))
        
        return x_hat, Px_hat
    
    def _PF(self, x_t, u_t, n_particles, particles, resample_method="systematic"):
        '''
        Particle Filter (PF) 
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_targets, ]
        u : the sensors, matrix of shape = [input_range, n_features]
        particles: particles, matrix of shape = [n_particles, z_dim]
        n_particles : number of sigma_points/particles to sample from prior distribution
        
        Returns
        -------
        mean_hypothesis : next state mean, matrix of shape = [n_targets, ]
        cov_hypothesis : next state covariance matrix, matrix of shape = [n_targets, n_targets] 
        particles : updated particles, matrix of shape = [n_particles, z_dim] 
        ''' 
        
        'Prediction step'
        if self.Q.shape==():
            self.Q = np.diag([self.Q]*len(self.z_pf))
        self.weights = np.ones(n_particles) / n_particles
        
        resample_proportion = 0.01
        n_eff_threshold = 1
        
        transformed_particles = self._state_transition_func(particles, u_t)
        self.transformed_particles_noisy = transformed_particles + multivariate_normal(self.z_pf, self.Q, n_particles)
        z_particles = self._measurement_func(self.transformed_particles_noisy)
        self.z_particles = z_particles + multivariate_normal(np.zeros(len(x_t)), self.R, n_particles)
        
        self.weights = np.clip(
                self.weights * np.array(
                    self.squared_error(z_particles.reshape(n_particles, -1),x_t.reshape(1, -1), sigma=100)),0,np.inf,)
        
        weight_normalisation = np.sum(self.weights)
        self.weights = self.weights / weight_normalisation
        n_eff = (1.0 / np.sum(self.weights ** 2)) / (n_particles)
        self.weight_entropy = np.sum(self.weights * np.log(self.weights))
        # preserve current sample set before any replenishment
        self.original_particles = np.array(self.transformed_particles_noisy)
        
        mean_hypothesis = np.sum(self.z_particles.T * self.weights, axis=-1).T
        cov_hypothesis = np.cov(self.z_particles, rowvar=False, aweights=self.weights)
        self.mean_state = np.sum(self.transformed_particles_noisy.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.transformed_particles_noisy, rowvar=False, aweights=self.weights)
        
        if n_eff < n_eff_threshold:
            
            if resample_method == "resample":
                indices = self.resample(self.weights)
                
            if resample_method == "systematic":
                indices = self.systematic_resample(self.weights)
                
            if resample_method == "residual":
                indices = self.residual_resample(self.weights)
                
            if resample_method == "stratified":
                indices = self.stratified_resample(self.weights)
                
            particles = self.transformed_particles_noisy[indices, :]
            self.weights = np.ones(n_particles) / (n_particles)

    # randomly resample some particles from the prior
        if resample_proportion > 0:
            random_mask = (
                np.random.random(size=(n_particles,)) < resample_proportion)
            particles[random_mask, :] = multivariate_normal(mean=self.z_pf, cov=self.P_pf, size=n_particles)[random_mask, :]
            
        return mean_hypothesis, cov_hypothesis, particles 
    
    def _UKFPF(self, x_t, u_t, n_particles, particles, resample_method="systematic"):
        '''
        Particle Filter (PF) 
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_targets, ]
        u : the sensors, matrix of shape = [input_range, n_features]
        particles: particles, matrix of shape = [n_particles, z_dim]
        n_particles : number of sigma_points/particles to sample from prior distribution
        
        Returns
        -------
        mean_hypothesis : next state mean, matrix of shape = [n_targets, ]
        cov_hypothesis : next state covariance matrix, matrix of shape = [n_targets, n_targets] 
        particles : updated particles, matrix of shape = [n_particles, z_dim] 
        ''' 
        
        'Prediction step'
        if self.Q.shape==():
            self.Q = np.diag([self.Q]*len(self.z_pf))
        self.weights = np.ones(n_particles) / n_particles
        
        resample_proportion = 0.01
        n_eff_threshold = 1
        
        particles_ukf = particles.T
        z_pf, P_pf = np.mean(particles,0), np.cov(particles.T)
        
        points = JulierSigmaPoints(n=len(particles_ukf),kappa=3-len(particles_ukf),sqrt_method=self._sqrt_func)
        sigmas = points.sigma_points(z_pf, P_pf)
        sigmas_f = self._state_transition_func(sigmas,u_t)
        z_hat, P_hat = unscented_transform(sigmas_f,points.Wm,points.Wc,self.Q)
        
        'Update step UKF'
        sigmas_h = self._measurement_func(sigmas_f)
        x_hat, Px_hat = unscented_transform(sigmas_h,points.Wm,points.Wc,self.R)       
        Pxz = np.zeros((len(z_hat),len(x_hat)))
        for i in range(len(sigmas)):
            Pxz += points.Wc[i] * np.outer(sigmas_f[i]-z_hat,sigmas_h[i]-x_hat)
        
        try:
            K = np.dot(Pxz,np.linalg.inv(Px_hat))
        except:
            K = np.dot(Pxz,np.linalg.pinv(Px_hat))
        z_ukf = z_hat + np.dot(K,x_t-x_hat)
        P_ukf = P_hat - np.dot(K,Px_hat).dot(np.transpose(K))
        
        x_particles_ukf = multivariate_normal(mean = x_hat, cov = Px_hat, size = n_particles, check_valid='ignore')
        z_particles_ukf = multivariate_normal(mean = z_ukf, cov = P_ukf, size = n_particles, check_valid='ignore')
        
        self.weights = np.clip(
                self.weights * np.array(
                    self.squared_error(x_particles_ukf.reshape(n_particles, -1),x_t.reshape(1, -1), sigma=100)),0,np.inf,)
        
        weight_normalisation = np.sum(self.weights)
        self.weights = self.weights / weight_normalisation
        n_eff = (1.0 / np.sum(self.weights ** 2)) / (n_particles)
        self.weight_entropy = np.sum(self.weights * np.log(self.weights))
        # preserve current sample set before any replenishment
        self.original_particles = np.array(self.transformed_particles_noisy)
        
# =============================================================================
#         self.mean_hypothesis = np.sum(self.x_particles_ukf.T * self.weights, axis=-1).T
#         self.cov_hypothesis = np.cov(self.x_particles_ukf, rowvar=False, aweights=self.weights)
#         self.mean_state = np.sum(self.x_particles_ukf.T * self.weights, axis=-1).T
#         self.cov_state = np.cov(self.x_particles_ukf, rowvar=False, aweights=self.weights)
# =============================================================================
        self.mean_hypothesis = np.sum(x_particles_ukf.T * self.weights, axis=-1).T
        self.cov_hypothesis = np.cov(x_particles_ukf, rowvar=False, aweights=self.weights)
        self.mean_state = np.sum(x_particles_ukf.T * self.weights, axis=-1).T
        self.cov_state = np.cov(x_particles_ukf, rowvar=False, aweights=self.weights)
        
        if n_eff < n_eff_threshold:
            
            if resample_method == "resample":
                indices = self.resample(self.weights)
                
            if resample_method == "systematic":
                indices = self.systematic_resample(self.weights)
                
            if resample_method == "residual":
                indices = self.residual_resample(self.weights)
                
            if resample_method == "stratified":
                indices = self.stratified_resample(self.weights)
                
            # particles = self.transformed_particles_noisy[indices, :]
            z_particles_ukf = z_particles_ukf[indices, :]
            self.weights = np.ones(n_particles) / (n_particles)

    # randomly resample some particles from the prior
        if resample_proportion > 0:
            random_mask = (
                np.random.random(size=(n_particles,)) < resample_proportion)
            z_particles_ukf[random_mask, :] = multivariate_normal(mean=self.z_ukf, cov=self.P_ukf, size=n_particles)[random_mask, :]
        
        z_particles_ukf = particles    
        
        return self.mean_hypothesis, self.cov_hypothesis, particles
       
    @override
    def predict(self,x,u):
        (x_recon,x_pred,_) = self.estimator.predict([x,u])
        return x_recon,x_pred
    
    @override
    def train(self, x, y, z_dim, hnet_hidden_layers=1, fnet_hidden_layers=1, fnet_hidden_dim=8, 
                uencoding_layers=1, uencoding_dim=8, z_activation='tanh', l2=0.0,
                optimizer='adam', batch_size=256, epochs=10,
                validation_split=0.2, save_best_only=True, verbose=0):
        '''
        Create Neural Network models
        Ref - https://github.com/NSIBF/NSIBF
        Parameters
        ----------
        x : input data = [x, u] where 
            x : the target varaibles, matrix of shape = [n_timesteps, n_targets]
            u : the sensors, matrix of shape = [n_timesteps, input_range, n_features]
        y : the ground truth output data = [x', u'] where
            x' : the target varaibles, matrix of shape = [n_timesteps, n_targets]
            u' : the sensors, matrix of shape = [n_timesteps, input_range, n_features]
        z_dim : hidden dimension
        hnet_hidden_layers : number of hidden layers for h_inv_nn (default is 1)
        fnet_hidden_layers: number of hidden layers for f_nn (default is 1)
        fnet_hidden_dim: number of hidden dimensions for f_nn (default is 8)
        uencoding_layers : number of encoding layers for sensors (default is 1)
        uencoding_dim : number of hidden dimensions for uencoding_layers (default is 8)
        z_activation : the activation function for hidden embedding for target variables (default is 'tanh')
        optimizer: the optimizer for gradient descent (default is 'adam')
        batch_size: the batch size (default is 256)
        epochs: the maximum epochs to train the model (default is 10)
        validation_split: the validation size when training the model (default is 0.2)
        save_best_only: save the model with best validation performance during training (default is True)
        verbose: 0 indicates silent, higher values indicate more messages will be printed (default is 0)
        
        Returns
        -------
        self (models)
        ''' 
        z = np.zeros((x[0].shape[0], z_dim))
        x_dim, u_dim = x[0].shape[1], x[1].shape[2]
        keras.backend.clear_session()
        model, h_nn, h_inv_nn, f_nn = self._make_nnwork(x_dim, u_dim, z_dim, 
                                                          hnet_hidden_layers, fnet_hidden_layers, 
                                                          fnet_hidden_dim, uencoding_layers,uencoding_dim,
                                                          z_activation,l2)
        model.compile(optimizer=optimizer, loss=['mse','mse','mse'], loss_weights=self.loss_weights)

        if save_best_only:
            checkpoint_path = tempfile.gettempdir()+'/BSSAD.ckpt'
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True, save_weights_only=True)                          
            model.fit(x, y+[z], batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[cp_callback], verbose=verbose)
            model.load_weights(checkpoint_path)
            h_nn.compile(optimizer=optimizer, loss='mse')
            h_inv_nn.compile(optimizer=optimizer, loss='mse')
            f_nn.compile(optimizer=optimizer, loss='mse')
        else:
            model.fit(x, y+[z], batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=verbose)
            h_nn.compile(optimizer=optimizer, loss='mse')
            h_inv_nn.compile(optimizer=optimizer, loss='mse')
            f_nn.compile(optimizer=optimizer, loss='mse')
        
        self.estimator = model
        self.h_nn = h_nn
        self.h_inv_nn = h_inv_nn
        self.f_nn = f_nn
        
        return self
    
    def score_samples_via_residual_error(self, x, u):
        '''
        Calculate anomalies scores for samples
        Parameters
        ----------
        x : the target variables, matrix of shape = [n_timesteps, n_targets]
        u : the sensors, matrix of shape = [n_timesteps, input_range, n_features]
        
        Returns
        -------
        recon_scores : matrix of shape = [n_timesteps, ]
        pred_scores : matrix of shape = [n_timesteps-1, ]
        '''        
        (x_recon,x_pred,_) = self.estimator.predict([x,u])
        
        recon_scores = np.mean(np.abs(x-x_recon),axis=1)

        pred_scores = np.mean(np.abs(x[:-1,:]-x_pred[1:,:]),axis=1)

        return recon_scores, pred_scores

    @override
    def score(self, neg_x, neg_y):
        '''
        Score the model based on datasets with uniform negative sampling 
        Parameters
        ----------
        neg_x : negative samples of x, matrix of shape = [n_timesteps, n_targets]
        neg_y : negative samples of x, matrix of shape = [n_timesteps, n_targets]
        
        Returns
        -------
        Best F1-score 
        ''' 
        
        _,pred_scores = self.score_samples_via_residual_error(neg_x[0],neg_x[1])
        pred_scores = -1*pred_scores
        t, _ = bf_search(pred_scores, neg_y[1:],start=np.amin(pred_scores),end=np.amax(pred_scores),step_num=1000,verbose=False)
        return t[0]
    
    @override
    def save_model(self,model_path=None):
        '''
        Save the model to files 
        Parameters
        ----------
        model_path : the target folder where the model files are saved (default is None)
                  If None, a tempt folder is created      
        Returns
        -------
        
        '''
        
        if model_path is None:
            model_path = tempfile.gettempdir()
        
        self.estimator.save(model_path+'/BSSAD.h5',save_format='h5')
        self.f_nn.save(model_path+'/BSSAD_f.h5',save_format='h5')
        self.h_nn.save(model_path+'/BSSAD_h.h5',save_format='h5')
        self.h_inv_nn.save(model_path+'/BSSAD_h_inv.h5',save_format='h5')
    
    @override
    def load_model(self,model_path=None):
        '''
        Load the model from files 
        Parameters
        ----------
        model_path : the target folder where the model files are locarted (default is None)
                  If None, a tempt folder is used to laod files      
        Returns
        -------
        self (models)
        '''

        if model_path is None:
            model_path = tempfile.gettempdir()
        self.estimator = keras.models.load_model(model_path+'/BSSAD.h5')
        self.f_nn = keras.models.load_model(model_path+'/BSSAD_f.h5')
        self.h_nn = keras.models.load_model(model_path+'/BSSAD_h.h5')
        self.h_inv_nn = keras.models.load_model(model_path+'/BSSAD_h_inv.h5')
        
        return self
    
    
    def _make_nnwork(self, x_dim, u_dim, z_dim, 
                      hnet_hidden_layers, fnet_hidden_layers, 
                      fnet_hidden_dim, uencoding_layers,uencoding_dim,z_activation,l2):
        '''
        Make neural network

        '''
        
        x_input = keras.Input(shape=(x_dim),name='x_input')
        u_input = keras.Input(shape=(self.input_range,u_dim),name='u_input')
        z_input = keras.Input(shape=(z_dim),name='z_input')
        
        interval = (x_dim-z_dim)//(hnet_hidden_layers+1)
        hidden_dims = []
        hid_dim = max(1,x_dim-interval)
        hidden_dims.append(hid_dim)
        g_dense1 = layers.Dense(hid_dim, activation='relu',name='g_dense1')(x_input)
        for i in range(1,hnet_hidden_layers):
            hid_dim = max(1,x_dim-interval*(i+1))
            if i == 1:
                g_dense = layers.Dense(hid_dim, activation='relu') (g_dense1)
            else:
                g_dense = layers.Dense(hid_dim, activation='relu') (g_dense)
            hidden_dims.append(hid_dim)
        if hnet_hidden_layers > 1:
            g_out = layers.Dense(z_dim, activation=z_activation,name='g_output',activity_regularizer=keras.regularizers.l2(l2))(g_dense)
        else:
            g_out = layers.Dense(z_dim, activation=z_activation,name='g_output',activity_regularizer=keras.regularizers.l2(l2))(g_dense1)
        h_nn = keras.Model(x_input,g_out,name='h_nn')
         
        h_dense1 = layers.Dense(hidden_dims[len(hidden_dims)-1], activation='relu',name='h_dense1')(z_input)
        for i in range(1,hnet_hidden_layers):
            if i == 1:
                h_dense = layers.Dense(hidden_dims[len(hidden_dims)-1-i], activation='relu') (h_dense1)
            else:
                h_dense = layers.Dense(hidden_dims[len(hidden_dims)-1-i], activation='relu') (h_dense)
        
        if hnet_hidden_layers > 1:
            h_out = layers.Dense(x_dim, activation='linear',name='h_output') (h_dense)
        else:
            h_out = layers.Dense(x_dim, activation='linear',name='h_output') (h_dense1)
        h_inv_nn = keras.Model(z_input,h_out,name='h_inv_nn')
         
        if uencoding_layers == 1:
            f_uencoding = layers.LSTM(uencoding_dim, return_sequences=False)(u_input)
        else:
            f_uencoding = layers.LSTM(uencoding_dim, return_sequences=True)(u_input)
            for i in range(1,uencoding_layers):
                if i == uencoding_layers-1:
                    f_uencoding = layers.LSTM(uencoding_dim, return_sequences=False)(f_uencoding)
                else:
                    f_uencoding = layers.LSTM(uencoding_dim, return_sequences=True)(f_uencoding)
        f_concat = layers.Concatenate(name='f_concat')([z_input,f_uencoding])
        f_dense = layers.Dense(fnet_hidden_dim, activation='relu')(f_concat)
        for i in range(1,fnet_hidden_layers):
            f_dense = layers.Dense(fnet_hidden_dim, activation='relu') (f_dense)
        f_out = layers.Dense(z_dim, activation=z_activation,name='f_output',activity_regularizer=keras.regularizers.l2(l2)) (f_dense)
        f_nn = keras.Model([z_input,u_input],f_out,name='f_nn')
        
        z_output = h_nn(x_input)
        x_output = h_inv_nn(z_output)
        z_hat_output= f_nn([z_output,u_input])
        x_hat_output = h_inv_nn(z_hat_output)
        smoothing = layers.Subtract(name='smoothing')([z_output,z_hat_output])
        model = keras.Model([x_input,u_input],[x_output,x_hat_output,smoothing])
         
        return model, h_nn,h_inv_nn,f_nn
    
    