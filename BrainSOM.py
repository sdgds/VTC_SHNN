#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4/16/2020
BrainSOM mapping AI to HumanBrain 
@author: Zhangyiyuan
"""
import h5py
import copy
import matplotlib as mpl
import brian2 as b2
from neurodynex3.leaky_integrate_and_fire import LIF
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.adex_model import AdEx
import PIL.Image as Image
import sys
from tqdm import tqdm
from time import time
from datetime import timedelta
import numpy as np
import itertools
from scipy.ndimage import zoom
#from scipy.misc import logsumexp
from scipy.integrate import odeint
from scipy.stats import zscore
from warnings import warn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import minisom
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial.distance import pdist, squareform



def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None):
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations

def _wrap_index__in_verbose(iterations):
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    sys.stdout.write(progress)
    beginning = time()
    sys.stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        sys.stdout.write(progress)

def fast_norm(x):
    return np.sqrt(np.dot(x, x.T))



class VTCSOM(minisom.MiniSom):
    #### Initialization ####
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=1,
                 neighborhood_function='gaussian', seed=0):
        """
        x : int
            x dimension of the feature map.
        y : int
            y dimension of the feature map.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float
            Spread of the neighborhood function (sigma(t) = sigma / (1 + t/T) where T is num_iteration/2)
        learning_rate : 
            initial learning rate (learning_rate(t) = learning_rate / (1 + t/T)
        neighborhood_function : function, optional (default='gaussian')
            possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'
        """
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = np.random.RandomState(seed)
        self.iterations = 0

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        
        # random initialization W
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._x = x
        self._y = y
        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)  # used to evaluate the neighborhood function
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        self.s = 0

        neig_functions = {'gaussian': self._gaussian,
                          'gaussian_truncate': self._gaussian_truncate,
                          'exp_decay': self._exp_decay,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle,
                          'circle': self._circle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        # initialization M
        self.M = np.random.normal(0,1,(4096,4096))
        self.Normalize_M()
        
        
    def Normalize_X(self, x):
        temp = np.sum(np.multiply(x, x))
        x /= np.sqrt(temp)
        return x
    
    def Normalize_W(self):
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)
        
    def Normalize_M(self):
        self.M /= np.linalg.norm(self.M, keepdims=True)

    
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T
    
    def _gaussian_truncate(self, c, sigma):
        d = 2*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        temp = (ax * ay).T
        return np.where(temp>0.1, temp, 0)
    
    def _exp_decay(self, c, lamda):
        ax = np.exp(-lamda*np.abs(self._xx-self._xx.T[c]))
        ay = np.exp(-lamda*np.abs(self._yy-self._yy.T[c]))
        return (ax * ay).T
    
    def _mexican_hat(self, c, exc_sigma=2, exc_mag=1, inh_sigma=4, inh_mag=0.8):
        d1 = 2*np.pi*exc_sigma*exc_sigma
        d2 = 2*np.pi*inh_sigma*inh_sigma
        ax_exc = exc_mag * np.exp(-np.power(self._neigx-c[0], 2)/d1)
        ay_exc = exc_mag * np.exp(-np.power(self._neigy-c[1], 2)/d1)
        exc = np.outer(ax_exc, ay_exc)
        ax_inh = inh_mag * np.exp(-np.power(self._neigx-c[0], 2)/d2)
        ay_inh = inh_mag * np.exp(-np.power(self._neigy-c[1], 2)/d2)
        inh = np.outer(ax_inh, ay_inh)       
        return exc - inh
    
    def _circle(self, c, radius):
        circle = np.zeros((self._x, self._y))
        for i in range(self._x):
            for j in range(self._y):
                d = np.sqrt((i-c[0])**2+(j-c[1])**2)
                if d<=radius:
                    circle[i,j] = 1
        return circle
    
    def plot_3D_constrain(self, Z):
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection='3d')
        xx = np.arange(0,self._x,1)
        yy = -np.arange(-self._y,0,1)
        X, Y = np.meshgrid(xx, yy)
        surf = ax.plot_surface(X,Y,Z, cmap='jet')
        ax.set_zlim3d(0)
        fig.colorbar(surf)
        plt.show()
        
    def plot_pinwheel(self):
        theta_range = np.linspace(0,2*np.pi,256)
        colorbar = mpl.cm.gist_rainbow(np.arange(256))
        color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
        Theta = np.zeros((self._x,self._y))
        R = np.zeros((self._x,self._y))
        for i in tqdm(range(self._x)):
            for j in range(self._y):  
                x,y = self._weights[i,j,:][[0,1]]
                a,b = x,y
                r = np.sqrt(a**2+b**2)
                R[i,j] = r
                if b>0 and a>0:
                    theta = np.arctan(b/a)
                if b>0 and a<0:
                    theta = np.pi + np.arctan(b/a)
                if b<0 and a<0:
                    theta = np.pi + np.arctan(b/a)
                if b<0 and a>0:
                    theta = 2*np.pi + np.arctan(b/a) 
                d = np.abs(theta_range-theta)
                Theta[i,j] = np.where(d==d.min())[0]
        plt.figure(figsize=(8,12))
        plt.subplot(121)
        plt.imshow(Theta,cmap=color_map)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(R)
        plt.axis('off')
    
    
    
    """ Train Model """  
    ###########################################################################
    ###########################################################################
    ### Training functions
    def Train(self, data, num_iteration, step_len, verbose):
        """Trains the SOM.
        data : np.array Data matrix (sample numbers, feature numbers).
        num_iteration : Maximum number of iterations.
        """            
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        self.iterations = iterations
        q_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            self.update(data[iteration], 
                        self.winner(data[iteration]), 
                        t, end_num) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, np.abs(self.change).sum())
            if (t+1) % 10000 == 0:
                self.plot_pinwheel()
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
        return q_error

    def Train_forward(self, data, num_iteration, step_len, verbose):
        """Trains the SOM.
        data : np.array Data matrix (sample numbers, feature numbers).
        num_iteration : Maximum number of iterations.
        """    
        start_num = num_iteration[0]
        end_num = num_iteration[1]        
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        self.iterations = iterations
        q_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            self.update(data[iteration], 
                        self.forward_winner(data[iteration]), 
                        t, end_num) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, np.abs(self.change).sum())
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
        return q_error
    
    def Train_exc_inh_with_hebb(self, data, num_iteration, step_len, 
                                exc_sigma=2, exc_mag=1, inh_sigma=6, inh_mag=0.5, verbose=False):          
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        self.iterations = iterations
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            gamma = 0.05
            self.update_mexican_M(data[iteration], self.forward_activate(data[iteration]), self.forward_winner(data[iteration]), 
                                  gamma, exc_sigma, exc_mag, inh_sigma, inh_mag) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def Train_with_hebb_based_on_firing_rate_model(self, data, num_iteration, step_len, parm):
        """
        1. Feedforward: Alexnet+SOM for all time in unstabel state
        2. Reccurent: firing model to stabel state
        3. Update W: SOM rule to update SOM weights, Hebb rule to update reccurent weights
                     Attention: neighborhood function caused by reccurent connection
        """
        def tune(x):
            return x/x.max()
        def _exponential_projection(c, lamda):
            """Returns a Exponential centered in c."""
            ax = np.power(self._xx-self._xx.T[c], 2)
            ay = np.power(self._yy-self._yy.T[c], 2)
            ds = np.sqrt(ax+ay)
            exp_proj = np.exp(-lamda*ds)
            return exp_proj
        def projection_constrain(parm):
            projection = np.zeros((self._x*self._y, self._x*self._y))            
            for i in tqdm(range(self._x)):
                for j in range(self._y):
                    index = i*self._x + j
                    p = _exponential_projection((i,j),parm).reshape(-1)
                    projection[index] = np.random.binomial(1, p, p.shape[0])
            return projection
            
        ### training
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,False, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        projection = projection_constrain(parm)
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            ## all-time feedforward + reccurent dynamic system
            solution = self._activate_recurrent_firing_rate(data[iteration], process_type='dynamics')
            # plt.figure()
            # plt.imshow(solution[0].reshape(64,64))
            # plt.colorbar()
            # plt.figure()
            # plt.imshow(solution[-1].reshape(64,64))
            # plt.colorbar()
            ## update weights of W and M
            # update M
            self.basic_hebb_with_M_update(data[iteration], solution[-1,:], projection)
            self.Normalize_M()
            # update W
            winner_neuron_index = np.argmax(solution[-1,:])
            neighbor_connection = self.M[winner_neuron_index,:].reshape(self._x, self._y)
            neighbor_connection = tune(neighbor_connection)
            self.update_structure_constrained_by_M(data[iteration], t, end_num, 
                                                   neighbor_connection)   
            # # prune the small M
            # if t%10==0:
            #     self.M = np.where((self.M>np.percentile(self.M,90))|(self.M<np.percentile(self.M,10)), self.M, 0)
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        return q_error, t_error
    
    
    
    ### Avtivation and select winner
    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        x = self.Normalize_X(x)
        self.s = np.subtract(x, self._weights)  # x - w
        self._activation_map = np.linalg.norm(self.s, axis=-1)
            
    def _forward_activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        x = self.Normalize_X(x)
        s = np.dot(self._weights, x)
        self._activation_map = s
        self.s = np.subtract(x, self._weights)  # x - w

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map
    
    def forward_activate(self, x):
        """Returns the activation map to x."""
        self._forward_activate(x)
        return self._activation_map
    
    def _activate_recurrent_firing_rate(self, x, process_type):
        if process_type=='dynamics':
            def F(x):
                #return np.where(x<=0, 0, 0.8*x)
                return x
            def diff_equation(v, t, M, H, tao, noise_level):
                h = H
                Change = F(np.dot(M,v)+h)
                dvdt = tao * (-v + Change)
                return dvdt
            ## all-time feedforward + reccurent dynamic system
            times = np.linspace(0,10,50)
            SOM_act = self.forward_activate(x)
            H = SOM_act.reshape(-1)
            solution = odeint(diff_equation, H, times, args=(self.M, H, 1.2, 0))
            return solution
        if process_type=='static':
            SOM_act = self.forward_activate(x)
            H = SOM_act.reshape(-1)
            K = np.eye(self.M.shape[0]) - self.M
            K_inv = np.linalg.inv(K)
            solution = np.dot(K_inv, H)
            return np.vstack((H,solution))
    
    def winner(self, x, k=0):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return np.unravel_index(self._activation_map.reshape(-1).argsort()[k],
                                self._activation_map.shape)
        
    def forward_winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._forward_activate(x)
        return np.unravel_index(self._activation_map.reshape(-1).argsort()[-1],
                                self._activation_map.shape)
        
    def activation_response(self, data, k=0):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x, k)] += 1
        return a
    
    
    
    ### Updating functions
    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        # structual constrain
        def asymptotic_decay(scalar, t, max_iter):
            return scalar / (1+t/(max_iter/2))
        eta = asymptotic_decay(self._learning_rate, t, max_iteration)
        g = self.neighborhood(win, self._sigma) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self.change = np.einsum('ij, ijk->ijk', g, self.s)
        self._weights += self.change
        self.Normalize_W()
        
    def update_mexican_M(self, x, activation, win, gamma, exc_sigma, exc_mag, inh_sigma, inh_mag):
        hebb_value = activation[win] * np.where(activation>=np.percentile(activation,95), activation, 0)
        self.g += gamma * hebb_value * self._mexican_hat(win, exc_sigma, exc_mag, inh_sigma, inh_mag)
        # update
        self._weights += np.einsum('ij, ijk->ijk', self.g, x-self._weights)
        self.Normalize_W()
        # control weights
        self.g = np.where(self.g>1, 1, self.g)
        self.g = np.where(self.g<-1, -1, self.g)        
        
    def update_structure_constrained_by_M(self, x, t, max_iteration, neighbor_connection):
        def asymptotic_decay(scalar, t, max_iter):
            return scalar / (1+t/(max_iter/2))
        eta = asymptotic_decay(self._learning_rate, t, max_iteration)
        g = neighbor_connection * eta
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        self.Normalize_W()
        
    def basic_hebb_update(self, x, H, update_object):
        h = H.reshape(-1,1)
        #h /= np.sqrt(np.sum(np.multiply(h, h)))
        #h = zscore(h)
        if update_object=='M':
            self.M += np.dot(h, h.T)
        if update_object=='W':
            for i in range(x.shape[0]):
                self._weights[:,:,i] += x[i]*h.reshape(self._x, self._y)
                
    def basic_hebb_with_M_update(self, x, H, projection):
        h = H.reshape(-1,1)
        self.M += np.dot(h, h.T) * projection
            
        


    ### Utilis
    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between data[i] and the j-th weight."""
        input_data = np.array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = np.power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(input_data, weights_flat.T)
        return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron) to each sample in data."""
        self._check_input_len(data)
        winners_coords = np.argmin(self._distance_from_weights(data), axis=1)
        return self._weights[np.unravel_index(winners_coords,
                                              self._weights.shape[:2])]
    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return np.linalg.norm(data-self.quantization(data), axis=1).mean()
    
    def wiring_cost(self):
        """
        wiring cost is the sum of functional weighted distance.
        """
        similarity_matrix = squareform(pdist(self._weights.reshape(-1,self._input_len),metric='euclidean')) 
        L = np.multiply(similarity_matrix, self.distance_matrix)
        return L.sum()
    
        
        
        
    #### Visulization ####
    ###########################################################################
    ###########################################################################  
    def U_avg_matrix(self, savedir=None):
        heatmap = self.distance_map()
        plt.figure(figsize=(8, 8))
        plt.title('U-avg-matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        if savedir!=None:
            plt.savefig(savedir)
        return heatmap
    
    def U_onefeature_avg_matrix(self, feature):
        def distance_map(self):
            um = np.zeros((self._weights.shape[0], self._weights.shape[1]))
            it = np.nditer(um, flags=['multi_index'])
            while not it.finished:
                for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                    for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                        if (ii >= 0 and ii < self._weights.shape[0] and
                                jj >= 0 and jj < self._weights.shape[1]):
                            w_1 = self._weights[ii, jj, feature]
                            w_2 = self._weights[it.multi_index][feature]
                            um[it.multi_index] += fast_norm(w_1-w_2)
                it.iternext()
            um = um/um.max()
            return um
        heatmap = distance_map(self)
        plt.figure(figsize=(7, 7))
        plt.title('U_onefeature_avg_matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        return heatmap
    
    def U_min_matrix(self):
        """Returns the distance map of the weights.
        Each cell is the normalised min of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance."""
        def min_dist(self):
            um = np.zeros((self._weights.shape[0], self._weights.shape[1]))
            it = np.nditer(um, flags=['multi_index'])
            while not it.finished:
                Dist_neig = []
                for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                    for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                        if (ii >= 0 and ii < self._weights.shape[0] and
                                jj >= 0 and jj < self._weights.shape[1]):
                            w_1 = self._weights[ii, jj, :]
                            w_2 = self._weights[it.multi_index]
                            Dist_neig.append(fast_norm(w_1-w_2))
                Dist_neig.remove(0)
                um[it.multi_index] = np.min(Dist_neig)
                it.iternext()
            um = um/um.max()
            return um
        heatmap = min_dist(self)
        plt.figure(figsize=(7, 7))
        plt.title('U-min-matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        return heatmap
        
    def Component_Plane(self, feature_index):
        """
        Component_Plane表示了map里每个位置的神经元对什么特征最敏感(或者理解为与该特征取值最匹配)
        """
        plt.figure(figsize=(7, 7))
        plt.title('Component Plane: feature_index is %d' % feature_index)
        plt.imshow(self._weights[:,:,feature_index], cmap='coolwarm')
        plt.colorbar()
        plt.show()
        
    def Winners_map(self, data, blur=None):
        if blur == None:
            plt.figure()
            plt.imshow(self.activation_response(data))
            plt.colorbar()
        if blur == 'GB':
            img = self.activation_response(data)
            plt.figure()
            plt.imshow(cv2.GaussianBlur(img,(5,5),0))
            plt.colorbar()
            
    def dynamics_pattern(self, data, represent_type='neuron'):
        def F(x):
            #return np.where(x<=0, 0, 0.8*x)
            return x
        def diff_equation(v, t, M, H, tao, noise_level):
            h = H
            Change = F((np.dot(M,v)+h))
            dvdt = tao * (-v + Change)
            return dvdt
        times = np.linspace(0,10,100)
        SOM_act = self.activate(data)
        H = 1/SOM_act.reshape(-1)
        solution = odeint(diff_equation, H, times, args=(self.M, H, 1.2, 0))        
        
        if represent_type=='neuron':
            plt.figure()
            neurons = np.random.choice(np.arange(self.M.shape[0]), 100)
            for neuron in neurons:
                plt.plot(solution[:,neuron])
                
        if represent_type=='pattern':
            plt.figure()
            plt.ion()     # 开启一个画图的窗口
            for i in range(solution.shape[0]):
                plt.imshow(solution[i,:].reshape(64,64), 'jet')
                plt.title('This is time: %d' %i)
                plt.axis('off')
                plt.pause(0.000000000000000001)       # 停顿时间
            plt.pause(0)   # 防止运行结束时闪退
            
        if represent_type=='state_space_pca':
            pca = PCA(n_components=3)
            solution_pca = pca.fit_transform(solution[:100,:])
            
            plt.figure(figsize=(7,7))
            x = solution_pca[:,0]
            y = solution_pca[:,1]
            z = solution_pca[:,2]
            plt.ion()
            for i in range(100):
                ax = plt.axes(projection='3d')
                ax.plot3D(x[:i], y[:i], z[:i])  
                ax.set_xlim(solution_pca[:,0].min(), solution_pca[:,0].max())
                ax.set_ylim(solution_pca[:,1].min(), solution_pca[:,1].max())
                ax.set_zlim(solution_pca[:,2].min(), solution_pca[:,2].max())
                ax.grid(False)
                plt.title('This is time: %d' %i)
                plt.pause(0.01)  
            plt.pause(0) 





if __name__ == '__main__':
    data = np.genfromtxt('/Users/mac/Desktop/TDCNN/minisom/examples/iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    # data normalization
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

    # Initialization and training
    som = VTCSOM(7, 7, 4, sigma=3, learning_rate=0.5, 
                      neighborhood_function='gaussian')
    som.pca_weights_init(data)
    q_error, t_error = som.Train(data, 100, verbose=False)
    
    plt.figure()
    plt.plot(q_error)
    plt.ylabel('quantization error')
    plt.xlabel('iteration index')
    
    plt.figure()
    plt.plot(t_error)
    plt.ylabel('som.topographic error')
    plt.xlabel('iteration index')
    
    


                