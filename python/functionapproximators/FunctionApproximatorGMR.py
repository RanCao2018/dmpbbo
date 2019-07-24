# This file is inspired by the paper "Many regression algorithms, one unified model: A review"(F.Stulp, 2015)
#

import os, sys

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from functionapproximators.FunctionApproximator import FunctionApproximator
from functionapproximators.Kmeans_GMM import *
from functionapproximators.EM_GMM import *

class FunctionApproximatorGMR(FunctionApproximator):
    
    def __init__(self,num_GM, regularization=0.0):
        
        self.num_GM_ = num_GM
        self.meta_regularization_ = regularization

        self.model_means_ = None
        self.model_covariance_ = None
        self.model_priors_ = None
        self.is_trained_ = False
        
    def train(self,inputs,targets):

        num_GM = self.num_GM_
        targets = np.array([targets]).T
        inputs_GMM = np.hstack((inputs, targets))
        initial_means, initial_covariance, initial_priors = kmeans_GMM(inputs_GMM,num_GM)
        means, covariance, priors = em_GMM(inputs_GMM,initial_means, initial_covariance, initial_priors)
       
        # Set the parameters of the GMM
        self.model_means_ = means
        self.model_covariance_ = covariance
        self.model_priors_ = priors
        self.is_trained_ = True


    def predict(self,inputs):
        #print('====================\npredict')
        #print(inputs.shape)

        if inputs.ndim==1:
            # Otherwise matrix multiplication below will not work
            inputs = np.atleast_2d(inputs).T
            
        timesize, inputs_dimension = inputs.shape
        target_dimension = np.size(self.model_means_,1)-inputs_dimension
        num_GMM = np.size(self.model_means_,0)
        in_index = range(inputs_dimension)
        out_index = range(inputs_dimension,inputs_dimension+target_dimension)
        diagRegFact = 1E-4 #防止协方差过小

        h = np.zeros([num_GMM,timesize])
        mvg = np.zeros([num_GMM,timesize])
        pre_target = np.zeros([timesize,target_dimension])
        variance = np.zeros([target_dimension,target_dimension,timesize])

        for i in range(num_GMM):
            mvg[i,:] = self.model_priors_[:,i]*multi_variant_gauss(inputs,self.model_means_[i,in_index], self.model_covariance_[in_index,in_index,i]).T
        h = mvg/np.sum(mvg,0)

        for i in range(num_GMM):
            inv_tmp = np.linalg.inv(np.atleast_2d(self.model_covariance_[in_index,in_index,i]))
            means_tmp = self.model_means_[i,out_index]+self.model_covariance_[out_index,in_index,i].dot(inv_tmp).dot((inputs-self.model_means_[i,in_index]).T)
            pre_target = pre_target+(means_tmp*h[i,:])[:,None]

        # for i in range(timesize):
        #     # compute activation weight
        #     for j in range(num_GMM):
        #         mvg[j,i] = self.model_priors_[:,j]*multi_variant_gauss(inputs[i,:],self.model_means_[j,in_index],
        #                                                                self.model_covariance_[in_index,in_index,j]).T
        #     h[:,i] = mvg[:,i]/np.sum(mvg[:,i],1)
        #     # compute conditional means/desired trajectory
        #     for j in range(num_GMM):
        #         tmp = self.model_covariance_[in_index,in_index,j]
        #         means_tmp = self.model_means_[j,out_index]+(inputs[i,:]-self.model_means_[j,in_index]).dot(self.model_covariance_[out_index,in_index,j])/tmp
        #         pre_target[:,i] += h[j,i]*means_tmp
        #     # compute conditonal variance(optional)
        #         variance_tmp = self.model_covariance_[out_index,out_index,j]-self.model_covariance_[out_index,in_index,j].dot(self.model_covariance_[out_index,in_index,j])/tmp
        #         variance[:,:,i] += h[j,i]**2*variance_tmp
        #     variance[:,:,i] += np.eye(target_dimension)*diagRegFact

        return np.squeeze(pre_target)
        
    def isTrained(self):
        return self.is_trained_

    def getParameterVectorSelected(self):
        if self.is_trained_:
            return self.model_means_, self.model_covariance_, self.model_priors_
        else:
            warning('FunctionApproximatorRBFN is not trained.')
            return [];
