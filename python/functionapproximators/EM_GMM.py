# inputs_GMM: T*D D means the dimension of datapoints, T means the time index
import numpy as np

def em_GMM(inputs_GMM,initial_means, initial_covariance, initial_priors):
    timesize, dimension = inputs_GMM.shape
    num_GM = initial_priors.size
    max_iter = 100
    min_iter = 5
    diagRegFact = 1E-4 #防止协方差过小
    converge = 1E-4 #判断是否收敛

    weight = np.zeros([num_GM,timesize])
    mvg = np.zeros([num_GM,timesize])
    l = 0
    l_old = 0
    #赋初值
    means = initial_means # num_GM*D
    covariance = initial_covariance # D*D
    priors = initial_priors # dimension: 1*num_GM

    for num_iter in range(max_iter):
        #E-step
        for i in range(num_GM):
            mvg[i,:] = priors[:,i]*multi_variant_gauss(inputs_GMM,means[i,:],covariance[:,:,i]).T
        weight = mvg/np.sum(mvg,0)
        weight_tmp = weight/np.expand_dims(np.sum(weight,1),1)
        #M-step

        means = np.einsum('ac,ci->ai', weight_tmp,inputs_GMM)# a num_GM, c timesize, i dimension

        dx = inputs_GMM.T[None,:] - means[:,:,None] # num_GM * dimension *timesize
        covariance = np.einsum('acj,aic->aij', np.einsum('aic,ac->aci', dx, weight_tmp), dx).T #a num_GM, c timesize, i-j dimension

        priors = np.mean(weight, axis=1)[None,:]

        # for i in range(num_GM):
        #     priors[:,i] = np.expand_dims(np.sum(weight[i,:],0),1)/timesize
        #     means[i,:] = weight[i,:].dot(inputs_GMM)/(priors[:,i]*timesize)
        #     temp = (inputs_GMM-means[i,:]).T.dot(np.diag(weight[i,:])).dot(inputs_GMM-means[i,:])
        #     covariance[:,:,i] = temp/(priors[:,i]*timesize)+np.eye(dimension)*diagRegFact
        #     print(covariance[:,:,i]-initial_covariance[:,:,i])

        # calculate log-likelihood
        l = np.sum(np.log(np.sum(mvg,1)),0)
        # print(l-l_old, num_iter)

        if num_iter > min_iter:
            if (l-l_old)<converge:
                return means, covariance, priors
        l_old = l


    return means, covariance, priors

def multi_variant_gauss(data, means, covariance):
    if means.size == 1:
        means = np.expand_dims(means,axis=1)
        covariance = np.expand_dims(covariance,axis=1)
    num_data,dimension = data.shape
    temp = np.zeros([num_data,1])
    for i in range(num_data):
        temp[i,:] = (data[i,:]-means).dot(np.linalg.inv(covariance)).dot(data[i,:]-means)
    if np.linalg.det(covariance)<0:
        print(covariance)
    prob = np.exp(-0.5*temp)/np.sqrt((2.0*np.pi)**dimension*(np.linalg.det(covariance))+np.finfo(float).eps)
    return prob
