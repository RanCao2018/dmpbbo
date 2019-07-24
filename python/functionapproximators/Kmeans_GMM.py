# inputs_GMM: T*D D means the dimension of datapoints, T means the time index
import numpy as np

def kmeans_GMM(inputs_GMM, num_GM):
    timesize, dimension = inputs_GMM.shape
    max_iter = 100
    regularizationFactor = 1E-2#防止协方差过小

    state = True #判断算法状态
    ran_index = np.random.randint(timesize-1, size=num_GM)
    mu = inputs_GMM[ran_index,:]
    distance = np.zeros([timesize,num_GM])
    initial_priors = np.zeros([1,num_GM])
    cluster = []
    cluster_old = []
    initial_covariance =np.zeros([dimension,dimension,num_GM])

    num_iter = 0

    while state:
        #寻找中心
        for i in range(num_GM):
            d = inputs_GMM-mu[i,:]
            distance[:,i] = np.sum(np.square(d),1)
        cluster = np.argmin(distance,1)
        #更新均值
        if np.all(cluster != cluster_old):
            for i in range(num_GM):
                if np.any(cluster==i):# 防止产生空的数组，传入mean中
                    mu[i,:] = np.mean(inputs_GMM[np.argwhere(cluster==i),:],0)
                #print(num_iter)
                #print(mu)

        if np.all(cluster == cluster_old) or num_iter >= max_iter:
            state = False
        else:
            num_iter += 1
            cluster_old = cluster

    initial_means = mu
    for i in range(num_GM):
        data_tmp = np.squeeze(inputs_GMM[np.argwhere(cluster==i),:]).T
        initial_priors[:,i] = np.argwhere(cluster==i).size
        initial_covariance[:,:,i] = np.cov(data_tmp)+np.eye(dimension)*regularizationFactor
    initial_priors /= np.sum(initial_priors)
    return initial_means, initial_covariance, initial_priors


