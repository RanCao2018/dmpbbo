# inputs_GMM: T*D D means the dimension of datapoints, T means the time index
import numpy as np

def kmeans_GMM(inputs_GMM, num_GM):
    timesize, dimension = inputs_GMM.shape
    max_iter = 100
    kmeans_iter = 5
    regularizationFactor = 1E-2#防止协方差过小

    initial_priors = np.zeros([1,num_GM])
    mu =[]
    cluster = []
    distanceToCenter = np.zeros([timesize,1])
    distanceToCenter_min = np.zeros([timesize,1])
    initial_covariance =np.zeros([dimension,dimension,num_GM])

    num_iter = 0


    for tmp in range(kmeans_iter):
        #随机初始化
        ran_index = np.random.randint(timesize-1, size=num_GM)
        mu_tmp = inputs_GMM[ran_index,:]
        distance = np.zeros([timesize,num_GM])
        cluster_old = []
        cluster_tmp =[]
        state = True #判断算法状态
        while state:
            #寻找中心
            for i in range(num_GM):
                d = inputs_GMM-mu_tmp[i,:]
                distance[:,i] = np.sum(np.square(d),1)
            cluster_tmp = np.argmin(distance,1)
            distanceToCenter = np.min(distance,1)
            #更新均值
            if np.all(cluster_tmp != cluster_old):
                for i in range(num_GM):
                    if np.any(cluster_tmp==i):# 防止产生空的数组，传入mean中
                        mu_tmp[i,:] = np.mean(inputs_GMM[np.argwhere(cluster_tmp==i),:],0)
                    #print(num_iter)
                    #print(mu)

            if np.all(cluster_tmp == cluster_old) or num_iter >= max_iter:
                state = False
            else:
                num_iter += 1
                cluster_old = cluster_tmp


        # print(np.sum(distanceToCenter),np.sum(distanceToCenter_min))
        if (np.sum(distanceToCenter) < np.sum(distanceToCenter_min)) or (tmp == 0):
            distanceToCenter_min = distanceToCenter
            cluster = cluster_tmp
            mu = mu_tmp

    initial_means = mu
    for i in range(num_GM):
        data_tmp = np.squeeze(inputs_GMM[np.argwhere(cluster==i),:]).T
        initial_priors[:,i] = np.argwhere(cluster==i).size
        initial_covariance[:,:,i] = np.cov(data_tmp)+np.eye(dimension)*regularizationFactor
    initial_priors /= np.sum(initial_priors)
    return initial_means, initial_covariance, initial_priors
