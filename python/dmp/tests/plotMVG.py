import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def plot_sample_gmm(means, covariance):
    num_GM, dimension = means.shape
    for i in range(num_GM):
        x,y = np.random.multivariate_normal(means[i,:], covariance[:,:,i], 500).T
        plt.plot(x,y,'x')


def plot_data(data):
    timesize,dimension = data.shape
    if dimension != 2:
        print('dimension must be 2')
        return
    plt.plot(data[:,0],data[:,1],'b')

def plot_spherical_gmm(ax, means, covariance, color='r', transparancy=255):
    means = means.T
    num_GM = means.shape[1]
    nbDrawingSeg = 35
    patches=[]

    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

    for i in range(num_GM):
        pts = (np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg]))
        R = np.real(sp.linalg.sqrtm(1.0 * covariance[:, :, i]))
        points = R.dot(pts)+means[:,i][:,np.newaxis]

       # points = (means[i]) * pts

       # points_int = (means[i] - covariance[:,:,i]**0.5	) * pts

        # if len(col) == 3:  # if alpha not already defined
        #     if isinstance(col, np.ndarray):
        #         c = np.append(col, transparancy[i])
        #     else:
        #         c = col + (transparancy[i],)
        # else:
        #     c = col

        # plt.plot(points[0, :], points[1, :])
        # plt.plot(points_ext[0, :], points_ext[1, :], lw=1, alpha=1, color=col)
        polygon = Polygon(points.transpose().tolist())
        patches.append(polygon)

        # plt.fill_between(points_ext[0], points_int[1], points_ext[1])
    p = PatchCollection(patches, alpha=0.4)
    ax.add_collection(p)
    plt.plot(means[0, :], means[1, :],'ro')
