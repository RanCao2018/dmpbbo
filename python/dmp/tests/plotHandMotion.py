import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from dmp.Trajectory import *
# from functionapproximators.FunctionApproximatorRBFN import *
# from functionapproximators.FunctionApproximatorGMR import *
from dmp.tests.plotMVG import *

def readFromMotionFile(filename, sensor, cutoff):# 按列排列：时间；位置。默认一个文件仅包含一条轨迹
    data = np.loadtxt(filename)

    (n_time_steps, n_cols) = data.shape
    n_dims = len(sensor)

    time = n_time_steps

    ts = data[0:time,0].copy()
    ts -= data[0,0]

    ys = data[0:time, sensor]
    ys_filter = butter_lowpass_filter(ys, cutoff, 200)# buttworth 低通滤波器（截止频率：50HZ，采样频率：2000）

    return Trajectory(ts, ys_filter)

def plotSensorData(trajectory,axs):
    """Plot a trajectory"""
    n_dims = (len(trajectory[0])-1)//3
    # -1 for time, /3 because contains y,yd,ydd
    time_index = 0;
    lines = axs.plot(trajectory[:,time_index],trajectory[:,1:n_dims+1], '-')
    axs.set_xlabel('time (s)');
    axs.set_ylabel('angle');


    x_lim = [min(trajectory[:,time_index]),max(trajectory[:,time_index])]
    axs.set_xlim(x_lim[0],x_lim[1])



    return lines

if __name__=='__main__':
    sensor = [4,5,6]#手册上的传感器编号+1
    cutoff = 50 #设置截止频率
    traj_fil = readFromMotionFile("test.txt", sensor, cutoff)
    ndim = len(sensor)

    fig, axs = plt.subplots(1,1)
    lines1 = plotSensorData(traj_fil.asMatrix(),axs)
    plt.setp(lines1, linestyle='-',  linewidth=1, label='demonstration')

    plt.show()
