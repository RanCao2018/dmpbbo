import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from dmp.dmp_plotting import *
from dmp.Dmp import *
from dmp.Trajectory import *
from functionapproximators.FunctionApproximatorRBFN import *

def readFromMotionFile(filename, sensor, cutoff):# 按列排列：时间；位置。默认一个文件仅包含一条轨迹
    data = np.loadtxt(filename)

    (n_time_steps, n_cols) = data.shape
    n_dims = len(sensor)

    time = n_time_steps

    ts = data[0:time,0].copy()
    ts -= data[0,0]

    ys = data[0:time, sensor]
    ys_filter = butter_lowpass_filter(ys, cutoff, 5000)# buttworth 低通滤波器（截止频率：100HZ，采样频率：5000HZ）

    return Trajectory(ts, ys_filter)

if __name__=='__main__':

    sensor = [4,5,6]#手册上的传感器编号+1
    cutoff = 50 #设置截止频率
    traj_fil = readFromMotionFile("handMotion1.txt", sensor, cutoff)
    # traj_fil2 = readFromMotionFile("handMotion1.txt", sensor, 500)
    # traj_fil3 = readFromMotionFile("handMotion1.txt", sensor, 50)
    n_dims = len(sensor)

    y_init = traj_fil.initial_y()
    y_attr = traj_fil.final_y()
    tau = traj_fil.duration()

    #function_apps = [None]*n_dims
    function_apps = [ FunctionApproximatorRBFN(30,0.7), FunctionApproximatorRBFN(30,0.7), FunctionApproximatorRBFN(30,0.7) ]
    dmp = Dmp(tau, y_init, y_attr, function_apps)

    dmp.train(traj_fil)

    tau_exec = traj_fil.duration()
    n_time_steps = tau_exec/0.005 #由间隔时间决定
    ts = np.linspace(0.0,tau_exec,n_time_steps) # 使用linespace时起始点和终止点应为浮点型，否则会有精度损失
    y_attr_scaled = np.array([10.0, 10.0, 10.0])
    dmp.set_attractor_state(y_attr_scaled) # 空间不变性
    dmp.set_tau(tau_exec) # 时间不变性
    ( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)
    traj_ana = dmp.statesAsTrajectory(ts, xs_ana, xds_ana)

    # 画子系统轨迹图
    fig = plt.figure(2)
    ts_xs_xds = np.column_stack((ts,xs_ana,xds_ana))
    plotDmp(ts_xs_xds,fig,forcing_terms_ana,fa_outputs_ana)
    fig.canvas.set_window_title('Analytical integration')

    print("Plotting")

    fig = plt.figure(1)
    axs = [ fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313) ]
    axs1, axs2, axs3 = axs
    lines1 = plotTrajectory(traj_fil.asMatrix(),axs)
    plt.setp(lines1, linestyle='-',  linewidth=1, color='y', label='demonstration')

    # lines1 = plotTrajectory(traj_fil.asMatrix(),axs)
    # p1 = plt.setp(lines1, linestyle='-',  linewidth=1, color='r', label='cutoff_100')
    # lines2 = plotTrajectory(traj_fil2.asMatrix(),axs)
    # p2 = plt.setp(lines2, linestyle='-',  linewidth=1, color='gray', label='cutoff_500')
    # lines3 = plotTrajectory(traj_fil3.asMatrix(),axs)
    # p3 = plt.setp(lines3, linestyle='-',  linewidth=1, color='green', label='cutoff_50')

    lines2 = plotTrajectory(traj_ana.asMatrix(),axs)
    plt.setp(lines2, linestyle='-',  linewidth=1, color='red', label='reproduced')

    axs1.legend([lines1[0], lines2[0]], ('demonstration', 'reproduced'))
    values = dmp.getParameterVectorSelected()
    print(values)

    # 绘制参数对系统影响
    # for ii in range(5):
    #     # Generate random vector with values between 0.5-1.5
    #     rand_vector = 0.5 + np.random.random_sample(values.shape)
    #     dmp.setParameterVectorSelected(rand_vector*values)
    #     values = dmp.getParameterVectorSelected()
    #     print(values)
    #     ( xs, xds, forcing_terms, fa_outputs) = dmp.analyticalSolution(ts)
    #
    #     traj = dmp.statesAsTrajectory(ts, xs, xds)
    #
    #     lines = plotTrajectory(traj.asMatrix(),axs)
    #     plt.setp(lines, linestyle='-',  linewidth=1, color=(0.3,0.3,0.3))
    #     if ii==0:
    #         plt.setp(lines, label='perturbed')

    plt.show()

    #traj_ana.saveToFile('.','handMotionGeneration_space.txt')
