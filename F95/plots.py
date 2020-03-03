# %% Import libs and set rcParams
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['axes.formatter.limits'] = [-2,3]
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = [3,3]
plt.rcParams['savefig.format'] = 'pgf'
plt.rcParams['pgf.texsystem'] = 'lualatex'

# %% image translation vs camera translation
distance = [5,10,15,20,35]
distance_err = [0.5, 0.5, 0.5, 0.5, 0.5]
pixels = [16.24,33.77,50.01,67.81,119.06]

def originlinfitf(x,m): return m*x
p_opt,p_err = curve_fit(originlinfitf, distance, pixels)
p_err = np.sqrt(np.diag(p_err))

plt.figure('image transl vs camera transl')
plt.xlabel('camera translation [cm]')
plt.ylabel('image translation [pixel]')
plt.xlim(0,40)
plt.ylim(0,140)
plt.errorbar(distance, pixels, xerr=distance_err, fmt='.', label='measurements')
x_array = np.linspace(0,40)
plt.plot(x_array, originlinfitf(x_array,*p_opt), label='fit')
plt.legend()
plt.savefig('Ausarbeitung/graphs/A1_translrel')

print(np.round(p_opt[0],2), '±', np.round(p_err[0],2), 'pixel/cm')

# %% ct2ct metric value vs parameter shift
ct2ct_params = [[],[],[],[],[],[]]
ct2ct_params[0] = np.loadtxt('Daten/ct2ct_parameter0.txt')
ct2ct_params[1] = np.loadtxt('Daten/ct2ct_parameter1.txt')
ct2ct_params[2] = np.loadtxt('Daten/ct2ct_parameter2.txt')
ct2ct_params[3] = np.loadtxt('Daten/ct2ct_parameter3.txt')
ct2ct_params[4] = np.loadtxt('Daten/ct2ct_parameter4.txt')
ct2ct_params[5] = np.loadtxt('Daten/ct2ct_parameter5.txt')

plt.figure('ct2ct metric vs param0 shift')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
ax1.label_outer()
plt.xlabel('Shift of paramter 0')
plt.xlim(0,15)
ax1.set_title('MI metric value', color='blue')
ax1.set_ylim(-1,0)
ax1.plot(ct2ct_params[0][:,0], ct2ct_params[0][:,1], color='blue')
ax2.set_title('MSD metric value', color='red')
ax2.set_ylim(0,3e5)
ax2.plot(ct2ct_params[0][:,0], ct2ct_params[0][:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2ct_param0')

plt.figure('ct2ct metric vs param1 shift')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
ax1.label_outer()
plt.xlabel('Shift of paramter 1')
plt.xlim(0,15)
ax1.set_title('MI metric value', color='blue')
ax1.set_ylim(-1,0)
ax1.plot(ct2ct_params[1][:,0], ct2ct_params[1][:,1], color='blue')
ax2.set_title('MSD metric value', color='red')
ax2.set_ylim(0,5e5)
ax2.plot(ct2ct_params[1][:,0], ct2ct_params[1][:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2ct_param1')

plt.figure('ct2ct metric vs param2 shift')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
ax1.label_outer()
plt.xlabel('Shift of paramter 2')
plt.xlim(0,15)
ax1.set_title('MI metric value', color='blue')
ax1.set_ylim(-1,0)
ax1.plot(ct2ct_params[2][:,0], ct2ct_params[2][:,1], color='blue')
ax2.set_title('MSD metric value', color='red')
ax2.set_ylim(0,10e5)
ax2.plot(ct2ct_params[2][:,0], ct2ct_params[2][:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2ct_param2')

plt.figure('ct2ct metric vs param3 shift')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
ax1.label_outer()
plt.xlabel('Shift of paramter 3')
plt.xlim(-25,25)
ax1.set_title('MI metric value', color='blue')
ax1.set_ylim(-1.5,0)
ax1.plot(ct2ct_params[3][:,0], ct2ct_params[3][:,1], color='blue')
ax2.set_title('MSD metric value', color='red')
ax2.set_ylim(0,3e5)
ax2.plot(ct2ct_params[3][:,0], ct2ct_params[3][:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2ct_param3')

plt.figure('ct2ct metric vs param4 shift')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
ax1.label_outer()
plt.xlabel('Shift of paramter 4')
plt.xlim(-25,25)
ax1.set_title('MI metric value', color='blue')
ax1.set_ylim(-1.5,0)
ax1.plot(ct2ct_params[4][:,0], ct2ct_params[4][:,1], color='blue')
ax2.set_title('MSD metric value', color='red')
ax2.set_ylim(0,3e5)
ax2.plot(ct2ct_params[4][:,0], ct2ct_params[4][:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2ct_param4')

plt.figure('ct2ct metric vs param5 shift')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
ax1.label_outer()
plt.xlabel('Shift of paramter 5')
plt.xlim(0,30)
ax1.set_title('MI metric value', color='blue')
ax1.set_ylim(-1.25,-0.25)
ax1.plot(ct2ct_params[5][:,0], ct2ct_params[5][:,1], color='blue')
ax2.set_title('MSD metric value', color='red')
ax2.set_ylim(0,1.5e5)
ax2.plot(ct2ct_params[5][:,0], ct2ct_params[5][:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2ct_param5')

# %% ct2mrt metric values vs parameter shift
ct2mrt_param0 = np.loadtxt('Daten/ct2mrt_parameter0_256bins.txt')
ct2mrt_param3 = np.loadtxt('Daten/ct2mrt_parameter3_256bins.txt')

plt.figure('ct2mrt metric vs param0 shift')
ax1 = plt.subplot(111)
plt.xlabel('Shift of paramter 0')
plt.xlim(0,1)
ax1.grid(False)
ax1.set_ylabel('MI value', color='blue')
ax1.set_ylim(-0.8,0)
ax1.tick_params(axis='y', colors='blue')
ax1.plot(ct2mrt_param0[:,0], ct2mrt_param0[:,1], color='blue')
ax2 = ax1.twinx()
ax2.grid(False)
ax2.set_ylabel('MSD value', color='red')
ax2.set_ylim(7e5,8.5e5)
ax2.tick_params(axis='y', colors='red')
ax2.plot(ct2mrt_param0[:,0], ct2mrt_param0[:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2mrt_param0')

plt.figure('ct2mrt metric vs param3 shift')
ax1 = plt.subplot(111)
plt.xlabel('Shift of paramter 3')
plt.xlim(0,1)
ax1.grid(False)
ax1.set_ylabel('MI value', color='blue')
ax1.tick_params(axis='y', colors='blue')
ax1.plot(ct2mrt_param3[:,0], ct2mrt_param3[:,1], color='blue')
ax2 = ax1.twinx()
ax2.grid(False)
ax2.set_ylabel('MSD value', color='red')
ax2.tick_params(axis='y', colors='red')
ax2.plot(ct2mrt_param3[:,0], ct2mrt_param3[:,2], color='red')
plt.savefig('Ausarbeitung/graphs/A2_ct2mrt_param3')

# %% ct2mrt metric value vs parameter using different bin sizes
ct2mrt_param0_bins = [[],[],[],[],[],[]]
ct2mrt_param0_bins_scale = np.loadtxt('Daten/ct2mrt_parameter0_5bins.txt')[:,0]
ct2mrt_param0_bins[0] = np.loadtxt('Daten/ct2mrt_parameter0_5bins.txt')[:,1]
ct2mrt_param0_bins[1] = np.loadtxt('Daten/ct2mrt_parameter0_10bins.txt')[:,1]
ct2mrt_param0_bins[2] = np.loadtxt('Daten/ct2mrt_parameter0_20bins.txt')[:,1]
ct2mrt_param0_bins[3] = np.loadtxt('Daten/ct2mrt_parameter0_50bins.txt')[:,1]
ct2mrt_param0_bins[4] = np.loadtxt('Daten/ct2mrt_parameter0_100bins.txt')[:,1]
ct2mrt_param0_bins[5] = np.loadtxt('Daten/ct2mrt_parameter0_256bins.txt')[:,1]

plt.figure('ct2mrt bin sizes param0')
plt.xlabel('Shift of parameter 0')
plt.ylabel('MI value')
plt.xlim(0,1)
plt.ylim(-0.7,0)
#plt.plot(ct2mrt_param0_bins_scale, ct2mrt_param0_bins[0],label='5 bins')
plt.plot(ct2mrt_param0_bins_scale, ct2mrt_param0_bins[1],label='10 bins')
plt.plot(ct2mrt_param0_bins_scale, ct2mrt_param0_bins[2],label='20 bins')
plt.plot(ct2mrt_param0_bins_scale, ct2mrt_param0_bins[3],label='50 bins')
plt.plot(ct2mrt_param0_bins_scale, ct2mrt_param0_bins[4],label='100 bins')
plt.plot(ct2mrt_param0_bins_scale, ct2mrt_param0_bins[5],label='256 bins')
plt.legend()
plt.savefig('Ausarbeitung/graphs/A2_ct2mrt_bins')

ct2mrt_param3_bins = [[],[],[],[],[],[]]
ct2mrt_param3_bins_scale = np.loadtxt('Daten/ct2mrt_parameter3_5bins.txt')[:,0]
ct2mrt_param3_bins[0] = np.loadtxt('Daten/ct2mrt_parameter3_5bins.txt')[:,1]
ct2mrt_param3_bins[1] = np.loadtxt('Daten/ct2mrt_parameter3_10bins.txt')[:,1]
ct2mrt_param3_bins[2] = np.loadtxt('Daten/ct2mrt_parameter3_20bins.txt')[:,1]
ct2mrt_param3_bins[3] = np.loadtxt('Daten/ct2mrt_parameter3_50bins.txt')[:,1]
ct2mrt_param3_bins[4] = np.loadtxt('Daten/ct2mrt_parameter3_100bins.txt')[:,1]
ct2mrt_param3_bins[5] = np.loadtxt('Daten/ct2mrt_parameter3_256bins.txt')[:,1]

plt.figure('ct2mrt bin sizes param3')
plt.xlabel('Shift of parameter 3')
plt.ylabel('MI value')
plt.xlim(0,1)
plt.ylim(-0.75,0)
#plt.plot(ct2mrt_param3_bins_scale, ct2mrt_param3_bins[0],label='5 bins')
plt.plot(ct2mrt_param3_bins_scale, ct2mrt_param3_bins[1],label='10 bins')
plt.plot(ct2mrt_param3_bins_scale, ct2mrt_param3_bins[2],label='20 bins')
plt.plot(ct2mrt_param3_bins_scale, ct2mrt_param3_bins[3],label='50 bins')
plt.plot(ct2mrt_param3_bins_scale, ct2mrt_param3_bins[4],label='100 bins')
plt.plot(ct2mrt_param3_bins_scale, ct2mrt_param3_bins[5],label='256 bins')
plt.legend()

# %% computation time vs sampling rate
sampling_rate = [1,0.8,0.6,0.4,0.2]
time = np.array([[47.2,45.5,50.2],[33.8,27.2,26.1],[25.6,28.3,26.6],[19.7,18.5,20.4],[9.7,11.7,14.2]])
time_err = np.std(time,axis=1)
time = np.mean(time,axis=1)

def originlinfitf(x,m): return m*x
p_opt,p_err = curve_fit(originlinfitf, sampling_rate, time)
p_err = np.sqrt(np.diag(p_err))

plt.figure('time vs sampling')
plt.xlabel('Sampling rate')
plt.ylabel('Computation time [s]')
plt.xlim(0,1.05)
plt.ylim(0,52.5)
plt.errorbar(sampling_rate, time, yerr=time_err, fmt='.', label='measurements')
x_array = np.linspace(0,1.05)
plt.plot(x_array, originlinfitf(x_array,*p_opt), label='fit')
plt.legend()
plt.savefig('Ausarbeitung/graphs/A3_sampling')

# %% Computation time vs grid points per dimension
gppd = [5,10,20,30,50,100]
time = [[0.12,0.12,0.12],[0.49,0.49,0.49],[2.01,2.01,2.01],[4.04,4.13,4.09],[6.16,6.18,6.14],[31.61,31.53,33.22]]
time_err = np.std(time,axis=1)
time = np.mean(time,axis=1)

def originparabfitf(x,m): return m*x**2
p_opt,p_err = curve_fit(originparabfitf, gppd, time)
p_err = np.sqrt(np.diag(p_err))

plt.figure('time vs gppd')
plt.xlabel('Grid points per dimension')
plt.ylabel('Computation time [s]')
plt.xlim(0,103)
plt.ylim(-1,35)
plt.errorbar(gppd, time, yerr=time_err, fmt='.', label='measurements')
x_array = np.linspace(0,103)
plt.plot(x_array, originparabfitf(x_array,*p_opt), label='fit')
plt.legend()
plt.savefig('Ausarbeitung/graphs/A4_gppd')

# %% Show plots
#plt.show()

# %%
