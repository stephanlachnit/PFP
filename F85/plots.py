import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import elementary_charge, Boltzmann, zero_Celsius
from scipy.optimize import curve_fit

# EOM

print('\nEOM')

# Values from the script

print(' Script values')

d=2e-3
L=20e-3
n0=2.286
ne=2.200
l=632.8e-9
r13_script=9.6e-12
r33_script=30.9e-12
Vpi_script=d/L*l/(r33_script*ne**3-r13_script*n0**3) 

print('  Vpi = '+str(Vpi_script))
print('  r13 = '+str(r13_script))
print('  r33 = '+str(r33_script))

# part 5.2

print(' Mach-Zehnder')

Vin = np.array([0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]) * 300
Vp45 = np.array([1.04,1.06,1.05,1.07,1.10,1.12,1.1,1.07,0.98,0.93,0.97,1.05,1.09])
Vm45 = np.array([362,362,360,366,363,366,363,367,363,362,359,366,357]) * 1e-3

x_array = np.linspace(0, 1.8e3, num=500)

def fitfunc(V, Upi, Ug, cvc, phi0):
  return cvc * np.cos(phi0/2 - np.pi/2 * V/Upi)**2 + Ug

[p_opt, p_cov] = curve_fit(fitfunc, Vin, Vp45, p0=[300, 1.0, 0.1, -5.0])

Upi0 = p_opt[0]
r13 = d/L * l/n0**3 / Upi0  #neglect r33

print('  V+45deg')
print('   Upi0 = '+str(Upi0))
print('   Ug   = '+str(p_opt[1]))
print('   cfc  = '+str(p_opt[2]))
print('   phi0 = '+str(p_opt[3]))
print('   r13  = '+str(r13))

plt.figure()
plt.title('MZ: +45 deg axis')
plt.ylabel('Intensity [V]')
plt.xlabel('Vin [V]')
plt.grid(True, which='both')
plt.errorbar(x=Vin,y=Vp45,fmt='x')
plt.plot(x_array, fitfunc(x_array, *p_opt))

[p_opt, p_cov] = curve_fit(fitfunc, Vin, Vm45, p0=[300, 0.365, 1.5e-3, -7.5])

Upie = p_opt[0]
r33 = d/L * l/ne**3 / Upie  # neglect r13

print('  V-45deg')
print('   Upie = '+str(Upie))
print('   Ug   = '+str(p_opt[1]))
print('   cfc  = '+str(p_opt[2]))
print('   phi0 = '+str(p_opt[3]))
print('   r33  = '+str(r33))

plt.figure()
plt.title('MZ: -45 deg axis')
plt.ylabel('Intensity [V]')
plt.xlabel('Vin [V]')
plt.grid(True, which='both')
plt.errorbar(x=Vin,y=Vm45,fmt='x')
plt.plot(x_array, fitfunc(x_array, *p_opt))

# part 5.3

print(' Transverse Amplitude Modulation')

Vp45 = np.array([2.93,3.02,3.09,3.00,2.91,2.84,2.91,3.01,3.03,2.94,2.84,2.85,2.95])
Vm45 = np.array([2.53,2.49,2.45,2.44,2.47,2.53,2.54,2.50,2.42,2.38,2.43,2.51,2.58])

plt.figure()
plt.title('Transverse Amplitude Modulation')
plt.ylabel('Intensity [V]')
plt.xlabel('Vin [V]')
plt.grid(True, which='both')
plt.errorbar(x=Vin,y=Vp45,fmt='x',label='+45 deg')
plt.errorbar(x=Vin,y=Vm45,fmt='x',label='-45 deg')

Vpi_calc = d/L*l/(r33*ne**3-r13*n0**3) 
print('  Vpic = '+str(Vpi_calc))

# AOM

print('\nAOM')

# speed of sound

f = np.array([85,90,95,100,105,110,115,120,125,130,135],dtype=float) * 1e6 # Hz
xl = np.array([2.7,2.8,2.9,3.1,3.3,3.5,3.6,3.7,3.9,4.1,4.3],dtype=float) # cm
xl2 = np.array([5.3,5.6,5.9,6.3,6.5,6.9,7.3],dtype=float) # cm
d = 205. # cm

l = 632.8e-9

def sin_ang(x,d):
  return np.sin(np.arctan2(x,d))

x_array = np.linspace(85e6, 135e6, num=500)

def fitfunc_1(f_s, v_s):
  return f_s * 1 * l / v_s

[p_opt_1, p_cov] = curve_fit(fitfunc_1, f, sin_ang(xl, d), p0=[4260])
v_s_1 = p_opt_1[0]

def fitfunc_2(f_s, v_s):
  return f_s * 2 * l / v_s

[p_opt_2, p_cov] = curve_fit(fitfunc_2, f[:7], sin_ang(xl2, d), p0=[4260])
v_s_2 = p_opt_2[0]

print(' speed of sound')
print('  script: 4260')
print('  m=1:    '+str(v_s_1))
print('  m=2:    '+str(v_s_2))

plt.figure()
plt.title('Speed of sound')
plt.xlabel('f [Hz]')
plt.ylabel('sin(angle)')
plt.grid(True, which='both')
plt.errorbar(x=f,y=sin_ang(xl,d),fmt='x')
plt.errorbar(x=f[:7],y=sin_ang(xl2,d),fmt='x')
plt.plot(x_array, fitfunc_1(x_array, *p_opt_1))
plt.plot(x_array, fitfunc_2(x_array, *p_opt_2))

# rel power (freq)

pM = 8.5 # V
p1 = np.array([0.6,1.25,1.85,1.3,0.86,0.59,0.31,0.16]) / pM

plt.figure()
plt.title('Relative Power (frequency)')
plt.xlabel('f [Hz]')
plt.ylabel('relative power')
plt.grid(True, which='both')
plt.errorbar(x=f[:8],y=p1,fmt='x')

print(' optimal power')
print('  f_max = 95 MHz')
print('  P_rel = 0.22')

# rel power (vco)

Ampl = np.array([2,1.8,1.6,1.4,1.2,1,0.8,0.7,0.6,0.5])
pM = 8.61
pDark = 0.16
p1 = np.array([1.26,1.23,1.21,1.17,1.04,0.91,0.5,0.26,0.24,0.2]) / pM

plt.figure()
plt.title('Relative Power (VCO)')
plt.xlabel('VCO [V]')
plt.ylabel('relative power')
plt.grid(True, which='both')
plt.errorbar(x=Ampl,y=p1,fmt='x')

print('  higher VCO results in higher rel power, no maximum')

# plot

plt.show()
