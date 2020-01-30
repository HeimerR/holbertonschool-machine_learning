#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.suptitle('All in One')
plt.subplots_adjust(wspace=0.5, hspace=1)
ax1 = plt.subplot(321)
ax1.plot(y0, 'r')
ax1.autoscale(axis='x', tight=True)

ax2 = plt.subplot(322)
ax2.scatter(x1, y1, c='magenta')
#ax2.set(xlabel='Height (in)', ylabel='Weight (lbs)')
ax2.set_xlabel('Height (in)', fontsize='x-small')
ax2.set_ylabel('Weight (lbs)', fontsize='x-small')
ax2.set_title("Men's Height vs Weight", fontsize='x-small')

ax3 = plt.subplot(323)
#ax3.set(xlabel='Time (years)', ylabel='Fraction Remaining')
ax3.set_xlabel('Time (years)', fontsize='x-small')
ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
ax3.set_title('Exponential Decay of C-14', fontsize='x-small')
ax3.labelsize='x-small'
ax3.set(yscale='log', xlim=(0, 28650))
ax3.plot(x2, y2)

ax4 = plt.subplot(324)
ax4.set_xlabel('Time (years)', fontsize='x-small')
ax4.set_ylabel('Fraction Remaining', fontsize='x-small')
ax4.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
ax4.set(xlim=(0, 20000), ylim=(0, 1))
ax4.plot(x3, y31, 'r--', x3, y32, 'g')
ax4.legend(['C-14', 'Ra-226'])

ax5 = plt.subplot(313)
#ax5.set(xlabel='Grades', ylabel='Number of Students')
ax5.set_xlabel('Grades', fontsize='x-small')
ax5.set_ylabel('Number of Students', fontsize='x-small')
ax5.hist(student_grades, edgecolor='black', bins=range(0, 110, 10))
ax5.set_title('Project A', fontsize='x-small')
ax5.labelsize='x-small'
ax5.set_xticks(np.arange(0, 100, step=10))
ax5.set(xlim=(0, 100), ylim=(0, 30))

plt.show()
