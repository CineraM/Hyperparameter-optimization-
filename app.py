import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)
print(t)

fig, ax = plt.subplots()

plt.xlim(0, 50)
plt.ylim(0, 5)

#ax.plot(t, s, '-', label='test')
ax.plot(t, s, 'm', label='Sine')
plt.plot([1, 2, 3], label="xd")
ax.set(xlabel='epoch', title='Optim')
ax.grid()
leg = ax.legend();

fig.savefig("xd")
ax.cla()

plt.xlim(0, 30)
plt.ylim(0, 2)



plt.plot([1, 2, 3], label="xd")
ax.set(xlabel='epoch', title='Optim')
ax.grid()
leg = ax.legend();
fig.savefig("xd2")
