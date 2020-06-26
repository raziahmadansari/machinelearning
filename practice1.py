import matplotlib.pyplot as plt
import numpy as np
import pandas

'''fig = plt.figure()  #an empty figure with no axes
fig.suptitle('No axes on this figure')  #add a title so we know which it is

fig, ax_lst = plt.subplots(2, 2)    #a figure with a 2x2 grid of axes
plt.show()'''



'''a = pandas.DataFrame(np.random.rand(4, 5), columns=list('abcde'))
a_asndarray = a.values
fig = plt.plot(a_asndarray)

b = np.matrix([[1, 2], [3, 4]])
b_asarray = np.asarray(b)
fig1 = plt.plot(b_asarray)

plt.show()'''



'''x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title('Sample Plot')
plt.legend()
plt.show()'''


'''x = np.arange(0, 10, 0.2)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()'''


# plt.plot([1, 2, 3, 4])
'''plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.ylabel('some numbers')
plt.show()'''


# evenly sampled time at 200ms intervals
'''t = np.arange(0., 5., 0.2)

#red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()'''


'''data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', s='d', c='c', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()'''



'''#CREATE DATA
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0, 0, 0)
area = np.pi*3

# PLOT
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()'''



# CREATE DATA
N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
g2 = (0.4 + 0.3 * np.random.rand(N), 0.5 * np.random.rand(N))
g3 = (0.3 * np.random.rand(N), 0.3 * np.random.rand(N))

data = (g1, g2, g3)
colors = ('red', 'green', 'blue')
groups = ('coffe', 'tea', 'water')

# CREATE PLOT
fig = plt.figure()


for data, color, group in zip(data, colors, groups):
    x, y = data
    fig.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('matplot scatter plot')
plt.legend(2)
plt.show()
