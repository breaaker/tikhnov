import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ["times new roman", "simsun"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
# 定义网格
class cube:
    def __init__(self, l, r, b, t, func):
        self.l = l
        self.r = r
        self.b = b
        self.t = t
        self.x = (l + r) / 2
        self.y = (b + t) / 2
        self.w = r - l
        self.h = t - b
        self.value = func(self.x, self.y)

# 定义弦
class line:
    def __init__(self, a, b_, c):
        self.a = a
        self.b_ = b_
        self.c = c

# 计算弦在网格内的长度
def length(line, cube):
    '''
    先找到交点。若有交点，则计算长度；若无交点，则返回0
    '''
    a = line.a
    b_ = line.b_
    c = line.c
    l = cube.l
    r = cube.r
    b = cube.b
    t = cube.t
    points = []
    if a == 0:
        y = -c / b_
        if y>b and y<t:
            points.append([l, y])
            points.append([r, y])
        elif y==b or y==t:
            return (r - l)/2
        else:
            return 0
    elif b_ == 0:
        x = -c / a
        if x>l and x<r:
            points.append([x, b])
            points.append([x, t])
        elif x==l or x==r:
            return (t - b)/2
        else:
            return 0
    else:
        y1 = (-a * l - c) / b_
        y2 = (-a * r - c) / b_
        x1 = (-b_ * b - c) / a
        x2 = (-b_ * t - c) / a
        if y1>=b and y1<=t:
            points.append([l, y1])
        if y2>=b and y2<=t:
            points.append([r, y2])
        if x1>=l and x1<=r:
            points.append([x1, b])
        if x2>=l and x2<=r:
            points.append([x2, t])
    if len(points) == 2:
        return np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
    else:
        return 0

# 给定边界生成网格
class grid:
    def __init__(self, l, r, b, t, func, m, n):
        self.cubes = []
        self.values = []
        dx = (r - l) / m
        dy = (t - b) / n
        for i in range(m):
            for j in range(n):
                self.cubes.append(cube(l + i * dx, l + (i+1) * dx, b + j * dy, b + (j+1) * dy, func))
                self.values.append(self.cubes[-1].value)

# 给定观察点和观察角度范围生成弦
def generate_line(x, y, theta_1, theta_2, num):
    lines = []
    theta_1 = theta_1 / 180 * np.pi
    theta_2 = theta_2 / 180 * np.pi
    dtheta = (theta_2 - theta_1) / (num-1)
    for i in range(num):
        theta = theta_1 + i * dtheta
        if theta == np.pi / 2 or theta == 3 * np.pi / 2:
            a = 1
            b_ = 0
            c = -x
        else:
            a = np.tan(theta)
            b_ = -1
            c = -a * x + y
        lines.append(line(a, b_, c))
    return lines

# 给定边界生成弦
def generate_line_2(l, r, b, t, theta, num):
    theta = theta / 180 * np.pi
    lines = []
    for i in range(num):
        if theta == np.pi / 2 or theta == 3 * np.pi / 2:
            a = 1
            b_ = 0
            c = -l - (r-l)/(num-1)*i
        else:
            a = np.tan(theta)
            b_ = -1
            if a > 0:
                c = -a * l + t - a*(r-l)/(num-1)*i - (t-b)/(num-1)*i
            elif a == 0:
                c = b + (t-b)/(num-1)*i
            else:
                c = -a * l + b - a*(r-l)/(num-1)*i + (t-b)/(num-1)*i
        lines.append(line(a, b_, c))
    return lines

# 绘图表示网格和弦
def plot(grid, lines):
    grid = grid.cubes
    fig = plt.figure(figsize=(21, 6))
    ax = fig.add_subplot(141)
    for i in range(len(grid)):
        ax.add_artist(plt.Rectangle((grid[i].l, grid[i].b), grid[i].w, grid[i].h, fill=False, alpha=0.5, edgecolor='gray', lw=0.8))
        
    l = grid[0].l
    r = grid[-1].r
    b = grid[0].b
    t = grid[-1].t

    for i in range(len(lines)):
        if lines[i].b_ == 0:
            x = -lines[i].c / lines[i].a
            plt.plot([x, x], [b, t], color='b', alpha=0.7, lw=1)
        else:
            y1 = (-lines[i].a * l - lines[i].c) / lines[i].b_
            y2 = (-lines[i].a * r - lines[i].c) / lines[i].b_
            plt.plot([l, r], [y1, y2], color='b', alpha=0.7, lw=1)

    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title("Grid and Lines")
    return fig

# 得到观察矩阵A
def get_A(grid, lines):
    grid = grid.cubes
    A = np.zeros((len(lines), len(grid)))
    for i in range(len(lines)):
        for j in range(len(grid)):
            A[i][j] = length(lines[i], grid[j])
    return A

##################################################
# 测试
##################################################

num_x = 40
num_y = 40
num_lines = 20
lambda_ = 1e-2

def func(x, y):
    return np.exp(-20*(x**2 + y**2))

grid_ = grid(-1, 1, -1, 1, func, num_x, num_y)

# 可在此处修改生成弦的方式 ########################################
lines = generate_line(0, -1, 0, 180, num_lines)
lines += generate_line(1, 0, 90, 270, num_lines)
lines += generate_line(0, 1, 180, 360, num_lines)

############################################################

A = get_A(grid_, lines)

ori = np.array(grid_.values)
b = A @ ori

noise = 0.1
for i in range(len(b)):
    b[i] *= 1 + np.random.normal(-noise, noise)

guess = np.linalg.inv(A.T @ A + lambda_ * np.eye(num_x * num_y)) @ A.T @ b
ori = ori.reshape(num_x, num_y).T
guess = guess.reshape(num_x, num_y).T

from scipy.signal import convolve2d
smooth_num = 5
kernal = np.ones((smooth_num, smooth_num)) / smooth_num**2
guess = convolve2d(guess, kernal, mode='same')

import cvxpy as cp
x = cp.Variable(num_x * num_y)
objective = cp.Minimize(cp.norm(A @ x - b) + lambda_ * cp.norm(x))
constraints = [x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)
guess2 = x.value
guess2 = guess2.reshape(num_x, num_y).T

guess2 = convolve2d(guess2, kernal, mode='same')

# 绘图
fig = plot(grid_, lines)

X, Y = np.meshgrid(np.linspace(-1, 1, num_x), np.linspace(-1, 1, num_y))
ax = fig.add_subplot(142, projection='3d')
ax.plot_surface(X, Y, ori, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Original")

ax = fig.add_subplot(143, projection='3d')
ax.plot_surface(X, Y, guess, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("$\Gamma = I$")

ax = fig.add_subplot(144, projection='3d')
ax.plot_surface(X, Y, guess2, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Reconstructed with x$\geq$0")

plt.savefig("tik2_1.png", dpi=300)
plt.show()
plt.close()

## laplace
from scipy.sparse import kron, eye

def laplace_2d(num_x, num_y):
    Dx = eye(num_x) - eye(num_x, k=-1)
    Dy = eye(num_y) - eye(num_y, k=-1)
    Lx = Dx.T @ Dx
    Ly = Dy.T @ Dy
    Gamma = kron(Lx, eye(num_y)) + kron(eye(num_x), Ly)

    Gamma = Gamma.toarray()
    return Gamma

def laplace(num):
    return -2 * np.eye(num) + np.eye(num, k=1) + np.eye(num, k=-1)

m, n = A.shape

H = laplace_2d(num_x, num_y).T @ laplace_2d(num_x, num_y)

guess = np.linalg.inv(A.T @ A + lambda_ * H) @ A.T @ b
guess = guess.reshape(num_x, num_y).T
guess = convolve2d(guess, kernal, mode='same')

## 下面是限制x>0的情况

L = laplace_2d(num_x, num_y)
x = cp.Variable(num_x * num_y)
objective = cp.Minimize(cp.norm(A @ x - b) + lambda_ * cp.norm(L @ x))
constraints = [x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)
guess2 = x.value
guess2 = guess2.reshape(num_x, num_y).T

guess2 = convolve2d(guess2, kernal, mode='same')

fig = plot(grid_, lines)

ax = fig.add_subplot(142, projection='3d')
ax.plot_surface(X, Y, ori, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Original")

ax = fig.add_subplot(143, projection='3d')
ax.plot_surface(X, Y, guess, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("$\Gamma = \\nabla^2$")

ax = fig.add_subplot(144, projection='3d')
ax.plot_surface(X, Y, guess2, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("$\Gamma = \\nabla^2$ and $x_j\geq$0")

plt.savefig("tik2_2.png", dpi=300)
plt.show()
plt.close()