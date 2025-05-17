import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ["times new roman", "simsun"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
# 定义网格
class cube:
    def __init__(self, l, r, b, t, func1, func2):
        self.l = l
        self.r = r
        self.b = b
        self.t = t
        self.x = (l + r) / 2
        self.y = (b + t) / 2
        self.w = r - l
        self.h = t - b
        self.I = func1(self.x, self.y)
        self.alpha = func2(self.x, self.y)
        self.value_x = self.I * np.cos(self.alpha)
        self.value_y = self.I * np.sin(self.alpha)

# 定义弦
class line:
    def __init__(self, a, b_, c, opoint):
        self.a = a
        self.b_ = b_
        self.c = c
        self.opoint = opoint

# 计算弦在网格内的长度
def length(line, cube):
    '''
    先找到交点。若有交点，则计算长度；若无交点，则返回0
    '''
    a = line.a
    b_ = line.b_
    c = line.c
    opoint = line.opoint
    l = cube.l
    r = cube.r
    b = cube.b
    t = cube.t
    vx = cube.value_x
    vy = cube.value_y
    half_sign = False

    points = []
    if a == 0:
        y = -c / b_
        if y>b and y<t:
            points.append([l, y])
            points.append([r, y])
        elif y==b or y==t:
            points.append([l, y])
            points.append([r, y])
            half_sign = True
        else:
            return 0, 0
    elif b_ == 0:
        x = -c / a
        if x>l and x<r:
            points.append([x, b])
            points.append([x, t])
        elif x==l or x==r:
            points.append([x, b])
            points.append([x, t])
            half_sign = True
        else:
            return 0, 0
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
        sorted_points = sorted(points, key=lambda x: np.linalg.norm(np.array(x) - np.array(opoint)))
        dx = sorted_points[0][0] - sorted_points[1][0]
        dy = sorted_points[0][1] - sorted_points[1][1]
        if dx*vx + dy*vy < 0:
            dx = 0
            dy = 0
        if half_sign:
            dx /= 2
            dy /= 2
        return dx, dy
    else:
        return 0, 0

# 给定边界生成网格
class grid:
    def __init__(self, l, r, b, t, func1, func2, m, n):
        self.cubes = []
        self.valuexs = []
        self.valueys = []
        dx = (r - l) / m
        dy = (t - b) / n
        for i in range(m):
            for j in range(n):
                self.cubes.append(cube(l + i * dx, l + (i+1) * dx, b + j * dy, b + (j+1) * dy, func1, func2))
                self.valuexs.append(self.cubes[-1].value_x)
                self.valueys.append(self.cubes[-1].value_y)

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
        lines.append(line(a, b_, c, np.array([x, y])))
    return lines

# 绘图表示网格和弦
def plot(grid, lines):
    grid = grid.cubes
    fig = plt.figure(figsize=(13, 9))

    ax = fig.add_subplot(231)
    for i in range(len(grid)):
        ax.add_artist(plt.Rectangle((grid[i].l, grid[i].b), grid[i].w, grid[i].h, fill=False, alpha=0.5, edgecolor='gray', lw=0.8))
        
    l = grid[0].l
    r = grid[-1].r
    b = grid[0].b
    t = grid[-1].t

    for i in range(len(lines)):
        if lines[i].b_ == 0:
            x = -lines[i].c / lines[i].a
            plt.plot([x, x], [b, t], color='b', alpha=0.8, lw=1)
        else:
            y1 = (-lines[i].a * l - lines[i].c) / lines[i].b_
            y2 = (-lines[i].a * r - lines[i].c) / lines[i].b_
            plt.plot([l, r], [y1, y2], color='b', alpha=0.8, lw=1)

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
    Ax = np.zeros((len(lines), len(grid)))
    Ay = np.zeros((len(lines), len(grid)))
    for i in range(len(lines)):
        for j in range(len(grid)):
            Ax[i][j], Ay[i][j] = length(lines[i], grid[j])
    A = np.concatenate((Ax, Ay), axis=1)
    return A

from scipy.signal import convolve2d
from scipy.sparse import kron, eye
import cvxpy as cp

##################################################
# 测试
##################################################

num_x = 40
num_y = 40
num_lines = 20
lambda_1 = 1e-3
lambda_2 = 1e-7

def func1(x, y):
    return np.exp(- 5 * (x**2 + y**2))

def func2(x, y):
    return np.arctan2(y, x)

grid_ = grid(-1, 1, -1, 1, func1, func2, num_x, num_y)

# 可在此处修改生成弦的方式 ########################################
lines = generate_line(0, -1, 0, 180, num_lines)
lines += generate_line(1, 0, 90, 270, num_lines)
lines += generate_line(0, 1, 180, 360, num_lines)

############################################################

A = get_A(grid_, lines)

orix = np.array(grid_.valuexs)
oriy = np.array(grid_.valueys)
ori = np.concatenate((orix, oriy))
b = A @ ori

orix = orix.reshape(num_x, num_y).T
oriy = oriy.reshape(num_x, num_y).T

noise = 0.05
for i in range(len(b)):
    b[i] *= 1 + np.random.normal(-noise, noise)

smooth_num = 5
kernal = np.ones((smooth_num, smooth_num)) / smooth_num**2

def laplace_2d(num_x, num_y):
    Dx = eye(num_x) - eye(num_x, k=-1)
    Dy = eye(num_y) - eye(num_y, k=-1)
    Lx = Dx.T @ Dx
    Ly = Dy.T @ Dy
    Gamma = kron(Lx, eye(num_y)) + kron(eye(num_x), Ly)
    Gamma = Gamma.toarray()
    x, y = Gamma.shape
    Gamma1 = np.concatenate((Gamma, np.zeros((x, y))), axis=1)
    Gamma2 = np.concatenate((np.zeros((x, y)), Gamma), axis=1)
    Gamma = np.concatenate((Gamma1, Gamma2), axis=0)
    return Gamma

m, n = A.shape
L = laplace_2d(num_x, num_y)

## cvxpy
x = cp.Variable(2 * num_x * num_y)
objective = cp.Minimize(cp.norm(A @ x - b) + lambda_1 * cp.norm(L @ x) + lambda_2 * cp.norm(x, 1))
prob = cp.Problem(objective)
prob.solve(solver=cp.SCS, verbose=False)
guess = x.value
guessx = guess[:num_x * num_y].reshape(num_x, num_y).T
guessy = guess[num_x * num_y:].reshape(num_x, num_y).T

guessx = convolve2d(guessx, kernal, mode='same')
guessy = convolve2d(guessy, kernal, mode='same')

fig = plot(grid_, lines)

ax = fig.add_subplot(232, projection='3d')
X, Y = np.meshgrid(np.linspace(-1, 1, num_x), np.linspace(-1, 1, num_y))
ax.plot_surface(X, Y, orix, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Original value$_x$")

ax = fig.add_subplot(235, projection='3d')
ax.plot_surface(X, Y, oriy, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Original value$_y$")

ax = fig.add_subplot(233, projection='3d')
ax.plot_surface(X, Y, guessx, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Reconstructed value$_x$")

ax = fig.add_subplot(236, projection='3d')
ax.plot_surface(X, Y, guessy, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Reconstructed value$_y$")

plt.savefig("tik3.png", dpi=300)
plt.show()
plt.close()

# 还原为I和alpha并绘图
I_ = np.sqrt(orix**2 + oriy**2)
alpha_ = np.zeros((num_x, num_y, 2))
for i in range(num_x):
    for j in range(num_y):
        alpha_[i][j] = np.array([orix[i][j], oriy[i][j]])/I_[i][j]

I = np.sqrt(guessx**2 + guessy**2)
alpha = np.zeros((num_x, num_y, 2))
for i in range(num_x):
    for j in range(num_y):
        alpha[i][j] = np.array([guessx[i][j], guessy[i][j]])/I[i][j]

fig = plot(grid_, lines)
ax = fig.add_subplot(232, projection='3d')
ax.plot_surface(X, Y, I_, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Original I")

ax = fig.add_subplot(235)
X2, Y2 = np.meshgrid(np.linspace(-1, 1, num_x//2), np.linspace(-1, 1, num_y//2))
U = alpha_[::2, ::2, 0]
V = alpha_[::2, ::2, 1]
ax.quiver(X2, Y2, U, V, color='b', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Original $\\alpha$")

ax = fig.add_subplot(233, projection='3d')
ax.plot_surface(X, Y, I, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel("intensity")
ax.set_title("Reconstructed I")

ax = fig.add_subplot(236)
U = alpha[::2, ::2, 0]
V = alpha[::2, ::2, 1]
ax.quiver(X2, Y2, U, V, color='b', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Reconstructed $\\alpha$")

plt.savefig("tik3_2.png", dpi=300)
plt.show()
plt.close()