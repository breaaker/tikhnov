import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ["times new roman", "simsun"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 定义网格
class cube:
    def __init__(self, r1, r2, func):
        self.r1 = r1
        self.r2 = r2
        self.x = (r1 + r2) / 2
        self.w = r2 - r1
        self.value = func(self.x)

# 定义弦
class line:
    def __init__(self, a, b, c, opoint):
        self.a = a
        self.b = b
        self.c = c
        self.opoint = opoint

# 计算弦和圆的交点
def inter(a, b, c, r):
    points = []
    if b == 0:
        x = -c / a
        if x > -r and x < r:
            points.append([x, np.sqrt(r**2 - x**2)])
            points.append([x, -np.sqrt(r**2 - x**2)])
        elif x == -r or x == r:
            points.append([x, 0])
    else:
        delta = (2*a*c)**2 - 4*(a**2 + b**2)*(c**2 - b**2*r**2)
        if delta == 0:
            x = -a*c / (a**2 + b**2)
            y = -(a*x + c) / b
            points.append([x, y])
        elif delta > 0:
            x = (-2*a*c + np.sqrt(delta)) / (2*(a**2 + b**2))
            y = -(a*x + c) / b
            points.append([x, y])
            x = (-2*a*c - np.sqrt(delta)) / (2*(a**2 + b**2))
            y = -(a*x + c) / b
            points.append([x, y])
    return points

# 计算弦在网格内的长度
def length(line, cube, rmin):
    '''
    d为弦距离圆心的距离
    '''
    a, b, c = line.a, line.b, line.c
    d = abs(c) / np.sqrt(a**2 + b**2)
    r1, r2 = cube.r1, cube.r2
    points = inter(a, b, c, r1)
    points += inter(a, b, c, r2)
    opoint = line.opoint

    if len(points) == 4 and d < rmin:
        points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
        p1, p2 = points[:2]
        return np.linalg.norm(np.array(p1) - np.array(p2))
    elif len(points) == 4 and d >= rmin:
        points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
        p1, p2 = points[0], points[3]
        return np.linalg.norm(np.array(p1) - np.array(p2))
    elif len(points) == 3:
        points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
        p1, p2 = points[0], points[2]
        return np.linalg.norm(np.array(p1) - np.array(p2))
    elif len(points) == 2:
        points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
        p1, p2 = points[0], points[1]
        return np.linalg.norm(np.array(p1) - np.array(p2))
    else:
        return 0

# 给定边界生成网格
class grid:
    def __init__(self, rmin, rmax, func, n):
        self.cubes = []
        self.values = []
        self.rmin = rmin
        self.rmax = rmax
        self.r = np.linspace(rmin, rmax, n+1)
        x = np.zeros(n, dtype=float)
        self.w = (rmax - rmin) / n
        for i in range(n):
            r1 = self.r[i]
            r2 = self.r[i+1]
            x[i] = (r1 + r2) / 2
            cube_ = cube(r1, r2, func)
            self.cubes.append(cube_)
            self.values.append(cube_.value)
        self.x = x

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
            b = 0
            c = -x
        else:
            a = np.tan(theta)
            b = -1
            c = -a * x + y
        lines.append(line(a, b, c, [x, y]))
    return lines

# 绘图表示网格和弦
def plot(grid, lines, lines_num):
    grid = grid.cubes
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(141)

    ax.add_artist(plt.Circle((0, 0), grid[0].r1, color='black', fill=False))
    ax.add_artist(plt.Circle((0, 0), grid[-1].r2, color='black', fill=False))

    ax.plot([-0.8, -0.2], [0, 0], color='green', linewidth=0.5, alpha=0.5)
    num_ticks = 10
    x_ticks = np.linspace(-0.8, -0.2, num_ticks + 1)
    for x in x_ticks:
        ax.plot([x, x], [-0.02, 0.02], color='green', linewidth=0.5)
    ax.text(-0.5, 0.05, f'{num}', color='green', fontsize=10)

    for i in range(len(lines)):
        line = lines[i]
        a, b, c = line.a, line.b, line.c
        opoint = line.opoint
        points = inter(a, b, c, grid[0].r1)
        
        if len(points) >= 1:
            points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
            p1 = points[0]
        else:
            points = inter(a, b, c, grid[-1].r2)
            points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
            p1 = points[1]
        ax.plot([opoint[0], p1[0]], [opoint[1], p1[1]], color='red', linewidth=0.5, alpha=0.7)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Grid and Lines (num={lines_num})")
    return fig

# 得到观察矩阵A
def get_A(grid, lines):
    grid = grid.cubes
    r_min = grid[0].r1
    A = np.zeros((len(lines), len(grid)))
    for i in range(len(lines)):
        for j in range(len(grid)):
            A[i][j] = length(lines[i], grid[j], r_min)
    return A

##################################################
# 测试
##################################################

num = 30
num_lines = 15
lambda_ = 1e-2

def func(r):
    return np.exp(-100*(r-0.5)**2)

grid_ = grid(0.2, 0.8, func, num)

# 可在此处修改生成弦的方式 ########################################
lines = generate_line(0, -0.8, 10, 80, num_lines)

############################################################

A = get_A(grid_, lines)
ori = np.array(grid_.values)
b = A @ ori

noise = 0.05
for i in range(len(b)):
    b[i] *= 1 + np.random.normal(-noise, noise)

guess = np.linalg.inv(A.T @ A + lambda_ * np.eye(num)) @ A.T @ b
guess_smooth = np.convolve(guess, np.ones(3)/3, mode='same')

# 绘图
fig = plot(grid_, lines, num_lines)

ax = fig.add_subplot(142)
ax.set_title('$\Gamma=I$')
ax.plot(grid_.x, ori, label='original')
ax.plot(grid_.x, guess, label='result', alpha=0.5, linestyle='--')
ax.plot(grid_.x, guess_smooth, label='smooth_result')
ax.set_xlabel('r')
ax.set_ylabel('Intensity')
ax.legend()

def laplace(n):
    return -2 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)

m, n = A.shape
L = laplace(n)
H = L.T @ L
guess = np.linalg.inv(A.T @ A + lambda_ * H) @ A.T @ b
guess_smooth = np.convolve(guess, np.ones(3)/3, mode='same')

ax = fig.add_subplot(143)
ax.set_title('$\Gamma=\\nabla^2$')
ax.plot(grid_.x, ori, label='original')
ax.plot(grid_.x, guess, label='result', alpha=0.5, linestyle='--')
ax.plot(grid_.x, guess_smooth, label='smooth_result')
ax.set_xlabel('r')
ax.set_ylabel('Intensity')
ax.legend()

import cvxpy as cp
x = cp.Variable(n)
obj = cp.Minimize(cp.norm(A @ x - b) + lambda_ * cp.norm(L @ x))
constraints = [x >= 0]
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.SCS, verbose=False)
guess2 = x.value

guess2_smooth = np.convolve(guess2, np.ones(3)/3, mode='same')
ax = fig.add_subplot(144)
ax.set_title('$\Gamma=\\nabla$ and $x\geq0$')
ax.plot(grid_.x, ori, label='original')
ax.plot(grid_.x, guess2, label='result', alpha=0.5, linestyle='--')
ax.plot(grid_.x, guess2_smooth, label='smooth_result')
ax.set_xlabel('r')
ax.set_ylabel('Intensity')
ax.legend()

plt.savefig("tik1.png", dpi=300)
plt.show()
plt.close()
