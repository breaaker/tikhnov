import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ["times new roman", "simsun"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 定义网格
class cube:
    def __init__(self, r1, r2, bo, to, func):
        self.r1 = r1
        self.r2 = r2
        self.r = (r1 + r2) / 2
        self.bo = bo
        self.to = to
        self.z = (to + bo) / 2
        self.value = func(self.r, self.z)

# 定义弦
class line:
    def __init__(self, a_, b_, c_, opoint):
        '''
        (a_, b_, c_)为弦的方向向量
        opoint为弦的端点
        '''
        self.a_ = a_
        self.b_ = b_
        self.c_ = c_
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

# 检查交点是否在层内
def check(points, cube, line):
    '''
    此处points为弦和圆的交点
    '''
    a_, b_, c_ = line.a_, line.b_, line.c_
    opoint = line.opoint
    real_points = []
    for point in points:
        x, y = point[0], point[1]
        z = opoint[2] + (x - opoint[0]) * c_ / a_ if a_ != 0 else opoint[2] + (y - opoint[1]) * c_ / b_
        if cube.bo <= z <= cube.to:
            real_points.append([x, y, z])
    return real_points

# 计算弦在网格内的长度
def length(line, cube, rmin):
    '''
    d为弦距离圆心的距离
    '''
    a_, b_ = line.a_, line.b_
    opoint = line.opoint
    a = b_
    b = -a_
    c = -b_ * opoint[0] + a_ * opoint[1]

    d = abs(c) / np.sqrt(a**2 + b**2)
    r1, r2 = cube.r1, cube.r2
    points = inter(a, b, c, r1)
    points += inter(a, b, c, r2)
    points = check(points, cube, line)

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
    def __init__(self, rmin, rmax, n_r, bo_, to_, n_z, func):
        self.cubes = []
        self.values = []
        self.rmin = rmin
        self.rmax = rmax
        self.bo = bo_
        self.to = to_
        self.r = np.linspace(rmin, rmax, n_r+1)
        self.z = np.linspace(bo_, to_, n_z+1)

        for i in range(n_r):
            for j in range(n_z):
                r1, r2 = self.r[i], self.r[i+1]
                z1, z2 = self.z[j], self.z[j+1]
                cube_ = cube(r1, r2, z1, z2, func)
                self.cubes.append(cube_)
                self.values.append(cube_.value)

# 把球面坐标转化成xyz坐标
def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# 给定观察点和观察角度范围生成弦
def generate_line(x, y, z, theta_1r, theta_2r, num_r, theta_1z, theta_2z, num_z):
    lines = []
    theta_1r = theta_1r / 180 * np.pi
    theta_2r = theta_2r / 180 * np.pi
    theta_1z = theta_1z / 180 * np.pi
    theta_2z = theta_2z / 180 * np.pi
    
    r1_z1 = sph2cart(1, theta_1z, theta_1r)
    r1_z2 = sph2cart(1, theta_2z, theta_1r)
    r2_z1 = sph2cart(1, theta_1z, theta_2r)
    dr = (r2_z1 - r1_z1) / (num_r - 1)
    dz = (r1_z2 - r1_z1) / (num_z - 1)
    for i in range(num_r):
        for j in range(num_z):
            vec = r1_z1 + i * dr + j * dz
            vec = vec / np.linalg.norm(vec)
            lines.append(line(vec[0], vec[1], vec[2], [x, y, z]))
    
    return lines

# 绘图表示网格和弦
def plot(grid, lines, lines_num):
    grid = grid.cubes
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(141, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Grid and Lines (num={lines_num[0]}(r)*{lines_num[1]}(z))")
    
    bo = grid[0].bo
    to = grid[-1].to
    r1 = grid[0].r1
    r2 = grid[-1].r2

    theta = np.linspace(0, 2 * np.pi, 100)

    for r in [r1, r2]:
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        z_bottom = np.ones_like(theta) * bo
        ax.plot(x, y, z_bottom, 'b-', alpha=0.8, linewidth=0.8)

        z_top = np.ones_like(theta) * to
        ax.plot(x, y, z_top, 'b-', alpha=0.8, linewidth=0.8)

        for t in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x_line = [r * np.cos(t), r * np.cos(t)]
            y_line = [r * np.sin(t), r * np.sin(t)]
            z_line = [bo, to]
            ax.plot(x_line, y_line, z_line, 'b-', alpha=0.5, linewidth=0.5)

    for i in range(len(lines)):
        line = lines[i]
        a_, b_, c_ = line.a_, line.b_, line.c_
        opoint = line.opoint
        a = b_
        b = -a_
        c = -b_ * opoint[0] + a_ * opoint[1]
        
        points = inter(a, b, c, r1)
        points = check(points, cube(r1, r2, -np.inf, np.inf, lambda r, z: 0), line)
        
        if len(points) >=1:
            points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
            p1 = points[0]
            if p1[2] < bo or p1[2] > to:
                z = bo if p1[2] < bo else to
                x = (z - opoint[2]) * a_ / c_ + opoint[0]
                y = (z - opoint[2]) * b_ / c_ + opoint[1]
                ax.plot([opoint[0], x], [opoint[1], y], [opoint[2], z], color='green', linewidth=0.7, alpha=0.7)
            else:
                ax.plot([opoint[0], p1[0]], [opoint[1], p1[1]], [opoint[2], p1[2]], color='green', linewidth=0.7, alpha=0.7)

        else:
            points = inter(a, b, c, r2)
            points = check(points, cube(r1, r2, -np.inf, np.inf, lambda r, z: 0), line)
            
            if len(points) >=1:
                points = sorted(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(opoint)))
                p1 = points[-1]
                if p1[2] < bo or p1[2] > to:
                    z = bo if p1[2] < bo else to
                    x = (z - opoint[2]) * a_ / c_ + opoint[0]
                    y = (z - opoint[2]) * b_ / c_ + opoint[1]
                    ax.plot([opoint[0], x], [opoint[1], y], [opoint[2], z], color='green', linewidth=0.7, alpha=0.7)
                else:
                    ax.plot([opoint[0], p1[0]], [opoint[1], p1[1]], [opoint[2], p1[2]], color='green', linewidth=0.7, alpha=0.7)

    ax.set_aspect('equal', adjustable='box')
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

num_r = 30
num_z = 80
num_lr = 10
num_lz = 15
lambda_ = 1e-2

def func(r, z):
    return np.exp(-70*((r-0.5)**2 + (z-0.3)**2)) + np.exp(-70*((r-0.5)**2 + (z+0.3)**2))

grid_ = grid(0.2, 0.8, num_r, -0.8, 0.8, num_z, func)

# 可在此处修改生成弦的方式 ########################################
lines = generate_line(0, -0.8, 0, 40, 72, num_lr, 45, 135, num_lz)

############################################################

A = get_A(grid_, lines)
ori = np.array(grid_.values)
b = A @ ori

'''noise = 0.05
for i in range(len(b)):
    b[i] *= 1 + np.random.normal(-noise, noise)'''

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

H = laplace_2d(num_r, num_z).T @ laplace_2d(num_r, num_z)

guess = np.linalg.inv(A.T @ A + lambda_ * H) @ A.T @ b
guess = guess.reshape(num_r, num_z).T
ori = ori.reshape(num_r, num_z).T

from scipy.signal import convolve2d
smooth_num = 5
kernal = np.ones((smooth_num, smooth_num)) / smooth_num**2
'''guess = convolve2d(guess, kernal, mode='same')'''

# 使用cvxpy进行优化
import cvxpy as cp
L = laplace_2d(num_r, num_z)
x = cp.Variable(num_r * num_z)
objective = cp.Minimize(cp.norm(A @ x - b) + lambda_ * cp.norm(L @ x))
constraints = [x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)
guess2 = x.value
guess2 = guess2.reshape(num_r, num_z).T
# 对guess2进行平滑处理
'''guess2 = convolve2d(guess2, kernal, mode='same')'''

# 绘图
fig = plot(grid_, lines, [num_lr, num_lz])

rs = np.linspace(grid_.rmin, grid_.rmax, num_r)
zs = np.linspace(grid_.bo, grid_.to, num_z)
rs, zs = np.meshgrid(rs, zs)

ax = fig.add_subplot(142, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_zlabel('intensity')
ax.plot_surface(rs, zs, ori, cmap='viridis')
ax.set_aspect('equal')
ax.set_title("Original")

ax = fig.add_subplot(143, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_zlabel('intensity')
ax.plot_surface(rs, zs, guess, cmap='viridis')
ax.set_aspect('equal')
ax.set_title("$\Gamma = \\nabla^2$")

ax = fig.add_subplot(144, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_zlabel('intensity')
ax.plot_surface(rs, zs, guess2, cmap='viridis')
ax.set_aspect('equal')
ax.set_title("$\Gamma = \\nabla^2$ and $x_j\geq$0")

plt.savefig("tik4.png", dpi=300, bbox_inches='tight')
plt.show()
