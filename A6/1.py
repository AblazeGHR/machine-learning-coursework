import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 5.5))

# ========== 左图：光路侧视图 ==========
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_aspect('equal')
ax1.set_xlim(-1, 11)
ax1.set_ylim(-3, 6)
ax1.axis('off')
ax1.set_title('点光源迈克尔逊干涉：光路侧视图', fontsize=13, fontweight='bold')

# 光轴 (z 轴)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
# 光屏
ax1.plot([10, 10], [-2.5, 5.5], color='black', linewidth=2, label='观察屏 (光屏)')
ax1.text(10.2, 5.2, '光屏', fontsize=11, ha='left', va='bottom')

# 两个虚点光源 S2 (靠前) 和 S1 (靠后，间距 2d)
ax1.plot(0, 0, 'o', color='darkorange', markersize=10, label='S₂ (靠前虚光源)')
ax1.text(-0.3, 0.3, 'S₂', fontsize=11, ha='right', color='darkorange')
ax1.plot(-3, 0, 'o', color='darkred', markersize=10, label='S₁ (靠后虚光源)')
ax1.text(-3.3, 0.3, 'S₁', fontsize=11, ha='right', color='darkred')

# 标注轴向距离 2d
ax1.annotate('', xy=(-3, -0.4), xytext=(0, -0.4),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(-1.5, -0.8, r'$2d$ (两虚光源轴向间距)', fontsize=12, ha='center')

# 光屏上一点 P (离轴有一定高度)
rho = 3.0
P = np.array([10, rho])
ax1.plot(*P, 'o', color='blue', markersize=8)
ax1.text(10.3, rho+0.2, 'P', fontsize=11, color='blue')

# 画出 S1->P 和 S2->P 的光线
S2 = np.array([0., 0.])
S1 = np.array([-3., 0.])
ax1.plot([S2[0], P[0]], [S2[1], P[1]], color='darkorange', linewidth=1.5, alpha=0.8)
ax1.plot([S1[0], P[0]], [S1[1], P[1]], color='darkred', linewidth=1.5, alpha=0.8)

# 画出视角 θ 的辅助线
# 以 P 为顶点，向光轴作垂足 (10,0)，构成直角三角形
ax1.plot([P[0], P[0]], [0, P[1]], 'gray', linestyle=':', linewidth=0.8)
ax1.plot([S2[0], P[0]], [0, 0], 'gray', linestyle=':', linewidth=0.8)  # 其实S2在轴上，这条线就是S2-P的水平投影
# 标记θ角: 在P点处，水平线与S2-P连线的夹角
angle_arc = np.linspace(0, np.arctan2(rho, 10), 20)
arc_r = 0.8
ax1.plot(P[0] - arc_r * np.cos(angle_arc), P[1] - arc_r * np.sin(angle_arc), 'blue', lw=1.5)
ax1.text(P[0]-1.2, P[1]-0.6, r'$\theta$', fontsize=13, color='blue', ha='center')

# 标注 L
ax1.annotate('', xy=(10, -0.2), xytext=(0, -0.2),
             arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
ax1.text(5, -0.6, r'$L$ (观察距离)', fontsize=11, ha='center', color='gray')

# 说明文字
ax1.text(0.5, 5.5, 'S₁ 和 S₂ 是一对相干虚点光源', fontsize=10, color='darkred')
ax1.text(0.5, 4.8, r'光程差 $\Delta \approx 2d \cos\theta$', fontsize=12, fontweight='bold')

# ========== 右图：干涉条纹模拟 (同心圆环) ==========
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_aspect('equal')
ax2.set_title('光屏上的干涉条纹 (同心圆环)', fontsize=13, fontweight='bold')

# 生成模拟的干涉强度
size = 500
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# 模拟公式：光程差 Δ = 2d * cosθ，这里 cosθ ≈ 1 - (r^2)/(2L^2)，光强 I = cos²(πΔ/λ)
# 选取参数使得能看到多个圆环
d = 50e-6   # 臂差 50 μm，那么虚光源间距 2d=100μm
lam = 632.8e-9  # He-Ne激光波长
L = 0.5     # 观察距离 0.5m
# 计算视角 θ 的余弦近似：cosθ ≈ L / sqrt(L^2 + (r*scale)^2)，设光屏尺寸10cm
screen_radius = 0.05  # 半径5cm
r_real = R * screen_radius
cos_theta = L / np.sqrt(L**2 + r_real**2)
Delta = 2 * d * cos_theta
I = np.cos(np.pi * Delta / lam)**2

ax2.imshow(I, extent=[-1, 1, -1, 1], cmap='gray', origin='lower')
ax2.set_xlabel('光屏位置 (归一化)', fontsize=10)
ax2.set_ylabel('光屏位置 (归一化)', fontsize=10)
ax2.text(0, -1.15, '中心对应 θ=0 (正入射)，越向外圆环越密', fontsize=10, ha='center')

plt.tight_layout()
plt.show()