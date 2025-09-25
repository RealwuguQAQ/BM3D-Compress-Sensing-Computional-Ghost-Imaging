import numpy as np
import scipy.io
import math
import time
from bm3d import bm3d,BM3DStages
from matplotlib import pyplot as plt
from FresnelTransform.FresnelTransform import FresnelTransform
import os

save_dir = "reconstruction_frames"
os.makedirs(save_dir, exist_ok=True)
'''
加载数据集函数，image=0读取TUT,1读取Lena
'''
def load(image=0): #0读取TUT,1读取Lena
    if image==0:
        data = scipy.io.loadmat('y_TUT_small.mat')
        y = data['y_TUT_small']
    else:
        data = scipy.io.loadmat('y_lena_small.mat')
        y = data['y_lena_small']
    return y
#y_abs = 1-y_true+0.1
#y_abs = 1-load(1)+0.1 #读取TUT
#y_abs = load(0)
y_abs = load(1)/210
#真实振幅物体 true amplitude object
u0 = y_abs


yN,xN = u0.shape #图像尺寸

#迭代次数
IterNumber = 100

#光学参数设置

d_factor = 3 #传播距离因子
mu = 2 #压缩比,mu越大，说明测量的越少，对图像压缩越强

#针对不同的图像需要调整的参数
threshold_ampl = 0.05 #BM3D阈值去噪参数
sigma = 0 #高斯噪声水平,为0表示只考虑泊松噪声
gamma_00 = 0.0002 #正则化强度

#BM3D-frame参数
Nstep = 1 #滑动步长
N11 = 4 #块大小
N22 = 8 #块大小
threshType = 'h' #阈值类型'h'或's'

#控制算法参数

filtering = 1 #是否使用BM3D去噪,1使用，0不使用
modulation_type = 1 #调制方式,1表示使用随机相位，0表示不使用

#4f系统参数
lam = 532e-6 #波长，单位mm
delta = 8e-3;  #像素间距，单位mm,SLM
d_f = delta*delta*yN/lam #4f系统的焦距
d = d_factor*d_f #传播距离
K = math.floor(yN*xN/mu) #实验次数，每次实验对应一个SLM相位掩膜
KAPPA = 10000; #泊松噪声强度

av = 1 #是否对传递函数进行平均,1表示平均，0表示不平均
Moduel_Up_Date = 1 #重新生成观测数据，0表示使用之前生成的观测数据，1表示重新生成

Gauss_var = 0 #1代表使用随迭代更新的变方差，0代表使用固定的变方差
IdealWeights=1        # 理想权重 (需要真值), 仅 Gauss_var=1 时有效
IdealWeights=0        # 自适应权重 (用估计值近似), 仅 Gauss_var=1 时有效
ss00=math.floor(IterNumber/4); # 在第 ss00 次迭代时切换到变方差算法

cache_path = ("AB.npz")

def Speckle_Generator():
    global gamma_00
    frame = []
    N = yN*xN
    B = np.empty((N, K), dtype=np.float64)
    zz = np.empty(K, dtype=np.float64)
    z_noiseless = np.empty(K, dtype=np.float64)
    if Moduel_Up_Date == 1: #重新生成观测数据
        print("Generation of Poissonian Observations")
        #与matlab外层rand对齐
        rng_mask_stream = np.random.default_rng(500001)

        for s_r in range(K):
            phase = rng_mask_stream.random((yN,xN))
            MASK = np.exp(1j*2.0*(np.pi-0.01)*phase) #这里相当于生成了在SLM处的光学空间调制掩膜

            u_r = FresnelTransform(MASK,d,lam,delta,av) #模拟通过Fresnel传播求SLM光学场到探测平面后的光学场




            b_r_col = (np.abs(u_r)**2).ravel(order='F') #列优先展开，获得光照强度向量

            # x_r = b_r ,*|u0(:)| ，matlab中冒号代表列优先展开
            x_r = b_r_col * np.abs(u0.ravel(order='F'))   #求出每个像素的光照贡献，光子数 b_r为SLM的光强，而u_0代表物体的光强

            #在求得o_r = b_r*c之后，可以认为针对每一次的o_r，桶探测器的观测值z_r服从泊松分布，均值为KAPPA*sum(o_r)
            rng_uniform_iter = np.random.default_rng(s_r + 100001)
            rng_normal_iter = np.random.default_rng(s_r + 1001)

            mean_bucket = KAPPA*float(np.sum(np.abs(x_r))) #计算每次实验的泊松分布输入，这个输入等于均值

            zz[s_r] = rng_uniform_iter.poisson(mean_bucket)/KAPPA #真实情况下，桶探测器的观测值服从泊松分布
            # 无噪声桶值，理想情况下的观测值
            z_noiseless[s_r] = float(np.sum(np.abs(x_r)))

            # 追加列
            B[:, s_r ] = b_r_col  #把每次实验的光强向量存入B矩阵的对应列中，组合成一个大的测量矩阵B,代表每次实验的光强分布
        ssigma_2 = np.mean(zz)/KAPPA #总体方差，实验次数足够大时，由于每次方差几乎不变，可以用总体方差取代样本方差
        gamma_0 = gamma_00/ssigma_2
        zz = zz.reshape(-1,1) #转为列向量

        BtB  = B.T@B
        A = BtB +(1.0/gamma_0) *np.eye(K)   #其实就是针对最优化方程(B.T*B/σ²+σ²/γ0*I)x = (1/σ*B.T*B*o+1/γ0*B.T*φθ)的左半部分
        FHI = np.linalg.pinv(A)   #求逆，为求解最优化方程做准备

        if FHI.ndim == 0:  # 标量的情况
            FHI = np.array([[FHI]])

        delta_x_zz = FHI@(BtB@zz)  #求解最优化方程的右半部分，初始化未进行稀疏操作，所以第二部分为0
        # save AB B zz ssigma_2 z_noiseless
        np.savez_compressed(cache_path, B=B, zz=zz, ssigma_2=ssigma_2, z_noiseless=z_noiseless)
    else:
        print("Using Old Poissonian Observations")
        if not cache_path.exists():
            raise FileNotFoundError("未找到缓存 AB_cache.npz，请先将 Model_Up_Date 设为 1 生成。")

        data = np.load(cache_path, allow_pickle=False)
        B_all = data["B"]
        zz_all = data["zz"]
        z_noiseless_all = data["z_noiseless"]
        ssigma_2 = float(data["ssigma_2"])

        gamma_0 = float(gamma_00 / ssigma_2)

        # B = B(:,1:K); zz = zz(1:K); z_noiseless = z_noiseless(1:K);
        B = B_all[:, :K]
        zz = zz_all[:K].reshape(-1, 1)
        z_noiseless = z_noiseless_all[:K]

        BtB = B.T @ B
        A = BtB + (1.0 / gamma_0) * np.eye(K)
        FHI = np.linalg.pinv(A)
        if FHI.ndim == 0:  # 标量的情况
            FHI = np.array([[FHI]])
        delta_x_zz = FHI @ (BtB @ zz)
    return B, zz, z_noiseless, ssigma_2, gamma_0, delta_x_zz,FHI

def csgi_demo_port(B,z_noiseless,ssigma_2,gamma_0,zz,delta_x_zz,FHI):
    t = time.perf_counter() #开始计时
    mean_B = B.T.mean(axis = 0) #用于sigma**2的更新

    ssigma_22 = np.zeros(IterNumber,dtype=np.float64)
    rmse_abs = np.zeros(IterNumber,dtype=np.float64)
    PSNR_hist = np.zeros(IterNumber,dtype=np.float64)
    #高斯估计算法
    v0 = np.ones((yN,xN),dtype = np.float64)/2.0 #初始振幅估计
    u0_est = np.copy(v0) #复制v0的矩阵模型
    D = np.diag((1.0 / z_noiseless + 1e-20) * KAPPA) * ssigma_2 #初始化D_diag
    for ss in range(IterNumber):
        if(Gauss_var == 1) and (ss>=ss00):
            if(ss==ss00) and (IdealWeights==1):
                print("理想权重")
                D = np.diag((1.0/z_noiseless+1e-20)*KAPPA)*ssigma_2
                FHI = np.linalg.pinv(B.T@B@D +np.eye(K)/gamma_0)
                delta_x_zz = FHI@(B.T@B)@D@zz
            if(ss>=ss00) and(IdealWeights==0):
                for ss1 in range(K):
                    base_vec = np.abs(u0_est).reshape(-1, order='F')  # 与 MATLAB 冒号展平一致（列优先）
                    z_noiseless[ss1] = np.sum(B[:, ss1] * base_vec)
                D = np.diag(1.0/(z_noiseless+1e-20))*KAPPA*ssigma_2
                FHI = np.linalg.pinv(B.T@B@D +np.eye(K)/gamma_0)
                delta_x_zz = FHI@(B.T@B)@D@zz
            x = delta_x_zz +FHI@B.T@v0.reshape(-1,1,order='F')/gamma_0
            u0_est = v0.reshape(-1,1,order='F')+B@D@(zz-x)*gamma_0

        else:  #当方差固定（通常方差都是固定的不需要更新方差）的时候，采取这种算法进行图像的重建
            # 固定方差更新图像
            # 计算最优化方程的解,求解方差(1/σ²*B.T*B+1/γ0*I)x = (1/σ²*B.T*B*o+1/γ0*B.T*φθ)，其中φθ=v0
            x = delta_x_zz +( FHI @(B.T @ v0.reshape(-1, 1, order='F'))) / gamma_0
            #完成图像求解后，对ct进行更新，公式为c_t = vt +γ0/σ²*B*(o - x) ,其中B为测量矩阵，o=bt*c
            u0_est = v0.reshape(-1,1,order='F') + (B @ (zz - x)) * gamma_0
            #求解均值B
            mean_B = B.mean(axis=1) #用于sigma**2的更新
            #更新方差
            ssigma_22[ss] = mean_B @ (np.abs(u0_est))/KAPPA
            #更新方差后，更新gamma_0=γ/σ²
            gamma_0 = gamma_00/ssigma_2
        #图像复原为二维
        u0_est = u0_est.reshape(yN,xN,order='F') #形状复原
        #去除负值为0
        u0_est = np.abs(u0_est) *(np.real(u0_est)>=0.0) #强制非负
        #计算图像误差
        ee_abs = u0_est - np.abs(u0)
        #求RMSE
        rmse_abs[ss] = np.sqrt(np.mean(ee_abs*ee_abs))

        #BM3D更新稀疏域




        if(ss>=1) and(ss<ss00):
            kk=10
            u0_est_filt = np.pad(u0_est,((kk,kk),(kk,kk)),'edge')
            v0tmp1 = bm3d(u0_est_filt,sigma_psd=threshold_ampl,stage_arg=BM3DStages.ALL_STAGES)
            v0 = v0tmp1[kk:-kk,kk:-kk] #去掉边框
        else:
            kk = 10
            u0_est_filt = np.pad(u0_est, ((kk, kk), (kk, kk)), 'edge')
            v0tmp1 = bm3d(u0_est_filt, sigma_psd= threshold_ampl, stage_arg=BM3DStages.ALL_STAGES)
            v0 = v0tmp1[kk:-kk, kk:-kk]  #去掉边框
    #可视化
        # ---- 计算 PSNR ----
        ymax = np.max(np.abs(y_abs))
        PSNR_val = 20 * np.log10(ymax / (rmse_abs[ss] + 1e-12))
        PSNR_hist[ss] = PSNR_val

        print(f"[PSNR, rmse] {ss} {PSNR_val:.3f} {rmse_abs[ss]:.3f}")

        # ---- 可视化 ----
        plt.figure(2, figsize=(9, 7))
        plt.clf()

        # 子图1：重建图像
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(u0_est), cmap='gray')
        plt.title(f"CSGI, RECONST, PSNR = {PSNR_val:.3f} dB")
        plt.axis('off')

        # 子图2：真实图像
        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(u0), cmap='gray')
        plt.title("TRUE IMAGE")
        plt.axis('off')

        # 子图3：横截面比较（第7行）
        plt.subplot(2, 2, 3)
        plt.plot(np.abs(u0[6, :]), 'r', label="TRUE")  # MATLAB 第7行 → Python 索引6
        plt.plot(np.abs(u0_est[6, :]), label="EST")
        plt.title("CROSS-SEC")
        plt.legend()

        # 子图4：PSNR 曲线
        plt.subplot(2, 2, 4)
        plt.plot(PSNR_hist[:ss + 1], lw=1.5)
        plt.title("PSNR, dB")
        plt.xlabel("ITER NUMBER")
        plt.grid(True)
        filename = os.path.join(save_dir, f"iter_{ss+1:03d}.png")
        plt.savefig(filename, dpi=200)
        plt.tight_layout()
        plt.pause(0.001)  # 实时更新图像



def main():
    # Step 1: 生成观测数据 (B, zz, …)
    B, zz, z_noiseless, ssigma_2, gamma_0, delta_x_zz ,FHI= Speckle_Generator()

    # Step 2: 运行 CSGI 重建
    csgi_demo_port(B, z_noiseless, ssigma_2, gamma_0, zz, delta_x_zz, FHI)

    print("重建完成 ✅")

if __name__ == "__main__":
    main()














