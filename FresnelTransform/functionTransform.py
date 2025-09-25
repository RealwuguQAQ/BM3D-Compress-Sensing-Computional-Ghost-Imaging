import math
import numpy as np

def kernel(x,t,d,delta):
    return (1-abs(x))*np.exp(t*delta**2*(d+x)**2)

def function_TransferFunction_FDDT(z,lam,delta,N,M,av):

    eps = 1e-20 # 避免除零错误
    k = 2*np.pi/lam
    f = float(z)
    t = 1j*k/(2.0*z+eps) #计算传递函数的系数

    #只由z决定的整体相位
    kf = -1j*np.exp(1j*2*np.pi*f/lam)/(z+eps)/lam*(delta*delta)

    #栅格索引
    xint = np.arange(-M//2,M//2,dtype = int)
    yint = np.arange(-N//2,N//2,dtype = int)
    Xint,Yint = np.meshgrid(xint,yint)
    a_ks = np.zeros((N,M),dtype = np.complex128)
    fx = np.zeros((len(xint)),dtype = np.complex128)
    fy = np.zeros((len(yint)),dtype = np.complex128)
    #计算积分
    def avg_kernel(d):
        xs = np.linspace(-1, 1, 401)
        vals = (1 - np.abs(xs)) * np.exp(t * delta**2 * (d + xs)**2)
        return np.trapz(vals, xs)

    for q,d in enumerate(xint):
        if av:
            fx[q] = avg_kernel(d)
        else:
            fx[q] = np.exp(t * delta**2 * (d**2))
    if N==M:
        smoothed = fx[Xint + M//2] * fx[Yint + N//2]
    else: #单独计算Y
        for q,d in enumerate(yint):
            if av:
                fy[q] = avg_kernel(d)
            else:
                fy[q] = np.exp(t * delta**2 * (d**2))
        smoothed = fx[Xint + M//2] * fy[Yint + N//2]
    y1 = kf*smoothed

    a_ks[1:,1:] = y1[1:,1:]

    #FFT，先circlshift，再做FFT
    a_shift = np.roll(np.roll(a_ks,-N//2,axis=0),-M//2,axis=1)
    AF = np.fft.fft2(a_shift)

    #条件数估计
    absA = np.abs(AF)
    min_nonzero = absA[absA>0].min() if np.any(absA>0) else 1.0
    cond_A = absA.max()/min_nonzero

    return AF, cond_A, a_ks

