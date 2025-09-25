import numpy as np
from FresnelTransform.functionTransform import function_TransferFunction_FDDT
def FresnelTransform(y,f,lbd,delta,av):
    row,col = y.shape
    rol_NN = row *2
    col_NN = col *2

    AF,cond_A,a_ks = function_TransferFunction_FDDT(f,lbd,delta,rol_NN,col_NN,av)

    temp_double = np.zeros((rol_NN,col_NN),dtype=complex)
    #将原始图像放在中心位置
    ys, ye = rol_NN//4, rol_NN*3//4
    xs, xe = col_NN//4, col_NN*3//4
    temp_double[ys:ye,xs:xe] = y
    #频域相乘+反傅里叶变换
    Y_double = np.fft.ifft2(np.multiply(AF, np.fft.fft2(temp_double))) #AF先和fft2(temp_double)做numpy的乘法，特别注意这里是点乘而不是矩阵乘法

    #取中心区域
    y_out = Y_double[ys:ye,xs:xe]
    return y_out