# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:25:22 2023

@author: MarqueLiu
"""
import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import minimize_scalar
from scipy.stats import logistic
from scipy.stats import rayleigh
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import beta
import math
import pandas as pd

### Weibull 抽样
def Weibull_Sample(mean_value,p5_value,p95_value,num):
    # 计算 Weibull 分布的形状参数和尺度参数
    shape_param, scale_param = weibull_params(mean_value, p5_value, p95_value)
    samples = sample_weibull(shape_param, scale_param, size=num)
    return samples

def weibull_params(mean, p5, p95):
    # 目标函数
    def objective_function(c):
        scale = mean / math.gamma(1 + 1/c)
        return (weibull_min.ppf(0.05, c, scale=scale) - p5)**2 + (weibull_min.ppf(0.95, c, scale=scale) - p95)**2
    # 初始猜测值
    initial_guess = 2.0
    # 最小化目标函数
    result = minimize_scalar(objective_function, bounds=(0.01, 10.0), method='bounded')
    # 提取结果中的形状参数
    shape_param = result.x
    # 计算尺度参数
    scale_param = mean / math.gamma(1 + 1/shape_param)
    return shape_param, scale_param

def sample_weibull(shape, scale, size=1):
    return weibull_min.rvs(shape, scale=scale, size=size)

### Beta抽样
def beta_sample(mean_value,p5_value,p95_value,num):
    alpha_param, beta_param = beta_pert_params(mean_value, p5_value, p95_value)
    samples = beta.rvs(alpha_param, beta_param, size=num)
    return samples

def beta_pert_params(mean, p5, p95):
    # 目标函数
    def objective_function(params):
        alpha, be = params
        return (beta.ppf(0.05, alpha, be) - p5)**2 + (beta.ppf(0.95, alpha, be) - p95)**2 + (alpha / (alpha + be) - mean)**2
    # 初始猜测值
    initial_guess = [2, 2]
    # 最小化目标函数
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    # 提取结果中的参数值
    alpha, be = result.x
    return alpha, be

### logistic 抽样
def logistic_sample(mean_value,p5_value,p95_value,num):
    # 计算 Logistic 分布的尺度参数
    scale_param = logistic_params(mean_value, p5_value, p95_value)
    samples=logistic.rvs(loc=mean_value, scale=scale_param, size=num)
    return samples

def logistic_params(mean, p5, p95):
    def objective_function(scale):
        return (logistic.ppf(0.05, loc=mean, scale=scale) - p5)**2 + (logistic.ppf(0.95, loc=mean, scale=scale) - p95)**2

    result = minimize_scalar(objective_function, bounds=(0.01, 10.0), method='bounded')
    scale_parameter = result.x
    
    return scale_parameter

### Rayleigh 分布
def rayleigh_sample(mean_value,p5_value,p95_value,num):
    scale_param = rayleigh_params(mean_value, p5_value, p95_value)
    samples=rayleigh.rvs(scale=scale_param, size=num)
    return samples

def rayleigh_params(mean, p5, p95):
    def objective_function(scale):
        return (rayleigh.ppf(0.05, scale=scale) - p5)**2 + (rayleigh.ppf(0.95, scale=scale) - p95)**2

    result = minimize_scalar(objective_function, bounds=(0.01, 10.0), method='bounded')
    scale_parameter = result.x
    
    return scale_parameter

### Gamma 分布
def gamma_sample(mean_value,p5_value,p95_value,num):
    shape_param, scale_param = gamma_params(mean_value, p5_value, p95_value)
    samples=gamma.rvs(a=shape_param, scale=scale_param, size=num)
    return samples

def gamma_params(mean, p5, p95):
    def objective_function(shape):
        scale = mean / shape
        return (gamma.ppf(0.05, a=shape, scale=scale) - p5)**2 + (gamma.ppf(0.95, a=shape, scale=scale) - p95)**2

    result = minimize_scalar(objective_function, bounds=(0.01, 10.0), method='bounded')
    shape_parameter = result.x
    scale_parameter = mean / shape_parameter
    
    return shape_parameter, scale_parameter

### Exponential 分布
def exponential_sample(mean_value,p5_value,p95_value,num):
    scale_param = exponential_params(mean_value, p5_value, p95_value)
    samples=np.random.exponential(scale=scale_param, size=num)
    return samples
    
def exponential_params(mean, p5, p95):
    scale_parameter = 1 / mean
    return scale_parameter

### Chi-squared分布
def Chi_sample(mean_value,p5_value,p95_value,num):
    samples=np.random.chisquare(mean_value, size=num)
    return samples

### lognormal 分布
def lognormal_sample(mean_value,p5_value,p95_value,num):
    mu_param, sigma_param = lognormal_params(mean_value, p5_value, p95_value)
    samples=np.random.lognormal(mean=mu_param, sigma=sigma_param, size=num)
    return samples

def lognormal_params(log_mean, p5, p95):
    # 计算基本正态分布的均值（μ_0）
    normal_mean = np.log(log_mean) - 0.5 * ((np.log(p95) - np.log(p5)) / (2 * 1.645))**2
    # 计算基本正态分布的标准差（σ_0）
    normal_std = (np.log(p95) - np.log(p5)) / (2 * 1.645)
    return normal_mean, normal_std

### normal 分布
def normal_sample(mean_value,p5_value,p95_value,num):
    mean_param, std_param = normal_params(mean_value, p5_value, p95_value)
    samples=np.random.normal(loc=mean_param, scale=std_param, size=num)
    return samples

def normal_params(mean, p5, p95):
    # 标准差的估计
    std_estimate = (p95 - p5) / (2 * norm.ppf(0.975))

    return mean, std_estimate

### triangle分布
def triangle_sample(mean_value,p5_value,p95_value,num):
    a_param, c_param, b_param = triangle_params(mean_value, p5_value, p95_value)
    samples=np.random.triangular(left=a_param, mode=c_param, right=b_param, size=num)
    return samples

def triangle_params(mean, p5, p95):
    # 均值的估计
    mu_estimate = mean
    # 峰值的估计
    c_estimate = 3 * mu_estimate - p5 - p95
    return p5, c_estimate, p95

###  inverse normal 分布
def inverse_normal_sample(mean, p5, p95, size=1):
    # 计算标准差
    std_dev = (mean - norm.ppf(0.05, loc=mean, scale=1)) / norm.ppf(0.95, loc=mean, scale=1)

    # 进行逆高斯分布的抽样
    samples = norm.ppf(np.random.uniform(norm.cdf(p5, loc=mean, scale=std_dev),
                                          norm.cdf(p95, loc=mean, scale=std_dev), size), loc=mean, scale=std_dev)
    return samples


def all_sample(df,idx,num):
    distribution=df.loc[idx,'distribution']
    mean=df.loc[idx,'mean'];low=df.loc[idx,'low'];up=df.loc[idx,'up']
    samples=0
    if distribution=='Triangle':
        samples=triangle_sample(mean,low,up,num)
    elif distribution=='Uniform':
        samples=np.random.uniform(low,up,num)
    elif distribution=='Normal':
        samples=normal_sample(mean,low,up,num)
    elif distribution=='Lognormal':
        samples=lognormal_sample(mean,low,up,num)
    elif distribution=='Inverse Normal':
        samples=inverse_normal_sample(mean,low,up,num)
    elif distribution=='Weibull':
        samples=Weibull_Sample(mean,low,up,num)
    elif distribution=='Beta-PERT':
        samples=beta_sample(mean,low,up,num)
    elif distribution=='Logistic':
        samples=logistic_sample(mean,low,up,num)
    elif distribution=='Rayleigh':
        samples=rayleigh_sample(mean,low,up,num)
    elif distribution=='Gamma':
        samples=gamma_sample(mean,low,up,num)
    elif distribution=='Exponential':
        samples=exponential_sample(mean,low,up,num)
    elif distribution=='Chi-squared':
        samples=Chi_sample(mean,low,up,num)
    
    samples=np.where(samples < 0, 0, samples)
    
    return samples
    

### test
if __name__=='__main__':
    num=5000
    inds=[]##use the name of uncertainty sources as index
    df=pd.DataFrame(index=inds,columns=list(range(0,5000,1)))##set your table framework here，and input the distraibution function and range (which we listed in the SI file)
    for idx in inds:
        distribution=df.loc[idx,'distribution']
        mean=df.loc[idx,'mean'];low=df.loc[idx,'low'];up=df.loc[idx,'up']
        samples=0
        if distribution=='Triangle':
            samples=triangle_sample(mean,low,up,num)
        elif distribution=='Uniform':
            samples=np.random.uniform(low,up,num)
        elif distribution=='Normal':
            samples=normal_sample(mean,low,up,num)
        elif distribution=='Lognormal':
            samples=lognormal_sample(mean,low,up,num)
        elif distribution=='Inverse Normal':
            samples=inverse_normal_sample(mean,low,up,num)
        elif distribution=='Weibull':
            samples=Weibull_Sample(mean,low,up,num)
        elif distribution=='Beta-PERT':
            samples=beta_sample(mean,low,up,num)
        elif distribution=='Logistic':
            samples=logistic_sample(mean,low,up,num)
        elif distribution=='Rayleigh':
            samples=rayleigh_sample(mean,low,up,num)
        elif distribution=='Gamma':
            samples=gamma_sample(mean,low,up,num)
        elif distribution=='Exponential':
            samples=exponential_sample(mean,low,up,num)
        elif distribution=='Chi-squared':
            samples=Chi_sample(mean,low,up,num)
        
        samples=np.where(samples < 0, 0, samples)
        df.loc[idx,4:5005]=samples
    resultPath=''#set your result path here
    df.to_excel(resultPath,sheet_name='sheet1')
