import torch

class ExtendedKalmanFilter:
    def __init__(self, shape, device, q_scale=0.01, r_scale=0.05):
        self.device = device
        self.shape = shape
        
        # P, Q, R 保持不变
        self.P = torch.full(shape, 0.1, device=device) 
        self.Q = torch.full(shape, q_scale**2, device=device)
        self.R = torch.full(shape, r_scale**2, device=device)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(42)

    def predict(self, flow_func, dvdx_func, x_curr, zt_tar, t_curr, t_prev):
        """
        EKF 预测步
        """
        dt = t_prev - t_curr

        v_tar, v_src = flow_func(x_curr, zt_tar, t_curr)
        v_curr = v_tar - v_src
        x_pred = x_curr + dt * v_curr
        
        dvdx = dvdx_func(zt_tar, v_tar, t_curr, t_prev)
        dvdx = dvdx.to(torch.float32) * dt
        dvdx = torch.clamp(dvdx, -10.0, 10.0)
        F_diag = 1.0 + dt * dvdx
        
        # P_pred = F * P * F^T + Q
        self.P = (F_diag ** 2) * self.P + self.Q
        
        return x_pred, v_curr

    def update(self, x_pred, z_obs):
        """
        EKF 更新步
        """
        if z_obs is None:
            return x_pred
        R = self.R
        # K = P / (P + R), H是单位矩阵
        K = self.P / (self.P + R + 1e-6)
        # x_new = x_pred + K * (z_obs - x_pred)
        innovation = z_obs - x_pred
        x_fused = x_pred + K * innovation
    
        # P_new = (1 - K) * P
        self.P = (1.0 - K) * self.P

        return x_fused

    def step(self, flow_func, dydx_func, x_curr, zt_tar, t_curr, t_prev, z_obs=None):
        """
        单步完整流程
        """
        x_pred, v_curr = self.predict(flow_func, dydx_func, x_curr, zt_tar, t_curr, t_prev)
        x_final = self.update(x_pred, z_obs)
        return x_final, v_curr
    
    def set_q_scale(self, q_scale):
        self.Q = torch.full(self.shape, q_scale**2, device=self.device)
        
    def set_r_scale(self, r_scale):
        self.R = torch.full(self.shape, r_scale**2, device=self.device)
