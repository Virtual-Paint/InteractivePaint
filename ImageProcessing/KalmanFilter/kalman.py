import numpy as np


class KalmanFilter:
    def __init__(self, 
                 dt: float, 
                 u_x: float, 
                 u_y: float, 
                 std_acc: float, 
                 x_std_meas: float, 
                 y_std_meas: float) -> None:
        self.dt = dt
        
        self.u = np.matrix([[u_x],
                            [u_y]])
        self.x = np.matrix([[0],
                            [0],
                            [0],
                            [0]])
        
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = np.matrix([[0.5*pow(self.dt, 2), 0],
                            [0, 0.5*pow(self.dt, 2)],
                            [self.dt, 0],
                            [0, self.dt]])
        self.H = np.matrix([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.Q = np.matrix([[pow(self.dt, 4) / 4, 0, pow(self.dt, 3) / 2, 0],
                            [0, pow(self.dt, 4) / 4, 0, pow(self.dt, 3) / 2],
                            [pow(self.dt, 3) / 2, 0, pow(self.dt, 2), 0],
                            [0, pow(self.dt, 3) / 2, 0, pow(self.dt, 2)]]) * pow(std_acc, 2)
        self.R = np.matrix([[pow(x_std_meas, 2), 0],
                            [0, pow(y_std_meas, 2)]])
        self.P = np.eye(self.A.shape[1])
        
    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        K = np. dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        
        I = np.eye(self.H.shape[1])
        
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]
    