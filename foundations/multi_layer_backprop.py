import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x); W1 = np.array(W1); b1 = np.array(b1)
        W2 = np.array(W2); b2 = np.array(b2); y_true = np.array(y_true)
        z1 = np.dot(x[None,:], W1.T) + b1
        a1 = np.maximum(0, z1)
        y_pred = np.dot(a1, W2.T) + b2
        loss = np.mean((y_true[None,:] - y_pred)**2)
        db2 = 2 * np.mean(y_pred - y_true[None,:]).reshape(1) 
        dW2 = np.dot(db2.T, a1)
        dW2 = dW2[None,:]
        A = np.where(np.arange(a1.shape[1])[:,None] == np.arange(a1.shape[1])[None,:], a1, 0)
        da1 = np.where(A > 0, 1, 0)
        db1 = (db2 @ W2) @ da1
        dW1 = db1[:,None] @ x[None, :]
        return {
  'loss': np.round(loss, 4),
  'dW1': np.round(dW1, 4),
  'db1': np.round(db1, 4),
  'dW2': np.round(dW2, 4),
  'db2': np.round(db2, 4)
}
