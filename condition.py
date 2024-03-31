import numpy as np


# W = np.load('Weights.npy')
W = np.array([[0.06895157],
 [1.12824395],
 [0.45450368],
 [0.15664798],
 [0.31429987],
 [0.69936847],
 [0.15362929],
 [0.44689867],
 [0.22104965],
 [0.26413506],
 [0.49712984]])
W = W.reshape(11,)
print(W)
print("\033[1;36;40m 1.  ^ --> NN       ==>             V_^ + W_NN > \u03F4   |  ", "\033[1;32;40mTrue" if (W[1] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 2.  ^ --> DT       ==>             V_^ + W_DT > \u03F4   |  ", "\033[1;32;40mTrue" if (W[1] + W[7] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 3.  ^ --> JJ       ==>             V_^ + W_JJ > \u03F4   |  ", "\033[1;32;40mTrue" if (W[1] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 4.  ^ --> OT       ==>             V_^ + W_OT > \u03F4   |  ", "\033[1;32;40mTrue" if (W[1] + W[9] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 5. DT --> NN       ==>        W + V_DT + W_NN < \u03F4   |  ", "\033[1;32;40mTrue" if (W[0] + W[3] + W[6] < W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 6. DT --> JJ       ==>        W + V_DT + W_JJ < \u03F4   |  ", "\033[1;32;40mTrue" if (W[0] + W[3] + W[8] < W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 7. JJ --> JJ       ==>            V_JJ + W_JJ < \u03F4   |  ", "\033[1;32;40mTrue" if (W[4] + W[8] < W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 8. JJ --> JJ       ==>        W + V_JJ + W_JJ < \u03F4   |  ", "\033[1;32;40mTrue" if (W[0] + W[4] + W[8] < W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m 9. JJ --> NN       ==>            V_JJ + W_NN < \u03F4   |  ", "\033[1;32;40mTrue" if (W[4] + W[6] < W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m10. JJ --> NN       ==>        W + V_JJ + W_NN < \u03F4   |  ", "\033[1;32;40mTrue" if (W[0] + W[4] + W[6] < W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m11. NN --> OT       ==>            V_NN + W_OT > \u03F4   |  ", "\033[1;32;40mTrue" if (W[2] + W[9] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m12. NN --> OT       ==>        W + V_NN + W_OT > \u03F4   |  ", "\033[1;32;40mTrue" if (W[0] + W[2] + W[9] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m13. OT --> NN       ==>            V_OT + W_NN > \u03F4   |  ", "\033[1;32;40mTrue" if (W[5] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m14. OT --> DT       ==>            V_OT + W_DT > \u03F4   |  ", "\033[1;32;40mTrue" if (W[5] + W[7] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m15. OT --> JJ       ==>            V_OT + W_JJ > \u03F4   |  ", "\033[1;32;40mTrue" if (W[5] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m16. OT --> OT       ==>            V_OT + W_OT > \u03F4   |  ", "\033[1;32;40mTrue" if (W[5] + W[9] > W[10]) else "\033[1;31;40mFalse")
