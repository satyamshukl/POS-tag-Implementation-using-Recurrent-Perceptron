import numpy as np


W = np.load('Weights.npy')

print("\033[1;36;40m1. ^ --> NN         ==>             V_^ + W_NN > \u03F4   |   " + "\033[1;32;40mTrue" if (W[1] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m2. ^ --> DT         ==>             V_^ + W_DT > \u03F4   |   " + "\033[1;32;40mTrue" if (W[1] + W[7] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m3. ^ --> JJ         ==>             V_^ + W_JJ > \u03F4   |   " + "\033[1;32;40mTrue" if (W[1] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m4. ^ --> OT         ==>             V_^ + W_OT > \u03F4   |   " + "\033[1;32;40mTrue" if (W[1] + W[9] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m5. DT --> NN        ==>        W + V_DT + W_NN > \u03F4   |  ", "\033[1;32;40mTrue" if (W[0] + W[3] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m6. DT --> JJ        ==>        W + V_DT + W_JJ > \u03F4   |   " + "\033[1;32;40mTrue" if (W[0] + W[3] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m7. JJ --> JJ        ==>            V_JJ + W_JJ > \u03F4   |   " + "\033[1;32;40mTrue" if (W[4] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m8. JJ --> JJ        ==>        W + V_JJ + W_JJ > \u03F4   |   " + "\033[1;32;40mTrue" if (W[0] + W[4] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m9. JJ --> NN        ==>            V_JJ + W_NN > \u03F4   |   " + "\033[1;32;40mTrue" if (W[4] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m10. JJ --> NN       ==>        W + V_JJ + W_NN > \u03F4   |   " + "\033[1;32;40mTrue" if (W[0] + W[4] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m11. NN --> OT       ==>            V_NN + W_OT > \u03F4   |   " + "\033[1;32;40mTrue" if (W[2] + W[9] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m12. NN --> OT       ==>        W + V_NN + W_OT > \u03F4   |   " + "\033[1;32;40mTrue" if (W[0] + W[2] + W[9] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m13. OT --> NN       ==>            V_OT + W_NN > \u03F4   |   " + "\033[1;32;40mTrue" if (W[5] + W[6] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m14. OT --> DT       ==>            V_OT + W_DT > \u03F4   |   " + "\033[1;32;40mTrue" if (W[5] + W[7] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m15. OT --> JJ       ==>            V_OT + W_JJ > \u03F4   |   " + "\033[1;32;40mTrue" if (W[5] + W[8] > W[10]) else "\033[1;31;40mFalse")
print("\033[1;36;40m16. OT --> OT       ==>            V_OT + W_OT > \u03F4   |   " + "\033[1;32;40mTrue" if (W[5] + W[9] > W[10]) else "\033[1;31;40mFalse")
