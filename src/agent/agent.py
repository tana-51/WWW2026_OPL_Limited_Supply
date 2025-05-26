import numpy as np

class PreviousAgent():
    def set_regret(self,fixed_q_x_a):
        self.q_max = fixed_q_x_a.max(axis=0) #user間で最も高いq(x,a)の値
        
    def select_arm(self, user_idx, fixed_q_x_a, supply):
        fixed_q_x_a_ = fixed_q_x_a*(supply>=1)
        regret_value = self.q_max[np.argmax(fixed_q_x_a_[user_idx])] - fixed_q_x_a[user_idx,np.argmax(fixed_q_x_a_[user_idx])]
        if (supply>=1).sum()==0:
            regret_value = 0
        return np.argmax(fixed_q_x_a_[user_idx]), regret_value


class NewAgent():
    def set_regret(self,fixed_q_x_a):
        self.q_max = fixed_q_x_a.max(axis=0) #user間で最も高いq(x,a)の値
        self.regret = np.zeros_like(fixed_q_x_a)
        n_user = fixed_q_x_a.shape[0]
        for i in range(n_user):
            self.regret[i]  = (fixed_q_x_a.sum(axis=0) - n_user*fixed_q_x_a[i,:]) / n_user
        
    def select_arm(self, user_idx, fixed_q_x_a, supply,):

        regret = self.regret.copy()
        regret[:,supply<1] = 1000000
        #regret /= supply
        min_regret_idx = np.where(regret[user_idx]==np.min(regret[user_idx]))
        min_regret_idx = np.where(fixed_q_x_a[user_idx] == np.max(fixed_q_x_a[user_idx,min_regret_idx]))
        
        regret_value = self.q_max[min_regret_idx[0][0]] - fixed_q_x_a[user_idx,min_regret_idx[0][0]]
        if (supply>=1).sum()==0:
            regret_value = 0
        return min_regret_idx[0][0], regret_value
    
        