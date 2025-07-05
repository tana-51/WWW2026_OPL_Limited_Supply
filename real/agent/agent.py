import numpy as np

class PreviousAgent():
    def set_regret(self,fixed_q_x_a):
        self.q_max = fixed_q_x_a.max(axis=0) #user間で最も高いq(x,a)の値
        
    def select_arm(self, user_idx, fixed_q_x_a, supply):
        #計算量を小さくするため
        #必要なのはuser_idxの行だけ
        #fixed_q_x_a_ = fixed_q_x_a.copy()
        #fixed_q_x_a_[:,supply<1] = -np.inf
        user_q = fixed_q_x_a[user_idx].copy()
        user_q[supply < 1] = -np.inf

        selected_arm = np.argmax(user_q)
        regret_value = self.q_max[selected_arm] - fixed_q_x_a[user_idx,selected_arm]
        if (supply>=1).sum()==0:
            regret_value = 0

        if ((supply>=1).sum()!=0) and (supply[selected_arm] < 1):
            raise ValueError(
                f"supply must be above 1, but got supply={supply[selected_arm]}" 
                f"about action {selected_arm}"
            )
        
        #return np.argmax(fixed_q_x_a_[user_idx]), regret_value
        return selected_arm, regret_value


class NewAgent():
    def set_regret(self,fixed_q_x_a, coef_=1.0):
        self.q_max = fixed_q_x_a.max(axis=0) #user間で最も高いq(x,a)の値
        self.regret = np.zeros_like(fixed_q_x_a)
        n_user = fixed_q_x_a.shape[0]
        #計算量を小さくするため
        #毎回同じfixed_q_x_a.sum(axis=0)を計算している
        #for i in range(n_user):
        #   self.regret[i]  = (coef_*fixed_q_x_a.sum(axis=0) - n_user*fixed_q_x_a[i,:]) / n_user
        q_x_a_sum = fixed_q_x_a.sum(axis=0)
        for i in range(n_user):
            self.regret[i]  = (coef_*q_x_a_sum - n_user*fixed_q_x_a[i,:]) / n_user
        
    def select_arm(self, user_idx, fixed_q_x_a, supply):
        regret = self.regret.copy()
        regret[:,supply<1] = 1000000
        
        #全体から最大のq_x_aを探しているので、
        #同じq_x_aがある場合min_regret_idxに複数のインデックスが入る、
        #在庫が無いaを選ぶ可能性がある。
        #min_regret_idx = np.where(regret[user_idx]==np.min(regret[user_idx]))
        #min_regret_idx = np.where(fixed_q_x_a[user_idx] == np.max(fixed_q_x_a[user_idx,min_regret_idx])
        min_regret_idx = np.where(regret[user_idx] == np.min(regret[user_idx]))[0]
        selected_arm = min_regret_idx[np.argmax(fixed_q_x_a[user_idx, min_regret_idx])]

        #regret_value = self.q_max[min_regret_idx[0][0]] - fixed_q_x_a[user_idx,min_regret_idx[0][0]]
        regret_value = self.q_max[selected_arm] - fixed_q_x_a[user_idx, selected_arm]
        if (supply>=1).sum()==0:
            regret_value = 0
            
        if ((supply>=1).sum()!=0) and (supply[selected_arm] < 1):
            raise ValueError(f"supply must be above 1, but got supply={supply[selected_arm]} about action {selected_arm}")
        return selected_arm, regret_value
    

class NewAgentStep():
    def set_regret(self,fixed_q_x_a, coef_=1.0):
        self.q_max = fixed_q_x_a.max(axis=0) #user間で最も高いq(x,a)の値
        self.regret = np.zeros_like(fixed_q_x_a)
        self.coef_ = coef_
        n_user = fixed_q_x_a.shape[0]
        #計算量を小さくするため
        #毎回同じfixed_q_x_a.sum(axis=0)を計算している
        #for i in range(n_user):
        #   self.regret[i]  = (coef_*fixed_q_x_a.sum(axis=0) - n_user*fixed_q_x_a[i,:]) / n_user
        q_x_a_sum = fixed_q_x_a.sum(axis=0)
        for i in range(n_user):
            self.regret[i]  = (self.coef_*q_x_a_sum - n_user*fixed_q_x_a[i,:]) / n_user
            
    def select_arm(self, user_idx, fixed_q_x_a, supply):
        
        #n_sold >0
        not_argmax = False
        if (supply*(np.ones(self.coef_.shape) - self.coef_)).sum()==0:
            not_argmax = True

        #計算量を小さくするため
        #必要なのはuser_idxの行だけ
        #fixed_q_x_a_ = fixed_q_x_a*(supply>=1)
        ##fixed_q_x_a_ *= np.ones(self.coef_.shape) - self.coef_
        #action_idx = np.argmax(fixed_q_x_a_[user_idx])
        user_q = fixed_q_x_a[user_idx].copy()
        user_q[supply < 1] = -np.inf
        user_q[self.coef_ == 1] = -np.inf#期待報酬が元々0だと不具合が発生するため
        action_idx = np.argmax(user_q)

        #n_sold <= 0
        not_regret = False
        if (supply*self.coef_).sum()==0:
            not_regret = True
        regret = self.regret.copy()
        regret[:,supply<1] = 1000000
        regret[:,self.coef_ == 0] = 1000000

        #全体から最大のq_x_aを探しているので、
        #同じq_x_aがある場合min_regret_idxに複数のインデックスが入る、
        #在庫が無いaを選ぶ可能性がある。
        #min_regret_idx = np.where(regret[user_idx]==np.min(regret[user_idx]))
        #min_regret_idx = np.where(fixed_q_x_a[user_idx] == np.max(fixed_q_x_a[user_idx,min_regret_idx]))
        min_regret_idx = np.where(regret[user_idx] == np.min(regret[user_idx]))[0]
        selected_action = min_regret_idx[np.argmax(fixed_q_x_a[user_idx, min_regret_idx])]

        #以降mini_regret_idx[0][0]　→　selected_action
        #min_regret_idxに複数のインデックスが入ると、エラーが発生する
        #if fixed_q_x_a[user_idx,min_regret_idx] >= fixed_q_x_a[user_idx,action_idx]:
        #    idx_ = min_regret_idx[0][0]
        #else:
        #    idx_ = action_idx
        if fixed_q_x_a[user_idx,selected_action] >= fixed_q_x_a[user_idx,action_idx]: 
            idx_ = selected_action
        else:
            idx_ = action_idx

        if self.coef_.sum() == len(self.coef_):
            idx_ = selected_action
        elif self.coef_.sum() == 0:
            idx_ = action_idx

        if not_regret == True:
            idx_ = action_idx
        if not_argmax == True:
            idx_ = selected_action

        regret_value = self.q_max[selected_action] - fixed_q_x_a[user_idx,selected_action]
        if (supply>=1).sum()==0:
            regret_value = 0
        
        if ((supply>=1).sum()!=0) and (supply[idx_] < 1):
            raise ValueError(f"supply must be above 1, but got supply={supply[idx_]} about action {idx_}")
        return idx_, regret_value
    
        