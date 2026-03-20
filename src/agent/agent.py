import numpy as np

class PreviousAgent():
    def set_regret(self,fixed_q_x_a):
        self.q_max = fixed_q_x_a.max(axis=0) 
        
    def select_arm(self, user_idx, fixed_q_x_a, supply):
        fixed_q_x_a_ = fixed_q_x_a.copy()
        fixed_q_x_a_[:,supply<1] = -np.inf
        regret_value = self.q_max[np.argmax(fixed_q_x_a_[user_idx])] - fixed_q_x_a[user_idx,np.argmax(fixed_q_x_a_[user_idx])]
        if (supply>=1).sum()==0:
            regret_value = 0
        
        if ((supply>=1).sum()!=0) and (supply[np.argmax(fixed_q_x_a_[user_idx])] < 1):
            raise ValueError(f"supply must be above 1, but got supply={supply[np.argmax(fixed_q_x_a_[user_idx])]} about action {np.argmax(fixed_q_x_a_[user_idx])}")
        

        return np.argmax(fixed_q_x_a_[user_idx]), regret_value


class NewAgent():
    def obtain_opls_value(self,fixed_q_x_a, user_idx, coef_=1.0):
        self.q_max = fixed_q_x_a.max(axis=0)
        self.opls_value = np.zeros_like(fixed_q_x_a)
        n_user = fixed_q_x_a.shape[0]
        for i in range(n_user):
            self.opls_value[i]  = fixed_q_x_a[i,:] - (coef_*fixed_q_x_a[user_idx].sum(axis=0) / user_idx.shape[0])

        
    def select_arm(self, user_idx, fixed_q_x_a, supply):

        opls_value = self.opls_value.copy()
        opls_value[:,supply<1] = -np.inf
        max_opls_value_idx = np.where(opls_value[user_idx]==np.max(opls_value[user_idx]))
        max_opls_value_idx = max_opls_value_idx[0][np.argmax(fixed_q_x_a[user_idx,max_opls_value_idx])]
        
        regret_value = self.q_max[max_opls_value_idx] - fixed_q_x_a[user_idx,max_opls_value_idx]
        
        if (supply>=1).sum()==0:
            regret_value = 0
        
        if ((supply>=1).sum()!=0) and (supply[max_opls_value_idx] < 1):
            raise ValueError(f"supply must be above 1, but got supply={supply[max_opls_value_idx]} about action {max_opls_value_idx}")
        
        return max_opls_value_idx, regret_value
    

class NewAgentStep():
    def obtain_opls_value(self,fixed_q_x_a, user_idx, coef_=1.0):
        self.q_max = fixed_q_x_a.max(axis=0) 
        self.opls_value = np.zeros_like(fixed_q_x_a)
        self.coef_ = coef_
        n_user = fixed_q_x_a.shape[0]
        for i in range(n_user):
            self.opls_value[i]  = fixed_q_x_a[i,:] - (coef_*fixed_q_x_a[user_idx].sum(axis=0) / user_idx.shape[0])


        
    def select_arm(self, user_idx, fixed_q_x_a, supply):
        
        #n_sold >0
        not_argmax = False
        if (supply*(np.ones(self.coef_.shape) - self.coef_)).sum()==0: 
            not_argmax = True
        fixed_q_x_a_ = fixed_q_x_a*(supply>=1)
        fixed_q_x_a_ *= np.ones(self.coef_.shape) - self.coef_ 
        action_idx = np.argmax(fixed_q_x_a_[user_idx])

        #n_sold <= 0
        not_regret = False
        if (supply*self.coef_).sum()==0:
            not_regret = True
        opls_value = self.opls_value.copy()
        opls_value[:,supply<1] = -np.inf
        opls_value[:,self.coef_ == 0] = -np.inf 

        max_opls_value_idx = np.where(opls_value[user_idx]==np.max(opls_value[user_idx]))
        max_opls_value_idx = max_opls_value_idx[0][np.argmax(fixed_q_x_a[user_idx,max_opls_value_idx])]

        if fixed_q_x_a[user_idx,max_opls_value_idx] >= fixed_q_x_a[user_idx,action_idx]: 
            idx_ = max_opls_value_idx
        else:
            idx_ = action_idx
        

        if self.coef_.sum() == len(self.coef_): 
            idx_ = max_opls_value_idx
        elif self.coef_.sum() == 0: 
            idx_ = action_idx

        if not_regret == True:
            idx_ = action_idx
        if not_argmax == True:
            idx_ = max_opls_value_idx

        regret_value = self.q_max[max_opls_value_idx] - fixed_q_x_a[user_idx, max_opls_value_idx]
        if (supply>=1).sum()==0:
            regret_value = 0
        
        if ((supply>=1).sum()!=0) and (supply[idx_] < 1):
            raise ValueError(f"supply must be above 1, but got supply={supply[idx_]} about action {idx_}")
        return idx_, regret_value
    
        
