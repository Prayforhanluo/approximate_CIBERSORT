# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:28:14 2019

@author: LuoHan
"""


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import NuSVR


##-----------------Loading Function----------------
def Get_SC_Meta_Info(fil):
    """
    """
    return pd.read_table(fil, header = 0)

def Get_SC_Count_Info(fil):
    """
    """
    return pd.read_table(fil, header = 0, index_col = 0)


def Get_bulk_Count_Info(fil):
    """
    """
    return pd.read_table(fil, header = 0, index_col = 0)


class QuantileNormalization:
    """
        QuantileNormalization in python way as what 
        preprocessCore.normalize.quantile do in R.
    """
    
    def __init__(self, X):
        """
            Initial Matrix(2D)
        """
        self.matrix = np.array(X, dtype = np.float64)
        assert len(self.matrix.shape) == 2
        
        self.rows, self.cols = self.matrix.shape
        self.index_matrix = np.argsort(self.matrix, axis = 0)
        self.quantilenormalize()
        
    def rowmeans(self, sorted_mat):
        """
        Given a dim=2 array, calculate its row means.
        As for Na, a float will be calculated by linear interpolation
        :param sorted_arr: sorted array of original arr. Na placed last on each column
        :param rows: rows of data, not counting header
        :param cols: rows of data, not counting header 
        :return: A new calloced row_means with size(rows) vector
        
        """
        eps = np.finfo(float).eps
        non_nas = self.rows - np.sum(np.isnan(sorted_mat), axis=0)
        row_means = np.zeros(self.rows, dtype=np.float64)

        for j in range(self.cols):
            non_na = non_nas[j]
            # if this column all is NA, skip. otherwise, row_means=nan, error
            if non_na == 0:
                continue
        
            if non_na == self.rows:
                for i in range(self.rows):
                    row_means[i] += sorted_mat[i, j]

            else:
                for i in range(self.rows):
                    sample_percent = float(i) / float(self.rows - 1)
                    index_f = 1.0 + (non_na - 1.0) * sample_percent
                    index_f_floor = np.floor(index_f + 4 * eps)
                    index_f = index_f - index_f_floor

                    if np.fabs(index_f <= 4 * eps):
                        index_f = 0.0

                    if index_f == 0.0:
                        row_mean_id = int(np.floor(index_f_floor + 0.5))
                        row_means[i] += sorted_mat[row_mean_id - 1, j]
                    elif index_f == 1.0:
                        row_mean_id = int(np.floor(index_f_floor + 1.5))
                        row_means[i] += sorted_mat[row_mean_id - 1, j]
                    else:
                        row_mean_id = int(np.floor(index_f_floor + 0.5))

                        if row_mean_id < self.rows and row_mean_id > 0:
                            row_means[i] += (1.0 - index_f) * sorted_mat[row_mean_id-1, j] + \
                                index_f * sorted_mat[row_mean_id, j]
                        elif row_mean_id >= self.rows:
                            row_means[i] += sorted_mat[non_na - 1, j]
                        else:
                            row_means[i] += sorted_mat[0, j]
        row_means /= float(self.cols)
        
        self.row_means = row_means
    
    def ranks(self, sorted_mat):
        """
        """
        ranks_matrix = np.zeros((self.rows, self.cols), dtype = np.float64)
        ranks_matrix[:, :] = np.NaN
        
        for c in range(self.cols):
            sorted_vec = sorted_mat[:, c]
            non_na = self.rows - np.sum(np.isnan(sorted_vec))
            
            i = 0
            while i < non_na:
                j = i
                while j < non_na - 1 and sorted_vec[j] == sorted_vec[j+1]:
                    j += 1
                if i != j:
                    for k in range(i, j+1):
                        ranks_matrix[k, c] = (i + j + 2.0) / 2.0
                else:
                    ranks_matrix[i, c] = i + 1
                
                i = j + 1
        
        self.rank_matrix = ranks_matrix
     
    def targeting(self):
        """
        """
        eps = np.finfo(float).eps
        non_nas = self.rows - np.sum(np.isnan(self.rank_matrix), axis = 0)
        normed = np.zeros((self.rows, self.cols), dtype = np.float64)
        normed[:, :] = np.NaN
        
        for j in range(self.cols):
            non_na = non_nas[j]
            if non_na == self.rows:
                for i in range(self.rows):
                    rank = self.rank_matrix[i, j]
                    ori_i = self.index_matrix[i, j]
                    if rank - np.floor(rank) > 0.4:
                        normed[ori_i, j] = 0.5 * self.row_means[int(np.floor(rank)) - 1] + \
                            0.5 * self.row_means[int(np.floor(rank))]
                    else:
                        normed[ori_i, j] = self.row_means[int(np.floor(rank)) - 1]
            else:
                for i in range(non_na):
                    ori_i = self.index_matrix[i, j]

                    sample_percent = (self.rank_matrix[i, j] - 1.0) / float(non_na - 1)
                    index_f = 1.0 + (self.rows - 1.0) * sample_percent
                    index_f_floor = np.floor(index_f + 4 * eps)
                    index_f -= index_f_floor

                    if np.fabs(index_f) <= 4* eps:
                        index_f = 0.0

                    if index_f == 0.0:
                        ind = int(np.floor(index_f_floor + 0.5))
                        normed[ori_i, j] = self.row_means[ind-1]
                    elif index_f == 1.0:
                        ind = int(np.floor(index_f_floor + 1.5))
                        normed[ori_i, j] = self.row_means[ind-1]
                    else:
                        ind = int(np.floor(index_f_floor + 0.5))
                        if (ind < self.rows) and ind > 0:
                            normed[ori_i, j] = (1.0 - index_f) * self.row_means[ind-1] + index_f * self.row_means[ind]
                        elif ind > self.rows:
                            normed[ori_i, j] = self.row_means[self.rows-1]
                        else:
                            normed[ori_i, j] = self.row_means[0]
    
        self.normed_matrix = normed
    
    
    def quantilenormalize(self):
        """
        """
        
        sorted_matrix = np.sort(self.matrix, axis = 0)
        self.ranks(sorted_matrix)
        self.rowmeans(sorted_matrix)
        self.targeting()


class SignatureMatrix:
    """
        Signature Matrix build.
        Not completed algorithm of CIBERSORT.
    """
    
    def __init__(self, sc_Meta, sc_Count):
        """
        """
        
        self.sc_Meta = sc_Meta
        self.sc_Count = sc_Count
        self.DEGs_Select()
        self.SigMatConstruction()
    
    def DEGs_Select(self):
        """
            Get the cell-specific genes from single-cell expression matrix.
            Welch Two Sample t-test for cell-specific gene.
        """
        cells = set(self.sc_Meta.cellType)
        genes = self.sc_Count.index
        
        DEGs_Raw = {}
        for cell in cells:
            
            cell_meta = self.sc_Meta[self.sc_Meta.cellType == cell]
            rest_meta = self.sc_Meta[self.sc_Meta.cellType != cell]
            
            cell_count = self.sc_Count[cell_meta.sampleName]
            rest_count = self.sc_Count[rest_meta.sampleName]
            
            print ('{} Specific Gene ... '.format(cell))
            P_value = []
            FC_value = []
            Candidate = []
            for gene in genes:
                g1 = cell_count.loc[gene]
                g2 = rest_count.loc[gene]
                v1 = g1[g1 != 0]
                v2 = g2[g2 != 0]
                if len(v1) < (0.2 * len(g1)) or len(v2) < (0.2 * len(g2)):
                    continue
                else:
                    fc = np.log2(v1.mean() / v2.mean())
                    p = stats.ttest_ind(v1, v2, equal_var = False)[1]
                    if abs(fc) > 1 and p < 0.01:
                        P_value.append(p)
                        FC_value.append(fc)
                        Candidate.append(gene)
            
            Info = pd.DataFrame(np.array([FC_value, P_value]).T, 
                                index = Candidate, columns = ['FC','P_value'])
        
            DEGs_Raw[cell] = Info
        
        print ('Cell_Specific_select ...')
        Final_DEGs = None
        
        for cell in cells:
            
            if Final_DEGs is None:
                Final_DEGs = DEGs_Raw[cell]
            else:
                tmp = DEGs_Raw[cell]
                for i in range(len(tmp)):
                    series = tmp.iloc[i]
                    gene = series.name
                    if gene not in Final_DEGs.index:
                        Final_DEGs = Final_DEGs.append(series)
                    else:
                        if Final_DEGs.loc[gene].P_value < series.P_value:
                            Final_DEGs.loc[gene].FC = series.FC
                            Final_DEGs.loc[gene].P_value = series.P_value
        
        print ('{} gene selected'.format(len(Final_DEGs)))

        self.DEGs = Final_DEGs.sort_values(by = ['P_value'], ascending = True)

    def SigMatConstruction(self, start = 200, end = 500):
        """
            SignatureMatrix Construction
        """
        
        ##Mean value of cells
        self.sc_Count_Normalize = self.sc_Count / self.sc_Count.sum(axis=0) * 10 **6
        
        Mean_count = None
        cells = set(self.sc_Meta.cellType)
        for cell in cells:
            sub_meta = self.sc_Meta[self.sc_Meta.cellType == cell]
            sub_count = self.sc_Count_Normalize[sub_meta.sampleName]
            if Mean_count is None:
                Mean_count = sub_count.mean(axis=1)
            else:
                Mean_count = pd.concat([Mean_count, sub_count.mean(axis=1)], axis = 1)
        
        Mean_count.columns = cells
        
        if Mean_count.shape[0] < start:
            self.ConditionNumber = np.linalg.cond(Mean_count)
            self.SigMat = Mean_count
            return 
       
        ##Condition Number control
        Condition_Num = []
        ##Iterator from top 200 genes to 500 genes
        for i in range(start,end):
            tmp = self.DEGs.iloc[:i]
            genes = tmp.index
            Matrix = Mean_count.loc[genes]
            CN = np.linalg.cond(Matrix, 2)
            Condition_Num.append(CN)
        
        index = range(start,end)[np.argmin(Condition_Num)]
        Matrix = Mean_count.loc[self.DEGs.iloc[:index].index]
        
        self.ConditionNumber = np.min(Condition_Num)
        self.SigMat = Matrix
        

class BulkMatrix:
    """
    """
    
    def __init__(self, bulk_Count, QN = False):
        """
        """
        self.bulk_Count = bulk_Count
        self.QN = QN
        self.BulkMatConstruction()
    
    def BulkMatConstruction(self):
        """
        """
        self.bulk_Count = self.bulk_Count / self.bulk_Count.sum(axis=0) * 10 ** 6
        
        if self.QN:
            QN_model = QuantileNormalization(self.bulk_Count)
            self.bulk_Count = pd.DataFrame(QN_model.normed_matrix, 
                                           index = self.bulk_Count.index,
                                           columns = self.bulk_Count.columns)
    
    
class NuSVR_regression:
    """
        Nu-support vector regression method for single cell 
        RNA deconvolution to determine cell type abundance.
    """
    
    def __init__(self, X, y, **kwargs):
        """
            init data
            X : signature matrix from SC-RNA.
            y : response matrix from bulk-RNA.

        """
        self.X = X
        self.y = y
        self.nu = kwargs.get('nu',[0.25,0.5,0.75])
        self.kernel = kwargs.get('kernel', 'linear')
    
    def coefficient_handle(self, coef):
        """
            Coefficient handle.
            Negative coefs  -> 0
            Remain coefs normalized -> sum to 1
        """
        
        coef = np.array(coef)
        coef[coef < 0] = 0
        tmp = coef.sum()
        coef = np.array([i / tmp for i in coef])
        
        return coef
            
        
    def fit(self):
        """
            Model.
        """
        scores = []
        models = []
        cors = []
        for nu in self.nu:
            reg = NuSVR(kernel=self.kernel, gamma='auto', nu=nu)
            reg.fit(self.X, self.y)
            models.append(reg)
            coef = self.coefficient_handle(reg.coef_)[0]
            
            # Using the final coefficant to calculate RMSE
            
            y_ = np.array((self.X * coef).sum(axis=1))
            score = np.sqrt(np.mean((y_- self.y)**2))
            
            #score = np.sqrt(sum((self.y - reg.predict(self.X)) ** 2) / len(self.y))
            scores.append(score)
            cor = stats.pearsonr(y_,self.y)
            cors.append(cor)
        
        self.RMSE = min(scores)
        index = scores.index(self.RMSE)
        bestmodel = models[index]
        self.cor = cors[index]
        
        rawcoef = bestmodel.coef_[0]
        self.coef = self.coefficient_handle(rawcoef)
        self.model = bestmodel
        
        self.Assin_P()

    
    def Assin_P(self):
        """
            Assign P value by Monte Carlo sampling.
            
        """
        y_ = self.model.predict(self.X)
        P_lst = []
        
        #---Monte Carlo sampling ---
        
        for i in range(50):
            # iterator 50
            random_seed = np.random.randint(0,high = (len(y_)-1), 
                                            size = (round(len(y_) / 3 * 2)))
            
            P_lst.append(stats.pearsonr(self.y[random_seed], y_[random_seed])[1])
        
        self.P_value = np.mean(P_lst)


class CIBERSORT:
    """
    Class for a CIBERSORT task.
    """
    
    def __init__(self, sc_Meta, sc_Count, bulk_Count, QN = False, ZS = False):
        """
        """
        self.sc_Meta = sc_Meta
        self.sc_Count = sc_Count
        self.bulk_Count = bulk_Count
        self.QN = QN
        self.ZS = ZS
        self.Preparation()
        self.Model()
    
    def Preparation(self):
        """
        """
        
        self.sc_basis = SignatureMatrix(self.sc_Meta, self.sc_Count)
        self.bulk_basis = BulkMatrix(self.bulk_Count, QN = self.QN)
        
        self.X = self.sc_basis.SigMat
        self.Y = self.bulk_basis.bulk_Count.reindex(self.X.index)
        
    def Model(self):
        """
        """
        
        Number = len(self.Y.columns)
        self.Model = {}
        self.coef = []
        ##Normalize
        if self.ZS:    
            self.X = (self.X - np.array(self.X).mean()) / np.array(self.X).std(ddof=1)
        for i in range(Number):
            Y = self.Y.iloc[:,i]
            print ('Deconvolution for {}'.format(Y.name))
            ##Normalize
            if self.ZS:
                Y = (Y - Y.mean()) / Y.std(ddof=1)
            model = NuSVR_regression(self.X, Y)
            model.fit()
        
            self.coef.append(model.coef)
            self.Model[Y.name] = model.model
        
        self.coef = pd.DataFrame(self.coef, index = self.Y.columns,
                                 columns = self.X.columns)

