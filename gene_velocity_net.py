import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import loompy
import velocyto as vcy
import logging
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import networkx as nx



class gene_velocity_net:
    
     
    # plotting utility functions
    def despline():
        ax1 = plt.gca()
        # Hide the right and top spines
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

    def minimal_xticks(start, end):
        end_ = np.around(end, -int(np.log10(end))+1)
        xlims = np.linspace(start, end_, 5)
        xlims_tx = [""]*len(xlims)
        xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
        plt.xticks(xlims, xlims_tx)
 

    def minimal_yticks(start, end):
        end_ = np.around(end, -int(np.log10(end))+1)
        ylims = np.linspace(start, end_, 5)
        ylims_tx = [""]*len(ylims)
        ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
        plt.yticks(ylims, ylims_tx)
    
    
    def __init__(self) -> None:  # data pre-process, execute what have been done before
         
        self.vlm = vcy.VelocytoLoom("/home/liz3/Desktop/DentateGyrus.loom")

        self.vlm.ts = np.column_stack([self.vlm.ca["TSNE1"], self.vlm.ca["TSNE2"]])
        
        
        colors_dict = {'RadialGlia': np.array([ 0.95,  0.6,  0.1]), 'RadialGlia2': np.array([ 0.85,  0.3,  0.1]), 'ImmAstro': np.array([ 0.8,  0.02,  0.1]),
              'GlialProg': np.array([ 0.81,  0.43,  0.72352941]), 'OPC': np.array([ 0.61,  0.13,  0.72352941]), 'nIPC': np.array([ 0.9,  0.8 ,  0.3]),
              'Nbl1': np.array([ 0.7,  0.82 ,  0.6]), 'Nbl2': np.array([ 0.448,  0.85490196,  0.95098039]),  'ImmGranule1': np.array([ 0.35,  0.4,  0.82]),
              'ImmGranule2': np.array([ 0.23,  0.3,  0.7]), 'Granule': np.array([ 0.05,  0.11,  0.51]), 'CA': np.array([ 0.2,  0.53,  0.71]),
               'CA1-Sub': np.array([ 0.1,  0.45,  0.3]), 'CA2-3-4': np.array([ 0.3,  0.35,  0.5])}
        self.vlm.set_clusters(self.vlm.ca["ClusterName"], cluster_colors_dict=colors_dict)

        """
        # Plot TSNE
        plt.figure(figsize=(10,10))
        vcy.scatter_viz(self.vlm.ts[:,0], self.vlm.ts[:,1], c=self.vlm.colorandum, s=2)
        for i in range(max(self.vlm.ca["Clusters"])):
            ts_m = np.median(self.vlm.ts[self.vlm.ca["Clusters"] == i, :], 0)
            plt.text(ts_m[0], ts_m[1], str(self.vlm.cluster_labels[self.vlm.ca["Clusters"] == i][0]),
                     fontsize=13, bbox={"facecolor":"w", "alpha":0.6})
        plt.axis("off");
        plt.savefig("/home/liz3/Desktop/TSNE.png")
        """

        self.vlm.filter_cells(bool_array=self.vlm.initial_Ucell_size > np.percentile(self.vlm.initial_Ucell_size, 0.4))
        self.vlm.ts = np.column_stack([self.vlm.ca["TSNE1"], self.vlm.ca["TSNE2"]])
        self.vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
        self.vlm.filter_genes(by_detection_levels=True)
        self.vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=35)
        self.vlm.filter_genes(by_cv_vs_mean=True)
        self.vlm.score_detection_levels(min_expr_counts=0, min_cells_express=0, min_expr_counts_U=25, min_cells_express_U=20)
        self.vlm.score_cluster_expression(min_avg_U=0.01, min_avg_S=0.08)
        self.vlm.filter_genes(by_detection_levels=True, by_cluster_expression=True)

        # best with sample and expression scaling
        self.vlm._normalize_S(relative_size=self.vlm.initial_cell_size,
                 target_size=np.mean(self.vlm.initial_cell_size))
        self.vlm._normalize_U(relative_size=self.vlm.initial_Ucell_size,
                 target_size=np.mean(self.vlm.initial_Ucell_size))


        self.vlm.perform_PCA()
        #plt.plot(np.cumsum(self.vlm.pca.explained_variance_ratio_)[:100])
        n_comps = np.where(np.diff(np.diff(np.cumsum(self.vlm.pca.explained_variance_ratio_))>0.002))[0][0]
        #plt.axvline(n_comps, c="k")
        

        k = 500
        self.vlm.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=16)

        self.vlm.fit_gammas(limit_gamma=False, fit_offset=False)
        self.vlm.predict_U()

        self.vlm.calculate_velocity()
        self.v = self.vlm.velocity
        print(self.v.shape)
        
        self.U = self.vlm.Ux_sz
        self.S = self.vlm.Sx_sz
        self.gene_names = self.vlm.ra['Gene']
        
    
    def plot_U_gene_hist(self):
        
        plt.figure(figsize=(10,10))
        plt.hist(self.U.sum(1), bins=20)
        plt.savefig("/home/liz3/Desktop/U_gene_deg.png")
        plt.close()
    
    
    def plot_S_gene_hist(self):
        
        plt.figure(figsize=(10,10))
        plt.hist(self.S.sum(1), bins=20)
        plt.savefig("/home/liz3/Desktop/S_gene_deg.png")
        plt.close()       
    
    
    def filter_gene_again1(self, num_of_gene_left):   # in terms of gene expression degree
        
        gene_index_U = self.U.sum(1).argsort()[-num_of_gene_left:][::-1]
        gene_index_S = self.S.sum(1).argsort()[-num_of_gene_left:][::-1]
        
        print(self.gene_names[np.intersect1d(gene_index_U, gene_index_S)])  # print genes both activated in U and S
        
        z= np.concatenate((gene_index_U, gene_index_S))
        _, i = np.unique(z, return_index=True)
        self.gene_index = z[np.sort(i)]
        
        self.U = self.U[self.gene_index]
        self.S = self.S[self.gene_index]
        self.gene_names = self.gene_names[self.gene_index]
        self.v = self.v[self.gene_index]
    
    
    def filter_gene_again2(self):  # based on experience
        
        self.gene_names = ["Tnc", "Gfap", "Tac2", "Igfbpl1", "Ptprn", "Sema3c", "Neurod6", "Stmn2", "Sema5a", "C1ql3", "Cpne4", "Cck"]
        
        self.gene_index = []
        
        for i, gn in enumerate(self.gene_names):
            
            self.gene_index.append(int(np.where(self.vlm.ra['Gene'] == gn)[0]))
        
        print(i)
        print(self.gene_index)
        self.U = self.U[self.gene_index]
        self.S = self.S[self.gene_index]
        self.v = self.v[self.gene_index]        
        
        
 
    def gene_US_corr_matrix(self):
        
        US = np.concatenate((self.U, self.S))
        self.corr_US = np.corrcoef(US)
        self.corr_US = np.where(np.isnan(self.corr_US)==True, 0, self.corr_US)
        np.fill_diagonal(self.corr_US, 0)
        self.corr_US = abs(self.corr_US)
        
        plt.figure(figsize=(10,10))
        plt.imshow(self.corr_US[:len(self.gene_index), len(self.gene_index):])
        plt.colorbar()
        plt.savefig("/home/liz3/Desktop/corr_US.png")
        plt.close()
        
        
    def gene_velocity_corr_matrix(self):
        
        self.corr_v = np.corrcoef(self.v)
        self.corr_v = np.where(np.isnan(self.corr_v)==True, 0, self.corr_v)
        np.fill_diagonal(self.corr_v, 0)
        self.corr_v = abs(self.corr_v)
        
        plt.figure(figsize=(10,10))
        plt.imshow(self.corr_v)
        plt.colorbar()
        plt.savefig("/home/liz3/Desktop/corr_v.png")
        plt.close()
        
        
    def gene_U_corr_matrix(self):
        
        self.U_corr_gene = np.corrcoef(self.U)
        self.U_corr_gene = np.where(np.isnan(self.U_corr_gene)==True, 0, self.U_corr_gene)
        np.fill_diagonal(self.U_corr_gene, 0)
        self.U_corr_gene = abs(self.U_corr_gene)
        
        plt.figure(figsize=(10,10))
        plt.imshow(self.U_corr_gene)
        plt.colorbar()
        plt.savefig("/home/liz3/Desktop/corr_U.png")
        plt.close()        
        
        
    def gene_S_corr_matrix(self):
        
        self.S_corr_gene = np.corrcoef(self.S)
        self.S_corr_gene = np.where(np.isnan(self.S_corr_gene)==True, 0, self.S_corr_gene)
        np.fill_diagonal(self.S_corr_gene, 0)
        self.S_corr_gene = abs(self.S_corr_gene)
        
        plt.figure(figsize=(10,10))
        plt.imshow(self.S_corr_gene)
        plt.colorbar()
        plt.savefig("/home/liz3/Desktop/corr_S.png")
        plt.close()   

    
    def gene_vs_corr_matrix(self):
        
        normal_factor = self.S.mean() / self.v.mean()
        self.vs = np.concatenate((self.S, normal_factor * self.v))
        self.vs_corr_gene = np.corrcoef(self.vs)
        self.vs_corr_gene = np.where(np.isnan(self.vs_corr_gene)==True, 0, self.vs_corr_gene)
        np.fill_diagonal(self.vs_corr_gene, 0)
        self.vs_corr_gene = abs(self.vs_corr_gene)
        self.vs_corr_gene = self.vs_corr_gene[:len(self.gene_index), len(self.gene_index):]
        
        plt.figure(figsize=(10,10))
        plt.imshow(self.vs_corr_gene)
        plt.colorbar()
        plt.savefig("/home/liz3/Desktop/corr_vs.png")
        plt.close() 
        
    
    def gene_causal_matrix(self, cut_off, weight=True):
        
        if weight:
        
            self.causal_matrix = self.vs_corr_gene
            print(self.causal_matrix)
            np.savetxt("/home/liz3/Desktop/causal_matrix_weight.txt", self.causal_matrix, delimiter=" ")
            np.savetxt("/home/liz3/Desktop/gene_names_weight.txt", self.gene_names, delimiter=" ", fmt="%s")    
        
        else:
        
            self.causal_matrix = np.where(self.vs_corr_gene > cut_off, 1, 0)
            print(self.causal_matrix)
            np.savetxt("/home/liz3/Desktop/causal_matrix.txt", self.causal_matrix, delimiter=" ")
            np.savetxt("/home/liz3/Desktop/gene_names.txt", self.gene_names, delimiter=" ", fmt="%s")      
    


if __name__ == '__main__':
    
    data = gene_velocity_net()
    
    """
    data.select_candidate_gene_index(20)
    data.filter_gene_again()
    data.gene_vs_corr_matrix()
    data.gene_causal_matrix(0.8)
    """
