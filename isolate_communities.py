import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, performance, coverage
import loompy
import velocyto as vcy
import matplotlib.pyplot as plt


class isolate_communities:
    
    
    def __init__(self):
        
        self.vlm = vcy.VelocytoLoom("/home/liz3/Desktop/DentateGyrus.loom")

        self.vlm.ts = np.column_stack([self.vlm.ca["TSNE1"], self.vlm.ca["TSNE2"]])
        
        
        colors_dict = {'RadialGlia': np.array([ 0.95,  0.6,  0.1]), 'RadialGlia2': np.array([ 0.85,  0.3,  0.1]), 'ImmAstro': np.array([ 0.8,  0.02,  0.1]),
              'GlialProg': np.array([ 0.81,  0.43,  0.72352941]), 'OPC': np.array([ 0.61,  0.13,  0.72352941]), 'nIPC': np.array([ 0.9,  0.8 ,  0.3]),
              'Nbl1': np.array([ 0.7,  0.82 ,  0.6]), 'Nbl2': np.array([ 0.448,  0.85490196,  0.95098039]),  'ImmGranule1': np.array([ 0.35,  0.4,  0.82]),
              'ImmGranule2': np.array([ 0.23,  0.3,  0.7]), 'Granule': np.array([ 0.05,  0.11,  0.51]), 'CA': np.array([ 0.2,  0.53,  0.71]),
               'CA1-Sub': np.array([ 0.1,  0.45,  0.3]), 'CA2-3-4': np.array([ 0.3,  0.35,  0.5])}
        self.vlm.set_clusters(self.vlm.ca["ClusterName"], cluster_colors_dict=colors_dict)

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
        
        self.S = self.vlm.Sx_sz
        self.gene_names = self.vlm.ra['Gene']
        self.alphas = [0.4+0.025*i for i in range(23)]


    def gene_vs_corr_matrix(self):
        
        normal_factor = self.S.mean() / self.v.mean()
        self.vs = np.concatenate((self.S, normal_factor * self.v))
        self.vs_corr_gene = np.corrcoef(self.vs)
        self.num_of_gene = self.S.shape[0]
        self.vs_corr_gene = self.vs_corr_gene[:self.num_of_gene, self.num_of_gene:]
        self.vs_corr_gene = np.where(np.isnan(self.vs_corr_gene)==True, 0, self.vs_corr_gene)
        self.vs_corr_gene = abs(self.vs_corr_gene)
    
    
    def gene_corr_matrix(self):
        
        self.s_corr_gene = np.corrcoef(self.S)
        self.s_corr_gene = np.where(np.isnan(self.s_corr_gene)==True, 0, self.s_corr_gene)
        np.fill_diagonal(self.s_corr_gene, 0)
        self.s_corr_gene = abs(self.s_corr_gene)
    
    
    def increase_threshold(self):
    
        size_of_cluster1_list = []
        num_of_cluster_list = []
        measure_list = []
        
        for alpha in self.alphas:
            
            causal_matrix = np.where(self.vs_corr_gene > alpha, 1, 0)
            G = nx.Graph(causal_matrix)
            group_class = list(greedy_modularity_communities(G))
            
            measure = coverage(G, group_class)
            num_of_cluster = len(group_class)
            indices_in_cluster1 = list(group_class[0])
            size_of_cluster1 = len(indices_in_cluster1)
            
            print(num_of_cluster)
            print(size_of_cluster1)
            
            size_of_cluster1_list.append(size_of_cluster1)
            num_of_cluster_list.append(num_of_cluster)
            measure_list.append(measure)
            
        plt.figure(figsize=(10,10))
        plt.plot(self.alphas, size_of_cluster1_list, linestyle='-', marker='o', color='b')
        plt.xlabel(r'$\tau$')
        plt.ylabel('Size of the largest community')
        plt.savefig("/home/liz3/Desktop/size_of_leading_community")
        plt.close()
        
        plt.figure(figsize=(10,10))
        plt.plot(self.alphas, num_of_cluster_list, linestyle='-', marker='o', color='b')
        plt.xlabel(r'$\tau$')
        plt.ylabel('Number of communities')
        plt.savefig("/home/liz3/Desktop/number_of_community")
        plt.close()        
        
        plt.figure(figsize=(10,10))
        plt.plot(self.alphas, measure_list, linestyle='-', marker='o', color='b')
        plt.xlabel(r'$\tau$')
        plt.ylabel('Coverage')
        plt.savefig("/home/liz3/Desktop/measure_of_partition")
        plt.close()
        
    
    def select_genes_s(self, cut_off):
        
        self.causal_matrix = np.where(self.s_corr_gene > cut_off, 1, 0)
        G = nx.Graph(self.causal_matrix)
        group_class = list(greedy_modularity_communities(G))
        indices_in_cluster1 = list(group_class[0])
        indices_in_cluster2 = list(group_class[1])
        indices_in_cluster3 = list(group_class[2])
        
        print(len(list(group_class[0])))
        print(len(list(group_class[1])))
        print(len(list(group_class[2])))
        print(len(list(group_class[3])))
        print(len(list(group_class[4])))
        
    
    def select_genes(self, cut_off):
        
        self.causal_matrix = np.where(self.vs_corr_gene > cut_off, 1, 0)
        G = nx.Graph(self.causal_matrix)
        group_class = list(greedy_modularity_communities(G))
        indices_in_cluster1 = list(group_class[0])
        indices_in_cluster2 = list(group_class[1])
        indices_in_cluster3 = list(group_class[2])
        
        print(len(list(group_class[0])))
        print(len(list(group_class[1])))
        print(len(list(group_class[2])))
        print(len(list(group_class[3])))
        print(len(list(group_class[4])))
            
        
        """
        np.savetxt("/home/liz3/Desktop/Dissertation_final/causal_matrix31.txt", self.causal_matrix[np.ix_(indices_in_cluster1,indices_in_cluster1)], delimiter=" ")
        np.savetxt("/home/liz3/Desktop/Dissertation_final/gene_names31.txt", self.gene_names[indices_in_cluster1], delimiter=" ", fmt="%s")
        np.savetxt("/home/liz3/Desktop/Dissertation_final/causal_matrix32.txt", self.causal_matrix[np.ix_(indices_in_cluster2,indices_in_cluster2)], delimiter=" ")
        np.savetxt("/home/liz3/Desktop/Dissertation_final/gene_names32.txt", self.gene_names[indices_in_cluster2], delimiter=" ", fmt="%s")
        np.savetxt("/home/liz3/Desktop/Dissertation_final/causal_matrix33.txt", self.causal_matrix[np.ix_(indices_in_cluster3,indices_in_cluster3)], delimiter=" ")
        np.savetxt("/home/liz3/Desktop/Dissertation_final/gene_names33.txt", self.gene_names[indices_in_cluster3], delimiter=" ", fmt="%s")
        """
        
    def intermediate_genes(self, start_gene_name, end_gene_name):
        
        start_index = np.where(self.vlm.ra['Gene'] == start_gene_name)
        end_index = np.where(self.vlm.ra['Gene'] == end_gene_name)
        index_list = []
        
        for i in range(self.num_of_gene):
            
           if self.vs_corr_gene[start_index, i]*self.vs_corr_gene[i, end_index]>0.49 and i!=start_index:
               
               index_list.append(i)
        
        print(self.vlm.ra['Gene'][index_list])
        
        
    def all_intermediate_genes(self, start_gene_name_list, end_gene_name):
        
        start_inter_dict = {gn : [] for gn in start_gene_name_list}
        end_index = np.where(self.vlm.ra['Gene'] == end_gene_name)[0]
        
        for gn in start_gene_name_list:
            
            start_index = np.where(self.vlm.ra['Gene'] == gn)[0]
            
            for i in range(self.num_of_gene):
                
                if self.vs_corr_gene[start_index, i]*self.vs_corr_gene[i, end_index]>0.45 and i!=start_index:
                    
                    start_inter_dict[gn].append(i)
                    
            start_inter_dict[gn] = list(self.vlm.ra['Gene'][start_inter_dict[gn]])
                    
        print(start_inter_dict)

    
    def interaction_with_selected_genes(self, start_gene_name_list, inter_gene_name, end_gene_name):
        
        end_index = int(np.where(self.vlm.ra['Gene'] == end_gene_name)[0])
        inter_index = int(np.where(self.vlm.ra['Gene'] == inter_gene_name)[0])
        
        for gn in start_gene_name_list:
            
            start_index = int(np.where(self.vlm.ra['Gene'] == gn)[0])
            print((self.vs_corr_gene[start_index, inter_index], self.vs_corr_gene[inter_index, end_index]))
    
    
    def two_intermediates(self, start_gene_name_list, layer1_cell_type, layer2_cell_type, end_gene_name):
        
        cell1_indices = np.where(self.vlm.ca['ClusterName']==layer1_cell_type)[0]
        cell2_indices = np.where(self.vlm.ca['ClusterName']==layer2_cell_type)[0]
        end_index = int(np.where(self.vlm.ra['Gene'] == end_gene_name)[0])
        
        S1 = self.S[:,cell1_indices]
        S2 = self.S[:,cell2_indices]
        
        gene_sum1 = S1.sum(1)
        gene_sum2 = S2.sum(1)
        
        leading_pos1 = gene_sum1.argsort()[-20:]
        leading_pos2 = gene_sum2.argsort()[-20:]
        
        layer1_gene_names = self.vlm.ra['Gene'][leading_pos1]
        layer2_gene_names = self.vlm.ra['Gene'][leading_pos2]
        
        
        for gn in start_gene_name_list:
            
            start_index = int(np.where(self.vlm.ra['Gene'] == gn)[0])
            
            for i in leading_pos1:
                
                for j in leading_pos2:
                    
                    if self.vs_corr_gene[start_index, i] > 0.4 and self.vs_corr_gene[i, j]>0.4 and self.vs_corr_gene[j, end_index] > 0.4:
                        
                        print((self.vs_corr_gene[start_index, i], self.vs_corr_gene[i, j], self.vs_corr_gene[j, end_index]))
                        print((gn, self.vlm.ra['Gene'][i], self.vlm.ra['Gene'][j]))
                        
                    
                    
      

           




if __name__ == '__main__':
    
    data = isolate_communities()
    data.gene_vs_corr_matrix()
    #data.increase_threshold()
    #data.select_genes(0.94)
    #data.gene_corr_matrix()
    #data.select_genes_s(0.992)
    #data.all_intermediate_genes(['Cenpm', 'Tk1', 'E2f8', 'Cenpn', 'Esco2', 'Pbk', 'Diaph3', 'Ncapg2', 'Melk', 'Spc24'],'Diaph3')
    #data.interaction_with_selected_genes(['Cenpm', 'Tk1', 'E2f8', 'Cenpn', 'Esco2', 'Pbk', 'Diaph3', 'Ncapg2', 'Melk', 'Spc24'],'Sema3c', 'Diaph3')
    #data.two_intermediates(['Cenpm', 'Tk1', 'E2f8', 'Cenpn', 'Esco2', 'Pbk', 'Ncapg2', 'Melk', 'Spc24'],'Nbl2', 'CA', 'Diaph3')
    data.two_intermediates(['Asrgl1', 'Nim1k', 'Nr3c1', 'Rfx4', 'Hepacam', 'Npas3', 'Gng12'], 'Nbl2', 'ImmGranule1', 'Ntrk2')