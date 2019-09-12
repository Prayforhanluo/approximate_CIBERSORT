## README

#### Approximate CIBERSORT in python.

Utilizes  scRNA-seq to estimate cell type proportions in bulk RNA-seq data by CIBERSORT algorithm. 

#### Notice ! ! !

This is a scripts  implement deconvolution by CIBERISORT algorithm, Due to the source code of CIBERSORT is not available. Some details are just implemented by author's way.So this sciprts may not reproduce CIBERSORT in all details.

More information about CIBERSORT here [CIBERSORT](https://cibersort.stanford.edu/)

Data formats as txt in data dirct

Use the function like :
	
	 sc_Meta = Get_SC_Meta_Info('sc_meta.txt')
	 sc_Count = Get_SC_Count_Info('sc_count.txt')
	 bulk_Count = Get_bulk_Count_Info('bulk_Count.txt')
	 
	 #do deconvolution
	 Results =  CIBERSORT(sc_meta, sc_count, bulk_count, ZS = True)



results :
	
	Results.coef    #CIBERSORT results
	...
	...
