import pandas as pd 

def bfs_trrust_database_search_tf(list_of_input_genes, trrust_filepath="../trrust_rawdata.human.tsv", column_names=["Transcription factor", "Target gene", "Relationship", "PubMED identifier"], return_all=False):
	"""BFS search of TRRUST database of TFs"""


	df = pd.read_csv(trrust_filepath, delimiter='\t', header=None)
	df.columns = column_names

	master_visited = []
	master_relationships = []

	if not return_all:
		for gene in list_of_input_genes:
			print(gene)

			queue = [gene]
			visited = []
			relationships = []

			while queue:
				current_gene = queue.pop(0)
				for tf_gene in df.loc[df["Target gene"]==current_gene.upper()]["Transcription factor"].values:
					if tf_gene not in visited:
						visited.append(tf_gene)
						queue.append(tf_gene)
						relationships.append([current_gene,tf_gene])

			master_visited.append(visited)
			master_relationships.append(relationships)

	else:
		for index, row in df.iterrows():
			master_relationships.append([row["Target gene"], row["Transcription factor"]])

	flat_visited = []
	for sub_list in master_visited:
		for gene_name in sub_list:
			flat_visited.append(gene_name)

	count_dict = {}
	for gene_name in flat_visited:
		if (gene_name in count_dict):
			count_dict[gene_name] += 1
		else:
			count_dict[gene_name] = 1

	return master_visited, master_relationships, count_dict


def bfs_trrust_database_search_target(list_of_input_genes, trrust_filepath="../trrust_rawdata.human.tsv", column_names=["Transcription factor", "Target gene", "Relationship", "PubMED identifier"], return_all=False):
	"""BFS search of TRRUST database of target genes"""


	df = pd.read_csv(trrust_filepath, delimiter='\t', header=None)
	df.columns = column_names

	master_visited = []
	master_relationships = []


	if not return_all:
		for gene in list_of_input_genes:
			print(gene)

			queue = [gene]
			visited = []
			relationships = []

			while queue:
				current_gene = queue.pop(0)
				for target_gene in df.loc[df["Transcription factor"]==current_gene.upper()]["Target gene"].values:
					if target_gene not in visited:
						visited.append(target_gene)
						queue.append(target_gene)
						relationships.append([current_gene,target_gene])

			master_visited.append(visited)
			master_relationships.append(relationships)

	elif return_all:
		for index, row in df.iterrows():
			master_relationships.append([row["Transcription factor"], row["Target gene"]])
			if index % 1000==0:
				print(index)
		master_relationships = [master_relationships]

	flat_visited = []
	for sub_list in master_visited:
		for gene_name in sub_list:
			flat_visited.append(gene_name)

	count_dict = {}
	for gene_name in flat_visited:
		if (gene_name in count_dict):
			count_dict[gene_name] += 1
		else:
			count_dict[gene_name] = 1

	return master_visited, master_relationships, count_dict

if __name__ == '__main__':
	
	m, r, d = bfs_trrust_database_search_tf(["STAT3", "SOX10", "AR"])
	print(d)