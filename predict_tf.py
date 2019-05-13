"""Function to predict TFs"""

# import modules
import numpy as np 
import json

# import custom modules
from process_sequences import *
from src import bfs_trrust

# math
from math import log
import random

# import file search
import glob
import csv

# Keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


class Predictor():
	"""Predictor class"""

	def __init__(self, list_of_target_gene_names, run_with_fraction=False):
		"""Initialize parameters"""
		self.list_of_target_gene_names = [target_gene_name.upper() for target_gene_name in list_of_target_gene_names]
		self.dict_of_promoters = {}
		self.dict_of_protein_aa_sequences = {}
		self.X_prediction_input = []
		self.aa_predictions_for_target_genes = []
		self.probability_matrix = np.load("src/target_tf_probability_matrix.npy")
		self.df_gene_locations_cleaned = pd.read_csv("datasets/overall_list_of_genes_cleaned.csv", delimiter=',')
		self.list_of_gene_names = list(self.df_gene_locations_cleaned["gene name"].unique())
		self.genes_information = []
		self.protein_aa_frequencies_dict = json.load(open("datasets/protein_aa_frequencies.json", 'r'))
		self.similarities_list = []
		self.scores = []
		self.sorted_list = []
		self.master_visited, self.master_relationships, self.dict_of_tf_frequency = bfs_trrust.bfs_trrust_database_search_tf(list_of_target_gene_names, trrust_filepath="trrust_rawdata.human.tsv")
		self.run_with_fraction = run_with_fraction
		self.list_of_all_files = glob.glob("static/*.csv")


	def load_amino_acid_data(self, filename="datasets/protein_aa_frequencies_divided_by_200.json"):
		"""Method to load data"""
		self.dict_of_protein_aa_sequences = json.load(open(filename, 'r'))


	def load_promoter_sequences(self, filename="datasets/gene_promoters_one_hot_raw_complete.json"):
		"""Method to read promoter sequences data"""
		self.dict_of_promoters = json.load(open(filename, 'r'))


	def compute_aa_predictions_for_target_genes(self, keras_model_path="saved_models/CNN_PUBMED_DATA-V3-_ALLGENES_ALLREL-48-0.06.hdf5"):
		"""Run Keras model"""
		for target_gene_name in self.list_of_target_gene_names:

			# get information on gene
			gene_information_list = self.df_gene_locations_cleaned.loc[self.df_gene_locations_cleaned['gene name'] == target_gene_name]
			self.genes_information.append(gene_information_list)

			# get one hot of promoter
			self.X_prediction_input.append(one_hot_dna(self.dict_of_promoters[target_gene_name]))

		# predict AA frequency for all genes
		self.X_prediction_input = np.array(self.X_prediction_input).reshape(len(self.X_prediction_input),1,2500,4)
		model = load_model(keras_model_path)
		print("Model loaded!")
		self.aa_predictions_for_target_genes = model.predict(self.X_prediction_input)


	def calculate_location_coefficient(self, target_gene_location, tf_gene_location):
		"""Method to calculate location coefficient"""
		return probability_matrix[target_gene_location-1][tf_gene_location-1]


	def generate_random(self, is_random=False):
		if not is_random:
			return False
		else:
			num = random.randrange(0,100)
			if num==0:
				return True
			else:
				return False


	def return_scores(self, random_select=False):
		"""Method to return scores"""

		# load models
		self.load_amino_acid_data(filename="datasets/protein_aa_frequencies_divided_by_200.json")
		# self.load_promoter_sequences(filename="datasets/gene_promoters_one_hot_raw_complete.json")
		self.load_promoter_sequences(filename="datasets/gene_promoters_one_hot_raw.json")

		# run predictions
		self.compute_aa_predictions_for_target_genes(keras_model_path="saved_models/CNN_PUBMED_DATA-V3-_ALLGENES_ALLREL-48-0.06.hdf5")
		self.similarities_list = []

		# for each AA prediction:
		for tf_aa_prediction, gene_name in zip(self.aa_predictions_for_target_genes, self.list_of_target_gene_names):
			print("TARGET GENE:", gene_name)
			temp_row = []

			gene_name_file = "static\\\\V5-activation_results"+gene_name+".csv"

			print(gene_name_file)
			print(self.list_of_all_files)

			print(gene_name_file in self.list_of_all_files)

			# if gene has already been searched
			for filename in self.list_of_all_files:
				if gene_name in filename:
					print("GENE FOUND!")
					with open(gene_name_file, 'r') as f:
						temp_list = []
						reader = csv.reader(f)
						for row in reader:
							temp_list.append(float(row[0]))
					self.similarities_list.append(temp_list)
					break

			else:

				target_gene_location = int(self.df_gene_locations_cleaned.loc[self.df_gene_locations_cleaned["gene name"] == gene_name].iloc[0]['chromosome'])

				# print("CHECK")
				# for matrix in self.master_relationships:
					# for row in matrix:
						# if row[0] == gene_name:
							# print(row)
				# print("CHECK DONE")

				counter = 0
				# for each possible protein (AA frequency)			
				for aa_sequence, aa_name in zip(list(self.protein_aa_frequencies_dict.values()), list(self.protein_aa_frequencies_dict.keys())):
				# for aa_name in list(self.protein_aa_frequencies_dict.keys()):

					if counter%100 == 0:
						print(counter)

					to_run = self.generate_random(self.run_with_fraction)

					if aa_name in list(self.df_gene_locations_cleaned["gene name"].values) and to_run:
						# cosine_score = cosine_similarity(gene, row)
						# get tf gene location
						tf_gene_location = int(self.df_gene_locations_cleaned.loc[self.df_gene_locations_cleaned["gene name"] == aa_name].iloc[0]['chromosome'])


						if tf_gene_location > -1:
							# print(cosine_similarity(gene, row))
							# print(probability_matrix[target_gene_location-1][tf_gene_location-1])
							# print(cosine_similarity(gene, row)*probability_matrix[target_gene_location-1][tf_gene_location-1])
							# print()

							# check if potential TF is in dict of breadth-frst-search
							tf_levels_value = 1
							if aa_name.upper() in list(self.dict_of_tf_frequency.keys()):
								tf_levels_value = self.dict_of_tf_frequency[aa_name.upper()]

							direct_relationship = 1

							# if [current gene (target gene), TF gene]
							for matrix in self.master_relationships:
								if [gene_name, aa_name] in matrix:
									# print("FOUND ROW! "+str([gene_name, aa_name]))
									direct_relationship += 1

							# values are accurate
							# temp_row.append(cosine_similarity(tf_aa_prediction, aa_sequence)*self.probability_matrix[target_gene_location-1][tf_gene_location-1])
							temp_row.append((log(cosine_similarity(tf_aa_prediction, aa_sequence)+1,2)+1)*(log(self.probability_matrix[target_gene_location-1][tf_gene_location-1]+1,2)+1)*(log(tf_levels_value+1,2)+1)*(log(direct_relationship+1,2)+1))
							# temp_row.append(log(tf_levels_value+1,2))


						else:
							temp_row.append(0)
					else:
						temp_row.append(0)

					counter += 1

				self.similarities_list.append(temp_row)

		print("self.similarities_list",self.similarities_list[0])
		for gene_name, row in zip(self.list_of_target_gene_names, self.similarities_list):
			np.savetxt("V5-activation_results"+str(gene_name)+".csv", row)

		self.similarities_list = np.array(self.similarities_list).T			# similarities is accurate

		for gene_name, row in zip(list(self.protein_aa_frequencies_dict.keys()), self.similarities_list): 

			# tem_row = [gene name, average score]
			temp_row = [gene_name, sum(row)/len(row)]
			self.scores.append(temp_row)

		self.sorted_list = sorted(self.scores, key=lambda x: x[1], reverse=True)
		return self.sorted_list		

if __name__ == '__main__':

	dict_of_target_genes = locations = {"ATF3": [1, 212565334, 212620777],
	"BASP1": [5, 17216823, 17276845],
	"CREB1": [2, 207529892, 207605989],
	"CREB3L2": [7, 137874979, 138002101],
	"GAP43": [3, 115623304, 115721487],
	"JUN": [1, 58780791, 58784113],
	"HSPB1": [7, 76302558, 76304301],
	"KLF4": [9, 107484852, 107489720],
	"NGF": [1, 115285915, 115338253],
	"POU3F1": [1, 38043851, 38046778],
	"SMAD1": [4, 145481306, 145559176],
	"SOX11": [2, 5692667, 5701385],
	"SPRR1A": [1, 152984088, 152985814],
	"STAT3": [17, 42313324, 42388503],
	"TP53": [17, 7668402, 7687550]}

	# promoter_dict = json.load(open("datasets/gene_promoters_one_hot_raw_complete.json", 'r'))
	# for target_gene_name in list(dict_of_target_genes.keys()):
	# 	if target_gene_name not in list(promoter_dict.keys()):
	# 		print(target_gene_name)



	predictor_object = Predictor(list(dict_of_target_genes.keys()))
	list_of_predictions = predictor_object.return_scores()
	print(list_of_predictions[:20])
	np.savetxt("V5-neuron-regeneration_saved_results_predictors_activation_cnn.csv", list_of_predictions, fmt="%s")