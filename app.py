# import modules
from flask import Flask, render_template, request
import pandas as pd 
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

from keras import backend as K
K.set_image_dim_ordering('th')

from process_sequences import *

# BioPython
# from Bio import Entrez
# from Bio.Seq import Seq
# from Bio.Alphabet import generic_dna
# from Bio import SeqIO

import json
import time
import random

import matplotlib.pyplot as plt

chromosome_keys = {0: 'NC_000001.11',
				 1: 'NC_000002.12',
				 2: 'NC_000003.12',
				 3: 'NC_000004.12',
				 4: 'NC_000005.10',
				 5: 'NC_000006.12',
				 6: 'NC_000007.14',
				 7: 'NC_000008.11',
				 8: 'NC_000009.12',
				 9: 'NC_000010.11',
				 10: 'NC_000011.10',
				 11: 'NC_000012.12',
				 12: 'NC_000013.11',
				 13: 'NC_000014.9',
				 14: 'NC_000015.10',
				 15: 'NC_000016.10',
				 16: 'NC_000017.11',
				 17: 'NC_000018.10',
				 18: 'NC_000019.10',
				 19: 'NC_000020.11',
				 20: 'NC_000021.9',
				 21: 'NC_000022.11',
				 22: 'NC_000023.11',
				 23: 'NC_000024.10',
				 24: 'NC_012920.1'}

# read data files
df_gene_locations_cleaned = pd.read_csv("datasets/overall_list_of_genes_cleaned.csv", delimiter=',')
list_of_gene_names = list(df_gene_locations_cleaned["gene name"].unique())

print("Loading CNN model...")
# neural network models
model = Sequential()
model = load_model("saved_models/V3-CNN-5.h5")
print("CNN model loaded!", end='\n\n')

# load probability matrix
probability_matrix = np.load("datasets/target_tf_probability_matrix.npy")

# load human genome

# dict_of_sequences = {}
# for seq_record in SeqIO.parse("datasets/genomic.fna", "fasta"):
#     print(seq_record.id)
#     dict_of_sequences[seq_record.id] = seq_record.seq
# for key in list(dict_of_sequences.keys()):
#     if key[0:2] != "NC":
#         dict_of_sequences.pop(key, None)
# chromosome_keys = dict(enumerate(list(dict_of_sequences.keys())))

print("Loading JSON file...")
# load JSON file with promoters
with open("datasets/gene_promoters_one_hot_raw.json", 'r') as jsonfile:
	gene_promoters_one_hot_dict = json.load(jsonfile)		# dictionary[gene name] = [onehot promoter]
print("JSON file loaded!", end='\n\n')

# dict_of_sequences = {'NC_000019.10':'A'*900000000}

# dict_of_sequences = {'A':'AAA'}
# chromosome_keys = {'A':'AAA'}

# load human protonome
with open('datasets/protein_aa_frequencies.json', 'r') as fp:
    protein_aa_frequencies_dict = json.load(fp)

# app
app = Flask(__name__)

# user class
class User:
	"""User class"""

	def __init__(self):
		self.list_of_target_genes = []
		self.list_of_gene_prediction_names = []
		self.list_of_gene_prediction_values = []
		self.gene_information = []
		self.run_with_ten_percent = False

	def add_gene_name(self, gene_name, verbose=True):
		self.list_of_target_genes.append(gene_name)
		if verbose:
			print(f"{gene_name} added to user object!")

	def add_gene_info(self, name, chromosome, start, end):
		temp_list = [name, chromosome, start, end]
		self.gene_information.append(temp_list)

	def add_gene_list(self, gene_list):
		self.gene_information.append(gene_list)

user = User()		# initialize user object

@app.route('/')
@app.route('/home')
def home():
	return render_template("home.html")

@app.route('/about')
def about():
	return render_template("about.html")

# form page
@app.route('/run')
def run():
	return render_template("number_of_genes.html")

@app.route('/form', methods=["GET", "POST"])
def form():

	global user

	form_string = ''		# string used to generate form based on number of genes user wants to submit

	if request.method == "POST":
		counter = 1
		print("Number of genes requested by user: ",request.form["number_of_target_genes"])

		gene_datalist_options = ""

		for gene_name in list_of_gene_names:
			gene_datalist_options += f"<option value={gene_name}>\n"

		# for each gene in the number of genes requested by the user:
		for gene_number in range(int(request.form["number_of_target_genes"])):
			form_string += """			<input list="list_of_genes" name=\"""" + "gene" + str(counter) + """\" placeholder="Gene"""+str(counter)+"""\">
			<datalist id="list_of_genes">""" + gene_datalist_options + """
			</datalist>
			<br>"""
			counter += 1


		try:
			print("10% option", request.form["run_with_ten_percent"])

			# if run with 10% option
			if request.form.get("run_with_ten_percent"):
				user.run_with_ten_percent = True
		except:
			user.run_with_ten_percent = False

		prediction_eta = (counter-1)*2 		# predicted time for prediction in minutes
	return render_template("form.html", list_of_genes=list_of_gene_names, form_string=form_string, prediction_eta=prediction_eta)

# loading page
@app.route("/loading", methods=["GET", "POST"])
def loading():
	global user
	user.list_of_target_genes = []
	print("LIST OF KEYS: ", str(list(request.form.keys())))

	# if POST request
	if request.method == "POST":

		# add gene names to user object
		for form_input_name in list(request.form.keys()):
			print("form_input_name: ", form_input_name)
			gene_name = request.form[form_input_name]
			user.add_gene_name(gene_name)
			print(gene_name)

		return results()
	return "DONE!"

# processing page
@app.route("/results")
def results():
	global user

	predictions_genes = []

	start = time.time()

	try:

		for gene_name in user.list_of_target_genes:
			# get gene position information from search
			print(df_gene_locations_cleaned.shape)
			print(gene_name)
			gene_information_list = df_gene_locations_cleaned.loc[df_gene_locations_cleaned['gene name'] == gene_name]
			print(gene_information_list)
			print(gene_information_list.iloc[0]["chromosome"])
			print(gene_information_list.iloc[0]["start"])
			print(gene_information_list.iloc[0]["end"])
			chromosome_index = int(gene_information_list.iloc[0]["chromosome"]-1)
			print(chromosome_keys[chromosome_index])

			predictions_genes.append(one_hot_dna(gene_promoters_one_hot_dict[gene_name]))

			user.add_gene_list(gene_information_list)		# add gene information to user object

		predictions_genes = np.array(predictions_genes)
		print("predictions_shape", predictions_genes.shape)
		prediction = model.predict(predictions_genes.reshape(len(user.list_of_target_genes),1,2500,4))		# prediction is accurate

		similarities = []
		# locations_keys = list(locations.keys())

		# used for counting
		counter = 0
		nb_gene = 0
		num_aa = 0

		print(user.run_with_ten_percent)

		if not user.run_with_ten_percent:

			# for gene prediction, gene name in submitted target genes
			for gene, gene_name in zip(prediction,user.list_of_target_genes):

				print(gene_name)
				print(gene)

				temp_row = []	# row with probabilities for each target gene
				print(gene_name)
				print(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == gene_name])
				print(int(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == gene_name].iloc[0]['chromosome']))
				target_gene_location = int(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == gene_name].iloc[0]['chromosome'])

				for tf_gene_name, row in zip(list(protein_aa_frequencies_dict.keys()), list(protein_aa_frequencies_dict.values())):
					if nb_gene%1000 == 0:
						print(nb_gene)
					if tf_gene_name in list(df_gene_locations_cleaned["gene name"].values):
						cosine_score = cosine_similarity(gene, row)
						# get tf gene location
						tf_gene_location = int(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == tf_gene_name].iloc[0]['chromosome'])


						if tf_gene_location > -1:
							# print(cosine_similarity(gene, row))
							# print(probability_matrix[target_gene_location-1][tf_gene_location-1])
							# print()
							temp_row.append(cosine_similarity(gene, row)*probability_matrix[target_gene_location-1][tf_gene_location-1])
						else:
							temp_row.append(0)
					else:
						temp_row.append(0)
					nb_gene += 1
				        
				similarities.append(temp_row)
			#         print(similarities)

				counter += 1
				num_aa += 1

		else:
			# for gene prediction, gene name in submitted target genes
			for gene, gene_name in zip(prediction,user.list_of_target_genes):

				print(gene_name)
				print(gene)			# gene prediction is unaltered

				temp_row = []		# row with probabilities for each target gene
				print("gene name", gene_name)
				print("row output", df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == gene_name])
				print("int of chromosome", int(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == gene_name].iloc[0]['chromosome']))
				target_gene_location = int(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == gene_name].iloc[0]['chromosome'])


				for tf_gene_name, row in zip(list(protein_aa_frequencies_dict.keys()),list(protein_aa_frequencies_dict.values())):
					random_seed = random.randrange(0,10)
					if nb_gene%1000 == 0:
						print("genes seen", nb_gene)

					# if True:
					if random_seed == 0:

						if tf_gene_name in list(df_gene_locations_cleaned["gene name"].values):
							cosine_score = cosine_similarity(gene, row)
							# get tf gene location
							tf_gene_location = int(df_gene_locations_cleaned.loc[df_gene_locations_cleaned["gene name"] == tf_gene_name].iloc[0]['chromosome'])


							if tf_gene_location > -1:
								# print(cosine_similarity(gene, row))
								# print(probability_matrix[target_gene_location-1][tf_gene_location-1])
								# print(cosine_similarity(gene, row)*probability_matrix[target_gene_location-1][tf_gene_location-1])
								# print()

								# values are accurate
								temp_row.append(cosine_similarity(gene, row)*probability_matrix[target_gene_location-1][tf_gene_location-1])
							else:
								temp_row.append(0)
						else:
							temp_row.append(0)
					else:
						temp_row.append(0)

					nb_gene += 1
					counter += 1
					num_aa += 1

						        
				similarities.append(temp_row)



		print("len temp row", len(temp_row))
		print("len aa dict", num_aa)

		similarities = np.array(similarities).T			# similarities is accurate
		print("length of similarities", similarities.shape)

		scores = []
		raw_scores = []

		print(similarities)
		print("|||")

		print("length of similarities" + str((len(similarities))) + ' ' + str(len(similarities[0])))

		# for gene name, row of probabilities per transcription factor
		for gene_name, row in zip(list(protein_aa_frequencies_dict.keys()), similarities): 
			# print(gene_name)
			temp_row = [gene_name, sum(row)/len(row)]
			scores.append(temp_row)
			raw_scores.append(temp_row[1])

		sorted_list = sorted(scores, key=lambda x: x[1], reverse=True)
		top_100_sorted_list = sorted_list[:100]
		print(top_100_sorted_list)

		end = time.time()

		elapsed = end - start

		print(f"Elapsed time:{elapsed} seconds.")

		plt.hist(raw_scores)
		plt.savefig("distribution.png")
		plt.close()

		table_of_results = """		<table class="table table-hover">
				<thead>
					<th scope="col">Rank</th>
					<th scope="col">Gene Name</th>
					<th scope="col">Score</th>
				</thead>
				<tbody>"""

		table_row_counter = 0
		for row in top_100_sorted_list:
			table_row_counter += 1
			row_string = f"""    <tr>
			<th scope="row">{table_row_counter}</th>
			<td><a target="_blank" href="https://www.ncbi.nlm.nih.gov/gene/?term={row[0]}">{row[0]}</a></td>
			<td>{row[1]}</td>
			</tr>"""
			table_of_results += row_string

		table_of_results = table_of_results + """</tbody>
			</table>"""


		return render_template("hoverable_results.html", table_of_results=table_of_results, elapsed_time=elapsed)

	except KeyError:
		return "You entered a gene that isn't yet included in this database. Please contact Laurence at https://adage.herokuapp.com/contact"


@app.route('/contact')
def contact():
	return render_template("contact.html")

# run app
if __name__ == '__main__':
	app.run()
