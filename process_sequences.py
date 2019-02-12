"""
Functions used for processing DNA and AA strings
"""

import pandas as pd
import numpy as np
import math

# BioPython
from Bio import Entrez
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio import SeqIO

print("loaded!")

df_loc = pd.read_csv("datasets/mart_gene_locations.txt", delimiter="\t")

def hello():
	print("hello")

def return_chromosome_number_minus_one(gene_name):

	try:
		row = df_loc.loc[df_loc['Gene name'] == gene_name].iloc[0]

		# if chromosome is labeled X
		if row["Chromosome/scaffold name"] == "X":
			chromosome_number = 23
			print("Chromosome X")

		# else if chromosome is labeld Y
		elif row["Chromosome/scaffold name"] == "Y":
			chromosome_number = 24
			print("Chromosome Y")

		# if chromosome is not X or Y, then get chromosome number
		else:
			chromosome_number = int(row["Chromosome/scaffold name"])

		return chromosome_number

	except ValueError:
		print("Invalid character!")
		return -1

	except IndexError:
		print("IndexError!")
		return -1

def one_hot_dna(sequence):
    """Outputs a 2D array based on a one-hot encoding of a string of a DNA ATCG sequence"""
    try:
        return_list = []
        encoding_dict = {'A':0,
                     'T':1,
                     'C':2,
                     'G':3}
        for char in sequence:
            temp_array = np.zeros(4)
            temp_array[encoding_dict[char]] = 1
            return_list.append(temp_array.tolist())
        return return_list
    except KeyError:
        return '!'

def get_promoter_sequence_raw(dict_of_sequences, chromosome, start_index, end_index):
	"""Return raw sequence of gene"""
	start_index_mod = start_index-2000
	end_index_mod = start_index + 500
	raw_string = str(dict_of_sequences[chromosome][start_index_mod:end_index_mod]).upper()	# error: 
	return raw_string

def get_one_hot_gene(dict_of_sequences, chromosome, start_index, end_index):
	"""Get gene and return one hot sequence"""
	start_index_mod = start_index-2000
	end_index_mod = start_index + 500
	raw_string = str(dict_of_sequences[chromosome][start_index_mod:end_index_mod]).upper()	# error: 
	one_hot_string = one_hot_dna(raw_string)
	return one_hot_string

def cosine_similarity(v1,v2):
    """compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"""
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def load_genome(filename="datasets/genomic.fna", fileformat="fasta", parse_first_chromosome_only=False):
	"""Function that loads genome"""

	print(f"Loading genome from {filename} ...")
	dict_of_sequences = {}

	if not parse_first_chromosome_only:
		for seq_record in SeqIO.parse(filename, fileformat):
		    print(seq_record.id)
		    dict_of_sequences[seq_record.id] = seq_record.seq
	else:
		for seq_record in SeqIO.parse(filename, fileformat):
		    print(seq_record.id)
		    dict_of_sequences[seq_record.id] = seq_record.seq			
		    break

	for key in list(dict_of_sequences.keys()):
	    if key[0:2] != "NC":
	        dict_of_sequences.pop(key, None)

	chromosome_keys = dict(enumerate(list(dict_of_sequences.keys())))	# chromsome keys

	return dict_of_sequences, chromosome_keys
