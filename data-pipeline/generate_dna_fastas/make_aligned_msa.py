import argparse, subprocess, glob, os, vcf, tracemalloc, pickle, time
import numpy as np
import pandas as pd
from Bio import SeqIO

# starting the memory monitoring
tracemalloc.start()


#################################### STEP 0: READ IN FILES AND INITIALIZE VARIABLES ####################################
    
    
######## IMPORTANT: START is 0-indexed, END is 1-indexed (i.e., 0-indexed half-open) to be consistent with other bioinformatics tools ########
        

parser = argparse.ArgumentParser()
# Add a required string argument for the paths file
parser.add_argument("-f", type=str, dest='PATHS_FILE', help='Text file of VCF paths to include in alignment', required=True)

# dest indicates the name that each argument is stored in so that you can access it after running .parse_args()
parser.add_argument('-start', type=int, dest='START', help='Start coordinate for alignment (1-indexed, inclusive)', required=True)
parser.add_argument('-end', type=int, dest='END', help='End coordinate for alignment (1-indexed, exlusive)', required=True)
parser.add_argument('-sense', type=str, dest='SENSE', help='Sense, must be one of pos, neg, POS, NEG', required=True)
parser.add_argument('-o', type=str, dest='OUT_FILE', help='Name of the output FASTA file', required=True)
parser.add_argument('-g', type=str, dest="GENOME_FILE", help="Full path to Genbank file for the reference genome", required=True)

parser.add_argument('--gene', type=str, dest='GENE', help='Gene name for the region of interest')
parser.add_argument('--f2', type=str, dest='ADDITIONAL_ISOLATES_FILE', help='Text file of additional VCFs to include in alignment')
parser.add_argument('--insilico-muts-file', dest='INSILICO_MUTS_FILE', type=str, help='Text file of insilico mutations to include in alignment')
parser.add_argument('--AF_thresh', dest='AF_THRESH', type=float, default=0.75, help='Alternative allele frequency threshold (exclusive) to consider variants present')
parser.add_argument('--resistance-groups', dest='RESISTANCE_GRPS', type=str, help='Category of resistance groups for selecting corresponding genes')

cmd_line_args = parser.parse_args()

# required arguments
PATHS_FILE = cmd_line_args.PATHS_FILE
START = cmd_line_args.START - 1
END = cmd_line_args.END
SENSE = cmd_line_args.SENSE
OUT_FILE = cmd_line_args.OUT_FILE
GENOME_FILE = cmd_line_args.GENOME_FILE

# optional arguments
GENE = cmd_line_args.GENE
ADDITIONAL_ISOLATES_FILE = cmd_line_args.ADDITIONAL_ISOLATES_FILE
INSILICO_MUTS_FILE = cmd_line_args.INSILICO_MUTS_FILE
AF_thresh = cmd_line_args.AF_THRESH # variants with an AF > AF_thresh are considered present. AF ≤ 1 - AF_thresh is absent. Everything else is missing
RESISTANCE_GRPS = cmd_line_args.RESISTANCE_GRPS

if START >= END:
    raise ValueError(f"START coordinate must be less than END coordinate. You passed in START = {START} and END = {END}")

SENSE = SENSE.upper()

if SENSE not in ["POS", "NEG", "pos", "neg"]:
    raise ValueError(f"SENSE argument must be one of pos, neg, POS, NEG. You passed in {SENSE}")

# must be a float less than or equal to 1
if AF_thresh > 1:
    AF_thresh /= 100

if not os.path.isfile(PATHS_FILE):
    raise ValueError(f"{PATHS_FILE} is not a file!")
    
# PATHS_FILE should be a text file of paths
if PATHS_FILE[-4:] != ".txt":
    raise ValueError(f"{PATHS_FILE} must be a text file!")

add_paths = []

if ADDITIONAL_ISOLATES_FILE is not None:
    
    if not os.path.isfile(ADDITIONAL_ISOLATES_FILE):
        raise ValueError(f"{ADDITIONAL_ISOLATES_FILE} is not a file!")
    
    # PATHS_FILE should be a text file of paths
    if ADDITIONAL_ISOLATES_FILE[-4:] != ".txt":
        raise ValueError(f"{ADDITIONAL_ISOLATES_FILE} must be a text file!")
    
    add_paths += list(pd.read_csv(ADDITIONAL_ISOLATES_FILE, sep="\t", header=None)[0].values)

if INSILICO_MUTS_FILE is not None:

    if not os.path.isfile(INSILICO_MUTS_FILE):
        raise ValueError(f"{INSILICO_MUTS_FILE} is not a file!")
    
    # PATHS_FILE should be a text file of paths
    if INSILICO_MUTS_FILE[-4:] != ".txt":
        raise ValueError(f"{INSILICO_MUTS_FILE} must be a text file!")
    
    add_paths += list(pd.read_csv(INSILICO_MUTS_FILE, sep="\t", header=None)[0].values)

paths = pd.read_csv(PATHS_FILE, sep="\t", header=None)[0].values
print(f"Making multiple sequence alignment for {len(paths)} sequences and {len(add_paths)} additional sequences")
paths = np.concatenate([paths, np.array(add_paths)], axis=0)

if ".fasta" not in OUT_FILE:
    OUT_FILE = OUT_FILE.split(".")[0] + ".fasta"
    
print(f"Output file: {OUT_FILE}")
    
# Define the output file with a full path
# OUT_FILE = 'results/output.fasta'  # Example path

# Create the directory if it does not exist
output_dir = os.path.dirname(OUT_FILE)
if output_dir and not os.path.isdir(output_dir):  # Check if output_dir is not empty
    os.makedirs(output_dir)

# reference genome
h37Rv = SeqIO.read(GENOME_FILE, "genbank")
genome_len = len(h37Rv)
print(f"Reference genome size: {genome_len}")

# get only the region of interest
h37Rv_region = list(str(h37Rv.seq[START:END]))
print(f"Unaligned region size: {len(h37Rv_region)}")
del h37Rv

def reverse_complement(seq):
    
    comp_dict = {'A': 'T', 
                 'C': 'G', 
                 'G': 'C', 
                 'T': 'A', 
                 'N': 'N', 
                 '-': '-'
                }
    
    # this is to turn it into a list where each element is of length 1
    seq = list("".join(seq))
    
    if len(np.unique(seq)) > 6:
        raise ValueError(f"More than 6 types of characters in the sequence!")

    if "X" in np.unique(seq):
        raise ValueError(f"There are Xs in the sequence!")
        
    seq = [comp_dict[base] for base in seq] 
    
    # reverse the sequence and return as a list
    return "".join(seq[::-1])
    
    
    
def allele_category(record, qualThresh=10, presentThresh=0.75):
    '''
    Returns "alt" or "ref" if the variant is low-quality or ambiguous. Otherwise this function returns "missing"
    
    Low-quality criteria:
    
        1. FILTER == Del, LowCov
        2. FILTER == Amb and 0.25 < AF <=0.75
        3. SNP quality < 10

    Criteria for not confident in a variant or can not be reliably inserted, so leave it as reference:

        1. IMPRECISE variant (in the INFO field)
        2. Indels longer than 15 bp where neither the REF nor the ALT are of length 1 (this case is handled in the next function)
        
    If FILTER contains Amb and the alternative allele fraction > 0.9, then it is a pure alternative call. 
    If FILTER contains Amb and the alternative allele fraction < 0.1, the it is a pure reference call. 
    '''

    alt_allele = "".join(np.array(record.ALT).astype(str))
    
    # fill in things that might be missing
    if "AF" not in record.INFO.keys():
        af = 0.76
    else:
        af_lst = record.INFO["AF"]
        
        # more than one alternative allele, which shouldn't happen for haploid organisms
        if len(af_lst) > 1:
            return "missing"
        else:
            af = float(af_lst[0])

    # QUAL field considers read depth, base quality, mapping quality. But it is also on the Phred scale
    if record.QUAL is None:
        qual = 11
    else:
        qual = record.QUAL
        
    # don't include IMPRECISE variants because they are difficult to reliably impute and often aren't reliable calls anyway
    # unreliability can be due to ambiguous alignments, complex genomic regions, low sequencing coverage, assembly gaps, or segmental duplications
    # basically these are breakpoints that the variant caller is not confident in. If we put Ns, often we get huge runs of Ns, which causes too much noise for the model.
    # pilon was not able to resolve the variants (usually due to large deletions), so leave as reference because we don't know what the variant is with high confidence

    if "IMPRECISE" in record.INFO.keys():
        return "ref"
    
    # the filter field is an empty list of it is PASS, else the list is non-empty
    # only consider the non-Amb cases here. Amb cases will be later, check the AF too for that
    if len(record.FILTER) > 0 and "Amb" not in record.FILTER:
        return "missing"
    
    # because IMPRECISE is taken care of above, this should only return missing for cases where REF = N or ALT = N
    if "N" in record.REF or "N" in alt_allele:
        return "missing"
    
    # check if there are any non alphanumeric characters. This would indicate a heterogeneous alternative allele
    if not alt_allele.isalnum():
        return "missing"

    # low SNP quality
    if qual < qualThresh:
        return "missing"

    # base quality, mapping quality, and read depth (measures of certainty about a variant)
    if 'DP' in record.INFO.keys():
        if record.INFO['DP'] < 5:
            return 'missing'

    if 'MQ' in record.INFO.keys():
        if record.INFO['MQ'] < 30:
            return 'missing'

    # base quality is 0 for indels, so include this step for only SNPs and MNPs (lengths are the same for REF and ALT)
    if len(record.REF) == len(alt_allele) and 'BQ' in record.INFO.keys():
        if record.INFO['BQ'] < 20:
            return 'missing'

    # PASS or Amb filters and an alternative allele fraction of less than 0.75 means we have a mixture of REF and ALT
    if "AF" in record.INFO.keys():
        # ≤ 25%, absent
        if af <= (1 - presentThresh):
            return "ref"
        elif af > presentThresh:
            return "alt"
        else:
            return "missing"
        
    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF <= 0.25
    return "alt"



def introduce_snps_indels_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=0.75):
    
    new_seq = h37Rv_region.copy()
    
    # create the tabix file if it doesn't exist
    if not os.path.isfile(f"{fName}.bgz.tbi"):

        print(f"Creating tabix file for {fName}")

        # bgzip the VCF file
        if not os.path.isfile(f"{fName}.bgz"):
            subprocess.run(f"bgzip -c {fName} > {fName}.bgz", shell=True)

        # tabix the bgzipped file, which will create fName.bgz.tbi
        subprocess.run(f"tabix -0 -p vcf {fName}.bgz -f", shell=True)

    # VCF file was indexed using 0-indexed half-open scheme (-0 flag above), so keep START and END coords as they are
    vcf_reader = vcf.Reader(filename=f"{fName}.bgz", compressed=True)

    print(f"Reading in {fName}.bgz")

    # need to read in the bgzipped file in order to use fetch because the tabix file is compatible only with the bgzipped format
    records = vcf_reader.fetch('NC_000962.3', start=START, end=END)

    # start is 0-indexed (exclusive) and end is 1-indexed (inclusive)
    # don't need to exclude any records based on position because already did region subsetting when reading in the VCF
    for record in records:

        # convert alternative allele from list to string
        alt_allele = "".join(np.array(record.ALT).astype(str))
        ref_allele = str(record.REF)

        # get the allele type: ref, alt, or missing
        single_allele_type = allele_category(record, qualThresh, presentThresh) 

        # only change the sequence if the type is not reference
        if single_allele_type != "ref":

            # the index to replace -- this is 0-indexed, consistent with Python
            idx = record.POS - (START + 1)

            # no length change -- SNP or MNP. Python will replace all elements if the original and new are the same length
            if len(ref_allele) == len(alt_allele):

                if single_allele_type == "alt":
                    new_seq[idx:idx+len(ref_allele)] = alt_allele
                elif single_allele_type == "missing":
                    new_seq[idx:idx+len(ref_allele)] = ["N"]*len(alt_allele)
                # the only other option is reference, so don't do anything
                else:
                    continue

            # indels
            else:
                
                # insertion -- insert both high- and low-quality insertions
                if len(alt_allele) > len(ref_allele):
                    
                    # replace the nucleotide at the reference index with the alternative nucleotides
                    # also add the number of gap characters needed (len(ALT) - len(REF) to insertion_dict at the appropriate index                        
                    # only input insertions up to 100 bp and also if they pass the QC filters. Leave the others as reference to avoid introducing long runs of Ns into the aln
                    if single_allele_type == "alt" or (single_allele_type == "missing" and (len(alt_allele) - len(ref_allele) <= 100)):
                            
                        if single_allele_type == "missing":
                            alt_allele = "N"*len(alt_allele)

                        # if REF > 1, then the entire REF allele must be removed (across all positions) and replaced with the ALT allele
                        # do this with a dummy character, X, which will be later removed. 
                        # This is generalizable to even the case where REF == 1 because it will just replace the first index

                        # replace everything with X first
                        new_seq[idx:idx+len(ref_allele)] = ["X"] * len(ref_allele)

                        # then make the first position the alternative allele
                        new_seq[idx] = alt_allele

                        # stores the maximum length of insertions at the corresponding index
                        insertion_dict[idx] = np.max([insertion_dict[idx], len(alt_allele) - len(ref_allele)])
    
                    # don't do anything if indels are missing and very long
                    else:
                        print(fName, record.POS, record) # print for information purposes
                        continue

                # deletion -- include only if they are high quality. Leave low-quality deletions as reference
                else:

                    if single_allele_type == "alt":
                        # case when the deletion starts outside the region of interest, but overlaps with the region
                        if record.POS <= START or record.POS > END:

                            # the last position (inclusive) that is deleted
                            end_deletion_pos = record.POS + len(ref_allele) - 1

                            # the ENTIRE region is deleted because the last deleted position is downstream of the last coordinate of the region
                            if end_deletion_pos > END:
                                new_seq = ["-"] * len(h37Rv_region)
                            else:
                                # everything from START to end_deletion_pos should become a gap character
                                # don't add 1 even though the second coordinate is exclusive because START is already 0-indexed, so it reduces the distance by 1 already
                                # can't replace multiple elements with a single character
                                new_seq[:end_deletion_pos - START] = ["-"]*(end_deletion_pos - START)

                        # deletion is fully contained within the region of interest
                        else:
                            # the replacement is the alternative allele padded with gap characters. # of gap characters = the length difference between them 
                            new_allele = list(alt_allele) + ['-'] * (len(ref_allele) - len(alt_allele))
                            assert len(new_allele) == len(ref_allele)
    
                            # Python will replace all elements if the original and replace string are the same length
                            # old_len is the allowed region length, which is the length of the region of interest
                            old_len = len(new_seq)
                            new_seq[idx:idx+len(ref_allele)] = new_allele

                            # add this step so that if the allele extends more than the region of interest, it is truncated. This is for large deletions
                            new_seq = new_seq[:old_len]
                            
                        
    # check lengths because both of them are lists right now 
    print(f"Length of new sequence: {len(new_seq)}")
    print(f"Length of h37Rv_region: {len(h37Rv_region)}")
    assert len(new_seq) == len(h37Rv_region)
    return new_seq


#################################### STEP 1: GET SNPS AND INDELS AND INSERT INTO EACH SEQUENCE USING THE FUNCTION ABOVE ####################################

# keep track of positions and the numbers of insertions relative to H37Rv
# after this main loop, these need to be introduced into h37Rv_region and also into 
global insertion_dict
insertion_dict = dict(zip(np.arange(0, END-START), np.zeros(END-START)))
print(f"Considering AFs > {AF_thresh} as present")

start_time = time.time()
seq_dict = {}

for i, fName in enumerate(paths):
    
    seq_dict[os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", "")] = introduce_snps_indels_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=AF_thresh)

    if (i+1) % 1000 == 0:
        print(f"{i+1} sequences read yet")

print(f"Finished reading {len(seq_dict)} sequences!")

# Create the directory if it doesn't exist
insertion_sites_dirName = os.path.join(os.path.dirname(OUT_FILE), "sites", GENE)
os.makedirs(insertion_sites_dirName, exist_ok=True)

seq_dict_fName = os.path.join(insertion_sites_dirName, f"{os.path.basename(OUT_FILE).split('.')[0]}_seq_dict.pkl")
pd.DataFrame(seq_dict).to_pickle(seq_dict_fName)

# convert to dataframe for easy querying. Convert everything to integers and keep only indices where gap characters need to be inserted (num_insertion > 0)
insertion_sites = pd.DataFrame(insertion_dict, index=[0]).T.reset_index().rename(columns={"index": "aln_idx", 0:"len_insertion"})
insertion_sites = insertion_sites.query("len_insertion > 0").reset_index(drop=True)
insertion_sites[insertion_sites.columns] = insertion_sites[insertion_sites.columns].astype(int)


insertion_sites_fName = os.path.join(insertion_sites_dirName, f"{os.path.basename(OUT_FILE).split('.')[0]}_insertion_sites.csv")

print(f"Saving dataframe of sites with insertions to {insertion_sites_fName}")
insertion_sites.to_csv(insertion_sites_fName, index=False)
    

#################################### STEP 2: FILL IN GAP CHARACTERS IN THE REFERENCE SEQUENCE ####################################

    
new_ref_seq = h37Rv_region.copy()

for _, row in insertion_sites.iterrows():
    
    # number of gap characters to add
    add_gap = row["len_insertion"]

    new_ref_seq[row["aln_idx"]] = new_ref_seq[row["aln_idx"]] + "-" * add_gap
        
# get the reverse complement if negative sense. This function returns the joined sequence. If not, 
if SENSE in ["NEG", "neg"]:
    new_ref_seq = reverse_complement(new_ref_seq)
else:
    new_ref_seq = "".join(new_ref_seq)

print(f"Aligned region size: {len(new_ref_seq)}")
    
    
#################################### STEP 3: FILL IN GAP CHARACTERS IN ALL SEQUENCES AND WRITE TO THE OUTPUT FILE ####################################


print(f"Writing aligned sequences to {OUT_FILE}")

with open(OUT_FILE, "w+") as file:
    # write the reference sequence 
    file.write(">MT_H37Rv\n")
    
    # get the reverse complement if negative sense
    # if SENSE in ["NEG", "neg"]:
    #     new_ref_seq = reverse_complement(new_ref_seq)
        
    file.write(new_ref_seq + "\n")

    for isolate, seq in seq_dict.items():
        print(f"Writing {isolate} to the alignment file")

        assert len(seq) == (END - START)

        for _, row in insertion_sites.iterrows():

            # the numbers in insertion_sites are the number of nucleotides to insert, i.e. len(alt) - len(ref)
            # for nearly all insertions, the length of the REF allele is 1 so add 1 to this.
            site_length = row["len_insertion"] + 1

            # if the length of the site in a given isolate is smaller than the number of inserted nucleotides, then pad the end with gap characters
            if len(seq[row["aln_idx"]]) < site_length:
                seq[row["aln_idx"]] = seq[row["aln_idx"]] + "-" * int(site_length - len(seq[row["aln_idx"]]))

            # these are the few cases where len(REF) > 1, so they would fail a check here. But the final sequence length will be checked against the reference below
            elif len(seq[row["aln_idx"]]) > site_length:
                continue
                
        # remove X characters, which are used for some insertions
        # check that the new length matches with the reference sequence, which has already had gap characters inserted
        seq = "".join(seq).replace("X", "")
        print("Length of sequence after removing Xs:", len(seq))
        assert len(seq) == len(new_ref_seq)
    
        # get the reverse complement if negative sense
        if SENSE in ["NEG", "neg"]:
            seq = reverse_complement(seq)
        
        # write the new sequence to the alignment file
        file.write(">" + isolate + "\n")
        file.write(seq + "\n")
        

# read in and check if all sequences are identical
seq_df = [(seq.id, seq.seq) for seq in SeqIO.parse(OUT_FILE, "fasta")]
seq_df = pd.DataFrame(seq_df)
seq_df.columns = ['Isolate', 'Seq']

#fix the input sequence 
if seq_df['Seq'].nunique() == 1:
    # Create the directory if it doesn't exist
    results_dirName = os.path.join(os.path.dirname(OUT_FILE), "logs")
    os.makedirs(results_dirName, exist_ok=True)
    results_fName = os.path.join(results_dirName, "identical_sequences_log.txt")

    # Log the output file name into a separate file
    with open(results_fName, "a") as log_file:
        log_file.write(f"{OUT_FILE}\n")

    raise Warning(f"All sequences in {OUT_FILE} are identical! Please exclude this locus from downstream models")

# Script runtime
end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = total_time % 60
print(f"Total script runtime: {minutes} min {seconds:.2f} sec")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Total memory used: {script_memory} GB\n")

print(f"Total files: {len(seq_df)}")