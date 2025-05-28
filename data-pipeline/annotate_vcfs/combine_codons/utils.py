import numpy as np

def merge_entries(combined_entry, record, coords_of_codon, h37Rv):
    """
    Merges two VCF records that belong to the same codon.

    Parameters:
        combined_entry (vcf.model._Record): The combined VCF record so far.
        record (vcf.model._Record): The new record to merge.
        coords_of_codon (list): List of coordinates in the codon.

    Returns:
        combined_entry (vcf.model._Record): The updated combined entry.
    """
    combined_entry = combine_ref_and_alt(combined_entry, record, coords_of_codon, h37Rv)

    # Regardless of whether the records are consecutive or not, update the position of the combined entry to the most upstream (smallest) position among the variants.
    combined_entry.POS = min(combined_entry.POS, record.POS)
    combined_entry.QUAL = combine_qual(combined_entry.QUAL, record.QUAL)
    combined_entry.FILTER = combine_filters(combined_entry.FILTER, record.FILTER)
    combined_entry.INFO.update(combine_info(combined_entry.INFO, record.INFO))
    return combined_entry


def combine_ref_and_alt(combined_entry, record, coords_of_codon, h37Rv):
    """
    Combines REF and ALT fields for two VCF records.

    Parameters:
        combined_entry (vcf.model._Record): The combined VCF record so far.
        record (vcf.model._Record): The new record to merge.
        coords_of_codon (list): List of coordinates in the codon.

    Returns:
        combined_entry (vcf.model._Record): The updated record with merged REF and ALT.
    """

    # consecutive case is easy
    # OR IF COMBINED_ENTRY.REF AND COMBINED_ENTRY.ALT ARE LONGER THAN 1, THAT MEANS THAT THERE IS A VARIANT AT ALL THREE SITES OF THE CODON            
    if abs(combined_entry.POS - record.POS) == 1 or (len(combined_entry.REF) > 1 and len(combined_entry.ALT[0]) > 1):
        combined_entry.REF += record.REF
        combined_entry.ALT = ["".join(map(str, combined_entry.ALT)) + "".join(map(str, record.ALT))]
    
    # if they are not consecutive records, then need to insert reference nucleotides to the REF and ALT fields, but don't need to change any of the other fields because it's not a variant
    else:
        # get all nucleotides in the codon coordinates list that are not in either the combined_entry or current record
        missing_coords = list(set(coords_of_codon) - {combined_entry.POS, record.POS})

        # Create a dictionary mapping missing codon coordinates to their reference nucleotides (adjusting for 1-based to -> 0-based indexing)
        fill_in_dict = {coord: str(h37Rv.seq)[coord - 1] for coord in missing_coords}
        fill_in_dict.update({combined_entry.POS: combined_entry.REF, record.POS: record.REF})

        combined_entry.REF = "".join(fill_in_dict[pos] for pos in coords_of_codon)
        combined_entry.ALT = ["".join(
                                "".join(map(str, record.ALT)) if pos == record.POS else
                                "".join(map(str, combined_entry.ALT)) if pos == combined_entry.POS else
                                fill_in_dict[pos]
                                for pos in coords_of_codon
                            )]

    return combined_entry


def combine_qual(qual1, qual2):
    """
    Combines quality scores by taking their mean.

    Parameters:
        qual1 (float): Quality score of the first entry.
        qual2 (float): Quality score of the second entry.

    Returns:
        float: Combined quality score.
    """
    if qual1 is not None and qual2 is not None:
        return int(round(np.mean([qual1, qual2])))
    return qual1 if qual1 is not None else qual2


def combine_filters(filter1, filter2):
    """
    Combines FILTER fields for two VCF records.

    Parameters:
        filter1 (list): FILTER field of the first entry.
        filter2 (list): FILTER field of the second entry.

    Returns:
        list: Combined FILTER field.
    """

    # Don't need to do anything if the filters are the same
    if filter1 == filter2:
        return filter1
    
    # If the combined entry is PASS and the new entry is not PASS, then keep the new one
    if len(filter1) == 0:
        return filter2
    if len(filter2) == 0:
        return filter1
    
    # FILTER field needs to be a list
    combined_filter = sorted(set(filter1 + filter2))

    # Options are Amb, Del, and LowCov. Sort, then take the last one, so LowCov is preferentially
    # kept over Del, which is kept over Amb. The sorted version is in order of decreasing quality,
    # so keep the last one (the lowest quality indicator).
    return [combined_filter[-1]]  


def combine_info(info1, info2):
    """
    Combines INFO fields for two VCF records.

    Parameters:
        info1 (dict): INFO field of the first entry.
        info2 (dict): INFO field of the second entry.

    Returns:
        dict: Combined INFO field.
    """
    updated_info = {}
    for key in set(info1.keys()).union(info2.keys()):
        if key in info1 and key in info2:
            if isinstance(info1[key], list):
                updated_info[key] = list(np.array(info1[key]) + np.array(info2[key]))
            else:
                updated_info[key] = np.round(np.mean([info1[key], info2[key]]), 2)
        else:
            updated_info[key] = info1.get(key, info2.get(key))
    return updated_info
