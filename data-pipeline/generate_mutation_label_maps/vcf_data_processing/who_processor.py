def process_who_data(who_df):
    """Processes WHO variant data to ensure correct types."""
    who_df['genome_index'] = who_df['genome_index'].astype(int)
    return who_df
