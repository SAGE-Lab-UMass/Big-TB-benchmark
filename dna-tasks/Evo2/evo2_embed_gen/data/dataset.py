"""DNABERT-compatible data preparation for Evo2 embedding generation.

This module intentionally mirrors the DNABERT2 pipeline:
1. Read the same aligned per-locus FASTA files in a fixed locus order.
2. Join genotypes to the same phenotype table by isolate ID.
3. Drop isolates with missing sequence data and isolates missing all drug labels.
4. Return batches without shuffling so embedding rows keep deterministic order.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from evo2_embed_gen.data.locus_order import DRUGS, locus_order

csv.field_size_limit(sys.maxsize)


RESISTANCE_CATEGORIES = {"R": 0, "S": 1, "-1.0": -1, "-1": -1, -1: -1}


@dataclass(frozen=True)
class DataConfig:
    datapath: str
    full_dataname: str
    train_dataname: str
    val_dataname: str
    phenotype_file: str
    genotype_input_directory: str
    test_split: float
    full_batch_size: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    max_isolates: int | None = None


class MultigeneMultidrugSamples(Dataset):
    def __init__(
        self,
        isolate_ids: list[str],
        sequences: list[list[str]],
        phenotypes: list[list[int]],
        gene_names: list[str],
        drug_names: list[str],
    ) -> None:
        self.isolate_ids = isolate_ids
        self.sequences = sequences
        self.phenotypes = phenotypes
        self.gene_names = gene_names
        self.drug_names = drug_names

    def __len__(self) -> int:
        return len(self.isolate_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "isolate_id": self.isolate_ids[idx],
            "gene_order": self.gene_names,
            "drug_order": self.drug_names,
        }
        item.update({f"gene_seq_{i + 1}": gene[idx] for i, gene in enumerate(self.sequences)})
        item.update(
            {
                f"res_phenotype_drug_{i + 1}": self.phenotypes[i][idx]
                for i in range(len(self.phenotypes))
            }
        )
        return item


def make_loader(config: DataConfig, is_single_gene_algo: bool, load_train: bool, n_gpu: int) -> DataLoader:
    create_multidrug_classification_data(config, is_single_gene_algo)

    if is_single_gene_algo:
        csv_filename = config.full_dataname
        batch_size = config.full_batch_size
    else:
        csv_filename = config.train_dataname if load_train else config.val_dataname
        batch_size = config.train_batch_size if load_train else config.val_batch_size

    csv_path = Path(config.datapath) / csv_filename
    print(f"Loading data from {csv_path}")
    with csv_path.open(newline="") as csvfile:
        reader = list(csv.reader(csvfile, delimiter=","))

    headers = reader[0]
    rows = reader[1:]
    if config.max_isolates is not None:
        rows = rows[: config.max_isolates]
    isolate_ids = [row[0] for row in rows]

    fasta_columns = [i for i, header in enumerate(headers) if header.endswith(".fasta")]
    gene_names = [headers[i] for i in fasta_columns]
    sequences = [[row[i] for row in rows] for i in fasta_columns]

    drug_columns = [i for i, header in enumerate(headers) if header in DRUGS]
    drug_names = [headers[i] for i in drug_columns]
    sorted_indices = np.argsort([DRUGS.index(name) for name in drug_names])
    drug_names = [drug_names[i] for i in sorted_indices]
    drug_columns = [drug_columns[i] for i in sorted_indices]

    phenotypes_raw = [[row[i] for row in rows] for i in drug_columns]
    phenotypes = [[RESISTANCE_CATEGORIES[value] for value in drug] for drug in phenotypes_raw]

    print(f"Number of isolates: {len(isolate_ids)}")
    print(f"Number of genes: {len(gene_names)}")
    print(f"Gene order: {gene_names}")
    print(f"Drug order: {drug_names}")

    dataset = MultigeneMultidrugSamples(
        isolate_ids=isolate_ids,
        sequences=sequences,
        phenotypes=phenotypes,
        gene_names=gene_names,
        drug_names=drug_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size * max(n_gpu, 1),
        shuffle=False,
        num_workers=config.num_workers * max(n_gpu, 1),
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
    )


def collate_samples(samples: list[dict]) -> dict:
    batch = {
        "isolate_id": [sample["isolate_id"] for sample in samples],
        "gene_order": samples[0]["gene_order"],
        "drug_order": samples[0]["drug_order"],
    }

    gene_keys = sorted(
        (key for key in samples[0] if key.startswith("gene_seq_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    drug_keys = sorted(
        (key for key in samples[0] if key.startswith("res_phenotype_drug_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )

    for key in gene_keys:
        batch[key] = [sample[key] for sample in samples]
    for key in drug_keys:
        batch[key] = torch.tensor([sample[key] for sample in samples], dtype=torch.long)

    return batch


def create_multidrug_classification_data(config: DataConfig, is_single_gene_algo: bool) -> pd.DataFrame:
    geno_pheno_df = create_genotype_phenotype_csv(config)
    split_data_into_train_val_sets(config, geno_pheno_df, is_single_gene_algo)
    return geno_pheno_df


def create_genotype_phenotype_csv(config: DataConfig) -> pd.DataFrame:
    Path(config.datapath).mkdir(parents=True, exist_ok=True)
    data_path = Path(config.datapath) / config.full_dataname

    if data_path.is_file():
        print(f"Genotype-phenotype CSV already exists: {data_path}")
    else:
        print("Creating genotype-phenotype CSV from FASTA and phenotype inputs")
        make_geno_pheno_csv(config, data_path)

    print(f"Reading genotype-phenotype CSV: {data_path}")
    return pd.read_csv(data_path, delimiter=",")


def make_geno_pheno_csv(config: DataConfig, output_path: Path, index_col: str = "New_ID") -> None:
    df_phenos = pd.read_csv(config.phenotype_file, index_col=index_col, sep=",", dtype=str).fillna(-1)
    df_genos = make_genotype_df(config.genotype_input_directory)

    isolate_ids = list(df_phenos.index)
    df_genos.index = df_genos.index.astype(str)
    df_genos = df_genos.loc[df_genos.index.intersection(isolate_ids)]
    df_genos = df_genos.dropna(axis="index")

    print(f"Genotype table shape after phenotype intersection/dropna: {df_genos.shape}")
    df_geno_pheno_full = df_genos.join(df_phenos, how="inner")
    df_geno_pheno_full.to_csv(output_path)
    print(f"Wrote {output_path}")


def make_genotype_df(genotype_input_directory: str) -> pd.DataFrame:
    dfs_list = []
    for locus in locus_order:
        pattern = f"{genotype_input_directory}/{locus}*.fasta"
        fasta_files = glob.glob(pattern)
        if len(fasta_files) != 1:
            raise ValueError(f"Expected exactly one FASTA for {locus}; found {fasta_files}")
        print(f"Reading locus {locus}: {fasta_files[0]}")
        dfs_list.append(sequence_dictionary(fasta_files[0]))

    df_genos = dfs_list[0].join(dfs_list[1:], how="outer")
    print(f"Combined genotype table shape: {df_genos.shape}")
    return df_genos


def sequence_dictionary(filename: str) -> pd.DataFrame:
    seq_dict = SeqIO.to_dict(
        SeqIO.parse(filename, "fasta"),
        key_function=lambda record: record.id.split("/")[-1].split(".cut")[0],
    )
    sequences = {str(identifier): str(sequence.seq) for identifier, sequence in seq_dict.items()}

    df = pd.DataFrame.from_dict(sequences, orient="index")
    gene_name = Path(filename).name.split("_")[0]
    df.columns = [gene_name if gene_name.endswith(".fasta") else f"{gene_name}.fasta"]
    return df


def split_data_into_train_val_sets(
    config: DataConfig,
    geno_pheno_df: pd.DataFrame,
    is_single_gene_algo: bool,
) -> None:
    geno_pheno_df = geno_pheno_df.reset_index(drop=True)
    geno_pheno_data = geno_pheno_df[geno_pheno_df[DRUGS].apply(lambda row: (row != -1).any(), axis=1)]

    if is_single_gene_algo:
        output_path = Path(config.datapath) / config.full_dataname
        geno_pheno_data.to_csv(output_path, index=False)
        print(f"Single-gene mode: wrote complete filtered data to {output_path}")
        print(f"Number of isolates: {len(geno_pheno_data)}")
        return

    all_indices = geno_pheno_df.index
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=config.test_split,
        random_state=42,
    )
    train_data = geno_pheno_df.loc[train_indices]
    val_data = geno_pheno_df.loc[val_indices]

    train_data = train_data[train_data[DRUGS].apply(lambda row: (row != -1).any(), axis=1)]
    val_data = val_data[val_data[DRUGS].apply(lambda row: (row != -1).any(), axis=1)]

    train_path = Path(config.datapath) / config.train_dataname
    val_path = Path(config.datapath) / config.val_dataname
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    print(f"Wrote train data to {train_path} ({len(train_data)} isolates)")
    print(f"Wrote validation data to {val_path} ({len(val_data)} isolates)")


def parse_gene_list(genes: str | Iterable[str]) -> list[str]:
    if isinstance(genes, str):
        return [gene.strip() for gene in genes.split(",") if gene.strip()]
    return [str(gene).strip() for gene in genes if str(gene).strip()]
