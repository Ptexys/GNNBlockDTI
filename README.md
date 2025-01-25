# GNNBlockDTI

![GNNBlockDTI](https://github.com/Ptexys/GNNBlockDTI/blob/main/GNNBlockDTI.jpg)

## Requirement
torch --2.0.0+ <br>
dgl --1.1.2+ <br>
rdkit <br>

## data process
### full_data.pkl : [Drugbank ID, Uniprot ID, label]
### drug_smi_raw.pkl : {Drugbank ID: SMILES}
### prot_seq_raw.pkl : {Uniprot ID: AAs}
### drug_graph.ipynb --create drug graph representation: {Drugbank ID: DGLgraph}
### protein_sequence.ipynb --create protein sequence representation: {Uniprot ID: vectors}
### protein_graph.ipynb --create protein graph representation: {Uniprot ID: DGLgraph}
