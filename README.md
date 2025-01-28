## GNNBlockDTI
![GNNBlockDTI](https://github.com/Ptexys/GNNBlockDTI/blob/main/GNNBlockDTI.jpg)

### Requirement
- `torch --2.0.0+` <br>
- `dgl --1.1.2+` <br>
- `rdkit` <br>
- `esm`: [esm](https://github.com/facebookresearch/esm)  <br>
- `protbert`: [prot-bert](https://github.com/agemagician/ProtTrans)  <br>

## Data preprocessing
- Run `python data_process.py`
- `full_data.pkl`    : [Drugbank ID, Uniprot ID, label] <br>
- `drug_smi_raw.pkl` : {Drugbank ID: SMILES} <br>
- `prot_seq_raw.pkl` : {Uniprot ID: AAs} <br>

## Training for a new model
- Run `python main.py`

## Testing with a pre-training model
- Run `python test.py`
