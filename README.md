# MolRetrieve

This repo offers many possible ways to retrieve molecules that are similar to a target molecule from a large molecule library.

- [MolRetrieve](#molretrieve)
  - [TODOs](#todos)
  - [Understanding this repo](#understanding-this-repo)
  - [Getting Started](#getting-started)
  - [Questions or Suggestions?](#questions-or-suggestions)

## TODOs 

- [ ] Clean up the hard-coding and unnecessary files.

- [ ] Support SMILES processing for the corpus.

- [ ] Benchmark (performance, efficiency) different retrieval methods.

- [ ] Support more promising retrieval methods.

## Understanding this repo

A retrieval process consists of three parts:

- **Build molecule corpus**. We need to have a large candidate moelcule library, where we can retreive molecules from. Here we provide the example of using [eMolecules](https://downloads.emolecules.com/free/), which consist of 231M commercially available molecules. You can download the latest version of `version.smi.gz` for all the smiles.
- **Choose an embedding type**. To accelerate the retrieval process, we have to convert each molecule to an embedding first. We provide various choices including SMILE-based fingerprints (`MACCSkeys`, `RDKFingerprint`, `EstateFingerprint`), molecule language models (`ChemBERTa`, `MolT5`, `BioT5`), and graph-based molecule representations (`Grover`, `AttrMask`, `GPT-GNN`, `GraphCL`, `GraphMVP`, `MolCLR`) learned via self-supervised learning. 

- **Choose a distance function**. Once we've got the embeddings of the corpus and our target molecule, we need to choose a distance function to measure the *similarity* of two embeddings. Then the top-k similar molecules from the corpus will be retrieved by searching (we use `heapq.nsmallest`). We provide many distance choices such as `tanimoto`, `dice`, `cosine`, `euclidean`, `sokal`, `russel`, `kulczynski`, `McConnaughey`. Notice that the majority of the distances are for fingerprint-based embeddings. If you are using neural representations, we suggest trying only `cosine` and `euclidean` distances.

## Getting Started

1. **Packages install**. The packages used are as follows (different version may also work).
```bash
tqdm==4.65.0
torch==2.0.0
torchvision==0.15.0
rdkit==2023.9.4
transformers==4.37.1
selfies==2.1.1
networkx==3.1
torch_geometric==2.4.0
torch-cluster==1.6.1+pt20cu117
torch_geometric==2.4.0
torch-scatter==2.1.1+pt20cu117
ogb==1.3.5
```

*Notice:* `ogb==1.3.5` is for SSL-based models likr GraphMVP. Here's an [issue](https://github.com/chao1224/GraphMVP/issues/23) about the version of ogb.

2. **Preprocessing corpus**. Download your molecule corpus from [eMolecules](https://downloads.emolecules.com/free/) in SMILES form. Process it into a line "SMILES size" (such as `COO 3`, `C1CCCCC1 6`) for each molecule, and save it in `SMILES_LIB_PATH` assigned in `utils/env_utils.py`. We will use the code snippet below to load this corpus:

   ```python
   with open(SMILES_LIB_PATH, 'r') as f:
       smiles = [x.strip().split()[0] for x in f.readlines()]
   ```

   The molecule sizes are used for prunning strategy. When retrieving from the corpus, we may sort the corpus first and only search the molecule with similar size. But it's ok not to use prunning.

3. **Prepare model checkpoints**. To run `ChemBERTa`, `BioT5`, `MolT5`, and other SSL-based graph models, you need to download the corresponding checkpoints. When they're done, make sure the paths in  `utils/env_utils.py` is correct.

   1. Download molecule language models: We suggest using `huggingface-cli` (check [this guide](https://huggingface.co/docs/huggingface_hub/guides/download), you need to install the latest `huggingface_hub` for downloading) to download the models. Use the following command (here's an examplar command for ChemBERTa):

      ```bash
      MODEL_DIR="ChemBERTa-77M-MTR"
      HF_PATH="DeepChem/ChemBERTa-77M-MTR"	# for biot5 and molt5, you can choose "laituan245/molt5-base" and "QizhiPei/biot5-base".
      mkdir $MODEL_DIR
      cd $MODEL_DIR
      huggingface-cli download $HF_PATH --local-dir ./
      ```

   2. Download SSL-based models. These models can be downloaded from the repo for [GraphMVP](https://github.com/chao1224/GraphMVP). They can be found in the [Google Drive](https://drive.google.com/drive/folders/1jvJ_n5z7XHouNxiv91gZHrAL5u5JRePY). You can download the corresponding checkpoints for `Grover` (`Motif.pth`), `AttrMask` (`AM.pth`), `GPT-GNN` (`GPT_TNN.pth`), `GraphCL` (`GraphCL.pth`). For `GraphMVP`, you should download the `GraphMVP_complate_features_for_regression.zip` from [here](https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6), where the model is `GraphMVP_complate_features_for_regression/GraphMVP/pretraining_model.pth`. For `MolCLR`, you can download it (`model.pth`) from this [repo](https://github.com/yuyangw/MolCLR/tree/master/ckpt/pretrained_gin/checkpoints). 

4. **Build embedding libs.** We provide code for building embedding library as `build_lib.py` and `retrieve/models/SSL/utils.py` and `retrieve/models/MolCLR/utils.py` (they will be merged together in the future). `build_lib.py` supports the embedding of fingerprints and molecule language models. The lib building may be time-consuming (maybe several hours). You can adjust the `blocksize` and `chunksize` according to your hardware (blocksize is like `batch_size` in ML training, and we will save the embeddings in one file for one chunk). If you don't want to retrieve using some specific embeddings, you can skip the lib building for those types of embedding.

5. **Begin Retrieval!** We'd like to use a `config.yaml` for argument parsing (see `retrieve/configs` for examples). A config file typically consists of the arguments such as `distance_type`, `embedding_type`,`prunning`, `query_path`, `save_path`, `topk`. 

   - The supported distance types are: 

   ```python
   distanceType = (
       'Tanimoto',
       'Dice',
       'Cosine',
       'Euclidean',
       'Sokal',
       'Russel',
       'Kulczynski',
       'McConnaughey',
       'random'
   )
   ```

   - The supported embedding types are:

   ```python
   embeddingTypes = (
       'RDKFingerprint', 
       'MACCSkeys',
       'EStateFingerprint',
       'ChemBERTa',
       'MolT5',
       'BioT5',
       'AttrMask',
       'GPT-GNN',
       'GraphCL',
       'MolCLR',
       'GraphMVP',
       'GROVER',
       'random'
   )
   ```

   - The `query_path` is a txt file that contains all the target molecules, which we want to retrieve similar molecules for them. One line is a SMILES of one molecule.

   - `save_path` is a dir that you want to save the retrieved results.
   - `top_k` is the expected number of retrieved molecules.

## Questions or Suggestions?

This repo is currently a very initial version. If you have any questions or you'd like to contribute to this repo, feel free to email [Haowei](mailto:linhaowei@pku.edu.cn) or just open an issue, or even make a pull request. Welcome contribute to this repo to make it more helpful!
