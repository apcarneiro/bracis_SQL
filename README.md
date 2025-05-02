# Somatic Q-Learning Experiment (BRACIS Paper)

This repository contains the source codes of the experiment cited in the article "Somatic Q-Learning" presented at the BRACIS 2025 congress (https://bracis.sbc.org.br/2025/).

To create environment:

```
conda env create -y --file environment.yml
```

To activate environment:

```
conda activate gymnasiumSRL
```

The file "FrozenLake.ipynb" contains the first part of the experiment, with the training of the proposed agents.

The file "FrozenLake_2.ipynb" contains the second part of the experiment with the already trained agents running in different environments.