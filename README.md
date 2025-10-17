# Faraway count solution

Goal: Count your points at the end of a Farway game.

## Installation

Need uv: [download here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

Then:

```sh
uv venv # to create the virtual env
uv sync # to install synchronize with uv lock
uv run python my_script.py # to run something
```

Download the model:

```sh
# if doesn't work use first:
# chmod 777 download_model.sh 
sh download_model.sh
```

Please follow git convention while committing. See [here](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13).

## Training metrics

To see training metrics, serve tensorboard and go to localhost on your browser:

```sh
uv run tensorboard --logdir rtdetr-v2-r18/
>>> Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
>>> TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

## TODO

- [X] Train a model detecting playing cards (both exploration and sancturary)
- [X] Extract numbers on cards
- [X] Reference all cards
- [ ] In sanctuary, extract :
  - [X] color
  - [ ] night
  - [ ] material
  - [ ] points

## Sanctuaires infos

Position :

- Points : Bas
- Couleur : Bas
- Indice : Haut Gauche
- Matériaux : Haut Droite

Scénarios possibles :

- Un indice + un matériau
- Une couleur + des points
- Un matériau + des points
- Un matériau + une couleur
- Des points
- Un matériau + la nuit
- Un indice + couleur
- Un indice + des points
- Couleur + nuit
