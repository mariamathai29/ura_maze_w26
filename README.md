# URA Maze Project, Winter 2026 Updates

> Acknowledgement: This project was done with the University of Waterloo's Vision and Image Processing lab and under the supervision of Yuhao Chen. Original codebase developed by Kellen Sun (kellen-sun).

This project looks into mazes (a matrix of 0 and 1s) and the length of the shortest path in a maze (from top left to bottom right). 

## Setup
To run the project create and activate a venv then install dependencies:
```
python3 -m venv venv
source ./venv/bin/activate
pip install numpy pytorch matplotlib
```
Note in the codebase ``.to(device='mps')`` is used. If this isn't running on an Apple Silicon Mac, it should be changed appropriately.
## Description
The maze.py library provides some functions to work with mazes, such as generating a realistic looking maze under certain rules (path exists, no double thickness walls or passages, etc) that look nice. It also provides functionality to calculate a maze's shortest path length as well as a function to perturb a maze. Perturbation, makes very little change to the look of the global maze (often just closing and opening 1 wall and passage), but can drastically change the length of the shortest path, as a detour might be necessary.

In testing_data.ipynb, we look into the generation of some mazes and how they look. As well as plotting the effect of a perturbation on a maze. This then allows us to generate data (pairs of lengths and mazes) to attempt and make an image model. We generate pairs of data while also adding maze perturbations to our data. This allows us to have mazes that look alike with different path lengths in our dataset. 

Our goal now is to generate a maze image, given a desired shortest path length. We aim at constructing and analyzing such a model. 

We analyze two models in Transformer&CVAE.ipynb. The first one is a transformer architecture, generating the whole structure of the maze in one step. It is very weak and quickly settles into local minima strategies that can be observed in the images provided. Often just repeating patterns of gridlines. The CVAE performs much better, it learns the texture of mazes, and at a local level generate very maze like structure. There's a provided plot of generated maze path lengths and our target lengths. There's barely any correlation, with many mistakes being made (often no path at all). This shows, that the CVAE model, can't learn the global phenomena of "length", but only local texture.

## RL Improvements
In cvae_decoder_reinforce.ipynb, we look further into using the CVAE to generate solvable mazes with RL. This uses a REINFORCE-based algorithm rewarded with solvability (path length > -1) to train the CVAE decoder.

In rl_latent_sampling.ipynb, we use RL to boost solvability by improving the sampling of the latent vector, and freezing the decoder. This is also implemented with a REINFORCE-based algorithm.

We also attempted to control maze conditions such as path length via reinforcement learning. We used two methods; firstly, a Gaussian distribution centered on the target length L for the reward depending on how close the path length was to the desired L (gaussian_length.ipynb); secondly, a tiered binary reward system, giving some reward for solvability, and some for matching path length (binary_length.ipynb).
