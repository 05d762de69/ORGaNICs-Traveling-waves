# ORGaNICs-Traveling-waves
This repo holds the code, documentation, and a toy notebook of a model of perceptual traveling waves in binocular rivalry, reimplemented with recurrent normalization in the ORGaNICs framework([Heeger & Mackey, 2019](https://doi.org/10.1073/pnas.1911633116), [Heeger & Zemlianova, 2020](https://doi.org/10.1073/pnas.2005417117)), with preliminary extensions beyond rivalry. We build on the model for perceptual waves using binocular rivalry from [Cardoso et al. (2025)](https://doi.org/10.1167/jov.25.12.18). 

## Quick start


```bash
git clone https://github.com/JayJawale/ORGaNICs-Traveling-waves.git
cd ORGaNICs-Traveling-waves
```

```bash
pip install -r requirements.txt
```
`ffmpeg` must be installed system-wide for the .mp4 to render correctly.

Expected output: 
Running the `Example Notebook.ipynb` leads to the creation of two `.mp4` files, with the first replicating the emergence of a traveling wave in binocular rivalry, and the second showing the emergence of an attentional wave in presence of a flashing stimulus. Running the whole notebook takes about 5min on a MacBook Air (M4), Python 3.9.6.

## The model

We are modeling the interactions of neurons in V1, and the emergence of traveling wave phenomena under different conditions. The response of each monocular neuron is dependent on its neighbours activities. This normalization step is implemeneted as a chaser with input and recurrent gains. The binocular rivlarly is realized by coupling the input from eyes by a subtractive opponency (mutual inhibition) and adaptation. An attention layer compares the two orientations at the binocular stage and modulates the monocular neurons' input gain, which leads to strengthening the winner and suppressing the loser. Finally, because the inhibition is spread across neighbouring modules, a dominance switch at one location pushes its neighbours past their own switching point, and the switch propagates as a wave.
Full equations, IDs and the report-to-code mapping are in
`docs/notation.md`.

The results of this model are written into an array of the shape `results[idx, q, lay, θ, t]`.  The quantities `q` holds the different state-variables, `lay` specifies the neuron type, `θ` is the orientation, and `t` the point in time. Full parameter descriptions are in `docs/notation.md`.

The model's logic is implemented in `BinocularRivalryWavesORGaNICs_vers1.py`.


