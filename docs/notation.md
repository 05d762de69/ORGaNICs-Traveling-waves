# Complementary notation to the ORGaNICs Traveling Waves repository

Only monocular and binocular neurons adapt/tire in this model. The comparators and attention neurons driving the binocular rivalry are gain free and memory-less .

## Equations

| ID    | Equation | Plain meaning |
| :--- | :------------: |  ------------: | 
| M1 | $\tau_{mo}\frac{dR_{l,1,i}}{dt}=-R_{l,1,i}+m\lfloor b_{l,1,i}\rfloor_+ E_{l,1,i}+a_{l,1,i}\hat{R}_{l,1,i}$ | Response of a monocular neuron, defined by the input gain times the drive, plus the recurrent gain times it's own last value|
|M2 | $E_{l,1,i}= \lfloor{ D_{l,1,i}^{n}-\sum\limits_{j}[w_o]_{i,j}O_{r,j}+\sum\limits_{i\ne j} [w_r]_{i,j} R_{l,1,j}} \rfloor_+$| Drive of the $i$-th monocular neuron, same form for the right eye (r) and for orientation 2; contrast, minus other-eye inhibition gathered from neighbours, plus same-eye neighbour excitation, clipped at zero|
| M3 | $\tau_{b}\frac{db_{l,1,i}}{dt}=-b_{l,1,i}+1+\sum\limits_{j}[w_a]_{i,j}R_{a,1,j}$| Input gain defined as chasing one plus attention gathered from neighboring attention neurons. When attention is silent the target is 1.
| M4 | $\tau_{a}\frac{da_{l,1,i}}{dt}=-a_{l,1,i}+1-(S_{m,i}+\sigma^n+(W_{H}H_{l,1,i})^n)$ | Recurrent gain; chases one minus the sum of its module's suppressive drive, the saturation constant σⁿ, and its own tiredness scaled by W_H.
| M5 | $\tau_{H}\frac{dH_{l,1,i}}{dt}=-H_{l,1,i}+R_{l,1,i}$ | The tiredness of the monocular neuron, a slow copy of its own response
| O1 | $\tau_{o}\frac{dR_{or,1,i}}{dt}=-R_{or,1,i}+E_{or,1,i}+a_{or,1,i}\hat{R}_{or,1,i}$ | Response of an opponency neuron as defined by the sum of its drive and the product of its recurrent gain and the recurrence (its own last value), has no input gain, so attention cannot reach it.
| O2 | $E_{or,1,i}= \lfloor (R_{r,1,i}-R_{l,1,i})^n\rfloor$ | comparator: how far ahead the left eye is ahead of the right eye; zero if it isn't
| O3 | $\tau_{a}\frac{da_{or,1,i}}{dt}=-a_{or,1,i}+1-(S_{or,i}+\sigma^n)$ | Recurrent gain of the opponency neuron; chases one minus the sum of the opponency neurons suppressive drive and the saturation constant $\sigma$. No tiredness term means that the comparator does not adapt.
| O4 | $O_{r,i}= R_{or,1,i}+R_{or,2,i}$ | Inhibition the left eye receives; the right-minus-left comparator's response summed over orientations 1 and 2. Summing inhibitions leads to suppression at eye-level.
| B1 | $\tau_{bi}\frac{dR_{b,1,i}}{dt} = -R_{b,1,i} + E_{b,1,i}+a_{b,1,i}\hat{R}_{b,1,i}$| Response of a binocular neuron, defined by chasing the sum of the drive and the product of its recurrent gain and the recurrence (its own last value), again no input gain, so attention cannot directly reach it.
| B2 | $E_{b,1,i}= (R_{l,1,i}+R_{r,1,i})^n$ | Drive of a binocular neuron defined as the summation of responses from left and right eyes for the same orientation, so it responds to its orientation whichever eye is showing it.
| B3 | $\tau_{a}\frac{da_{b,1,i}}{dt}=-a_{b,1,i}+1-(S_{b,i}+\sigma^n+H_{b,1,i}^n)$ | Recurrent gain of a binocular neuron, defined as chasing one minus the sum of its suppressive drive, a saturation constant $\sigma$ and its own tiredness term.
| B4 | $\tau_{H}\frac{dH_{b,1,i}}{dt}=-H_{b,1,i}+W_HR_{b,1,i}$ | Tiredness term of the binocular neuron, a slow, scaled copy of its own response.
| A1 | $\tau_{at}\frac{dR_{a,1,i}}{dt}=-R_{a,1,i}+E_{a,1,i}+a_{a,1,i}\hat{R}_{a,1,i}$  | Response of an attention neuron, defined by chasing the sum of the drive and the product of its recurrent gain and the recurrence (its own last value), again no input gain, so nothing modulates it.
| A2 | $E_{a,1,i}= (R_{b,1,i}-R_{b,2,i})^n$ |  comparator; how far the binocular response for orientation 1 is ahead of the one for orientation 2. Signed: positive when 1 leads, negative when 2 leads.
| A3 | $\tau_{a}\frac{da_{a,1,i}}{dt}=-a_{a,1,i}+1-(S_{a,i}+\sigma_a^n)$ | Recurrent gain of an attention neuron, defined by chasing one minus the sum of its suppressive drive and a saturation constant $\sigma_a$; doesn't include a tiredness term so attention neurons do not adapt.




## State variables

## State variables

`q` and `lay` are the quantity and layer axes of the `results` array,
`results[idx, q, lay, θ, t]`. Layer numbers are the code's, 0-based (the header
comment in the source counts from 1). Orientation θ and time t are implicit in
every row. Gains have their own layers: `a` at lay + 1, `b` at lay + 2.

| Report symbol | Plain meaning | q | lay | Eq. ID | Notes |
| :-----------: | :------------ | :-: | :-- | :----: | :---- |
| $D_{l,1,i}$ / $D_{r,1,i}$ | Stimulus contrast driving the monocular neuron | – | – | – | Lives in `input[idx, eye, θ, t]`, not in `results`; eye 0 = L, 1 = R; read at t |
| $E_{l,1,i}$ / $E_{r,1,i}$ | Excitatory drive of the monocular neuron | 0 | 0 (L), 3 (R) | M2 | Algebraic; rectified by `half_exp`; report's single $w_r$ is three kernels (`wc`, `wr`, `wr_local`) |
| $S_{m,i}$ | Suppressive drive (pool) of the monocular neurons | 1 | 0 (L), 3 (R) | – | Not defined in the report. Code: Σ of E over both eyes and all orientations at module i; same value stored in both layers |
| $b_{l,1,i}$ / $b_{r,1,i}$ | Input gain of the monocular neuron | 3 | 2 (L), 5 (R) | M3 | Rectified by `half_exp` where it multiplies E in M1 |
| $a_{l,1,i}$ / $a_{r,1,i}$ | Recurrent gain of the monocular neuron | 3 | 1 (L), 4 (R) | M4 | Code: $(w_h H)^{n_m}$; report: $W_H H^n$ |
| $R_{l,1,i}$ / $R_{r,1,i}$ | Response of the monocular neuron | 3 | 0 (L), 3 (R) | M1 | $\hat{R}$ = R at t−1 |
| $H_{l,1,i}$ / $H_{r,1,i}$ | Adaptation of the monocular neuron | 4 | 0 (L), 3 (R) | M5 | |
| $O_{r,i}$ / $O_{l,i}$ | Inhibition received by the left / right eye | 5 | 0 (received by L, written from layer 10), 3 (received by R, written from layer 8) | O4 | Σ over orientations of the comparator's R; the only value written into another neuron's layer; read at t−1 by M2 across modules via `wo` |
| $E_{b,1,i}$ | Excitatory drive of the binocular neuron | 0 | 6 | B2 | Built from both eyes' R at t−1 |
| $S_{b,i}$ | Suppressive drive of the binocular neuron | 1 | 6 | – | Not defined in the report. Code: its own E only (no pooling across orientations) |
| $a_{b,1,i}$ | Recurrent gain of the binocular neuron | 3 | 7 | B3 | Code: binocular neuron's own H, as $(w_h H)^n$; report prints $H_{l,1,i}$ and no $W_H$ |
| $R_{b,1,i}$ | Response of the binocular neuron | 3 | 6 | B1 | Code reads E at t−1 here; every other layer reads E at t |
| $H_{b,1,i}$ | Adaptation of the binocular neuron | 4 | 6 | B4 | Code: chases R, no $W_H$; report places $W_H$ here |
| $E_{or,1,i}$ (and unwritten mirror $E_{ol,1,i}$) | Excitatory drive of the comparator | 0 | 8 (L−R), 10 (R−L) | O2 | Rectified *before* the power (`half_exp(...)**n`); report writes only the R−L neuron, layer 8 is its mirror |
| $S_{or,i}$ | Suppressive drive of the comparator | 1 | 8 (L−R), 10 (R−L) | – | Not defined in the report. Code: Σ over orientations of its own E |
| $a_{or,1,i}$ | Recurrent gain of the comparator | 3 | 9 (L−R), 11 (R−L) | O3 | No adaptation term: comparators don't adapt |
| $R_{or,1,i}$ | Response of the comparator | 3 | 8 (L−R), 10 (R−L) | O1 | Summed over orientations into the O row |
| $E_{a,1,i}$ | Excitatory drive of the attention neuron | 0 | 12 | A2 | Signed (`aSign * aDrive**n`); built from $R_b$ at t via `aKernel` |
| $S_{a,i}$ | Suppressive drive of the attention neuron | 1 | 12 | – | Not defined in the report. Code stores Σ aDriveⁿ + $\sigma_a^n$, so $\sigma_a^n$ is counted a second time in A3 |
| $a_{a,1,i}$ | Recurrent gain of the attention neuron | 3 | 13 | A3 | No adaptation term |
| $R_{a,1,i}$ | Response of the attention neuron | 3 | 12 | A1 | Read at t−1 by M3 across modules via `wa` |


## Parameters

| Report Symbol              | Plain meaning | Code name | parameter value | Notes|
| :---------------- | :------: | :---------: |  :-: | ----------------: |
| $\tau_{mo}$       |   how fast the monocular response chases its target   | `tau_v1` |  5 | used in M1, orgin |
| $\tau_{b}$  or  $\tau_{at}$ |     | `tau_attention` |  5 | used in M3 and A1 |
| $\tau_{a}$   |     | `tau_recurrence` | 1 | used in M4, O2, and B3|
| $\tau_{H}$ |    | `tau_adaptation` | 1500 | used in M5|
| $\tau_{o}$ |    | `tau_opponency` |  | used in O1 |
| $\tau_{bi}$ |    | `tau_binocular` | 5 | used in B1 |

