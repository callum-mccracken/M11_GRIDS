# M11 experiment from GRIDS

Contains a couple files for doing these experiments with the M11 beam:
- calculating beam momentum (using `beam_momentum.py`, specify file with `-f`)
- testing PMT things (see the jupyter notebooks)

Requirements:
- python (3, I used 3.10.5 but others should work)
- uproot, numpy, matplotlib, scipy (all are pip- or conda-installable)
- if you know how to use Docker I included a Dockerfile :)

Root files in this repo:
- `output_000496.root` -- data from a run with a working beam
- `output_000497.root` -- data from a short cosmics run
- `output_000498.root` -- data from a longer cosmics run

The PNGs in this repo are made by `beam_momentum.py`,
and will be overwritten if you run that script with another file.
