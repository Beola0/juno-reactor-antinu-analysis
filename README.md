# JUNO Reactor Antineutrino Analysis

Contact: Beatrice Jelmini
beatrice.jelmini@studenti.unipd.it

## Macros

You can run the macros in interactive mode with `python -i macro_name.py` or `ipython -i macro_name.py`.
Some plots will appear.

Macros:
* `macro_reactor.py`: shows comparison between the three reactor models (V, HM, DYB); SNF and NonEq corrections
* `macro_oscillation.py`: shows comparison between NO/IO, vacuum/matter, NuFIT2019/PDG2020

## Inputs

The directory `Inputs` contains all inputs needed.

## Antinuetrino Spectrum

The directory `AntineutrinoSpectrum` contains the four classes used to obtain the reactor antineutrino spectrum:
* `reactor.py`: isotopic spectra; reactor models: Vogel, Huber+Mueller, DYB; cross section: Vogel+Beacom, Strumia+Vissani;
SNF and NonEq corrections; unoscillated spectrum
* `oscillation.py`: normal and inverted ordering; in vacuum or matter
* `spectrum.py`: oscillated spectrum; sum over list of reactors
* `detector_response.py`: energy resolution; non-linearity

## Backgrounds

The directory `Backgrounds` contains some old backgrounds.