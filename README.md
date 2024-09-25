# timbre-simulation

Now contains a timbre simulation of guitar.

The 'metal' feeling of guitar relies on high harmonics, a file of sr=44100 may not be able to fully simulate the timbre.
(need further experiments)

## properties of fft

1. argmax of fft represents the frequency of wave, regardless of phase and changes of amplitude over time

2. given constant amplitude, max of fft is relevant of phase and length of the array

3. phase can only be calculated when the time of array is an integer times of the period, while there's noise in calculation

## assumptions

We only consider harmonics of base, and neglect waves that are not integer times of base frequency 
(noting that base doesn't always holds the largest amplitude).
Assume that amplitude over time can be expressed by a poynomial over the exponential.

## method

1. get frequency of each harmonic

2. get phase of each harmonic

3. stft

  consider the uncertainty principle of stft(same as heissenbberg's law),

4. get amplitude over time for each harmonic

  noting that each harmonic have its own amp-t function.

5. find a function of time to express amplitude

6. write audio

## future work

it's now only tested on one simple audio and is restricted by noise in data. need more and purified data with high sr.
