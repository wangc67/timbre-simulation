# timbre-simulation

Now contains a timbre simulation of guitar.

The 'metal' feeling of guitar relies on high harmonics, a file of sr=44100 may not be able to fully simulate the timbre.
(need further experiments)

## properties of discrete fourier transform

1. argmax of fft represents the frequency of wave, regardless of phase and changes of amplitude over time

2. given constant amplitude, max of fft is relevant of phase and length of the array

3. phase can only be calculated when the time of array is an integer times of the period, while there's noise in calculation

## assumptions

We only consider harmonics of base, and neglect waves that are not integer times of base frequency 
(noting that base doesn't always holds the largest amplitude).
Assume that amplitude over time can be expressed by a poynomial over the exponential.

## method

1. get frequency of each harmonic

  use the whole process to fft, choose argmax as base temporarily(need further work), slice the array near the harmonic's index and get its argmax as the frequency of harmonic
  
2. get phase of each harmonic

  for each frequency of harmonics, search the length for time * frequency closest to an integer. considering noises when calculating phase, when distance between time * frequency and the integer is less than a given parameter, we add the calculated phase to a list. we choose the average of the largest cluster in the list as the phase of the given frequency.
  
3. stft

  consider the uncertainty principle of stft(same as heissenbberg's law),

4. get amplitude over time for each harmonic

  noting that each harmonic have its own amp-t function, we assume that amplitude within the given period is a constant(which is not a good hypothesis), get the max of slices near the harmonic as the amplitude of the given time. 

5. find a function of time to express amplitude

  we apply polynomial regression to time and log(amplitude). if the coefficient of the highest power is positive, we add the degree of polynomial. if the degree is to high, we use linear regression.

6. write audio

## future work

it's now only tested on one simple audio and is restricted by noise in data. need more and purified data with high sr.
