# Bezier Word-gsture Keyboard

Word-gesture keyboards use a swiping motion over a keyboard layout to write a word. The gesture is then compared to perfect gesture graphs (PGGs) of each word in a dictionary to see which one most closely resembles the gesture. The idea of this project is to construct PGGs of the words using more complex Bezier curves, rather than straight lines as is customary.

The repository offers a Unity project with an experiment scene where the user has to write out phrases correctly using either Bezier PGGs or Line PGGs and their performance is recorded.

# Software

This repository comes with:
- WG Keyboard - A Unity project with a scene that can directly run the word gesture keyboard and the experiment. The keyboard is taken from [here](https://github.com/Philipp1202/WGKeyboard).
- word_graph_generator.py - A Python script to generate new PGGs for a given lexicon. Originally from the same repository as the keyboard, adapted to feature Bezier PGGs
- data_analysis.py - Script that analyses user data generated from the experiment. It gives useful statistics on the data and also outputs
- curve_fitter.py - Script that fits data of writing gestures to a Bezier curve as close as possible.

# About

This repository contains the code used and produced during the TU Delft CSE3000 Research Project (2023/2024) course.
