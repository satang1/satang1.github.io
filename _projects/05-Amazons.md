---
title: "Amazons"
layout: single

header:
  teaser: /assets/img/Amazons_board.png
---

## Intro: 
The Amazons game is a rather simple board game in which the starting board has 4 black queens and 4 white queens positioned symmetrically on opposite sides of the board. Each Queen piece is able to move in any direction horizontally, vertically, and diagonally...that is, as long as there is nothing blocking its way. On the same turn, the Queen piece that had just moved is then able to throw an arrow/spear in any direction horizontally, vertically, or diagonally given nothing is in its way again. 

This continues until one side's Queen pieces are unable to make any more moves because all of its Queen pieces are surrounded, either by arrows or other Queen pieces. 

A brief introduction as to how the game is played: 
<center><img src="https://media.giphy.com/media/br6yn6i3OrrjQ9CFA9/giphy.gif"/></center>

This version of Amazons also consists of an AI component which was implemented using a minimax alpha-beta pruning algorithm. 

In the terminal, if you type in "Auto white" the white player will play against the black player automatically. (Note: Black is initially set as an automatic player)
<center><img src="https://media.giphy.com/media/2hgw1iDh1pkgV9rj84/giphy.gif"/></center>

Notice the moves that are printed out are in the following format: "a7-a6(a7)". Since the 10 x 10 board is represented in something similar to a matrix (or more formally a 2D array), each column is denoted by a letter (from a to j) and each row is denoted by a number (from 1 to 10). 

In the mentioned example ("a7-a6(a7)"), "a7" refers to the starting position of a Queen piece. "a6" refers to the position that the Queen piece will move to, and "a7" refers to the position that the Queen piece will throw its spear after it has moved (to "a6"). 

At the end, the program will print out a message stating which side/player has won. You can also input a command "dump" to see the contents of the board in the terminal. "B", "W", and "S" signify Black, White, and Spear respectively. 

The "manual [Black or White]" command returns the player to rely on manual user input instead of the AI commands. The "new" command clears the board and returns all pieces to its starting positions. 

<center><img src="https://media.giphy.com/media/br2nYq81mc2TDkhgjZ/giphy.gif"/></center>

