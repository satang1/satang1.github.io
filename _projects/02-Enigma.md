---
layout: single
title: "The Enigma Machine"
date: 2018-10-16

header:
  teaser: /assets/img/enigma.png
---

This was the second project for my Data Structures class. We were tasked to recreate the Enigma machines used in Germany during WWII to encrypt messages. The premise of the machine was to encode each letter/message using alphabetic rotors, which turned only when its ratchet was engaged at a notch. 

More specifically, when a key is pressed on the Engima machine plugboard, the letter of that key gets translated/converted into a different letter. This converted letter, then goes into a set of rotors (which contains at least 1 reflector, any number of non-moving rotors, and any number of moving rotors). As the converted letter goes through these rotors, which are each set to a specific "setting", they get translated/converted again via a permutation cycle that maps each letter to a different letter in the alphabet. This conversion continues until the letter reaches the reflector, which "reflects" the converted letter backwards to the very first rotor. 

For example, consider the following sequence of rotors and the plugboard: 

(R = reflector, N = non-moving rotor, M = moving rotor)

R N M1 M2 M3 (plugboard)

A letter goes in from the right, into the plugboard, gets translated and moves onto the "first" rotor, M3. Then, the letter gets translated/converted inside M3, and moves onto M2 . . . and so on, until the letter reaches R. R will then "reflect" the letter "backwards" to M1, where the letter will get translated/converted again. Then move on to M2, and so on until we reach the plugboard again, where it will be translated again, and outputted. 


Due note, that as each letter gets translated, the rotors will shift/rotate into a different "setting" if the rotor is at a "notch". But the details of this concept/condition will not be discussed here. 

All in all, the basics of this enigma machine simulator was to mimic the encryption of messages used by the Germans during WWII. You can read more about the mechanics of the machine [here](https://en.wikipedia.org/wiki/Enigma_machine). 

<b>Example Input (left) and Output (right)</b>
Note: Encriptions are based on a given configuration file which specifies the "settings", types of rotors, and permutation cycles. 
<center><p><img src="/assets/img/enigma.png" alt=""></p></center>