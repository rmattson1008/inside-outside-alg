# Inside-Outside algorithm for PCFG
An implementation of the canonical algorithm using numpy for rapid calculation.


Work in progress. Tasks left:

- Functionality
   - Test cases for EM and Chart edge cases

- Management
    - requirements or better package set up
    - input arg manager

- Additions
    - show parse
    - maybe handle bad user input
    - handle more variation in user input

There are also a lot of hanging comments/notes to self, so looking through the code may be confusing at this point in time.

________
This time for this algorithm scales rapidy with the size of the sentence input. Some work arounds are to truncate each sentence to a short length or to mitigate the problem by using fine-to-coarse PCFG structures. Another addition could be to automatically break a sentence into two separate subtrees if it is a certain length, and don't keep track of the relationship between all rules.
