# Inside-Outside algorithm for PCFG
An implementation of the canonical algorithm (cite) using numpy for rapid calculation.

text input -> fitted pcfg output

TODO 
- Plan test cases

- Management
    - make package somewhere
        - requirements.txt or something
    - should be much better documentation.... hmm docstrings, typing
    - parameter passing is messy and sometimes unnecesary. 
    - a couple big functions could be split smaller

- Additions
    - show parse
    - clarify likelihood in EM
    - decide how "final" to make code for it.
    - maybe handle pcfg bad input, catch gracefully
    - tool to convert np->text? nice and modular function
    - initialize with weight
    - generate different types of grammars? control possibilities... 


- Test
- take canonical astronomer example.... 


Either binary rules and unary rules, 
Or non-terminals, but this makes no sense without pos. 
so nonterminals + unary rules. (suggest tagging) 


________
Testing

This repo contains a few small files to use test existing functions. The toy PCFG is a caonical example borrowed from [link to repo].

________
Installation
Todo - requirements.txt



________
This time for this algorithm scales rapidy with the size of the sentence input. Some work arounds are to truncate each sentence to a short length or to mitigate the problem by using fine-to-course PCFG structures. Another addition could be to automatically break a sentence into two separate subtrees if it is a certain length, and don't keep track of the relationship between all rules.
