# Inside-Outside algorithm for PCFG
An implementation of the canonical algorithm (cite) using numpy for rapid calculation.

text input -> fitted pcfg output

TODO 
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


