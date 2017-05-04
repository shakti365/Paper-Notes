# [Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms](https://arxiv.org/pdf/1702.01776.pdf)

## Summary

The aim of this paper is to extract aspect and opinion words for a review text.<br>
*“This little place has a cute interior decor and affordable prices”*<br>
Aspect terms: interior decor, prices<br>
Opinion terms: cute, affordable<br>

The proposed model uses a coupled attention over GRUs to learn aspect and opinion terms.

## Notes

* Two attention models one for aspect and one for opinion, each attention aims to learn:
	- a prototype vector for aspect or opinion
	- a high-level feature vector for each token
	- an attention score for each token in the sentence.

* To exploit the relations between aspect and opinion terms the attentions are coupled such that their learning effects one another

* To capture indirect relations they have proposed multiple layers of attention

* A a tensor operation is performed to capture different syntactic representations of the same input. This is similar to the different kinds of parse trees that can be formed for a sentence.

* To solve the problem of multiple word or phrase as aspect words they have considered a BIO encoding scheme where a word can wither be Beginning of the aspect, In the aspect or Out of the aspect phrase.

## Model

1. Use GRU to find hidden state representation $`h_i`$ of every word $`w_i`$ in a sentence
2. Randomly initalize prototype vector $`u^m ~ U[-0.2,0.2]`$, where m can be aspect or opinion term
3. Calculate $`\beta_i^m`$ = $`f^a`$($`h_i`$, $`u^m`$, $`u^{\bar{m}}`$) = $`\tanh(h_i^T G^m u^m ; h_i^T D^m u^{\bar{m}})`$
4. $`r_i^m`$ = GRU($`\beta_i^m`$) - attention vector - high level representation of each input token
5. $`e_i^m`$ = $`v^{m^{T}}`$ $`r_i^m`$ - attention score

[TODO: add the multi-layer part]

## Inference

* This seems like a good model since it is able to capture semantic and syntactic nature of a text.
* The results are shown to perform across different domains of reviews. This proves that the model actually learns the syntactic placement of aspect and opinion words in a sentence.
* Need to verify whether the model needs to be so complex.
* Memory networks can be explored to check if they perform better

## Implementation details
* The dataset found for this task only has aspect terms and not opinion terms. Thus only a single attention model is used in  the implementation.
* The word2vec could not be trained for yelp restaurant review dataset as it is huge and yelp dataset set contains mixed reviews and not restaurant reviews seperately
*
* The model is underfitting 
