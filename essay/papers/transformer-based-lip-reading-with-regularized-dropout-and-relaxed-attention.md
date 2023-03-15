# TRANSFORMER-BASED LIP-READING WITH REGULARIZED DROPOUT AND RELAXED ATTENTION

The paper introduces an architecture based on the pure Transformer model, with no special additions.
The author also applies the regularized dropout (R-Drop) method to improve the training inference consistency. It also applies relaxed attention during training to better integration with a language model.

## Regularized Dropout

When using vanilla dropouts, a random selection of units are deactivated during training to reduce overfitting.
During inference, however, all the units are used. This inconsistency between training and inference can cause undesired side-effects.
Regularized dropout (R-Drop) is designed to reduce the training-inference inconsistency by forcing similar outputs of sub-models