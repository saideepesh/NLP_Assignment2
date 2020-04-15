Name of students : Hammouch Ouassim, El Hajji Mohamed, POKALA Sai Deepesh, de BROGLIE Philibert

Libraries used : 

pandas
transformers
numpy
tensorflow
sklearn

Feature representation : 

For each element of [sentences, aspect terms, aspect category], we pass it in a pretrained BERT 
model from transformers, in order to get the last hidden state as inputs for the classifier.
we do not apply any preprocess to the text as in our test, this lead to worse results.
We limit the number of tokens to 32, pad if needed, and we concatenate the 3 representation :
we have a final input vector(for the classifier) of shape (batch, 96, 768).

Classifier:
The classifier used is a Dense NN with dropout layers in order to not overfit. We used no activation
functions except for the last layer (softmax). The optimizer is Adam and we clip the gradients' value. 

Metrics : 
With sklearn's classification report, in our test, we scored 0.85 on the dev set.
However, it is likely that the scores will be a little lower with tester, as we used an earlystopping method with the dev set in order to stop
the training before the model overfits.

In order to do it in the training, as we don't have access to dev data in tester, we use train_test_split from sklearn to 
create a dev set containing 10% of train data, and use this set as validation set for earlystopping.
We know this lead to worse performance, but in our test this method was more robust.

Other approach tested : 

Here are the other approaches tested, they all leaded to worse results so we didn't include them in the final code

- Use Word2vec instead of BERT ==> 0.81 accuracy on the dev set
- Include some preprocess with spacy ==> 0.75 accuracy on the dev set
- Use other transformers (DistillBERT, Roberta) ==> 0.82 acc for Roberta, 0.81 for DistillBERT
- Use another feature representation (multiplying the representations of aspect term and sentences for example) ==> 0.82 acc
- Add class weights ( without it, the model simply ignores the neutral class) ==> 0.84 accuracy, but this is not robust.



