__author__ = 'peter'

# ToDo:
# general exploration of topic model concepts
#
# topic model as receptive Fields
## any character as a potential topic component
## any character has a receptive field across text
### a smoothed count of word appearence
## for the most part this is probably very sparse with some exceptions
### names, stop words, specific proper nouns
## assume these are Poission processes, estimate lambda in time
### looking at frequncy over variance can help distinguish topic variability
## make thresholds based on high, frequency high variance?
## correlate terms past certain frequencies
## try to cluster these?
## extend to autocorelations and lagged correlations
## build a RNN around these?
### train an RNN to predict future topic appearence based on past topic appearence
## different RNNs with different timescales feeding into an embeddig matrix
### embedding matrix passed into a word for word RNN
### consider these pre RNN's as having sepperate error flow? or allow for continued error?
### add additional error in the word for word RNN based on distance from topic by topic correlation
## ultimatly want to abstract topics more then just characters, like instead of a specific name,
##      just that its the most frequent name, or the nth frequent name
##      many terms are related, or have short distances and should be treated as a cluster in the
##      same topic, can combine words via wordnet, LSI, or similar
