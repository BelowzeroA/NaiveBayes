from classifier import NBClassifier


def get_features0(sample):
    return (sample[-1],) # get last letter

def get_features(sample):
    return (
        'll: %s' % sample[-1],          # get last letter
        'fl: %s' % sample[0],           # get first letter
        'sl: %s' % sample[-2],           # get second letter
        )

nb = NBClassifier()
#samples = (line.decode('utf-8').split() for line in open('names.txt'))
samples = (line.split() for line in open('names.txt'))
#feats = get_features(samples)
features = [(get_features(feat), label) for feat, label in samples]
nb.train(features)

print ('gender: ', nb.classify(get_features(u'Дима')))
