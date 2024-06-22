# Environment
## python(2/3): 
Both support numpy, scipy, TensorFlow under CPU, theano, pytorch (0.4.0, except python 3.6) and mxnet (0.12.0), as well as keras (2.1.6), lasagne, scikit-image and h5py.
```
python 3.6 does not support theano and lasagne.
The pytorch version under python 3.6 is 1.4.0.
The mxnet version under python 3.6 is 1.4.0.
```

## Recommened version
```
Python = 3.6.5
```

# Checkpoint
Every user on Botzone can have a isolated storage of 268435456 bytes (256 MB) maximum for its bots to read or write. To access, the path is `data` folder under current working directory when running bot.
## Recommened name
```
tractor_model.pt
```

# Upload files
Zip all python codes and make sure `__main__.py` is included in the root.

```
zip package.zip __main__.py mvGen.py wrapper.py model.py
```

Then upload `package.zip` and make sure package size is less than 4 MB.
