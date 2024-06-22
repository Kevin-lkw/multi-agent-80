# Environment
## python(2/3): 
Both support numpy, scipy, TensorFlow under CPU, theano, pytorch (0.4.0, except python 3.6) and mxnet (0.12.0), as well as keras (2.1.6), lasagne, scikit-image and h5py.
```
python 3.6 does not support theano and lasagne.
The pytorch version under python 3.6 is 1.4.0.
The mxnet version under python 3.6 is 1.4.0.
```

# Checkpoint
Botzone 上的每个用户可以为其所有 Bot 准备一个大小不超过 268435456 Byte(256MB)的独立存储空间，Bot 可以随意读写其中的文件。文件路径是 Bot 运行时目录下的 `data` 文件夹。

# Upload files
Zip all python codes and make sure `__main__.py` is included in the root.

```
zip package.zip __main__.py mvGen.py wrapper.py model.py
```

Then upload `package.zip` and make sure package size is less than 4 MB.
