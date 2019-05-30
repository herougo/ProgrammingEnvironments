class MyLogger:
    def __init__(self, pkl_path):
        import pickle
        import os

        # assert not os.path.isfile(pkl_path), f"File {pkl_path} already exists"
        self._file = open(pkl_path, 'ab')

    def __call__(self, *args, **kwargs):
        try:
            pickle.dump(args[0], self._file)
        except Exception as ex:
            print(ex)
            # REDO: ^^^^^^^^^^^^^^^^
        print(*args)

    def __del__(self):
        self._file.close()
log = MyLogger("/home/henri/ml_results/exp10-pickle")


def pkl_load_all(file_path):
    with open(file_path, 'rb') as f:
        l = []
        try:
            while True:
                l.append(pickle.load(f))
        except:
            pass

    return l



from keras.layers import Dense, Input
from keras.models import Model
a = Input((2,))
b = Dense(3, name='b')(a)
c = Dense(2, name="c")(a)
model = Model(a, [b, c])
model.compile(
    optimizer='adam',
    loss=['mse', 'mse'])

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = [
    np.zeros((4, 3)),
    np.zeros((4, 2))
]

model.test_on_batch(X, y)



# EXAMPLE

import time
from multiprocessing import Process, Queue

# DO NOT USE time.sleep or time.time() with multiprocessing from personal tests

def work(q=None):
    for i in range(10_000_000):
        for j in range(100):
            pass
        if i % 1_000_000 == 0:
            print('hello world', q is None)
    if q is not None:
        q.put([1, 2, 3])
        
q = Queue()
p = Process(target=work, args=(q,))
p.start()
work()
print(q.get())    # prints "[42, None, 'hello']"
p.join()



# OPEN IN OTHER THREAD

from multiprocessing import Process, Queue
class ParallelFunc:
    def __init__(self, fn, args={}):
        assert isinstance(args, dict)
        self._done = False
        self._fn = self._create_fn_wrapper(fn, args)
        
    def start(self):
        self._q = Queue()
        self._p = Process(target=self._fn, args=(self._q,))
        self._p.start()
        return self
        
    def _create_fn_wrapper(self, fn, args):
        def fn_wrapper(q):
            q.put(fn(**args))
        return fn_wrapper
    
    def get(self):
        assert not self._done
        result = self._q.get()
        self._p.join()
        self._done = True
        return result
            
p = ParallelFunc(work).start()
work()
p.get()




class ThreadSafeGenerator:
    def __init__(self, generator):
        import threading
        self._generator = iter(generator)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self._generator)

def keras_stuff():
    vocab_size = 26
    embedding_size = 8
    input_length = 15
    hidden_units = 8

    #model = Sequential()
    x1 = Input((input_length,))
    emb = Embedding(vocab_size, embedding_size, input_length=input_length)(x1)
    left = LSTM(units=hidden_units, return_sequences=True)(emb)
    left = Dense(hidden_units)(left)
    right = LSTM(units=hidden_units, return_sequences=True, go_backwards=True)(emb)
    right = Dense(hidden_units, use_bias=False)(right)

    yo = add([left, right])

    model = KerasModel(x1, yo)


'''
def myshape(yo, prefix=''):
    if isinstance(yo, tuple):
        print(prefix, 'tuple:', len(yo))
    elif isinstance(yo, list):
        print(prefix, 'list:', len(yo))
    elif isinstance(yo, np.ndarray):
        print(prefix, yo.shape)
        return
    else:
        return
    for i in yo:
        myshape(i, prefix+'|-')
'''

def cmd_output(cmd):
    import subprocess
    result = subprocess.run(list(cmd.split()), stdout=subprocess.PIPE)
    # result.stdout is a byte string (ie b'abced')
    return result.stdout.decode("utf-8")


def keras_gate():
    a = Input((2,), name='a')
    b = Dense(2, name='b')(a)
    g = Dense(1, activation='sigmoid')(a)
    g = RepeatVector(2)(g)
    g = Flatten(name='g')(g)
    c = Add(name='c')([Multiply()([g, b]), Multiply()([Lambda(lambda x: 1 - x)(g), a])])
    model = Model(a, c)
    model.compile(optimizer='adam', loss='mse')
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    model.summary()
    model.test_on_batch(X, X)

def keras_stack_models():
    # with Model
    model = Sequential()
    model.add(Dense(2, input_shape=(2,)))

    a = Input((2,))
    b = model(a)

    m = KerasModel(a, b)
    m.compile(loss='mse', optimizer='adam')
    m.summary()
    m.test_on_batch(np.zeros((1, 2)), np.zeros((1, 2)))

    # with Sequential
    model = Sequential()
    model.add(Dense(2, input_shape=(2,)))

    m2 = Sequential()
    m2.add(model)
    m2.add(model)
    m2.compile(loss='mse', optimizer='adam')
    m2.summary()
    m2.train_on_batch(np.zeros((1, 2)), np.zeros((1, 2)))


def check_shape_consistent(v, shape):
    assert v.shape == shape
    if len(shape) > 1:
        l, rest = shape[0], shape[1:]
        assert len(v) == l
        for row in v:
            check_shape_consistent(row, rest)


def split_tt(data):
    test_data = XHashFilteredStream(data, seed=2, is_test=True, test_partition_size=5)
    used_data = XHashFilteredStream(data, seed=2, test_partition_size=5)
    return used_data, test_data

def split_tvt(data):
    test_data = XHashFilteredStream(data, seed=2, is_test=True, test_partition_size=5)
    used_data = XHashFilteredStream(data, seed=2, test_partition_size=5)
    
    train_data = XHashFilteredStream(used_data, seed=1, test_partition_size=5)
    val_data = XHashFilteredStream(used_data, seed=1, is_test=True, test_partition_size=5)
    return train_data, val_data, test_data

def keras_change_constants():
    import keras.backend as K

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = [
        np.zeros((4, 3)),
        np.zeros((4, 2))
    ]



    alpha = K.variable(0.5)

    a = Input((2,))
    b = Dense(3, name='b')(a)
    middle = Lambda(lambda x: x * alpha)(a)
    c = Dense(2, name="c")(middle)
    model = Model(a, [b, c])
    model.compile(
        optimizer='adam',
        loss=['mse', 'mse'])

    K.set_value(alpha, 0)
    print(model.test_on_batch(X, y))
    K.set_value(alpha, 1)
    print(model.test_on_batch(X, y))

    save_model(model, 'deleteme.model')
    m = load_model('deleteme.model', custom_objects={'alpha': alpha})

import inspect

# Tests
def yo1():
    pass
def yo2(hi2):
    pass
def yo3(hi3=0):
    pass
def yo4(hi4, yo=0):
    pass

# Exmple: fn_to_arg_names(Hi.__init__)
# Example: fn_to_arg_names(Hi.yo)
def fn_to_arg(f):
    return inspect.getargspec(f)[0]


# ArgSpec(args=['hi', 'yu', 'yellow'], varargs='args', keywords='kwargs', defaults=(0, 'hi'))
#               all named args          name of variable # of args, name of kwargs, default values

def fn_to_req_arg(f):
    arg_spec = inspect.getargspec(f)
    arg_names = arg_spec[0]
    default_values = arg_spec[3]
    if default_values is None:
        return arg_names
    else:
        n_default = len(default_values)
        return arg_names[:-n_default]

def test_inspect():
    test_functions = [
        (yo1, []),
        (yo2, ['hi2']),
        (yo3, []),
        (yo4, ['hi4'])
    ]

    errors = []
    for a, b in test_functions:
        result = fn_to_req_arg(a)
        if result != b:
            errors.append((result, b))
    print(errors)

def check_shape_match(v1, v2):
    types = (tuple, list, np.ndarray)
    for t in types:
        if isinstance(v1, t):
            assert isinstance(v2, t), f'{v1} {v2}'
            assert len(v1) == len(v2), f'{len(v1)} {len(v2)}'

            for v1row, v2row in zip(v1, v2):
                check_shape_match(v1row, v2row)
            break

def myshape2(yo):
    if isinstance(yo,  (tuple, list, np.ndarray)):
        return (len(yo),) + myshape2(yo[0])
    else:
        return ()


def split_last(s, substring):
    i = s.rfind(substring)
    if i == -1:
        return [s]
    return s[:i], s[i+len(substring):]
    

def split_first(s, substring):
    i = s.find(substring)
    if i == -1:
        return [s]
    return s[:i], s[i+len(substring):]



def shape_match_helper(arr, shape, axis=0):
    ''' Tests
    bad = np.array([[1, 2], 3])
    shape_match_helper(bad, (2,))
    shape_match_helper(bad, (2,2))
    
    '''
    
    if isinstance(arr, list):
        assert type(arr) == type(shape), \
            f'Shape mismatch axis={axis}, type1={type(arr)}, type2={type(shape)}'
        assert len(arr) == len(shape), \
            f'Length mismatch axis={axis}, type1={len(arr)}, type2={len(shape)}'
        for v, yo in zip(arr, shape):
            shape_match_helper(v, yo, axis=axis + 1)
    elif isinstance(arr, np.ndarray):
        assert arr.shape == shape, \
            f'Shape mismatch axis={axis}, type1={arr.shape}, type2={shape}'
            

class OneMinusLayer(Layer):
    def __init__(self, **kwargs):
        super(OneMinusLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return 1 - inputs
    
class GateControlLayer(Layer):
    def __init__(self, gate_multiplier, gate_bias, **kwargs):
        super(GateControlLayer, self).__init__(**kwargs)
        self.gate_multiplier = K.variable(gate_multiplier)
        self.gate_bias = K.variable(gate_bias)

    def call(self, inputs, **kwargs):
        return inputs * self.gate_multiplier + self.gate_bias
    
    def get_config(self):
        config = {
            'gate_multiplier': self.gate_multiplier.eval(),
            'gate_bias': self.gate_bias.eval()
        }
        base_config = super(GateControlLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def keras_load_variable():
    alpha = K.variable(0.5)

    a = Input((2,))
    b = Dense(3, name='b')(a)
    middle = Lambda(lambda x: x * alpha)(a)
    c = Dense(2, name="c")(middle)
    model = KerasModel(a, [b, c])
    model.compile(
        optimizer='adam',
        loss=['mse', 'mse'])

    save_model(model, 'deleteme.model')
    m = load_model('deleteme.model', custom_objects={'alpha': alpha})

def keras_load_custom_layer():
    from keras.models import save_model, load_model
    a = Input(shape=(2,))
    print(K.ndim(a))
    one_minus_layer = OneMinusLayer(name='one_minus')
    y = one_minus_layer(a)
    gate_control_layer = GateControlLayer(0.5, 0.5, name='gate_control')
    y = gate_control_layer(y)
    model = KerasModel(a, y)
    save_model(model, 'm')
    m = load_model('m', custom_objects={'OneMinusLayer': OneMinusLayer, 
        'GateControlLayer': GateControlLayer})
    model.predict(np.zeros((2, 2)))


def pdf_to_text(pdfname):
    """ Source: https://gist.github.com/jmcarp/7105045
        Dependencies: pip install pdfminer.six
    """
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter # process_pdf
    from pdfminer.pdfpage import PDFPage
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams

    from io import StringIO

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text

def keras_custom_loss():
    def vector_mse(y_true, y_pred):
        from keras import backend as K
        return K.mean(K.square(y_pred - y_true))

    from keras.layers import Dense, Input
    from keras.models import Model as KerasModel
    a = Input((2,))
    b = Dense(3, name='b')(a)
    c = Dense(2, name="c")(a)
    model = KerasModel(a, [b, c])
    model.compile(
        optimizer='adam',
        loss=[vector_mse, vector_mse])

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]*20)
    y = [
        np.zeros((4*20, 3)),
        np.zeros((4*20, 2))
    ]

    model.test_on_batch(X, y)

def keras_transfer_weights():
    model.set_weights(model2.get_weights())

def keras_test_constructing_with_other_models():
    # CAUTION: not sure if y[0] is correct
    # if you build using a model, then train the bigger model, the component
    # will also have changed weights
    from keras.layers import Dense, Input
    a = Input((2,))
    b = Dense(3, name='b')(a)
    c = Dense(2, name="c")(a)
    model = KerasModel(a, [b, c])
    model.compile(
        optimizer='adam',
        loss=['mse', 'mse'])

    print(model.get_weights(), '\n')

    x = Input((2,))
    y = model(x)
    y1 = Dense(3)(y[0])
    y2 = Dense(2)(y[1])
    model2 = KerasModel(x, [y1, y2])
    model2.compile(
        optimizer='adam',
        loss=['mse', 'mse'])

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]*20)
    y = [
        np.zeros((4*20, 3)),
        np.zeros((4*20, 2))
    ]

    print(model.get_weights(), '\n')
    model2.train_on_batch(X, y)
    print(model.get_weights())

def keras_drop_output():
    # CAUTION: not guaranteed
    from keras.layers import Dense, Input
    from keras.models import Model
    a = Input((2,), name='a')
    b = Dense(3, name='b')(a)
    c = Dense(2, name="c")(a)
    model = Model(a, [b, c])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    a = model.get_layer('a').output
    b = model.get_layer('b').output
    model2 = Model(a, b)
    model2.summary()

    a = model.input
    b = model.output[0]
    model2 = Model(a, b)
    model2.summary()

    a = model.input_layers[0].output
    b = model.output_layers[0].output
    model2 = Model(a, b)
    model2.summary()

def hdf5():
    batch_size = 32
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('data_x', shape=(50, 20, 60))
    data = h5f['data_x']
    for i in range(0, 64, batch_size):
        data[i:i+batch_size] = np.ones((20, 60)) + i
    h5f.attrs['length'] = 0
    h5f.close()

    h5f = h5py.File('data.h5', 'r')
    print(h5f['data_x'][0])
    print(type(h5f.attrs['length']), h5f.attrs['length'])
    print(h5f['data_x'].shape)
    h5f.close()

def hdf5_resize():
    batch_size = 32
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('data_x', shape=(50, 20, 60), 
                       chunks=(32, 20, 60), maxshape=(None, 20, 60))
    h5f['data_x'].resize((64, 20, 60))
    print(h5f['data_x'].shape)
    h5f.close()

def keras_gradient
    warnings.warn('This code requires tf backend and still gives an error')
    # model.total_loss, model.trainable_weights
    from keras.layers import Dense, Input
    from keras.models import Model
    a = Input((2,), name='a')
    b = Dense(3, name='b')(a)
    c = Dense(2, name="c")(a)
    model = Model(a, b)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    gradients = K.gradients(model.total_loss, model.trainable_weights)
    import tensorflow as tf
    trainingExample = np.random.random((1,2))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    evaluated_gradients = sess.run(gradients,feed_dict={model.input :trainingExample})

import pickle, gzip
def cpkl_save_all(filename, objects):
    ''' save objects into a compressed diskfile '''
    fil = gzip.open(filename, 'wb')
    for obj in objects: pickle.dump(obj, fil)
    fil.close()

def cpkl_load_all(filename):
    ''' reload objects from a compressed diskfile '''
    fil = gzip.open(filename, 'rb')
    while True:
        try: yield pickle.load(fil)
        except EOFError: break
    fil.close()

def compressed_pkl_experiments():
    def compute_size(save_fn, n_arr):
        path = 'hi.pkl'
        save_fn(path, [np.ones(100,)]*n_arr)
        return os.path.getsize(path)
    sizes = [compute_size(pkl_save_all, x) for x in range(1, 50)]
    c_sizes = [compute_size(save, x) for x in range(50)]

    pw.altLines(c_sizes)
    pw.altLines(sizes)


def tests(stream):
    for i, val in enumerate(stream):
        pass
    print(i, 'total examples')
    for i, (x,y,w) in enumerate(stream):
        if i >= 3:
            break
        print('x:', x)
        print('\ty:', y)
        print('\tw:', w)
    from deepengine.streams import Batcher
    from deepengine.streams.stream_utils import BatcherUtil
    batcher = Batcher(stream, 32)
    BatcherUtil.check_batcher(batcher)


def check_equal(x, y):
    assert type(x) == type(y)
    if isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for a, b in zip(x, y):
            check_equal(a, b)
    elif isinstance(x, np.ndarray):
        assert x.shape == y.shape
        assert np.allclose(x, y)
    else:
        assert x == y

def filePathToFileName(path):
    return os.path.basename(path)

def getParentDir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


                
def my_random_search(choice_lists, size):
    for choice_list in choice_lists:
        print(list(np.random.choice(choice_list, size=size)))

def mygetfloat(text):
    return re.findall('\d+(\.\d+)?', text)[0]

def trymap(fn, arr):
    def try_fn(x):
        try:
            return fn(x)
        except Exception as ex:
            print(str(ex))
            return None

    return list(map(try_fn, arr))



def mychoice(arr, size=1, p=None):
    return np.random.choice(arr, size=size, p=p)

def myisnumber(val):
    # alternatively implement a try catch block
    return isinstance(np.isfinite(val), np.bool_)

def test_myisnumber():
    pos = [0, 1.0, np.array(1.5)]
    neg = [[0, 0], np.array([1])]
    for val in pos:
        assert myisnumber(val), val
    for val in neg:
        assert not myisnumber(val), val

def myindex(arr, i):
    for j, val in enumerate(arr):
        if j == i:
            return val



def array_to_img(x, dim_ordering='default', scale=True):
    # copied from https://www.kaggle.com/hexietufts/easy-to-use-keras-imagedatagenerator
    from PIL import Image
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    # copied from https://www.kaggle.com/hexietufts/easy-to-use-keras-imagedatagenerator
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


# PYTHON HACKING
def print_imports(module):
    l = dir(__import__(module, fromlist=['*']))
    remove = set('np,os,json'.split(','))
    l = set(l) - remove
    return ', '.join([name for name in l if not name.startswith('_')])

d = {'a':1, 'b':1}
def assign_dict(d):
    import inspect, re
    keys = list(d.keys())
    lhs = ', '.join(keys)
    dict_name = ''
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        # Line is the actual text of the line of code which called this function.
        # We use regex to extract the argument.
        # \s means a whitespace character
        m = re.search(r'\bassign_dict\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            dict_name = str(m.group(1))
            break
    assert len(dict_name) > 0
    rhs = ', '.join([f'{dict_name}["{key}"]' for key in keys])
    return lhs + ' = ' + rhs

# PYTHON HACKING ^^^^^^

def re_overview():
    import re

    ''' Regex patterns
    .+
    .*
    [^hi]+
    ^[hi]?$
    (abc|ghi)?def # -> abcdef
    [A-Za-z_]+
    \s # unicode whitespace
    # m.group(i) seems equivalent to m[i]
    # m[0] = whole match
    # m[1] = first group (I think first parentheses)
    \d # a single digit
    \w # unicode characters
    \s # whitespace
    '''

    re.search('(abc|ghi)?def', 'abcdef def').group(0) # 'abcdef'

    re.split('aa', 'abacadaaaf') # ['abacad', 'af']
    # returns a list of strings

    m = re.search('(abc|bc)?def', 'abcdef')
    # m is None if not match, otherwise a match object
    # returns first match (I think)
    m[0]
    m.span() # (0, 6)

    re.split('aa', 'abacadaaaf')
    # returns a list of strings
    # ['abacad', 'af']

    re.findall('aa', 'abacadaaaf') # ['aa']
    # non-overlapping matches as a list of strings
    # DON'T use this findall('a(b)', 'ab') gives ['b']


    for i in re.finditer('[a-z]', 'abacadaaaf'):
        print(i.group(0))

    # sub
    # more regex testing
    # assert check before and afterwards
    print(re.sub(r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):',
        r'static PyObject*\npy_\1(void)\n{',
        'def myfunc():'))
    # static PyObject*
    # py_myfunc(void)
    # {

    pattern = re.compile('abc?')
    pattern.findall('abcd') # ['abc']

    re.search('(?<=abc)def', 'abcdef')[0] # 'def'
    re.search('(?<!abc)def', 'def')[0] # 'def'


def overlapping_hist():
    import random
    import numpy
    import matplotlib.pyplot as plt

    x = [random.gauss(3,1) for _ in range(400)]
    y = [random.gauss(4,2) for _ in range(400)]

    bins = numpy.linspace(-10, 10, 100)

    plt.hist(x, bins, alpha=0.5, label='x')
    plt.hist(y, bins, alpha=0.5, label='y')
    plt.legend(loc='upper right')
    plt.show()

def filldl(dl):
    dl = dict(dl)
    if len(dl.keys()) == 0:
        return dl
    max_len = 1
    for v in values:
        if isinstance(v, list):
            max_len = max(max_len, len(v))
    # unfinished
    pass

def keras_custom_regularizer():
    class CustomRegularizer:
        def __init__(self, shape, multiplier=None, offset=None):
            self._shape = shape
            if multiplier is None:
                multiplier = 0.0
            if offset is None:
                offset = np.zeros(shape)
            else:
                offset = np.array(offset)
            self._offset = offset
            self.offset = K.variable(offset)
            self._multiplier = multiplier
            self.multiplier = K.variable(multiplier)
            
        def __call__(self, weight_matrix):
            return self.multiplier * K.sum(K.abs(weight_matrix - self.offset))
        
        def set_offset(self, offset):
            self._offset = offset
            K.set_value(self.offset, offset)
            
        def set_multiplier(self, multiplier):
            self._multiplier = multiplier
            K.set_value(self.multiplier, multiplier)
        
        def get_config(self):
            config = {
                'shape': self._shape,
                'multiplier': self._multiplier,
                'offset': self._offset.tolist()
            }
            base_config = super(CustomRegularizer, self).get_config()
            return {**base_config, **config}
            #return config


    a = Input((2,))
    kernel_reg = CustomRegularizer((2, 3))
    bias_reg = CustomRegularizer((3,))
    b = Dense(3, name='b', kernel_regularizer = kernel_reg, bias_regularizer = bias_reg)(a)
    #b = Dense(3, name='b')(a)
    model = KerasModel(a, b)
    #model.get_layer('b').kernel_regularizer = kernel_reg
    #model.get_layer('b').bias_regularizer = bias_reg
    model.compile(
        optimizer='adam',
        loss='mse')
    model.save_weights('temp4.h5')
    model.load_weights('temp4.h5')

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]*1000)
    y = np.ones((4000, 3))

    print('Before training')
    print(model.test_on_batch(X, y))
    K.set_value(kernel_reg.offset, model.get_layer('b').get_weights()[0])
    K.set_value(bias_reg.offset, model.get_layer('b').get_weights()[1])
    print(model.test_on_batch(X, y))
    print(model.predict(np.ones((1, 2))))

    model.fit(X, y, epochs=2)
    print('After training')
    print(model.test_on_batch(X, y))
    print(model.predict(np.ones((1, 2))))

    K.set_value(kernel_reg.multiplier, 100.0)
    K.set_value(bias_reg.multiplier, 100.0)

    model.fit(X, y, epochs=2)
    print('After training')
    print(model.test_on_batch(X, y))
    print(model.predict(np.ones((1, 2))))


def list_cross_product(*l):
    import itertools
    return itertools.product(*l)


def set_xticks(axis, index, labels):
    # Works for axis
    index_min = min(index)
    
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in index:
            return labels[int(tick_val) - index_min]
        else:
            return ''
    
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    axis.xaxis.set_major_formatter(FuncFormatter(format_fn))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

def test_set_xticks():
    fig, axes = plt.subplots(2, 2)
    axis = axes.reshape((-1,))[0]
    axis.scatter([6, 7, 8], [5, 6, 7])
    set_xticks(axis, [6, 7, 8], 'a,b,c'.split(','))
    plt.show()

def get_closest_points(train, test, metric='euclidean', n=None, frac=1):
    """get n closest points or the fraction of train that is closest to the 
       points in test
      :returns: 3 lists
    """
    from sklearn.neighbors import NearestNeighbors
    import warnings

    if metric is None:
        warnings.warn("'metric' shouldn't be None. Correcting to be 'euclidean'.")
        metric = 'euclidean'

    if n is None:
        n = int(len(test) * frac)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=metric).fit(train)
    distances, labels = nbrs.kneighbors(test)

    indices = np.array(list(range(len(test)))).reshape((-1, 1))
    concat = np.concatenate([distances, labels, indices], axis=1)
    
    # Don't use np.sort! It doesn't preserve rows.
    concat = concat[concat[:,0].argsort()]

    distances = list(concat[:n, 0].reshape((-1,)))
    labels = list(concat[:n, 1].reshape((-1,)).astype(int))
    indices = list(concat[:n, 2].reshape((-1,)).astype(int))

    return indices, distances, labels

def print_traceback():
    import traceback
    traceback.print_stack()

from keras.engine.topology import Layer
class BiasLayer(Layer):
    def __init__(self, bias_initializer='zeros', **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.bias_initializer = bias_initializer
        
    def build(self, input_shape):
        self.offset = self.add_weight(name='bias',
                                      shape=input_shape[1:],
                                      initializer=self.bias_initializer,
                                      trainable=True)
        super(BiasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.offset

''' Version 2 with regularizer and constraint
class BiasLayer(Layer):
    def __init__(self, initializer='zeros', regularizer=None, 
                 constraint=None, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint
        
    def build(self, input_shape):
        self.offset = self.add_weight(
            name='bias', shape=input_shape[1:], initializer=self.initializer,
            regularizer=self.regularizer, constraint=self.constraint, trainable=True)
        super(BiasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.offset

'''

def test_keras_bias_layer():
    a = Input((2,), name='a')
    g = BiasLayer(name='grad')(a)
    c = Dense(3)(g)
    model = KerasModel(a, c)
    model.compile(loss='mse', optimizer='adam')

    layer = model.get_layer('grad')
    print(layer.get_weights()) # [array([ 0.,  0.], dtype=float32)]
    model.train_on_batch(np.array([[1, 2]]), np.array([[1, 2, 3]]))
    print(layer.get_weights()) # [array([ 0.001,  0.001], dtype=float32)]


def get_req_args(arg_names):
    # hi, there = get_required_args(['hi', 'there'])
    import argparse
    parser = argparse.ArgumentParser()
    for arg_name in arg_names:
        parser.add_argument(arg_name)
    args = parser.parse_args()
    result = []
    for arg_name in arg_names:
        result.append(args.__getattribute__(arg_name).strip())
    return result


def compute_gradients(x, y, model, weight_tensor):
    input_tensors = [model.inputs[0],
                     model.sample_weights[0],
                     model.targets[0],
                     K.learning_phase()]

    inputs = [x, np.ones(len(x)), y, 0]
    
    gradients = model.optimizer.get_gradients(model.total_loss,
                                              weight_tensor)
    get_gradients = K.function(input_tensors, gradients)
    grad_result = get_gradients(inputs)
    return grad_result

def print_shapes(model, stream):
    shapes = {}
    shapes['stream_shape'] = custom_shape(stream)
    point = list(islice(stream, 1))[0]
    shapes['point_shape'] = custom_shape(stream)
    try:
        batcher = Batcher(stream, 5)
        batch = list(islice(batcher, 1))[0]
        shapes['batch_shape'] = custom_shape(batch)
    except Exception as ex:
        print(f'Error occurred with batcher: {ex}')
    
    shapes['model_input_shape'] = fix_keras_shape(model.input_shape)
    shapes['model_input_shape'] = fix_keras_shape(model.output_shape)
    print(shapes, '\n')
    
    print('Point-stream match:', shapes['stream_shape'] == shapes['point_shape'])
    print('input-stream_x match:', shapes['model_input_shape'] == shapes['stream_shape'][0])
    print('output-stream_y match:', shapes['model_input_shape'] == shapes['stream_shape'][1])

def tensorboard_experiment():
    from keras.callbacks import TensorBoard
    set_keras_backend('tensorflow')

    m = custom_keras_model()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = [
        np.zeros((4, 3)),
        np.zeros((4, 2))
    ]
    m.fit(X, y, epochs=3, callbacks=[TensorBoard('/Users/hromel/tf_logs/')])


'''
******************************************* 
EXPERIMENT HELPER FUNCTIONS ***************
*******************************************
'''

def parse_fake_yaml(text):
    """ Parse yaml-like text of the form
    hi:
      yo > hi
      hello
    """
    lines = reversed(text.split('\n'))
    lines = filter(lambda line: len(line.strip()) > 0, lines)
    tag_files = {}
    rules = []
    for line in lines:
        if re.match('\s.+', line) is not None:
            # is rule in tag file
            splitted = line.strip().split('>')
            assert len(splitted) in [1, 2]
            if len(splitted) == 1:
                rule = (splitted[0], '')
            else:
                rule = tuple(splitted)
            rules.append(rule)
        elif re.match('.+:\s*', line) is not None:
            # is tag file
            tag_files[line.strip()[:-1]] = rules
            rules = []
    return tag_files

import os
from collections import Counter

class FileDatasetManager:
    _WORKING_DIRS = ('tags', 'rules', 'datasets', 'labels')
    _WORKING_FILES = ('filelist.txt',)
    
    def __init__(self, input_dir, working_dir):
        self._input_dir = input_dir
        self._working_dir = working_dir
        for d in self._WORKING_DIRS:
            self._create_missing_dir(d)
        for f in self._WORKING_FILES:
            self._create_missing_file(f)
            
        print('Working directory is', working_dir)
        print('\tsize =', self._get_dir_size())
        print('Input directory is', input_dir)
        self._existing_file_paths = list(self._get_paths(input_dir))
        self._existing_file_names = list(map(os.path.basename, self._get_paths(input_dir)))

        # Consider duplicate file names
        file_name_counter = Counter(self._existing_file_names)
        dup_file_names = {}
        for k, v in dup_file_names.items():
            if v >= 2:
                dup_file_names[k] = []
        if len(dup_file_names.keys()) > 0:
            for path, name in zip(self._existing_file_paths, self._existing_file_names):
                if name in dup_file_names.keys():
                    dup_file_names[name].append(path)
            print('There are duplicate files!')
            for k, v in dup_file_names.items():
                print(k)
                for path in v:
                    print('\t{}'.format(path))
        self._dup_file_names = dup_file_names
    
    def _get_dir_size(self):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self._working_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        size_names = ['B', 'KB', 'MB', 'GB', 'TB']
        for i, name in enumerate(size_names):
            if total_size < 1000:
                return f'{int(total_size)} {size_names[i]}'
                break
            total_size /= 1000
        raise BaseException(f"Directory is too big: {total_size} PB")
        
    def _get_paths(self, directory):
        for paths, subdirs, files in os.walk(directory):
            for file in files:
                #print(name, paths)
                pure_path = os.path.join(paths, file)
                yield pure_path
    
    def _parent_dir(self, path):
        return os.path.abspath(os.path.join(path, os.pardir))
    
    def _create_missing_dir(self, rel_path):
        path = os.path.join(self._working_dir, rel_path)
        if not os.path.exists(path):
            os.makedirs(path)
            print('Created folder path', path)
    
    def _create_missing_file(self, rel_path):
        path = os.path.join(self._working_dir, rel_path)
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                pass
    
    @property
    def rule_folder_path(self):
        os.path.join(self._working_dir, 'rules')
    
    @property
    def tag_folder_path(self):
        os.path.join(self._working_dir, 'tags')
        
    @property
    def labels_folder_path(self):
        os.path.join(self._working_dir, 'labels')
        
    @property
    def labels_folder_path(self):
        os.path.join(self._working_dir, 'datasets')
        
    def _input_rel_path(self, path):
        return os.path.relpath(path, self._input_dir)
    
    # IN PROGRESS *******
    def _get_keras_labels(self):
        # compute the label for each file assuming the keras folder format
        labels = [''] * len(self._existing_file_paths)
        for i, path in enumerate(self._existing_file_paths):
            category = os.path.basename(self._parent_dir(path))
            label[i] = category
        return label
    
    
    # IN PROGRESS ********
    def create_tags(self, text):
        tag_files = parse_fake_yaml(text)
        
        for tag_file_name, rules in tag_files.items():
            tag_file_path = os.path.join(self.tag_folder_path, tag_file_name)
            
            with open(os.path.join(file_name), 'w') as f:
                for file_path in self._existing_file_paths:
                    rel_path = self._input_rel_path(file_path)
                    parent_dir = self._parent_dir(rel_path)
                    
                    if parent_dir in rules.keys():
                        f.write(rel_path + ',' + rules[parent_dir] + '\n')
            print('Created tag file:', tag_file_name)

    
# parse_fake_yaml('''
# hi:
#     yo
# ''')
# yields {'hi': [('yo', '')]}

'''
FileDatasetManager
* check for duplicate file names
* get file paths (can then shuffle list then split)
* get total folder size
* get keras labels
- resolve duplicates
- TODO: reading csv labels, moving files to a new structure (afterwards save a move map so 
it's possible to reverse the map), save raw labels in a label folder, which can be converted 
to more useful forms, save data set file lists
- tag by folder
- construct mapping of folder names to folder paths
  - folder rules mapping folders to labels (chould be recursive)

** File types
file maps
labels for 1 dataset
tags

m = FileDatasetManager('input_dir', 'working_dir')
# > there are these duplicates
m.resolve_duplicates(mode='....blah.....')
m.create_label('label_1.txt', folders=['cats'], default='cat')
# > Label created in labels/label.txt
m.create_tags('
tag_1.txt:
    cats > cat,mammal
tag_2.txt:
    yo
')
m.save_keras_labels('label_2.txt')
m.save_labels(lambda path: 'hi', 'labels_1.csv')

m.update()
m.copy_dataset('dataset_1.csv')
m.combine_labels(['label_1.csv', 'label_2.csv'])

Folders
- tags
- labels
- filelist.txt
- datasets


init manager
correct duplicates
print a tag map based on parent folders
save folder format as a label file
have a function which takes in a function and saves a label

add new files
create incomplete label file (specify default and folders)
copy based on dataset map
combine label files


from sklearn.metrics import accuracy_score

accuracy_score(clf.predict(data), labels)
'''

0.7


'''
- value fill implications
- warn broken conventions
- colmap (ie change column names)
  input can be a dict, or a function such as strip, or a list of function
- select, filter, mutate
- toreuse.py functions

Not Started
- describe missing data (by column, in terms of rows)
- detect categorical columns
- convert string to int/float
- delete empty column
- IsString, IsFloat, IsInt
- apply model and append predictions
- df['train_acc > 0.6']
- warn if column names are contained in other column names
- maybe: informative errors for yo (ie missing key)
- possible bugs
  - execute which reads 'class' as a variable as opposed to the key to df

- pandas melt, pivot, concat

Unsorted
dfw
- look for -> 

'''
def quotes_wrap(col_name):
    if col_name[0] == col_name[-1]:
        if col_name[0] == '"':
            return col_name
        elif col_name == "'":
            return col_name
    return '"' + col_name + '"'

def df_wrap(col_name, result):
    # wrap in df["..."]; second part needs to be r'...'
    result = re.sub(f"('{col_name}')", r'df[\1]', result)
    result = re.sub(f'("{col_name}")', r'df[\1]', result)
    return re.sub(f'(?<![\'"])({col_name})', r'df["\1"]', result)

def df_values_wrap(col_name, result):
    return re.sub(f'({col_name})', r'df["\1"].values', result)

#result = '"hello" = there == '+ "'hello'"
#for val in ['hello', 'there', 'yu']:
#    result = df_wrap(val, result)
#print(result)

def quotes_wrap(col_name):
    if col_name[0] == col_name[-1]:
        if col_name[0] == '"':
            return col_name
        elif col_name == "'":
            return col_name
    return '"' + col_name + '"'

def get_quote_match(text, i):
    # i is the first quote
    quote = text[i]

    for j in range(i + 1, len(text)):
        if text[j] == quote:
            return j
    assert False, 'reached end of {} without a matching quote'.format(text[i:])

def next_non_whitespace(text, i):
    j = re.search('[^\s]', text[i:]).start() + i
    #print('next_non_whitespace', text[j], 'from', text[i:j+1])
    return j

def col_name_end(text, i):
    for j in range(i, len(text)):
        if text[j] in ',>':
            #print('col_name_end', text[j], 'from', text[i:j+1], 'in', text)
            return j
    assert False, 'reached end of {} without an end'.format(text[i:])

def process_column_rename(text):    
    text = text.strip()
    text_forwards = text
    text_backwards = ''.join(list(reversed(text)))
    lhs = []
    rhs = []
    n = len(text)
    quotes = "'" + '"'
    forward_iter = next_non_whitespace(text_forwards, 0)
    backward_iter = next_non_whitespace(text_backwards, 0)
    loop_counter = 0
    
    while forward_iter + backward_iter < n - 1:
        # while there exists a pair
        # get lhs part
        
        # BUG: must col_name_end after get_quote_match
        lhs_begin = forward_iter
        if text_forwards[lhs_begin] in quotes:
            lhs_end = get_quote_match(text_forwards, lhs_begin) + 1
        else:
            lhs_end = col_name_end(text_forwards, forward_iter + 1)
        lhs_part = text[lhs_begin:lhs_end].strip()
        lhs.append(quotes_wrap(lhs_part))
        
        # get rhs part
        rhs_end = backward_iter
        if text_backwards[rhs_end] in quotes:
            rhs_begin = get_quote_match(text_backwards, rhs_end) + 1
        else:
            rhs_begin = col_name_end(text_backwards, rhs_end + 1)
        
        forward_iter = col_name_end(text_forwards, lhs_end) + 1
        backward_iter = col_name_end(text_backwards, rhs_begin) + 1
        forward_iter = next_non_whitespace(text_forwards, forward_iter)
        backward_iter = next_non_whitespace(text_backwards, backward_iter)
        
        rhs_begin = n - rhs_begin
        rhs_end = n - rhs_end
        rhs_part = text[rhs_begin:rhs_end].strip()
        rhs.append(quotes_wrap(rhs_part))
        
        assert lhs_end <= forward_iter, (lhs_end, forward_iter)
        assert (n - rhs_end) <= backward_iter
        
        loop_counter += 1
        assert loop_counter < 10
    
    return dict(zip(lhs, rhs))


import warnings
from sklearn.preprocessing import LabelEncoder
class DataFrameWrapper:
    _BAD_COLUMN_CHARS = [' ', '>', '=', '-', ',']
    
    def __init__(self, data_source, header=0, columns=None, verbose=1):
        # verbose = 0 => no logging
        # verbose = 1 => regular logging
        self.verbose = verbose
        if isinstance(data_source, str): # file path
            self.df = pd.read_csv(data_source, header=header)
        else:
            self.df = pd.DataFrame(data_source, columns=columns)
        self._check_col_format()
        if verbose > 0:
            self.summary()
        
    def _check_col_format(self):
        for col in self.columns: # REDO with re *******
            for char in self._BAD_COLUMN_CHARS:
                if char in col:
                    warnings.warn(f'"{char}" character in column {col}')
            if len(col) > 0:
                if col[0] == col[-1]:
                    if col[0] == "'":
                        warnings.warn(f"column {col} is wrapped in ' quotes")
                    if col[0] == '"':
                        warnings.warn(f'column {col} is wrapped in " quotes')
    
    def _print_columns(self):
        print('Columns:')
        columns = self.df.columns.values
        for col in columns:
            line = f'\t{col}\t{self.df.dtypes[col]}'
            if len(self.df[col].values) > 0:
                
                line += f'\t{self.df[col].values[:2]}'
            print(line)
        
    def summary(self):
        self._print_columns()
        print('\nDataFrame describe')
        print(self.df.describe())
        # ** missing values
        print('\nDataFrame head')
        print(self.df.head())
        
    def __repr__(self):
        return self.df.__repr__()
    
    def __str__(self):
        return self.df.__str__()
        
    def drop_col(self, columns):
        if isinstance(columns, str):
            columns = [x.strip() for x in columns.split(' ')]
        missing_columns = set(columns) - set(self.columns)
        assert len(missing_columns) == 0, (
            f'DataFrame does not have columns: {missing_columns}' +
            f'\n- DataFrame columns {self.columns}')
        self.df = self.df.drop(list(columns), 1)
        
    def to_csv(self, file_path, index=False):
        self.df.to_csv(file_path, index)
        
    def __len__(self):
        return len(self.df.index)
    
    @property
    def dtypes(self):
        return dict(self.df.dtypes)
    
    @property
    def columns(self):
        return self.df.columns.values
    
    @property
    def values(self):
        return self.df.values
    
    def to_numeric(self, errors='ignore'):
        for col in self.df.columns:
            if self.df[col].dtype == object:
                self.df[col] = LabelEncoder().fit_transform(self.df[col].values)
        self.df = self.df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        if self.verbose > 0:
            self._print_columns()
        return self
    
    # Experimental
    def _process_rule(self, rule):
        # basic
        parts = [x.strip() for x in rule.split(' > ')]
        if len(parts) == 2:
            return parts
        else:
            assert False
        
    def __getitem__(self, key):
        if key in self.df.columns:
            return self.df[key]
        elif len(self) == 0:
            print('Empty DataFrame')
            return None
        elif ',' in key:
            keys = [x.strip() for x in key.split(',')]
            return self.df[keys]
        else:
            try:
                return self.df.query(key)
            except as ex:
                print('dfw getitem failure:', ex)
                print('Columns:', list(self.columns))
                raise ex
    
    def _change_col_name(self, old_col, new_col):
        self.df = self.df.rename(columns={old_col: new_col})
    
    def rename_col(self, rules): # adding/deleting columns ******
        # *** look at each line before applying
        if isinstance(rules, dict):
            self.df = self.df.rename(rules)
        elif isinstance(rules, str):
            lines = rules.split('\n')
            for line in lines:
                yo = line.split('>') # ** rename yo
                if len(yo) == 2:
                    old_col = yo[0].strip()
                    if old_col in self.df.columns.values:
                        new_col = yo[1].strip()
                        self._change_col_name(old_col, new_col)
                        print('Applied:', line)
                    else:
                        print(f'Ignored ({old_col} not a column):', line)
                elif len(yo) == 0:
                    pass
                else:
                    print('Ignored:', line)
        elif isinstance(rules, list):
            for rule in rules:
                self.colmap(rule)
        print('\nColumns:', self.columns)
        
    def yo(self, text):
        df = self.df.copy()
        log_to_print = []
        code_to_print = []
        errors = []
        for line in text.split('\n'):
            line = line.strip()
            if len(line) == 0:
                continue
            elif line.startswith('#'):
                continue # ignore comments
            elif line.startswith('del '):
                code_line = line
                for col_name in df.columns.values:
                    code_line = df_wrap(col_name, code_line)
            elif True: # assignment
                lhs, rhs = line.split('=', 1)
                lhs, rhs = lhs.strip(), rhs.strip()
                # wrap lhs in df[...]
                if lhs.startswith('"') or lhs.startswith("'"):
                    lhs = 'df[' + lhs + ']'
                else:
                    lhs = 'df["' + lhs + '"]'
                
                # wrap rhs in df[...]
                for val in df.columns.values:
                    rhs = df_values_wrap(val, rhs)

                try: # does the rhs have a length?
                    len(eval(rhs))
                except:
                    rhs = 'np.array([' + rhs + f'] * len(df.index))'
                    
                code_line = lhs + ' = ' + rhs
                        
            try:
                exec(code_line)
                log_to_print.append('Executed: ' + line)
                code_to_print.append(code_line)
            except Exception as ex:
                errors.append('ERROR: Failed this line ' + line)
                errors.append('\tTransformed Code: ' + code_line)
                errors.append('\t' + str(ex))
        
        if len(errors) > 0:
            print('yo failed')
            for line in errors:
                print(line)
        else:
            self.df = df
            print('yo succeeded')
            for line in log_to_print:
                print(line)
            print('')
            print('Transformed Code')
            for line in code_to_print:
                print(line)
            
    
    def split_tvt(self, proportions):
        pass
    
    def split_tt(self, proportions):
        pass
        
        
def dataframe_wrapper_test():
    DFW = DataFrameWrapper
    DF = pd.DataFrame

    df = DFW({'a >': ['1', '32'], 'b': ['hi', 0]})
    df.to_numeric()
    #df.rename_col('''
    #b > c
    #''')
    df = DFW({'class': [1], 'length': [1],'width': [1],'my_list': [1]}, verbose=0)
    df.yo('''
    # New Columns
    class = length / width
    #class = my_function(class)
    class = np.mean(my_list, axis=-1)
    class = 0

    # Delete Old Columns
    del ['class', "length", width]
    ''')

    df.yo('''
    # Change Column Names
    'a >' > a
    "b >" > b

    # Transformations
    class == 1 and my_list > 0
        -> class = 2

    # Delete Old Columns
    del (a, b)
    del c
    del [d]
    del ['e', 'f']

    ''')

def keras_add_loss():
    a = Input((2,))
    b = Dense(3, name='b')(a)
    #c = Dense(2, name="c")(a)
    model = KerasModel(a, b)
    model.compile(
        optimizer='adam',
        loss='mse')
    layer = model.get_layer('b')
    weight_tensor = layer.__getattribute__('kernel')
    layer.add_loss(l2(1_000_000)(weight_tensor))
    # recompiling is necessary for the loss to be added!
    model.compile(
        optimizer='adam',
        loss='mse')

    x, y = fake_data(model, 50000)
    print(layer.get_weights()[0].reshape((-1,)))
    model.fit(x, y)
    print(layer.get_weights()[0].reshape((-1,)))
    model.fit(x, y)
    print(layer.get_weights()[0].reshape((-1,)))

def _concat(a, b):
    assert type(a) == type(b), (type(a), type(b))
    if isinstance(a, tuple):
        return tuple(_concat(a_val, b_val) for a_val, b_val in zip(a, b))
    elif isinstance(a, list):
        return [_concat(a_val, b_val) for a_val, b_val in zip(a, b)]
    elif isinstance(a, np.ndarray):
        return np.concatenate([a, b], axis=0)  # axis=0 not necessary
    else:
        raise BaseException(f'Unhandled type {type(a)}')


class DataSet:
    """
    Manages datasets of numpy arrays x, y, and w

    Future Improvements:
    - slicing returns a DataSet
    - rename as DataArrayTuple
    - handle tuple of lists of arrays
    """

    def __init__(self, x, y, w, one_hot=True, max_nbytes=100_000_000):
        self._x = x
        self._y = y
        self._w = w
        self._one_hot = one_hot
        self._max_nbytes = max_nbytes

    def __getitem__(self, item):
        x = self._x[item]
        y = self._y[item]
        w = self._w[item]
        return x, y, w

    def __len__(self):
        x = self._x
        if isinstance(x, list):
            x = x[0]
        return len(x)

    def __iter__(self):
        for x, y, w in zip(*self.xyw):
            yield x, y, w

    def shuffle(self):
        permutation = np.random.permutation(len(self))
        self._x, self._y, self._w = self._shuffle_helper(self.xyw, permutation)

    def _shuffle_helper(self, to_shuffle, permutation):
        if isinstance(to_shuffle, tuple):
            return tuple(self._shuffle_helper(elem, permutation) for elem in to_shuffle)
        elif isinstance(to_shuffle, list):
            return [self._shuffle_helper(elem, permutation) for elem in to_shuffle]
        elif isinstance(to_shuffle, np.ndarray):
            return to_shuffle[permutation]
        else:
            raise BaseException(f'Unhandled type {to_shuffle}')

    def append_dataset(self, other):
        if isinstance(other, DataSet):
            self._x, self._y, self._w = _concat(self.xyw, other.xyw)
        else:
            raise BaseException(f'Unhandled type {type(other)}')

    def append_point(self, other):
        import warnings
        warnings.warn('This may be wrong since Dataset shape has n_examples first')
        if isinstance(other, (DataStream, tuple)):
            self._x, self._y, self._w = _concat(self.xyw, other)
        else:
            raise BaseException(f'Unhandled type {type(other)}')

    @property
    def xyw(self):
        return self._x, self._y, self._w

    @property
    def xy(self):
        return self._x, self._y

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __str__(self):
        return self.xyw.__str__()

    def __repr__(self):
        return self.xyw.__repr__()

    @property
    def nbytes(self):
        return self._x.nbytes + self._y.nbytes + self._w.nbytes

    def __add__(self, other):
        result = self.copy()
        result.append_dataset(other)
        return result

    def split_by_p(self, p):
        """
        Partition the data into DataSet objects according to distribution p
        Example: p = [0.25, 0.75] means the first DataSet is the first 0.25 of the data and the
                 second DataSet is the rest of the data
        :param p: list of >= 0 floats which sum to 1
        :return: list of DataSets corresponding to splitting according to p
        """
        assert np.isclose(np.sum(p), 1)
        data_len = len(self)
        lengths = [int(data_len * frac) for frac in p]
        # missing points at the end
        lengths[-1] += data_len - sum(lengths)
        ptr = 0
        result = []
        for length in lengths:
            result.append(DataSet(*self[ptr:ptr + length]))
            ptr += length
        return result

    def split_uniformly(self, n):
        # split into alist of n roughly, equally sized DataSet objects
        return self.split_by_p([1.0 / n] * n)

    def split_by_category(self, categories):
        """
        The categories argument labels each example with an category id.
        Split self so that each DataSet corresponds to all examples with a particular category id.
        :param categories: list of >= 0 integers with length the same as self
        :return: list of DataSet objects
        """
        assert len(categories) == len(self), (len(categories), len(self))

        category_indices = dict([(category, []) for category in set(categories)])
        for i, category in enumerate(categories):
            category_indices[category].append(i)

        return [DataSet(*self[category_indices[category]]) for category in
                sorted(category_indices.keys())]

    @property
    def int_category_y(self):
        """ If the y is format is categorical output return it in its non-one-hot vector form"""
        y_values = self._y.astype(int)
        if self._one_hot:
            y_values = np.argmax(y_values, axis=1)
        return y_values

    def split_by_y(self):
        """
        Partition self into multiple DataSets based on the categorial class indicated by y
        :return: list of DataSet objects
        """
        y_values = self.int_category_y

        assert len(y_values.shape) == 1, y_values.shape
        return self.split_by_category(y_values)

    def split_tvt(self, p):
        """
        Split into train, validation, and test DataSets
        :param p: list of 3 floats (ie [train_frac, val_frac, test_frac]) which sum to 1
        :return: list of 3 DataSet objects
        """
        assert len(p) == 3, p
        return self.split_by_p(p)

    def split_tt(self, p):
        """
        Split into train and test DataSets
        :param p: list of 2 floats (ie [train_frac, test_frac]) which sum to 1
        :return: list of 2 DataSet objects
        """
        assert len(p) == 2, p
        return self.split_by_p(p)

    def split_by_index(self, index):
        """
        Given an iterable of indices, partition self into 2 DataSet objects:
        - one including examples with indices in index
        - other data
        :param index: indices of examples to include
        :return: tuple of 2 DataSet objects
        """
        out_index = list(set(range(len(self))) - set(index))
        return DataSet(*self[index]), DataSet(*self[out_index])

    def sample_index(self, n=None, frac=None, p=None):
        """
        Sample data point indices without replacements
        :param n: number of examples to sample
        :param frac: fraction of examples in self to sample
        :param p: distribution over all examples when sampling
        :return: 1-dimensional numpy array of sampled indices
        """
        if frac is not None:
            n = int(len(self) * frac)
        return np.random.choice(list(range(len(self))), size=n, p=p, replace=False)

    def sample(self, n=None, frac=None, p=None):
        # sample data points without replacement
        index = self.sample_index(n=n, frac=frac, p=p)
        return DataSet(*self[index])

    def filter_by_y(self, y_filter):
        """
        Return a DataSet with examples filtered by whether the y value is in argument y
        :param y_filter: integer or list of integers
        :return: DataSet of filtered results
        """
        if not isinstance(y_filter, list):
            y_filter = [y_filter]
        y_values = self.int_category_y
        mask = np.isin(y_values, y_filter)
        return DataSet(*self[mask])

    def closest_points(self, test, metric='euclidean', n=None, frac=1):
        """
        Get n closest points or the fraction of the data that is closest to the
        points in test with respect to x

        :returns: 3 lists
        """
        from sklearn.neighbors import NearestNeighbors
        import warnings

        if metric is None:
            warnings.warn("'metric' shouldn't be None. Correcting to be 'euclidean'.")
            metric = 'euclidean'

        if n is None:
            n = int(len(test) * frac)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=metric).fit(self._x)
        distances, labels = nbrs.kneighbors(test._x)

        indices = np.array(list(range(len(test)))).reshape((-1, 1))
        concat = np.concatenate([distances, labels, indices], axis=1)

        # Don't use np.sort! It doesn't preserve rows.
        concat = concat[concat[:, 0].argsort()]

        distances = list(concat[:n, 0].reshape((-1,)))
        labels = list(concat[:n, 1].reshape((-1,)).astype(int))
        indices = list(concat[:n, 2].reshape((-1,)).astype(int))

        return indices, distances, labels

    def class_count(self):
        """
        :return: dictionary mapping classes (as integers) to the occurrence count in self
        """
        categories = self.int_category_y
        from collections import Counter
        return dict(Counter(categories))

    @classmethod
    def from_stream(self, stream, max_len=100_000):
        result_len = stream_len(stream, max_len)
        assert result_len > 0
        batcher = Batcher(stream, result_len)
        x, y, w = list(islice(batcher, 1))[0]
        x, y, w = copy.deepcopy((x, y, w))
        return DataSet(x, y, w)

    def to_stream(self, shape=None):
        if shape is None:
            warnings.warn('to_stream is unstable without the shape argument')
        xyw_iter = copy.deepcopy(list(self))
        return IterDataStream(xyw_iter, shape=shape)

    @classmethod
    def from_hdf(self, file_path):
        NotImplementedError()

    def to_hdf(self, file_path):
        NotImplementedError()
        
        
class IterDataStream(DataStream):
    """
    Given a normal iterable of (x, y, w) tuples, this serves as a DataStream with a shape
    property, so it is compatible with Batcher
    """

    def __init__(self, iterable_xyw, shape=None):
        if isinstance(iterable_xyw, zip):
            warnings.warn('zip objects cannot be iterated over twice. Converting to list.')
            iterable_xyw = list(iterable_xyw)
        if shape is None:
            first = list(islice(iterable_xyw, 1))[0]
            shape = custom_shape(first)
        self._iterable_xyw = iterable_xyw
        self._shape = shape

    def __iter__(self):
        for point in self._iterable_xyw:
            yield point

    @property
    def shape(self):
        return self._shape




def _getitems(d, items):
    items = items.split(',')
    items = [item.trim() for item in items]
    result = [d.get(item) for item in items]
    assert all([item is not None for item in items]), result
    return result


def _setitems(d, keys, values):
    keys = keys.split(',')
    keys = [key.trim() for key in keys]
    for key, val in zip(keys, values):
        d[key] = val


class HyperparameterSearcher:
    """
    Module for automating hyperparameter search.
    Search plans are returned as lists of dictionaries.
    A hyperparameter generator can be
    - list: values are generated by selecting a random element (default is the first element)
    - function with one input: values are generated by calling the function (no default)
    - other type: the generator is returned as the value (default is the generator)

    Example use:

    hyp = {
        'mixing_rate': np.random.uniform(0, 1)
        'batch_size': 32,
        'learning_rate': [0.01, 0.001]
    }
    hyp_defaults={'mixing_rate': 0.5}
    searcher = HyperparameterSearcher(hyp, hyp_defaults)
    for hyp_dict in searcher.random_search(100):
        for repetition in range(3):
            run_experiment(hyp_dict)

    """
    def __init__(self, hyperparams, hyp_defaults=None):
        """
        :param hyperparams: dict mapping hyperparameter names to hyperparameter generators
        :param hyp_defaults: dict mapping hyperparameter names to values which override the
                             defaults of the hyperparameters in the hyperparams argument
        """
        if hyp_defaults is None:
            hyp_defaults = {}
        self.hyperparams = hyperparams.copy()
        self.hyp_defaults = hyp_defaults
        self._check_config()

    def _check_config(self):
        errors = []

        all_hyp = set(self.hyperparams.keys())
        callable_hyp = set([hyp_name for hyp_name, hyp_gen in self.hyperparams.items()
                            if callable(hyp_gen)])
        non_callable_hyp = all_hyp - callable_hyp
        
        # check that hyp_defaults references proper hyperparameters
        default_errors = set(self.hyp_defaults.keys()) - all_hyp
        if len(default_errors) != 0:
            errors.append(f'{default_errors} are not hyperparameters')

        # check that callable hyperparameters take no arguments
        for hyp_name in callable_hyp:
            try:
                self.hyperparams[hyp_name]()
            except Exception as ex:
                errors.append(f'{hyp_name} call error: {ex}')
            
        # check that all hyperparameters are defaults
        covered_defaults = (non_callable_hyp | set(self.hyp_defaults.keys()))
        not_covered_defaults = set(self.hyperparams.keys()) - covered_defaults
        if len(not_covered_defaults) != 0:
            warnings.warn(f'{not_covered_defaults} have no default values')

        assert len(errors) == 0, errors

    def _choose_random(self, hyp_name):
        generator = self.hyperparams[hyp_name]
        return self._generate_random(generator)

    @staticmethod
    def _generate_random(generator):
        if isinstance(generator, list):
            # random.choice returns a numpy type (not json serializable)
            ix = np.random.randint(0, len(generator))
            return generator[ix]
        elif callable(generator):
            return generator()
        else:
            return generator

    def _choose_default(self, hyp_name):
        if hyp_name in self.hyp_defaults.keys():
            return self.hyp_defaults[hyp_name]
        generator = self.hyperparams[hyp_name]
        if isinstance(generator, list):
            return generator[0]
        elif callable(generator):
            warnings.warn('Missing default for hyperparameter {}'.format(hyp_name))
            return generator()
        else:
            return generator

    def default_dict(self):
        # default hyperparameter dict
        return dict((hyp_name, self._choose_default(hyp_name))
                    for hyp_name in self.hyperparams.keys())

    def random_dict(self):
        # random hyperparameter dict
        return dict((hyp_name, self._choose_random(hyp_name))
                    for hyp_name, hyp_gen in self.hyperparams.items())

    def default_search(self, n=1):
        # list of n default hyperparameter dictionaries
        return [self.default_dict() for i in range(n)]

    def random_search(self, n=1):
        # list of n random search hyperparameter dictionaries
        return [self.random_dict() for i in range(n)]

    def one_value_search(self, interests, n=1):
        """
        With k hyperparameters, return the concatenation of (n / k) chunks, where chunk i is a
        hyperparameter search where
        - hyperparameter i is chosen randomly
        - other hyperparameters are chosen using their default values
        :param interests: list of hyperparameter names
        :param n: number of hyperparameter dictionaries in total
        :return: list of n hyperparameter dictionaries
        """
        if interests is None:
            interests = list(self.hyperparams.keys())
        assert isinstance(interests, list)
        n_rep = n // len(interests)
        assert n_rep > 0, f'Error: n_rep=={n_rep} when n={n} and len(interest)={len(interests)}'
        result_search = self.default_search(n=n)
        for i in range(n):
            hyp_name = interests[i // n_rep]
            result_search[i][hyp_name] = self._choose_random(hyp_name)
        return result_search
    
    def one_value_grid_search(self, interests):
        """
        Return a subset of grid search of the variables in interest where only 1 variable is
        different from the default at a time
        :param interests: list of hyperparameter names
        :return: list of n hyperparameter dictionaries
        """
        if interests is None:
            interests = list(self.hyperparams.keys())
        assert isinstance(interests, list)
        assert all([isinstance(self.hyperparams[interest], list) for interest in interests])

        n_unique_tuples = (sum([len(self.hyperparams[interest]) for interest in interests]) +
                           1 - len(interests))

        # construct hyperparameter dicts without default values
        non_default_hyperparams = {}
        for hyp_name in interests:
            non_default_list = list(self.hyperparams[hyp_name])
            non_default_list.pop(non_default_list.index(self._choose_default(hyp_name)))
            non_default_hyperparams[hyp_name] = non_default_list

        result_search = self.default_search(n=n_unique_tuples)
        search_i = 1  # first dictionary is the default
        for hyp_name in interests:
            for value in non_default_hyperparams[hyp_name]:
                result_search[search_i][hyp_name] = value
                search_i += 1
        assert search_i == n_unique_tuples
        return result_search

    def custom_search(self, overwritten_hyp, n=1):
        """
        default search but with overwritten generators
        :param overwritten_hyp: dict mapping hyperparameter names to hyperparameter generators,
                                which overrides the corresponding hyperparameter generators for
                                this search only
        :param n: number of hyperparameter dictionaries in total
        :return:list of n hyperparameter dictionaries
        """
        result_search = self.default_search(n=n)
        for i in range(n):
            for hyp_name, hyp_gen in overwritten_hyp.items():
                result_search[i][hyp_name] = self._generate_random(hyp_gen)
        return result_search

class DictLogger:
    """ A logger for dictionaries into a json file.
    CAUTION: the final json file may not be json-serializable (use the .read function)
    """

    def __init__(self, path, use_index=False, verbose=0):
        self._file_path = path
        self._use_index = use_index
        self._verbose = verbose
        if os.path.isfile(path):
            file_mode = 'a'
        else:
            file_mode = 'w'
        with open(path, file_mode) as f:
            f.write(',\n{"start_log": True, "json_log_id": -1}')

    def filter_json_serializable(self, d):
        # returns a copy of dictioary d, where non-json-serializable entries are removed
        result = {}
        missing = []
        for k, v in d.items():
            try:
                json.dumps(v)
                result[k] = v
            except Exception as ex:
                missing.append(k)

        if len(missing) > 0:
            warnings.warn('Missing keys: {}'.format(missing))
        return result

    def log(self, d, json_log_id=0):
        if not isinstance(d, dict):
            warnings.warn(f'DictLogger expects a dictionary to be passed to the log function: {d}')
            return
        with open(self._file_path, 'a') as f:
            f.write(',\n')
            json_serializable = self.filter_json_serializable(d)
            if self._use_index:
                json_serializable['json_log_id'] = json_log_id
            json.dump(json_serializable, f)
            if self._verbose > 0:
                print('{} : {}'.format(json_log_id, d))

    def read(self):
        # returns a list of dictionaries based on the json log content
        with open(self._file_path) as f:
            data = str(f.read())
            if len(data) > 0:
                data = '[' + data[1:] + ']'
        return json.loads(data)
        
        
def test_json_logger():
    logger = DictLogger('hi.json', use_index=True)
    logger.log({'hi': 0}, 0)
    logger.log({'hi': 0}, 1)
    logger.read()


# Copy-paste of Keras' Callback object
from keras.callbacks import Callback
class Callback(object):
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class CB(Callback):
    def __init__(self):
        pass
    
    def on_batch_end(self, batch, logs={}):
        """
         :batch: int corresponding to which batch
         :logs: dict with ie {'batch': 0, 'size': 4, 'loss': array(50.8, dtype=float32)}
        """
        print(batch, logs)


class MySGD(Optimizer):
    def __init__(self, lr=0.01, n_outputs=100, **kwargs):
        super(MySGD, self).__init__(**kwargs)
        self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.lr = K.variable(lr, name='lr')
        def mean_squared_error(y_true, y_pred):
            return K.sum(K.square(y_pred - y_true), axis=-1)
        self._loss_fn = mean_squared_error
        
    def set_model(self, model):
        self.model = model
        y_true = model.targets[0]
        y_pred = model.outputs[0]
        #masks = model.compute_mask(model.inputs, mask=None)
        #if not isinstance(masks, list):
        #    masks = [masks]
        #sample_weight = K.placeholder(ndim=1, name='ewc_sample_weights')
        sample_weight = K.ones((1,), name='ewc_sample_weights')
        weighted_loss = _weighted_masked_objective(self._loss_fn)
        self.custom_loss = weighted_loss(y_true, y_pred, sample_weight, None)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(self.custom_loss, params)
        self.updates = []

        lr = self.lr

        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            new_p = p - lr * g

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates
    

def get_links_on_imdb():
    # pip install Scrapy
    from scrapy.http.response.html import HtmlResponse
    from scrapy.linkextractors import LinkExtractor
    from urllib import request
    url = "http://www.imdb.com"
    r = request.urlopen(url)
    bytecode = r.read()
    page = HtmlResponse(url=url, body=bytecode)
    for link in LinkExtractor().extract_links(page):
        print(link)


def get_index_map(l1, l2):
    # index to map l1 to l2
    # index i of the output is the index of l1's element which is equal to l2[j]
    # :return: list of integers
    n = len(l1)
    result = [-1] * n
    assert len(l1) == len(l2), (len(l1), len(l2))
    for i_1 in range(n):
        for i_2 in range(n):
            if l1[i_1] == l2[i_2]:
                result[i_2] = i_1
    return result

def nltk_experimentation():
    ''' nltk wordnet
    https://stackoverflow.com/questions/15330725/how-to-get-all-the-hyponyms-of-a-word-synset-in-python-nltk-and-wordnet
    https://stackoverflow.com/questions/26222484/determining-hypernym-or-hyponym-using-wordnet-nltk
    http://www.nltk.org/howto/wordnet.html
    group labelling interface'''

    import nltk
    from nltk.corpus import wordnet as wn
    nltk.download('wordnet')

    dog = wn.synset('dog.n.01')
    cat = wn.synset('cat.n.01')
    print(dog.hypernyms())

    print(wn.synsets('cat'))

    print(cat.lemmas())
    def counts(synset):
        count = 0
        for l in synset.lemmas():
            count += l.count()
        return count


    print(list(zip(hmap(counts, wn.synsets('jump')), wn.synsets('jump'))))

    l = wn.synsets('jump')
    def synonyms(word):
    l = wn.synsets(word)
    result = set()
    for s in l:
        result |= set(s.lemmas())
    return result
        
    print(synonyms('dog'))


        
def mix_tensorflow_and_keras():
    import tensorflow as tf

    a = Input((2,), name='a')
    b = Dense(3, name='b')(a)
    m1 = KerasModel(a, b)
    m1.compile(optimizer='adam', loss='mse')

    m1_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    m1_y = np.zeros((4, 3))

    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.initialize_all_variables())
    grad = tf.gradients(
        K.categorical_crossentropy(m1.outputs[0], m1.targets[0]), 
        m1.get_layer('b').kernel)

    gradients = sess.run(grad, feed_dict={
        m1.inputs[0]: m1_X,
        m1.targets[0]: m1_y
    })[0]

def construct_mnist():
    # Load MNIST data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return (X_train, y_train), (X_test, y_test)

def construct_cifar10():
    # Load CIFAR10 data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    # X_train = X_train.reshape(-1, 3, 32, 32)
    # X_test = X_test.reshape(-1, 32**2)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    no = X_train.max()
    X_train /= no
    X_test /= no

    return (X_train, y_train), (X_test, y_test)

def construct_cifar100():
    # Load CIFAR100 data and normalize
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    m = np.max( (np.max(X_train), np.max(X_test)))
    X_train /= m
    X_test /= m

    return (X_train, y_train), (X_test, y_test)


# Proposed Solution to the model path problem

def experiment_manager():
    class HypPlan:
        def __init__(self, search, file_path):
            json.dump(search, open(file_path, 'w'))

    # train_model_a.py

    log_file_path = 'exp{}-{}-train-model-a-log.json'
    # store in ~/ml_results
    model_a_path = 'exp{}-{}-train-model-a-model-a'
    # store in ~/models

    import time


    class ExperimentResultManager:
        RESULTS_PATH = '/home/henri/ml_results'
        MODELS_PATH = '/home/henri/models'
        N_EXPERIMENTS_FILE_NAME = 'n_experiments.txt'
        
        def __init__(self, log_file_name=None, verbose=0, use_index=True):
            self._log_path = os.path.join(RESULTS_PATH, log_file_name)
            
            self._logger = DictLogger(self._log_path, verbose=verbose, use_index=use_index)
            self._exp_nb_path = os.path.join(RESULTS_PATH, N_EXPERIMENTS_FILE_NAME)
            self._init_exp_nb()
            
        def log(self):
            self._logger.log
        
        def _init_exp_nb(self):
            if not os.path.isfile(self._exp_nb_path):
                result = 1
            with open(self._exp_nb_path, 'r') as f:
                result = int(str(f.read()).strip()) + 1

            # update n_experiments
            with open(self._exp_nb_path, 'w') as f:
                f.write(str(result))
            
            self._exp_nb = result
        
        @property
        def exp_nb(self):
            return self._exp_nb
        
        def save_model(self, model, file_name):
            # ********
            # import datetime
            exp_nb = self.exp_nb
            # time_str = datetime.datetime.now().strftime("%b-%d-y")
            prefix = 'exp{}-'.format(exp_nb)
            formatted_file_path = os.path.join(MODELS_PATH, prefix + file_name)
            try:
                import keras
                keras.models.save_model(model, file_path)
            except Exception as ex:
                print(ex)
                print('Saving weights only')
                model.save_weights(file_path)
            
            
        

    em = ExperimentResultManager()
    em.log
    em.save_model(model, 'model-a')


class DataStreamDebugger:
    def __init__(self, stream):
        streams = []
        streams.append(stream)
        stream = self._get_prev_stream(stream)
        while stream is not None:
            streams.append(stream)
            stream = self._get_prev_stream(stream)
        self._streams = list(reversed(streams))

    def _get_prev_stream(stream):
        if hasattr(stream, '_stream'):
            return stream._stream
        elif hasattr(stream, '_data_stream'):
            return stream._data_stream
        else:
            return None
        
    def _myshape(self, arr):
        if isinstance(arr, list):
            return [self._myshape(val) for val in arr]
        elif isinstance(arr, np.ndarray):
            return arr.shape
        elif isinstance(arr, tuple):
            return tuple(self._myshape(var) for var in arr)
        else:
            raise BaseException(f'Unseen shape type {type(arr)}')
    
    def step(self, full=False):
        myshape = self._myshape
        for i, stream in enumerate(self._streams):
            print('Stream', i, type(stream))
            for x,y,w in stream:
                break
            if full:
                print('\tx', myshape(x), x)
                print('\ty', myshape(y), y)
                print('\tw', myshape(w), w)
            else:
                print('\tx', myshape(x))
                print('\ty', myshape(y))
                print('\tw', myshape(w))


def get_tensor_ddl(model):
    TENSOR_NAMES = ('kernel', 'recurrent_kernel', 'bias', 'embeddings')
    result = {}

    for layer in model.layers:
        layer_name = layer.name
        # account for TimeDistributed
        while isinstance(layer, Wrapper):
            layer = layer.layer
        for tensor_name in TENSOR_NAMES:
            if hasattr(layer, reg_name):
                weight_tensor = layer.__getattribute__(tensor_name)
                if layer_name not in result.keys():
                    result[layer_name] = {}
                result[layer_name][tensor_name] = weight_tensor

    return result

def get_weights_ddl(model):
    warnings.warn('This uses logic which layer get_weights wraps around')
    TENSOR_NAMES = ('kernel', 'recurrent_kernel', 'bias', 'embeddings')
    result = {}

    key_pairs = []
    weight_tensors = []

    for layer in model.layers:
        layer_name = layer.name
        # account for TimeDistributed
        while isinstance(layer, Wrapper):
            layer = layer.layer
        for tensor_name in TENSOR_NAMES:
            if hasattr(layer, tensor_name):
                weight_tensor = layer.__getattribute__(tensor_name)
                key_pairs.append((layer_name, tensor_name))
                weight_tensors.append(weight_tensor)

    # what layer get_weights uses
    weight_values = K.batch_get_value(weight_tensors)

    for (layer_name, tensor_name), weight_value in zip(key_pairs, weight_values):
        if layer_name not in result.keys():
            result[layer_name] = {}
        result[layer_name][tensor_name] = weight_value


    return result

def set_weights_ddl(model, weight_ddl):
    warnings.warn('This uses logic which layer set_weights wraps around')
    TENSOR_NAMES = ('kernel', 'recurrent_kernel', 'bias', 'embeddings')

    weight_value_tuples = []

    # try accessing all weights
    for layer_name, dl in weight_ddl.items():
        for tensor_name, weight_arr in dl.items():
            model.get_layer(layer_name).__getattribute__(tensor_name)
            
    for layer_name, dl in weight_ddl.items():
        for tensor_name, weight_arr in dl.items():
            weight_tensor = model.get_layer(layer_name).__getattribute__(tensor_name)
            weight_value_tuples.append((weight_tensor, weight_arr))

    # what layer get_weights uses
    K.batch_set_value(weight_value_tuples)


def transfer_weights(model1, model2):
    # transfer weights by layer name from model1 into model2
    for layer in model1.layers:
        if isinstance(layer, InputLayer):
            continue
        layer_name = layer.name
        model2.get_layer(layer_name)

    for layer in model1.layers:
        if isinstance(layer, InputLayer):
            continue
        layer_name = layer.name
        layer_weights = layer.get_weights()
        model2.get_layer(layer_name).set_weights(layer_weights)




# Experimental
def plot_stream():
    while True:
        fig, axes = plt.subplots(2, 2)
        axes = axes.reshape((-1,))
        for axis in axes:
            yield axis

def plot_patterns(data, offset=20, goal='avg', list_goal='avg', interest=None):
    assert goal in data.columns
    learning_rate = ['lr', 'learning_rate']
    epochs = ['nb_epochs', 'epochs']
    loss = ['loss']
    assert isinstance(data, pd.DataFrame)
    plots = iter(plot_stream())
    
    if interest is None:
        interest = data.columns
    
    for i, col in enumerate(interest):
        try:
            df = data.iloc[i*offset: i*offset+offset]
            values = df[col].values
            first_val = values[0]
            if col in learning_rate:
                values = np.log(values)

            if isinstance(first_val, int) or isinstance(first_val, str):
                # box stuff
                axis = next(plots)
                axis.set_title(col)
                to_plot = twol2dl(df[col], df[list_goal])
                index = range(1, 1 + len(to_plot.keys()))
                axis.boxplot(list(to_plot.values()))
                set_xticks(axis, index, list(to_plot.keys()))
            elif np.isreal(first_val):
                # scatter
                axis = next(plots)
                axis.set_title(col)
                axis.scatter(df[col].values, df[goal].values)
            else:
                print('Ignored', col)
        except Exception as ex:
            print('Error', col, ex)
    plt.show()

def get_dl(n):
    ld = filter_ld(data, match={'exp_nb': n, 'json_log_id': 2})
    d = ld2dl(ld)
    d = {k: d[k] for k in 'train1_acc,test1_acc,train2_acc,test2_acc'.split(',')}
    return d

def informative_json_loads(s):
    try:
        s = s.replace("'", '"').replace("None", "0").replace('False', '0').replace('True', '1')
        s = s.replace('true', '1').replace('false', '0')
        return json.loads(s)
    except Exception as ex:
        msg = f'Json Parse Error: {ex.pos}, {s[ex.pos:min(len(s), ex.pos + 30)]}'
        start = max(0, ex.pos - 100)
        end = min(len(s), ex.pos + 100)
        msg += '\n\n' + s[start:end]
        print(msg)
        raise BaseException(msg)

class DictLogger:
    """
    Problems: error if non-dict is passed to .log
    """
    def __init__(self, path, use_index=False):
        self._file_path = path
        self._use_index = use_index
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                pass
    
    def filter_json_serializable(self, d):
        result = {}
        missing = []
        for k, v in d.items():
            try:
                json.dumps(v)
                result[k] = v
            except Exception as ex:
                missing.append(k)
        
        if len(missing) > 0:
            import warnings
            warnings.warn('Missing keys: {}'.format(missing))
        return result
        
    def log(self, d, json_log_id=0):
        with open(self._file_path, 'a') as f:
            f.write(',\n')
            json_serializable = self.filter_json_serializable(d)
            if self._use_index:
                json_serializable['json_log_id'] = json_log_id
            json.dump(json_serializable, f)
            
    def read(self):
        with open(self._file_path) as f:
            data = str(f.read().strip())
            if len(data) > 0:
                data = '[' + data[1:] + ']'
            
        return informative_json_loads(data)
    
def filter_ld(ld, match=None, below_match=None, below_not_match=None):
    if match is None:
        match = {}
    if below_match is None:
        below_match = {}
    if below_not_match is None:
        below_not_match = {}
        
    result = []
    
    for i, d in enumerate(ld):
        include = True
        for k, v in match.items():
            if d.get(k) != v:
                include = False
                break
        
        if i + 1 < len(ld):
            for k, v in below_match.items():
                if ld[i + 1].get(k) != v:
                    include = False
                    break
            
            for k, v in below_not_match.items():
                if ld[i + 1].get(k) == v:
                    include = False
                    break

        if include:
            result.append(d)
            
    return result

def mod_split(l, n=1):
    result = []
    for i in range(n):
        result.append(l[i:][::n])
    return result

def read_json(file_path):
    logger = DictLogger(file_path)
    data = logger.read()
    yo = filter_ld(data, below_not_match={'json_log_id': 2})
    yo = mod_split(yo, 2)
    yo = [{**x, **y} for x, y in zip(*yo)]
    return yo


class DataParser:
    def __init__(self):
        pass

    def download(self):
        return self

class NlpDataParser(DataParser):
    def __init__(self):
        # self.nbytes = 1000
        pass
    
    def build(self):
        self.download()
        self.raw_stream
        self.token_vectorizer
        self.output_vectorizer
        self.embedding_weights
        self.transformed_stream

        return self

    def to_dataset(self):
        pass

    @property
    @abstractmethod
    def raw_stream(self):
        pass
    
    @property
    @abstractmethod
    def transformed_stream(self):
        pass
    
    def split_tt_stream(self):
        pass
    
    @property
    @abstractmethod
    def token_vectorizer(self):
        pass
    
    @property
    @abstractmethod
    def output_vectorizer(self):
        pass
    
    @property
    @abstractmethod
    def embedding_weights(self):
        pass


all_data_parsers = {
    'hello': MyNlpDataParser()
}
dp = all_data_parsers['hello']
train, test = dp.split_tt_stream



@pytest.fixture(name='hyp_searcher')
def get_hyp_searcher():
    hyperparameters = {
        'a': [0, 1],
        'b': [8, 9, 10],
        'c': [True],
        'd': 15,
        'e': lambda: 20,
        'a2': [29, 30],
        'b2': [51, 52, 53],
        'c2': [True],
        'd2': 1,
        'e2': 5
    }
    hyp_defaults = {
        'a': 51,
        'b': 26,
        'c': True,
        'd': 31,
        'e': 55,
    }

    return HyperparameterSearcher(hyperparameters, hyp_defaults=hyp_defaults)


def check_dict_isclose(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    assert d1_keys == d2_keys
    for k in d1_keys:
        assert np.isclose(d1[k], d2[k])


class TestHyperparameterSearcher:
    def test_default_dict(self, hyp_searcher):
        default_dict = hyp_searcher.default_dict()
        expected = {'a2': 29, 'b2': 51, 'c2': True, 'd2': 1, 'e2': 5,
                    'a': 51, 'b': 26, 'c': True, 'd': 31, 'e': 55}
        check_dict_isclose(default_dict, expected)

    def test_random_dict(self, hyp_searcher):
        np.random.seed(0)
        random_dict = hyp_searcher.random_dict()
        expected = {'a': 0, 'a2': 29, 'b': 9, 'b2': 52, 'c': True,
                    'c2': True, 'd': 15, 'd2': 1, 'e': 20, 'e2': 5}
        check_dict_isclose(random_dict, expected)

    def test_default_search(self, hyp_searcher):
        search = hyp_searcher.default_search(n=30)
        assert len(search) == 30

    def test_random_search(self, hyp_searcher):
        search = hyp_searcher.random_search(n=31)
        assert len(search) == 31

    def test_one_value_search(self, hyp_searcher):
        interests = ['a', 'b']
        search = hyp_searcher.one_value_search(interests, n=32)
        assert len(search) == 32

    def test_custom_search(self, hyp_searcher):
        search = hyp_searcher.custom_search({'a': 1000}, n=34)
        assert len(search) == 34


###############################
# 2018 Onwards ################
###############################

def tensorflow_stuff():
    l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.nn.relu(l1)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def tf_tensorboard_summaries():
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/train',
        sess.graph)

    iteration = 0
    summary, acc = sess.run([merged, accuracy], feed_dict=...)
    train_writer.add_summary(summary, iteration)

def tf_basic_nn():
    pass

def tf_embeddings():
    # Keras uses 0.05
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -0.05, 0.05)
    )
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

def tf_saving():
    # tf.reset_default_graph() if you want to run save and restore in python notebook

    path = "..."
    ckpt_ext = 'model.ckpt'

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else: # train neural network
            os.mkdir(path)
            saver.save(sess, path + ckpt_ext)

def conv2d(x, W):
    # x = tf.placeholder('float', [None, 784])
    # y = tf.placeholder('float')
    return tf.nn.conv2d(x, W, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def dense(x, input_size, output_size):
    # use this instead: tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.sigmoid)
    # seems to apply after and before rnns correctly (ie [1, 2] gives same first output as [1, 1])

    init = tf.glorot_uniform_initializer()

    W = tf.Variable(init([input_size, output_size]))
    b = tf.Variable(init([output_size]))

    result = tf.matmul(x, W) + b
    result = tf.nn.relu(result)
    # d = tf.nn.softmax(d)
    return result

def tf_rnn_example():
    def RNN(x, n_timesteps, n_hidden):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # create a list of n_timesteps elements via axis=1
        x = tf.unstack(x,n_timesteps,1)

        if True:
            # 1-layer LSTM with n_hidden units.
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        else:
            num_units = [5, 3]
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in num_units]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)

        # generate prediction (n_timesteps outputs)
        outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_timesteps outputs but
        # we only want the last output
        return outputs[-1]

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).reshape((4, 2, 1))
    x = tf.placeholder('float', [None, 2, 1])

    rnn = RNN(x, 2, 3)

    sess.run(tf.global_variables_initializer())
    sess.run(rnn, feed_dict={x: X})
    sess.close()

def tf_bidirectional():
    # no tf.unstack
    x = tf.placeholder('float', [None, 3, 1])
    lengths = tf.placeholder('int32', [None])
    X = np.ones((5, 3, 1))
    L = [1, 2, 3, 2, 1]

    cell1 = tf.contrib.rnn.BasicLSTMCell(2)
    cell2 = tf.contrib.rnn.BasicLSTMCell(2)
    
    outputs, states  = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell1,
        cell_bw=cell2,
        sequence_length=lengths, # can be a fixed list
        inputs=x,
        dtype=tf.float32 # necessary
    )
    
    output_fw, output_bw = outputs
    states_fw, states_bw = states

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(outputs, feed_dict={x: X, lengths: L})

def tf_extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.

    eg. tf_extract_axis_1(rnn_result, sequence_lengths - 1)
        # gives last outputs for each batch
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

def tf_rnn_example():
    x_bidir_gate = tf.concat([x, bidir, use_soft_gate], axis=-1)

    outputs, states = tf.nn.dynamic_rnn(self, x_bidir_gate, sequence_length=sequence_lengths,
                                        dtype=tf.float32)

    final_outputs = extract_axis_1(outputs[:,:,:-1], sequence_lengths - 1)
    
    gates = outputs[:,:,-1:]
    
    return final_outputs, gates, outputs, states


def tf_gate():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([1, 0])
    expected = [[1, 2], [0, 0]]
    # because of broadcasting, with tf.multiply, if ranks are not equal
    # the tensor is duplicated until they have the same dimensions
    gated = tf.multiply(tf.reshape(b, shape=(2, 1)), a)
    # b => [[1], [0]] -> [[1, 1], [0, 0]]
    

def tf_seq2seq_sparse_crossentropy(logits, targets, weights=None, sequence_lengths=None):
    if sequence_lengths is not None:
        batch_maxlen = tf.shape(logits)[1]
        sequence_len_weights = tf.sequence_mask(lengths=sequence_lengths,
            maxlen=batch_maxlen, dtype=tf.float32)

    if weights is None and sequence_lengths is None:
        weights = tf.ones(shape=tf.shape(targets))
    elif weights is None:
        weights = sequence_len_weights
    else:
        weights *= sequence_len_weights

    # Note: sequence loss applies the activation to the logits
    # Evidence: logits being both 0.5 and -50 give the same sequence_loss

    return tf.contrib.seq2seq.sequence_loss(
        logits=logits,  # [batch_size, sequence_length, num_decoder_symbols]
        targets=targets,  # [batch_size, sequence_length]
        weights=weights)  # [batch_size, sequence_length]


def tf_seq2seq_sigmoid_crossentropy(logits, labels, weights=None, sequence_lengths=None):
    """ Expected shapes
    logits: [batch_size, n_timesteps]
    targets: [batch_size, n_timesteps]
    """
    batch_maxlen = tf.shape(logits)[1]
    if sequence_lengths is not None:
        sequence_len_weights = tf.sequence_mask(lengths=sequence_lengths,
                                                maxlen=batch_maxlen, dtype=tf.float32)

    if weights is None and sequence_lengths is None:
        weights = tf.ones(shape=tf.shape(labels))
    elif weights is None:
        weights = sequence_len_weights
    else:
        weights *= sequence_len_weights

    if sequence_lengths is None:
        sequence_lengths = batch_maxlen

    # Note: sequence loss applies the activation to the logits
    # Evidence: logits being both 0.5 and -50 give the same sequence_loss

    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    weighted_loss = tf.reduce_mean(  # reduce over everything else
        tf.divide(
            tf.reduce_sum(losses * weights, axis=1),  # reduce over timesteps
            tf.cast(sequence_lengths, dtype=tf.float32)))

    # does not use sequence lengths for averaging \/
    # weighted_loss = tf.losses.compute_weighted_loss(losses, weights)

    return weighted_loss

def tf_compute_gradients():
    gate_grads = gate_training_optimizer.compute_gradients(
        self.reconstruction_loss, var_list=gate_training_variables)
    for v, (g, v2) in zip(gate_training_variables, gradients):
        try:
            if hasattr(g, 'values'): # handle IndexedSlicesValue
                g = g.values
            print(np.mean(np.abs(g)), '\t\t', np.max(np.abs(g)), '\t\t', v)
        except Exception as ex:
            print(v, str(ex), g)

def tf_hinge_loss():
    x = tf.zeros((1,  1))
    pred = tf.sigmoid(tf.zeros((1,  1)))
    y = tf.ones((1,  1))

    tf_run(tf.losses.hinge_loss(logits=pred, labels=y))
    # shapes of pred and y must be equal
    # CAUTION: pred must already be between 0 and 1 to work


""" tensorflow learn list

RNN
GRU
bi-directional
training
word embedding

fit generator

loss categorical crossentropy

tf sequence loss (dynamic)

"""

def tf_reuse_variables():
    x = tf.zeros((2, 2, 4))

    with tf.variable_scope('dense') as scope:
        d = tf.layers.dense(inputs=x, units=4)
        
    with tf.variable_scope('dense', reuse=True) as scope:
        d2 = tf.layers.dense(inputs=x, units=4)

    print(tf.global_variables())


def fn_binary_search(fn, lower, upper, target):
    # lower <= x < upper
    while lower < upper:   # use < instead of <=
        x = lower + (upper - lower) // 2
        val = fn(x)
        if target == val:
            return x
        elif target > val:
            if lower == x:   # these two are the actual lines
                break        # you're looking for
            lower = x
        elif target < val:
            upper = x

    return -1

def arr_kl_divergence(p_arr, q_arr):
    result = 0
    for p_i, q_i in zip(p_arr, q_arr):
        result += p_i * np.ln(p_i / q_i)
    if np.isnan(result):
        raise ValueException('zero values')

    return result

def word_similarity(word1, word2):
    pass

    
def char_filtering():
    all_chars = set('IÜó*\x9dX×uDM7—32+fz%UC4$·GW¹m°àJQ&xVraä,’@[oq<1´"T\x95w½vndïp gkü\x93;–eíéÈyl²\xadYc_ñø-(E:6=\x94H¾RçB¼50…\')™s#9!tFö/A\xa0ºÉ8?¿Z.]N®|LSb>êißOPjKè•\t‘h')
allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')



def tf_idf(list_of_list_of_words):
    import collections

    n_doc = len(list_of_list_of_words)
    words = set()
    for doc in list_of_list_of_words:
        words |= set(doc)

    for word in words:
        df_t = len([1 for doc in list_of_list_of_words if word in doc])
        idf = np.log10(n_doc / df_t)
        for doc in list_of_list_of_words:
            tf = doc.count(word)
            tf_idf = tf * idf


def pearson_correlation_coefficient(x, y):
    # between -1 and 1
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.dot((x - x_mean), (y - y_mean)) / len(x)
    x_sample_std = np.sqrt(np.sum(np.square(x - x_mean)) / (len(x) - 1))
    y_sample_std = np.sqrt(np.sum(np.square(y - y_mean)) / (len(y) - 1))

    return covariance / (x_sample_std * y_sample_std)


def spearman_correlation_coefficient(x, y):
    """
    http://www.statstutor.ac.uk/resources/uploaded/spearmans.pdf
    between -1 and 1
    .00-.19 “very weak”
    .20-.39 “weak”
    .40-.59 “moderate”
    .60-.79 "strong"
    .80-1.0 "very strong"
    """
    rank_x = [ix for val, ix in sorted(zip(x, range(1, len(x) + 1)))]
    rank_y = [ix for val, ix in sorted(zip(y, range(1, len(y) + 1)))]

    # pearson correlation coefficient of rank values
    x_mean = np.mean(rank_x)
    y_mean = np.mean(rank_y)
    covariance = np.dot((rank_x - x_mean), (rank_y - y_mean))
    x_sample_std = np.sqrt(np.sum(np.square(rank_x - x_mean)) / (len(rank_x) - 1))
    y_sample_std = np.sqrt(np.sum(np.square(rank_x - x_mean)) / (len(rank_x) - 1))

    return covariance / (x_sample_std * y_sample_std)

def custom_split_tvt(data, split, lengths=None):
    # sets starts later
    assert np.isclose(sum(split), 1), (sum(split), 1)
    assert len(split) > 0
    data_len = len(data)
    split_len = len(split)
    split_sums = [sum(split[:i+1]) for i in range(len(split) - 1)] + [1]
    if lengths is None:
        starts = [0] + [int(data_len * s) for s in split_sums]
    else:
        acc_sum = 0
        total_sum = sum(lengths)
        start_total_sums = [int(total_sum * split_sums[i]) for i in range(split_len)]
        starts = [0]
        
        for split_i in range(split_len):
            for i in range(starts[-1], data_len):
                acc_sum += lengths[i]
                if acc_sum >= start_total_sums[split_i]:
                    starts.append(i + 1)
                    break
                
    return tuple([data[starts[i]:starts[i+1]] for i in range(split_len)])


def tf_rnn_example():
    x_bidir_gate = tf.concat([x, bidir, use_soft_gate], axis=-1)

    outputs, states = tf.nn.dynamic_rnn(self, x_bidir_gate, sequence_length=sequence_lengths,
                                        dtype=tf.float32)

    final_outputs = extract_axis_1(outputs[:,:,:-1], sequence_lengths - 1)
    
    gates = outputs[:,:,-1:]
    
    return final_outputs, gates, outputs, states