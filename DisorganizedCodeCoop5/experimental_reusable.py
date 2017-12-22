import numpy as np
import pandas as pd
import warnings
from keras.models import Model as KerasModel
import matplotlib.pylab as plt
from keras.layers import Input, Dense
import os
import sys


def set_keras_backend(backend='tensorflow'):
    import warnings
    warnings.warn('This does not work in Jupyter because of optimizer')
    import keras.backend as K
    import os
    import tensorflow as tf
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        try:
            reload(K)
        except:
            import importlib
            importlib.reload(K)
        assert K.backend() == backend

def openy(file = '.'):
    os.system('open ' + file)

def sendy(file_path):
    path = os.path.abspath(file_path)
    cmd = f'scp {path} henri@192.168.0.109:/home/henri/ml_results/'
    print(os.system(cmd))

def gety():
    return os.system('rsync -auv henri@192.168.0.109:ml_results /Users/hromel/Documents')

def first(l):
    for i in l:
        return i

def second(l):
    for i, val in enumerate(l):
        if i == 1:
            return val

def third(l):
    for i, val in enumerate(l):
        if 2 == 1:
            return val

def myshuffle(data):
    #warnings.warn('Use "np.random.shuffle(arr)" instead')
    warnings.warn('This does not distribute a matching shuffle across tuple elements.')
    if isinstance(data, (list, np.ndarray)):
        np.random.shuffle(data)
        # random.shuffle(l) # also works
    elif isinstance(data, tuple):
        for val in data:
            myshuffle(val)
    elif isinstance(data, pd.DataFrame):
        warnings.warn('You must assign this to the DataFrame')
        # when sampling, the indices move with the rows, so reset_index resets the index
        # to be 0, 1, 2, ... again
        return data.sample(frac=1).reset_index(drop=True)
    else:
        raise BaseException('Unseen data type {}'.format(type(data)))
        
def mysample(data, n=1, frac=None, p=None):
    #warnings.warn('Use "np.random.choice(arr, size=.., replace=False)" instead')
    # TODO: work for range & map objects, multidimensional numpy arrays
    if isinstance(data, list):
        warnings.warn('List is currently being converted to ndarray')
    if isinstance(data, (list, np.ndarray)):
        if frac is not None:
            n = int(len(data.shape) * frac)
        data = np.array(data)
        return np.random.choice(data, size=n, p=p, replace=False)
    elif isinstance(data, tuple):
        result = ()
        for val in data:
            result += (mysample(val),)
        return result
    elif isinstance(data, pd.DataFrame):
        if frac is None:
            return data.sample(n=n, replace=False) # replace=False shouldn't be needed
        else:
            return data.sample(frac=frac, replace=False)
    else:
        raise BaseException('Unseen data type {}'.format(type(data)))

def mysplit(data_and_labels, split=[0.6, 0.2, 0.2]):
    """
    Take a data set and split it according to split

    :param data_and_labels: tuple of 2 numpy arrays representing the data and labels
    :param split: list of 2 or 3 floats which sum to one
    :return: tuple of data-label tuples according to the split

    TODO: support DataStream
    """
    from sklearn.model_selection import train_test_split
    assert isinstance(data_and_labels, tuple) and len(data_and_labels) == 2
    data = data_and_labels[0]
    labels = data_and_labels[1]
    assert type(data, np.ndarray)
    assert type(labels, np.ndarray)
    x, x_test, y, y_test = train_test_split(data,labels,test_size=split[-1],train_size=(1-split[-1]))

    if len(split) == 2:
        return (x, y), (x_test, y_test)
    
    train_and_validation = split[0] + split[1]
    train = split[0]
    train_validation_ratio = train / train_and_validation
    x_train, x_cv, y_train, y_cv = train_test_split(
        x,y,test_size = 1-train_validation_ratio,train_size = train_validation_ratio)
    
    return (x_train, y_train), (x_cv, y_cv), (x_test, y_test)

def myconcat(a, b):
    # concat data batches
    assert type(a) == type(b), (type(a), type(b))
    if isinstance(a, tuple):
        return tuple(myconcat(a_val, b_val) for a_val, b_val in zip(a, b))
    elif isinstance(a, list):
        return [myconcat(a_val, b_val) for a_val, b_val in zip(a, b)]
    elif isinstance(a, np.ndarray):
        return np.concatenate([a, b], axis=0)  # axis=0 not necessary
    else:
        raise BaseException(f'Unhandled type {type(a)}')

def custom_keras_model():
    a = Input((2,))
    b = Dense(3, name='b')(a)
    c = Dense(2, name="c")(a)
    model = KerasModel(a, [b, c])
    model.compile(
        optimizer='adam',
        loss=['mse', 'mse'])
    return model

m = custom_keras_model()
m_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
m_y = [
    np.zeros((4, 3)),
    np.zeros((4, 2))
]

def custom_keras_model1():
    a = Input((2,))
    b = Dense(3, name='b')(a)
    model = KerasModel(a, b)
    model.compile(optimizer='adam', loss='mse')
    return model

m1 = custom_keras_model1()
m1_X = m_X
m1_y = m_y[0]

def myshape(arr):
    """This is a more generalized version of the .shape property which can be used for
    numpy arrays, lists, tuples, and other variables which have a shape"""
    if isinstance(arr, list):
        return [myshape(val) for val in arr]
    elif isinstance(arr, tuple):
        return tuple(myshape(var) for var in arr)
    elif isinstance(arr, dict):
        return dict((k, myshape(v)) for k, v in arr.items())
    else:
        return arr.shape


def myencodelabels(arr):
    # onehot
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder().fit_transform(arr)

def myflatten(arr):
    try:
        result = []
        assert not isinstance(arr, str)
        for row in arr:
            result += myflatten(row)
        return result
    except:
        return [arr]

def mytrysizes():
    sizes = [
        (6, 4),
        (10, 6),
        (14, 9)
    ]
    for size in sizes:
        print(f'plt.rcParams["figure.figsize"] = {size}')
        plt.rcParams['figure.figsize'] = size
        plt.scatter([1], [1])
        plt.show()

def myprint(*args):
    if len(args) > 1:
        for arg in args:
            myprint(arg)
    else:
        arg = args[0]
        if isinstance(arg, (list, tuple)):
            for val in arg:
                print(val)
        else:
            print(arg)

def myadd(x, y):
    # myadd(([1, 2], [3, 4]), ([1, 2], [3, 4]))
    assert type(x) == type(y), (x, y, type(x), type(y))
    if isinstance(x, list):
        assert len(x) == len(y), (len(x), len(y))
        return [myadd(a, b) for a, b in zip(x, y)]
    elif isinstance(x, tuple):
        assert len(x) == len(y), (len(x), len(y))
        return tuple(myadd(a, b) for a, b in zip(x, y))
    elif isinstance(x, dict):
        assert set(x.keys()) == set(y.keys()), (set(x.keys()), set(y.keys()))
        return dict((k, myadd(x[k], y[k])) for k in x.keys())
    else:
        return x + y

def myminus(x, y):
    # myadd(([1, 2], [3, 4]), ([1, 2], [3, 4]))
    assert type(x) == type(y), (x, y, type(x), type(y))
    if isinstance(x, list):
        assert len(x) == len(y), (len(x), len(y))
        return [myminus(a, b) for a, b in zip(x, y)]
    elif isinstance(x, tuple):
        assert len(x) == len(y), (len(x), len(y))
        return tuple(myminus(a, b) for a, b in zip(x, y))
    elif isinstance(x, dict):
        assert set(x.keys()) == set(y.keys()), (set(x.keys()), set(y.keys()))
        return dict((k, myminus(x[k], y[k])) for k in x.keys())
    else:
        return x - y

def mymultiply(x, y):
    # myadd(([1, 2], [3, 4]), ([1, 2], [3, 4]))
    assert type(x) == type(y), (x, y, type(x), type(y))
    if isinstance(x, list):
        assert len(x) == len(y), (len(x), len(y))
        return [myminus(a, b) for a, b in zip(x, y)]
    elif isinstance(x, tuple):
        assert len(x) == len(y), (len(x), len(y))
        return tuple(myminus(a, b) for a, b in zip(x, y))
    elif isinstance(x, dict):
        assert set(x.keys()) == set(y.keys()), (set(x.keys()), set(y.keys()))
        return dict((k, myminus(x[k], y[k])) for k in x.keys())
    else:
        return x * y


def myallclose(x, y):
    # np.allclose does not work for everything
    # this fails: np.allclose([1, np.array([1, 2, 3])], [1, np.array([1, 2, 3])])
    pass

def messy_get_item(d, key):
    # MAYBE REFACTOR???????
    # - return an index path
    # - for lists maybe instead consider looking through lists of dictionaries
    if isinstance(key, list):
        return [messy_get_item(k) for k in key]
    elif isinstance(key, tuple):
        return tuple(messy_get_item(k) for k in key)
    elif isinstance(d, dict):
        if key in d.keys():
            answers = [d[key]]
            return 
        answers = []
        for k, val in d.items():
            new_result = messy_get_item(val, key)
            if new_result is not None:
                answers.append(new_result)
        if len(answers) > 0:
            if len(answers) > 1:
                warnings.warn(f'key {key} has multiple values for {d}')
            return answers[0]
    return None

def multiscatter(data):
    fig, axes = plt.subplots(2, 2)
    axes = axes.reshape((-1,))
    for axis, x, y, title in zip(axes, data['x'], data['y'], data['titles']):
        axis.scatter(x, y)
        axis.set_title(title)
    plt.show()


def ld2dl(ld, remove_identical=True, get_mean_std=True):
    # list of dictionaries to dictionary of lists
    # possible one-liner: dict(zip(LD[0],zip(*[d.values() for d in LD])))
    #  - source: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    import collections
    result = collections.defaultdict(list)
    for d in ld:
        for k, v in d.items():
            result[k].append(v)
    result = dict(result)
    new_result = dict(result)
    for k, v in result.items():
        if remove_identical and v[1:] == v[:-1]:  # identical elements
            result[k] = v[0]
        elif get_mean_std and np.isreal(v[0]):
            new_result['mean_' + k] = np.mean(v)
            new_result['std_' + k] = np.std(v)
    return new_result

def dl2ld(dl, accommodate_values=True):
    dl = dict(dl)
    n_values = 1
    # get list length
    for k, v in dl.items():
        if isinstance(v, list):
            n_values = len(v)
            break

    for k, v in dl.items():
        if isinstance(v, list):
            # check same length
            assert len(v) == n_values
        else:
            # replace non-lists
            dfl[k] = [v] * n_values

    # One-liner from https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    return [dict(zip(dl,t)) for t in zip(*dl.values())]
    '''
    result = []
    for i in range(n_values):
        new_dict = {}
        for k, v in dl.items():
            result[k] = v[]
        result.append(new_dict)
    return result
    '''


def ld2d(ld):
    pass

def dl2d(dl):
    pass

def twol2dl(l1, l2):
    # input: ['hi', 'there'], [1, 2]
    # output: {'hi': [1], 'there': [2]}
    from collections import defaultdict
    result = defaultdict(list)
    for val1, val2 in zip(l1, l2):
        result[val1].append(val2)
    return dict(result)

def ll2twol(ll):
    # input: [[1, 2], [4, 5]]
    # output: [0, 0, 1, 1], [1, 2, 4, 5]
    l1, l2 = [], []
    for i, l in enumerate(ll):
        for val in l:
            l1.append(i)
            l2.append(val)
    return l1, l2

def ddlswitch(ddl):
    # switch first d structure with 2nd d
    if len(ddl) == 0:
        return {}
    new_keys = list(ddl[list(ddl.keys())[0]].keys())
    result = dict([(new_key, {})for new_key in new_keys])
    for k1, dl in ddl.items():
        assert new_keys == list(k)
        for k2, v2 in dl.items():
            result[k2][k1] = v2
    return result

def ddx2dx(ddx):
    # dict of dict of x
    result = {}
    for k1, v1 in ddx.items():
        for k2, v2 in v1.items():
            result[k1 + k2] = v2
    return result


def issame(l):
    return l[1:] == l[:-1]
    # return len(set(iterator)) <= 1

def save_model(model, file_path):
    # try try .save and .load from the class
    # If it is a keras model, use save_model
    # Default choice is pkl
    # Parameter to check if load works properly
    pass

def sizeof(val):
    return sys.getsizeof(val)


def fix_keras_shape(keras_shape):
    """
    Convert shape in keras's format to the DataStream format
    :param keras_shape: a shape in the keras format (either input or output shape)
    :return: keras_shape in the DataStream format
    """
    if isinstance(keras_shape, list):
        return [fix_keras_shape(shape) for shape in keras_shape]
    elif isinstance(keras_shape, tuple):
        assert keras_shape[0] is None
        return keras_shape[1:]
    else:
        raise BaseException(f'Unrecognized type {type(model_output_shape)}')

def stream_len(data_stream, max_length=5e7):
    # return the length of a data_stream
    result = 0
    for i, point in enumerate(data_stream):
        if i == max_length:
            break
        result += 1
    return result

def fake_data(model, n=1, kind='ones'):
    input_shapes = fix_keras_shape(model.input_shape)
    output_shapes = fix_keras_shape(model.output_shape)
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]
    if not isinstance(output_shapes, list):
        output_shapes = [output_shapes]
    x = []
    y = []
    for input_shape in input_shapes:
        x.append(np.ones((n,) + input_shape))
    for output_shape in output_shapes:
        y.append(np.ones((n,) + output_shape))
    return x, y

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

def to_categorical(x, n):
    # example: to_one_hot([np.array([1, 2, 3]), np.array(1)], 4) 
    if isinstance(x, list):
        return [to_categorical(val, n) for val in x]
    elif isinstance(x, tuple):
        return tuple(to_categorical(val, n) for val in x)
    elif isinstance(x, dict):
        return dict((k, to_categorical(x[k], n)) for k in x.keys())
    elif isinstance(x, np.ndarray):
        return keras.utils.np_utils.to_categorical(x, n)
    else:
        raise BaseException(f'to_one_hot unhandled type {type(x)}')


def readable_bytes(n_bytes):
    total_size = n_bytes
    size_names = ['B', 'KB', 'MB', 'GB', 'TB']
    for i, name in enumerate(size_names):
        if total_size < 1000:
            return f'{int(total_size)} {size_names[i]}'
            break
        total_size /= 1000
    raise BaseException(f"Directory is too big: {total_size} PB")


'''
******************************************* 
EXPERIMENT HELPER FUNCTIONS ***************
*******************************************
'''

import pandas as pd
#from experimental_reusable import gety
import re, json, os, copy

def informative_json_loads(s):
    try:
        s = s.replace("'", '"').replace("None", "0").replace('False', '0').replace('True', '1')
        s = s.replace('true', '1').replace('false', '0')
        return json.loads(s)
    except Exception as ex:
        msg = f'Json Parse Error: {ex.pos}, {s[ex.pos:min(len(s),  + 30)]}'
        start = max(0, ex.pos - 100)
        end = min(len(s),  + 100)
        msg += s[start:end]
        print(msg)
        raise BaseException(msg)

def read_exp(n, grep_exp="{"):
    def my_replace(line):
        #i = line.find('{')
        #line =line[i:]
        line = re.findall('{.*}', line)[0]
        return line.replace("'", '"').replace("None", "0").replace('False', '0').replace('True', '1')

    print('download nohup:', gety())
    temp_file_path = '/Users/hromel/Documents/Jupyter/temp.out'
    print('grep to temp.out:', os.system('cat /Users/hromel/Documents/ml_results/exp' + str(n) + '-nohup.out | grep -E "' + 
                    grep_exp + '" > ' + temp_file_path))
    #l = fileToLines('/Users/hromel/Documents/ml_results/exp50-nohup.out')
    #l = list(filter(lambda x: x.startswith('{'), l))
    with open(temp_file_path) as f:
        l = list(f.read().splitlines())
    l = [informative_json_loads(x) for x in l]
    return pd.DataFrame(l)

def view_exp(n):
    os.system('open ' + '/Users/hromel/Documents/ml_results/exp' + 
              str(n) + '-nohup.out')

'''
def see_end(n, file_name):    
    import subprocess
    result = subprocess.run(['tail', '-n', str(n), file_name], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8") 

def see_progress(n, file_name):
    import re
    yo = see_end(n, file_name)
    print(re.sub(r"[^\n]+\r", "", yo))

def get_log(n=24, log_type="train"):
    submit_cmd = (f"ssh henri@192.168.0.109 " + '"'
                  + "tail -10000 ~/github/deepengine/deepengine.log | grep -oE 'Henri" + str(n) + ": " + log_type + " batch.+'"
                  + " | sed 's/:/,/;s/ -/,/;s/ -/,/' > ~/ml_results/" + log_type + f"_batch{n}.csv" + '"')
    #print(submit_cmd)
    #assert False
    print(os.system(submit_cmd))
    
    print(os.system('rsync -auv henri@192.168.0.109:ml_results /Users/hromel/Documents'))


def my_plots2(n=24):
    df = pd.read_csv(f'/Users/hromel/Documents/ml_results/next_batch{n}.csv', header=None)
    pw.altLines(df[3].values[1:])
    print(readable_sec(df[3].values[1:].sum()))
    
    df2 = pd.read_csv(f'/Users/hromel/Documents/ml_results/train_batch{n}.csv', header=None)
    pw.altLines(df2[3].values[1:])
    print(readable_sec(df2[3].values[1:].sum()))
'''

def parse_log(file_path):
    empty_losses =  {
        'loss': [],
        'a_loss': [], 
        'b_loss': [],
        'a_acc': []
    }
    patterns = {
        'exp': '(?<=Experiment: )[\d]+',
        'rep': '(?<=Repetition )[\d]+',
        'loss': '(?<=^loss=)[\d]+\.?[\d]*', 
        'a_loss': '(?<=model_b_a_output_loss:)[\d]+\.?[\d]*', 
        'b_loss': '(?<=model_b_b_output_loss:)[\d]+\.?[\d]*',
        'a_acc': '(?<=model_b_a_output_acc:)[\d]+\.?[\d]*'
    }
    
    with open(file_path, 'r') as f:
        look_for = set(patterns.keys())
        
        exp = -1
        rep = -1
        losses = {}
        
        for line in f:
            for key in look_for:
                extracted = re.findall(patterns[key], line)
                if len(extracted) > 0:
                    extracted = extracted[0]
                    if key == 'exp':
                        exp = int(extracted)
                    elif key == 'rep':
                        rep = int(extracted)
                        losses[(exp, rep)] = copy.deepcopy(empty_losses)
                    else:
                        losses[(exp, rep)][key].append(float(extracted))
    
    return losses

def parse_losses(n):
    gety()

    file_path = f'/Users/hromel/Documents/ml_results/exp{n}-nohup.out'
    empty_losses =  []
    patterns = {
        'exp': '(?<=Experiment: )[\d]+',
        'rep': '(?<=[Rr]epetition )[\d]+',
        'loss': '(?<=^loss=)[\d]+\.?[\d]*'
    }
    
    with open(file_path, 'r') as f:
        look_for = set(patterns.keys())
        
        exp = -1
        rep = -1
        losses = {}
        
        for line in f:
            for key in look_for:
                extracted = re.findall(patterns[key], line)
                if len(extracted) > 0:
                    extracted = extracted[0]
                    if key == 'exp':
                        exp = int(extracted)
                    elif key == 'rep':
                        rep = int(extracted)
                        losses[(exp, rep)] = copy.deepcopy(empty_losses)
                    else:
                        losses[(exp, rep)].append(float(extracted))
    
    return losses

#losses = parse_log(hresults + 'exp61-nohup.out')

