from __future__ import print_function
import numpy as np
import argparse
import os
import pandas as pd
import xgboost as xgb
import joblib
from sagemaker_containers import _content_types, _errors
from sagemaker_xgboost_container.constants import xgb_content_types
import csv
import logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    print('args train', args.train)
    train_input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    print('len(train_input_files)', len(train_input_files))
    if len(train_input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_train_data = [ pd.read_csv(file, header=None, engine="python") for file in train_input_files ]
    train_data = pd.concat(raw_train_data)
    print('train_data.shape', train_data.shape)

    print('args validation', args.validation)
    val_input_files = [ os.path.join(args.validation, file) for file in os.listdir(args.validation) ]

    print('len(val_input_files)', len(val_input_files))
    if len(val_input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_val_data = [ pd.read_csv(file, header=None, engine="python") for file in val_input_files ]
    val_data = pd.concat(raw_val_data)
    print('val_data.shape', val_data.shape)
    # labels are in the first column
    y = train_data.iloc[:,0]
    X = train_data.iloc[:,1:]

    val_y = val_data.iloc[:,0]
    val = val_data.iloc[:,1:]
    
    print('train data shape -1', (train_data.shape[1]-1))
    print(train_data.iloc[0,:])
    print('new')
    dtrain = xgb.DMatrix(X, label=y, feature_names = tuple([('f' + str(i)) for i in np.arange(train_data.shape[1]-1)]))
    dval = xgb.DMatrix(val, label=val_y, feature_names = tuple([('f' + str(i)) for i in np.arange(train_data.shape[1]-1)]))
    
    #l = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32']
    
    #print('l',len(l))
    #print('y', len(y))
    #print('X', len(X))
    
    #dtrain = xgb.DMatrix(X, label=y)
    #dval = xgb.DMatrix(val, label=val_y)
    
    num_round =100 



    evallist = [(dtrain, 'train'),(dval, 'eval')]
    #evallist = [(dval, 'eval'),(dtrain, 'train')]
    params = {'alpha': 0.28, 'eta' : 0.32, 'max_depth': 3, 'min_child_weight' :8.52, 'objective':'binary:logistic' }
    #params = {'alpha': 0.78, 'eta' : 0.82, 'max_depth': 7, 'min_child_weight' :8.78,'objective':'binary:logistic'}
    
    
    # lock num_boost_round to 10. This gives the best performance.
    bst = xgb.train(params, dtrain, evals= evallist, early_stopping_rounds=50, num_boost_round = 50,
                     verbose_eval = False)
    print('live')
    print('score' ,bst.best_score)
    print('iteration', bst.best_iteration)
    print('ntree limit', bst.best_ntree_limit)

    # Print the coefficients of the trained classifier, and save the coefficients
    response = joblib.dump(bst, os.path.join(args.model_dir, "model.joblib"))
    print('joblib dump', response)
    print(os.path.join(args.model_dir, "model.joblib"))



def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    print('model_fn dir', model_dir)
    bst = joblib.load(os.path.join(model_dir, "model.joblib"))
    print(os.path.join(model_dir, "model.joblib"))
    
    print('score' ,bst.best_score)
    print('best iteration', bst.best_iteration)
    print('ntree limit', bst.best_ntree_limit)

    return bst


# input-test
def predict_fn(input_data, model):
    print('predict_fn model', model)
    print('predict_fn input data', input_data)
    
    output = model.predict(input_data, validate_features=False)
    print("no ntree:", model.predict(input_data, validate_features=False))
    print("best ntree:", model.predict(input_data, validate_features=False, ntree_limit = model.best_ntree_limit))
    return output


def _clean_csv_string(csv_string, delimiter):
    return ['nan' if x == '' else x for x in csv_string.split(delimiter)]


def my_csv_to_dmatrix(string_like, dtype=None):  # type: (str) -> xgb.DMatrix
    """Convert a CSV object to a DMatrix object.
    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (xgb.DMatrix): XGBoost DataMatrix
    """
    sniff_delimiter = csv.Sniffer().sniff(string_like.split('\n')[0][:512]).delimiter
    delimiter = ',' if sniff_delimiter.isalnum() else sniff_delimiter
    logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))

    try:
        print('trying to convert to np payload')
        print(string_like)
        print('trying to convert tpye', type(string_like))
        #print([string_like])
        print('split', string_like.split('\n'))
        print(1)
        print(map(lambda x: _clean_csv_string(x, delimiter), string_like.split('\n')))
        print(2)
        print(map(lambda x: _clean_csv_string(x, delimiter), string_like.split('\n')[:-1]))
        #print('splitt', string_like.split('\n')[:-1])
        print(3)
        print(list(map(lambda x: _clean_csv_string(x, delimiter), string_like.split('\n')[:-1])))
        print('without minus 1', list(map(lambda x: _clean_csv_string(x, delimiter), string_like.split('\n'))))
        print(4)
        np_payload = np.array(list(map(lambda x: _clean_csv_string(x, delimiter), string_like.split('\n')[:-1]))).astype(dtype)
        
        print('converted to payload')
    except:
        print('Problem in npy payload conversion')

    return xgb.DMatrix(np_payload)

_my_dmatrix_decoders_map = {
    _content_types.CSV: my_csv_to_dmatrix}#,
    #xgb_content_types.LIBSVM: libsvm_to_dmatrix,
    #xgb_content_types.X_LIBSVM: libsvm_to_dmatrix}


def my_decode(obj, content_type):
    # type: (np.array or Iterable or int or float, str) -> xgb.DMatrix
    """Decode an object ton a one of the default content types to a DMatrix object.
    Args:
        obj (object): to be decoded.
        content_type (str): content type to be used.
    Returns:
        np.array: decoded object.
        
    """
    print('inside decoder')
    print(content_type)
    try:
        decoder = _my_dmatrix_decoders_map[content_type]
        return decoder(obj)
    except KeyError:
        raise _errors.UnsupportedFormatError(content_type)

def input_fn(input_data, content_type):
    print('inside default input') 
    print(input_data)
    print(content_type)
    return my_decode(input_data, content_type)


