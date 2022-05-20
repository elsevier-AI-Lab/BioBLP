import sys
import time
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        torch.save(obj, output, pickle_module=dill)


def load_object(filename):
    with open(filename, 'wb') as object:
        obj = torch.load(object, pickle_module=dill, encoding='utf-8')


def read_query(query_filename):
    """
    Read a query from file and return as a string
    Parameters
    ----------
    query_filename: str name of the query. It will be looked for in the queries folder of this project
    Returns
    -------
    query: str the query with placeholders for the query parameters, as a string to be formatted
    """
    # query_filepath = Path(RAW_DIR / QUERY_DIR / query_filename)

    with open(query_filename) as fr:
        query = fr.read()
    return query

    
def loading_animation(process, message="Loading") :
    while process.isAlive() :
        chars = "/—\|" 
        for char in chars:
            sys.stdout.write('\r' + f'{message} {char} ')
            time.sleep(.1)
            sys.stdout.flush()


def write_dict_as_pkl(dict_object, filename):
    """
    filename: path to pickle file, should include appropiate .pkl extension
    """
    with open(filename, "wb") as pkl_handle:
        pickle.dump(dict_object, pkl_handle)


def load_dict_from_pkl(filename):
    """
    filename: path to pickle file, should include appropiate .pkl extension
    """
    with open(filename, "rb") as pkl_handle:
        dict_object = pickle.load(pkl_handle)

    return dict_object

