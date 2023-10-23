import os
import time
import pickle
import pandas as pd
import tqdm
import dill

class FileCache:
    def __init__(self):
        self.cache = {}  # This will hold the data

    def __contains__(self, filename):
        if filename not in self.cache:
            return False
        if self._is_file_modified(filename):
            del self.cache[filename]
            return False
        return True

    def __getitem__(self, filename):
        if filename not in self.cache or self._is_file_modified(filename):
            self.invalidate(filename)
            return None
        self.cache[filename]['last_access_time'] = time.time()
        return self.cache[filename]['data']

    def __setitem__(self, filename, value):
        with open(filename, "wb") as f:
            dill.dump(value, f)

        current_time = os.path.getmtime(filename)

        self.cache[filename] = {
            'data': value,
            'last_access_time': current_time,
            'last_modified_time':  current_time # Update in-memory modified time
        }


    def __delitem__(self, filename):
        if filename in self.cache:
            del self.cache[filename]

    def _is_file_modified(self, filename):
        if filename not in self.cache:
            print('INVALIDTE MODIFIED')
            return True
        return os.path.getmtime(filename) > self.cache[filename]['last_modified_time']

    def invalidate(self, filename):
        print(f"INVALIDATE!!: {filename}")
        if filename in self.cache:
            del self.cache[filename]

    def keys(self):
        # Clean-up invalidated keys
        filenames = list(self.cache.keys())
        for filename in filenames:
            if self._is_file_modified(filename):
                self.invalidate(filename)
        return self.cache.keys()

    def commit(self):
        with tqdm.tqdm(total = len(self.cache), desc = 'Comitting in memory chunks') as pbar:
            for filename, data in self.cache.items():
                # If in-memory modified time is newer than the file's modified time
                if data['last_modified_time'] > os.path.getmtime(filename):
                    if isinstance(data, (pd.DataFrame,)):
                        with open(filename, "wb") as f:
                            dill.dump(data['data'], f)
                        data['last_modified_time'] = os.path.getmtime(filename)
                    else:
                        with open(filename, 'wb') as file:
                            dill.dump(data['data'], file)
                        os.utime(filename, (data['last_access_time'], data['last_modified_time']))
                pbar.update(1)

