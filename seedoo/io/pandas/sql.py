import os
import traceback

import psutil
import collections
import multiprocessing
import uuid

import seedoo.logger_helpers.logger_setup
import ast
import logging
import torch
import sqlite3
import pandas as pd
import os
import pickle
import numpy as np
import re
import inspect
import dill
import asyncio
import threading
import tqdm
import ast
import json
import time
from seedoo.io.pandas.utils import FileCache

from concurrent.futures import ThreadPoolExecutor, as_completed
pd.options.mode.chained_assignment = None

# Store a reference to the original `sqlite3.connect`.
_original_sqlite3_connect = sqlite3.connect

def set_pragmas_for_conn(conn):
    """Set optimized pragmas for a sqlite3 connection."""

    cur = conn.cursor()
    cur.execute("PRAGMA page_size;")
    page_size = cur.fetchone()[0]
    desired_cache_size_bytes = 30 * 1024 ** 3

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA wal_autocheckpoint = 500;")

    # Calculate the number of pages required for the desired cache size
    num_pages = desired_cache_size_bytes // page_size

    # Set the cache size
    cur.execute(f"PRAGMA cache_size={num_pages};")


    conn.execute("PRAGMA synchronous=NORMAL;")
    mmap_size = desired_cache_size_bytes  # Setting it to the same size as the cache for simplicity
    cur.execute(f"PRAGMA mmap_size={mmap_size};")
    cur.close()

def optimized_sqlite3_connect(*args, **kwargs):
    """Wrapper around sqlite3.connect that sets optimized pragmas."""
    conn = _original_sqlite3_connect(*args, **kwargs)
    set_pragmas_for_conn(conn)
    return conn

# Monkey-patch sqlite3.connect with our optimized version.
sqlite3.connect = optimized_sqlite3_connect

original_to_sql = pd.DataFrame.to_sql


def custom_read_sql(sql, con, index_col = None, chunk_size=500):
    logger = logging.getLogger(__name__)
    # Create a cursor from the connection
    cur = con.cursor()

    cur.execute(sql)

    # Fetch the column names from the cursor description
    columns = [col[0] for col in cur.description]

    cur_pragma = con.cursor()

    split = 'from' if 'from' in sql else 'FROM'
    # If we want to infer the types from SQLite's table schema:
    cur_pragma.execute(f"PRAGMA table_info(data)")
    type_mapping = {row[1]: row[2].lower() for row in cur_pragma.fetchall()}
    cur_pragma.close()

    def get_dtype(col_name):
        sqlite_type = type_mapping.get(col_name, 'float')
        if sqlite_type in ["int", "integer", "tinyint", "smallint", "mediumint", "bigint", "unsigned big int", "int2", "int8"]:
            return "int"
        elif sqlite_type in ["real", "double", "double precision", "float"]:
            return "float"
        elif sqlite_type in ["text", "char", "varchar", "clob"]:
            return "object"
        elif sqlite_type in ["blob"]:
            return "bytes"
        else:
            # Default to object for unrecognized or mixed types.
            return "object"

    dtypes = {col: get_dtype(col) for col in columns}

    dfs = []

    #with tqdm.tqdm(desc="Fetching rows") as pbar:
    while True:
        data_chunk = cur.fetchmany(chunk_size)
        if not data_chunk:
            break

        n = np.array(data_chunk)
        dfs.append(n)
    #pbar.update(len(data_chunk))

    logger.info('finished fetching rows, building result data frame')
    cur.close()
    if len(dfs) == 0:
        return pd.DataFrame([], columns=columns).astype(dtypes)

    df = pd.DataFrame(np.concatenate(dfs, axis = 0), columns = columns).astype(dtypes)
    logger.info('finished building result dataframe')
    return df


def custom_to_sql(
    df,
    name: str,
    con,
    schema=None,
    if_exists: str = "fail",
    index: bool = True,
    index_label=None,
    chunksize=None,
    dtype=None,
    method=None,
) -> None:
    """
    Custom to_sql function to improve SQLite writing performance.
    """

    # If 'replace', use original pandas to_sql for the first few rows to set up table schema
    if if_exists == 'replace':
        # We do this because for some reason executemany (later in the code) has a bug that when vals is of length 1
        # it just inserts Null. I have no idea why, but this is a hack to overcome it.
        if len(df.columns.values) < 2:
            df['extra_spare_column'] = 'extra'

        logger = logging.getLogger(__name__)
        start = time.time()
        logger.info('Started insert with to_sql')

        if "index" in df:
            del df['index']

        original_to_sql(df.iloc[:1], name, con, if_exists=if_exists, index=False, index_label=index_label,
                        dtype=dtype)

        con.commit()
        df = df.iloc[1:]
        if_exists = 'append'  # Switch to 'append' mode for the remaining rows
        end = time.time()
        logger.info(f'Finished insert with to_sql, it took: {(end - start) * 1000} ms')


    # Create a connection and cursor
    if if_exists == 'append':
        cur = con.cursor()
        # Calculate chunk size, if not provided
        chunksize = chunksize or 100_000
        column_names = ", ".join([f'{col}' for col in df.columns])
        # Prepare the placeholders
        num_columns = len(df.columns)
        placeholders = ", ".join(["?"] * num_columns)

        with tqdm.tqdm(total = int(len(df) // chunksize) + 1, desc = 'SQLLite batch insert') as pbar:
            for start in range(0, len(df), chunksize):
                end = start + chunksize
                batch = df.iloc[start:end]
                # Extract column names from the DataFrame
                data = [tuple(row) for row in batch.values]

                cur.execute("BEGIN;")
                # Execute the in sert with explicit column names
                cur.executemany(f"INSERT INTO {name} ({column_names}) VALUES ({placeholders})", data)
                logging.getLogger(__name__).info('Committing for batch')
                cur.execute("END;")

                con.commit()
                logging.getLogger(__name__).info('Done committing for batch')
                pbar.update(1)
        cur.close()

    con.commit()


# Monkey patch pandas DataFrame's to_sql with our custom version
pd.DataFrame.to_sql = custom_to_sql
origin_read_sql = pd.read_sql
pd.DataFrame.read_sql = custom_read_sql
pd.read_sql = custom_read_sql

def pandas_query_to_sqlite(query_str):
    # Access calling frame's local and global variables
    frame = inspect.stack()[1]
    calling_locals = frame[0].f_locals
    calling_globals = frame[0].f_globals

    # Convert `.str.contains("value")` to `LIKE "%value%"`
    query_str = re.sub(r"\.str\.contains\('(.*?)'\)", r' LIKE "%\1%"', query_str)
    query_str = re.sub(r'\.str\.contains\("(.*?)"\)', r' LIKE "%\1%"', query_str)

    # Convert double equals to a single equal for SQL syntax
    query_str = query_str.replace("==", "=")

    # Convert @variable to its value
    for match in re.findall(r'@\w+', query_str):
        var_name = match[1:]  # remove @
        value = calling_locals.get(var_name) or calling_globals.get(var_name)
        if isinstance(value, (list, tuple)):
            value = ', '.join([f'"{v}"' if isinstance(v, str) else str(v) for v in value])
            query_str = query_str.replace(match, f"({value})")
        else:
            value = f'"{value}"' if isinstance(value, str) else str(value)
            query_str = query_str.replace(match, value)

    return query_str

class SQLDataFrameAttrWrapper:
    def __init__(self, attr_generators):
        self.attr_generators = attr_generators

    def __getattr__(self, item):
        def method(*args, **kwargs):
            results = []

            for attr in self.attr_generators:
                method = getattr(attr, item)
                results.append(method(*args, **kwargs))

            # Concatenate results from all chunks
            return pd.concat(results, axis=0)

        return method

class DataFrameGroupByWrapper:
    def __init__(self, groupby_generators):
        self.groupby_generators = groupby_generators

    def __getitem__(self, key):
        # Define a generator that applies getitem on each groupby object
        def item_generator():
            for groupby_obj in self.groupby_generators:
                yield groupby_obj[key]

        return DataFrameGroupByWrapper(item_generator())

    def parallel_apply(self, func, *args, **kwargs):
        results = []

        for index, groupby_obj in self.groupby_generators:
            result_chunk = groupby_obj.apply(func, *args, **kwargs)
            results.append(result_chunk)

        # Concatenate results from all chunks
        return pd.concat(results, axis=0)

    def __getattr__(self, attr):
        # If the attribute exists in SQLDataFrameWrapper or it's not part of pandas DataFrame, don't delegate
        if hasattr(self, attr):
            return super().__getattribute__(attr)

        elif not hasattr(pd.DataFrame(), attr):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        else:
            # Define a generator that applies getattr on each chunk
            def attr_generator():
                for chunk_df in self.chunked_dataframes:
                    yield getattr(chunk_df, attr)

            return SQLDataFrameAttrWrapper(attr_generator())

    def transform(self, func, *args, **kwargs):
        results = []

        for groupby_obj in self.groupby_generators:
            result_chunk = groupby_obj.transform(func, *args, **kwargs)
            results.append(result_chunk)

        # Concatenate results from all chunks
        return pd.concat(results, axis=0)


def optimize_sqlite(conn, desired_cache_size_gb=30):
    # Convert desired cache size from GB to bytes
    desired_cache_size_bytes = desired_cache_size_gb * 1024 ** 3

    # Connect to the SQLite database
    cur = conn.cursor()

    # Get the page size
    cur.execute("PRAGMA page_size;")
    page_size = cur.fetchone()[0]

    # Calculate the number of pages required for the desired cache size
    num_pages = desired_cache_size_bytes // page_size

    # Set the cache size
    cur.execute(f"PRAGMA cache_size={num_pages};")

    # Set other performance enhancing pragmas
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # Set memory-mapped I/O size
    mmap_size = desired_cache_size_bytes  # Setting it to the same size as the cache for simplicity
    cur.execute(f"PRAGMA mmap_size={mmap_size};")

    cur.close()
    # Commit changes and close the connection
    conn.commit()


class SQLDataFrameWrapper:
    class LocIndexer:
        def __init__(self, wrapper):
            self.wrapper = wrapper
        def __setitem__(self, *args):
            print(args)

        def __getitem__(self, idx):
            # If idx is a slice, it's equivalent to df.loc[start:stop]
            if isinstance(idx, slice):
                # Handle the case where the stop value in the slice might be None.
                # If it's None, then we'll need a default value.
                # A typical default could be the length of the DataFrame, but it depends on your context.
                default_stop = len(self.wrapper.chunk_files)  # Assuming this gives the length of the DataFrame
                start, stop, step = idx.start, idx.stop, idx.step

                # Create an array of indices based on the slice's start, stop, and step
                row_idx = np.arange(start, stop, step).tolist()
                return self.wrapper._loc_method(row_idx, None)

            # Handle single label, list of labels, or tuple with (row_labels, col_labels)
            elif isinstance(idx, tuple):
                row_labels, col_labels = idx
                if isinstance(row_labels, (int, np.int32, np.int64)):
                    row_labels = [row_labels]

                if isinstance(row_labels, slice):
                    # Ensure the slice start and stop are valid labels in the DataFrame
                    # This check can be expanded further based on your specific implementation
                    assert row_labels.start in self.wrapper.chunking_index and row_labels.stop in self.wrapper.chunking_index
                return self.wrapper._loc_method(row_labels, col_labels)
            else:
                if isinstance(idx, list):
                    row_labels = idx
                elif isinstance(idx, (int, np.int32, np.int64)):
                    row_labels = [idx]

                col_labels = None
                return self.wrapper._loc_method(row_labels, col_labels)

    class IlocIndexer:
        def __init__(self, wrapper):
            self.wrapper = wrapper

        def __getitem__(self, idx):
            # If idx is a slice, it's equivalent to df.iloc[start:stop]
            if isinstance(idx, slice):
                # Handle the case where the stop value in the slice might be None.
                # If it's None, then we'll need a default value.
                # A typical default could be the length of the DataFrame, but it depends on your context.
                default_stop = len(self.wrapper.chunk_files)  # Assuming this gives the length of the DataFrame
                start, stop, step = idx.start, idx.stop, idx.step

                # Create an array of indices based on the slice's start, stop, and step
                row_idx = np.arange(start, stop, step).tolist()
                return self.wrapper._iloc_method(row_idx, None)

            # Handle single integer, list of integers, or tuple with (row_idx, col_idx)
            elif isinstance(idx, tuple):
                row_idx, col_idx = idx
                # Ensuring the slice start and stop are valid integer indices can be done in _iloc_method
                # Here, we simply pass them along
                return self.wrapper._iloc_method(row_idx, col_idx)

            else:
                row_idx = idx
                col_idx = None
                return self.wrapper._iloc_method(row_idx, col_idx)

    def __init__(self, df=None, db_name="database.db", path = os.getcwd(), chunk_size = 1024):
        self.db_name = os.path.join(path, db_name)
        self.complex_columns = []
        self._chunk_cache = FileCache()
        self.eval_cache = {}
        self.path = path
        self.chunk_size = chunk_size
        self.always_commit = False
        self._iloc_indexer = SQLDataFrameWrapper.IlocIndexer(self)
        self.append_lock = threading.Lock()
        self._loc_indexer = SQLDataFrameWrapper.LocIndexer(self)
        self.thread_executor = ThreadPoolExecutor(8)  # Initializes a thread pool executor
        self.special_types = {}
        self.logger = logging.getLogger(__name__)
        self._connection = {}

        if df is not None:
            # If append mode, fetch the max chunking_index from the DB and adjust the new df's chunking_index accordingly
            df['chunking_index'] = df.core_index.values.astype(np.int32)

            self._simple_columns, self.complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df)
            self.special_types.update(special_types)
            self._store_data(df, append = False)

    @property
    def connection(self):
        thread_id =  threading.get_ident() % 3

        if thread_id not in self._connection:
            self._connection[thread_id] = sqlite3.connect(self.db_name, check_same_thread=False)
            optimize_sqlite(self._connection[thread_id])

        return self._connection[thread_id]

    def __len__(self):
        return self.connection.execute("select count(*) from data").fetchone()[0]

    def __del__(self):
        self.connection.close()

    def _loc_method(self, row_labels, col_labels=None):
        # Handle the logic for .loc indexer
        columns_query = None
        if col_labels:
            simple_columns = self.simple_columns
            if 'chunking_index' not in col_labels:
                col_labels.append('chunking_index')
            if 'chunk_id' not in col_labels:
                col_labels.append('chunk_id')

            columns_query = [c for c in col_labels if c in simple_columns]

        if col_labels:
            columns_query = ",".join([f"data.{c}" for c in columns_query])
        else:
            columns_query = "data.*"

        if row_labels is None:
            # Join the temp table with the main data table on chunking_index
            query = f"""
                SELECT {columns_query}
                FROM data
            """
            conn = self.connection
            subset_df = pd.read_sql(query, conn)

        # Convert row_labels to a DataFrame for insertion to temp table
        elif len(row_labels) < 100:
            ids = ",".join([str(i) for i in row_labels])
            subset_df = pd.read_sql(f"select * from data where chunking_index in ({ids})", self.connection)
        else:
            temp_df = pd.DataFrame({'chunking_index': row_labels})

            # Insert row_labels into a temp table
            conn = self.connection
            with self.append_lock:
                temp_df.to_sql("temp_table", conn, if_exists="replace", index=False, method='multi')

                conn.execute(f"CREATE INDEX chunking_index_idx_temp_table ON temp_table (chunking_index)")

                # Join the temp table with the main data table on chunking_index
                query = f"""
                    SELECT {columns_query}
                    FROM data
                    JOIN temp_table ON data.chunking_index = temp_table.chunking_index
                """
                subset_df = pd.read_sql(query, conn)

                conn.execute(f"DROP TABLE temp_table")

        subset_df = self._restore_types(subset_df)

        if col_labels is None or set(col_labels) & (set(self.complex_columns) - set(['chunking_index', 'chunk_id'])):
            subset_df = self.fetch_all_for_df(subset_df)
        return subset_df

    @property
    def columns(self):
        return pd.Series(list(self.complex_columns + self.simple_columns))

    @staticmethod
    def identify_column_types(df):
        """
        Identifies and returns lists of simple and complex columns from the dataframe.

        Parameters:
        - df: DataFrame to analyze

        Returns:
        - tuple of two lists: (simple_columns, complex_columns)
        """

        logger = logging.getLogger(__name__)
        simple_cols = []
        complex_cols = []
        special_types = {}

        def safely_convert(i):
            if isinstance(i, (int, np.float64, np.float32, np.float16,)):
                return np.int32(i)
            elif isinstance(i, (float, np.float64, np.float32, np.float16,)):
                return np.float32(i)
            else:
                return i


        for c in df.columns.values:
            index = 0
            # Check for numeric types and adjust data types if needed
            while index < len(df):
                try:
                    val = df[c].values[index]
                    if isinstance(val, (np.ndarray, pd.DataFrame)):
                        break
                    elif isinstance(val, type(None)) or \
                                       (not isinstance(val, (np.ndarray, pd.DataFrame, list, dict)) and pd.isnull(val)):
                        index +=1
                        continue
                    else:
                        break
                except Exception as exc:
                    logger.error(f'Error while evaluating type of column: {c}')
                    raise


            if index >= len(df):
                df[c].fillna('', inplace=True)
                simple_cols.append(c)
                continue

            if isinstance(df[c].values[index], (str, float, int, np.float32, np.int32, np.float64, np.int64, bool, np.bool_)):
                simple_cols.append(c)
                if isinstance(df[c].values[index], (np.bool_, bool,)):
                    df[c] = df[c].apply(lambda x: bool(x))
                elif isinstance(df[c].values[index], np.int64):
                    df[c] = df[c].astype(np.int32)
                elif isinstance(df[c].values[index], np.float64):
                    df[c] = df[c].astype(np.float32)
            else:

                if isinstance(df[c].values[index], (np.ndarray,)):
                    df[c] = df[c].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [0])


                if isinstance(df[c].values[index], (dict,)):
                    df[c] = df[c].apply(lambda x: str(x))
                    simple_cols.append(c)
                    special_types[c] = dict
                if isinstance(df[c].values[index], (tuple,torch.Size, list)):
                    if isinstance(df[c].values[index], (torch.Size, list)):
                        special_types[c] = type(df[c].values[index])
                        df[c] = df[c].apply(lambda x: str(tuple(x)) if isinstance(x, torch.Size) else str(x))
                    else:
                        special_types[c] = type(df[c].values[index])
                        df[c] = df[c].apply(lambda x: str(x))
                    simple_cols.append(c)
                else:
                    complex_cols.append(c)

        complex_cols.append("chunk_id")
        complex_cols.append("chunking_index")

        simple_cols = set(simple_cols)
        complex_cols = set(complex_cols)

        overlap = simple_cols & (complex_cols - set(['chunk_id', 'chunking_index']))
        if len(overlap) > 0:
            logging.getLogger(__name__).warning(f'We have overlap of complex and simple columns: {overlap} removing the overlap from the complex columns')
            complex_cols = complex_cols - overlap


        return list(simple_cols), list(complex_cols), special_types

    def __getstate__(self):
        state = {
            'complex_columns': self.complex_columns,
            'always_commit' : self.always_commit,
            'path': self.path,
            '_chunk_cache' : {},
            'special_types' : self.special_types,
            'chunk_size' : self.chunk_size,
            'db_name': self.db_name
        }

        for key, connection in self._connection.items():
            connection.commit()
            connection.close()

        self._connection = {}

        self.thread_executor.shutdown(wait = True)
        self.thread_executor = ThreadPoolExecutor(4)
        return state

    def update(self, df, swap = False):
        """
        Updates the SQLite table with new data from the provided DataFrame.

        Parameters:
        - df: DataFrame with new data.
        """

        df = df.copy()
        if "index" in df:
            del df['index']


        # Step 1: Identify simple and complex columns in the provided dataframe
        provided_simple_cols, provided_complex_cols, special_types = self.identify_column_types(df)
        self.special_types.update(special_types)

        # Step 2: Identify new columns compared to existing data
        columns_to_use = list(set(self.simple_columns) - (set(provided_simple_cols) - set(['chunking_index', 'chunk_id'])))
        columns_to_use = ", ".join([f"a.{col} as {col}" for col in columns_to_use])
        columns_to_add_or_update = ", ".join([f"b.{col} as {col}" for col in provided_simple_cols if col != 'chunking_index' and col != 'chunk_id'])

        conn = self.connection
        # Step 3: Insert the new DataFrame into a temporary table
        self.logger.info(f'Inserting df of length {len(df)} into temp table for update')
        temp_table_name = f'temp_new_data_{str(uuid.uuid1()).replace("-", "")}'
        with self.append_lock:
            conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

            df[provided_simple_cols].to_sql(temp_table_name, conn, if_exists='replace', index=False, method='multi')

            self.logger.info(f'Creating index for temp table on bulk insert')
            # Step 6: Re-add indices to the new data table
            conn.execute(f"CREATE INDEX chunking_index_idx_{temp_table_name} ON {temp_table_name} (chunking_index)")
            self.logger.info(f'Done creating index for temp table on bulk insert')

        if swap:
            with self.append_lock:
                conn.execute("DROP TABLE data")
                conn.execute(f"ALTER TABLE {temp_table_name} RENAME TO data")
                conn.commit()
        else:
            self.logger.info(f'Joining temp table with main table for update')
            # Step 4: Join the temp table with the main data table
            conn.execute(f"""
                CREATE TABLE temp_combined_data AS
                SELECT {columns_to_use},{columns_to_add_or_update}
                FROM data AS a
                INNER JOIN {temp_table_name} AS b
                ON a.chunking_index = b.chunking_index
            """)

            # Step 5: Swap tables
            self.logger.info('Dropping data and renaming the temp table')

            with self.append_lock:
                conn.execute("DROP TABLE data")
                conn.execute("ALTER TABLE temp_combined_data RENAME TO data")
                conn.execute(f"DROP TABLE {temp_table_name}")
                conn.commit()

        self.logger.info(f'Creating indexexes for simple columns, except those in special types')
        for col in provided_simple_cols:
            if col != 'index' and col not in self.special_types:
                try:
                    self.logger.info(f'Creating index for {col}')
                    with self.append_lock:
                        conn.execute(f"CREATE INDEX idx_{col} ON data ({col})")
                        conn.commit()
                    self.logger.info(f'Done creating index for {col}')

                except sqlite3.OperationalError as exc:
                    if "already exists" in exc.args[0]:
                        pass
                    else:
                        raise

        self.logger.info(f'Done creating indexexes for simple columns, except those in special types')

        self.logger.info('Creating indexes')
        remaining_complex_columns = set(provided_complex_cols) - set(["chunking_index", "chunk_id"])

        if remaining_complex_columns:
            self.logger.info('Handling complex columns')
            df = df.drop_duplicates(subset = ['chunking_index', 'chunk_id'])
            self.update_chunked_dataframes(df[provided_complex_cols])


    def _update_complex_columns(self):
        chunk = self.fetch_raw_chunk(0)
        simple_columns = self.simple_columns
        self.complex_columns = list([c for c in chunk.columns.values if c not in simple_columns])
        self.complex_columns.append('chunk_id')
        self.complex_columns.append('chunking_index')

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=None, observed=False, dropna=True):
        # This will be a generator

        complex_overlap = set(by) & set(self.complex_columns)
        complex_overlap = complex_overlap - set(['chunking_index', 'chunk_id'])

        if complex_overlap:
            raise ValueError(f'Only simple columns can be used in a group by. You provided these complex columns: {list(complex_overlap)}')

        by = [i for i in by if i in self.simple_columns]
        columns_list = ",".join(by)

        df = self._inner_query(f"select distinct {columns_list} from data", fetch_complex=False)

        def fetch_value(row, column):
            val = row[column]
            if isinstance(val, (str,)):
                return f"'{val}'"
            elif column in self.special_types:
                return str(val)
            else:
                return val

        def groupby_gen():
            nonlocal df
            for index, row in df.iterrows():
                query = " and ".join([f"{column}=={fetch_value(row, column)}" for column in by])
                df = self.query(query)
                if group_keys:
                    yield tuple([row[column] for column in by]), df
                else:
                    yield index, df
        return DataFrameGroupByWrapper(groupby_gen())

    @property
    def simple_df(self):
        conn = self.connection
        query = "SELECT {cols} FROM data".format(cols=", ".join(self.simple_columns))
        df = pd.read_sql(query, conn)
        return df

    def to_pandas(self):
        """
        :return: Only the simple columns table as a simple pandas dataframe
        """
        conn = self.connection

        # Using a parameterized query to fetch records based on the chunking indices
        df_simple = pd.read_sql("SELECT * FROM data", conn)

        df_simple = self._restore_types(df_simple)
        return df_simple.set_index(df_simple.chunking_index.values)

    def update_chunked_dataframes(self, df):
        """
        Iterates through each chunk, updates with new data from the DataFrame and saves it back.

        Parameters:
        - df: DataFrame with new data.
        """
        self.thread_executor.shutdown(wait = True)
        self.thread_executor = ThreadPoolExecutor(8)

        # Ensure chunking_index is present in the DataFrame
        if "chunking_index" not in df.columns:
            raise ValueError("DataFrame must contain 'chunking_index' for updating.")

        with tqdm.tqdm(total=len(self.chunk_files), desc='Updating chunk files') as pbar:
            for i, chunk_file in self.chunk_files.items():
                # Load the chunk using read_pickle with compression
                pbar.update(1)
                chunk_df = self.fetch_raw_chunk(chunk_file = chunk_file)
                if chunk_df is None:
                    continue

                # Update the chunk using merge on the 'chunking_index'
                # Hack to fix data corruption bug of already existing data, remove after first iteation. Temp
                had_fixes = False
                for c in chunk_df.columns.values:
                    vals = chunk_df[c].values
                    if len(vals.shape) > 1 and vals.shape[1] > 1:
                        print(f'BAD DOUBLE COLUMN: {c}, DUPS: {vals.shape}')
                        vals = vals[:, 0]
                        del chunk_df[c]
                        chunk_df[c] = vals
                        had_fixes = True

                if had_fixes:
                    chunk_df = chunk_df.copy()
                    self._chunk_cache[chunk_file] = chunk_df.drop_duplicates(subset = ['chunk_id', 'chunking_index'])

                updated_chunk = pd.merge(chunk_df, df, on=["chunking_index", "chunk_id"], how="inner", suffixes=['_old', '_new'])
                updated_chunk = updated_chunk[[c for c in updated_chunk.columns.values if "_old" not in c]]

                rename = {}
                renamed_to_orig = collections.defaultdict(lambda: [])
                for c in updated_chunk.columns.values:
                    if '_new' in c:
                        new_name = c.replace("_new", "")
                        if len(renamed_to_orig[new_name]) == 0:
                            rename[c] = new_name
                            renamed_to_orig[rename[c]].append(c)
                        else:
                            if new_name in updated_chunk:
                                del updated_chunk[new_name]

                updated_chunk = updated_chunk.rename(columns = rename)

                # Save the updated chunk back using to_pickle with compression
                self._chunk_cache[chunk_file] = updated_chunk.drop_duplicates(subset = ['chunk_id', 'chunking_index'])

            self.commit()


    def chunked_dataframes(self):
        """ Generator to produce dataframes from the stored chunks."""
        with tqdm.tqdm(total = len(self.chunk_files), desc = 'Reading chunk files') as pbar:
            for i, chunk_file in self.chunk_files.items():
                chunk = self.fetch_raw_chunk(None, chunk_file=chunk_file)
                if chunk is None:
                    continue

                df_chunk = self.fetch_raw_chunk(chunk_file=chunk_file)
                pbar.update(1)
                yield df_chunk

    def __setstate__(self, state):
        self.complex_columns = state['complex_columns']
        self.path = state['path']
        self.db_name = state['db_name']
        self._cached_chunk_files = None
        self.special_types = state['special_types']
        self._last_modified_time = None
        self.thread_executor = ThreadPoolExecutor(8)
        self._chunk_cache = FileCache()
        self.always_commit = state.get('always_commit', False)
        self._loc_indexer = SQLDataFrameWrapper.LocIndexer(self)
        self._connection = {}
        self.append_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.chunk_size = state['chunk_size']
        self.eval_cache = {}
        self._iloc_indexer = SQLDataFrameWrapper.IlocIndexer(self)
        self._update_complex_columns()

    def _store_data(self, df, append=False):
        df['chunk_id'] = df['chunking_index'].apply(lambda x: int(np.floor(x / self.chunk_size)))
        _simple_columns, complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df)
        self.special_types.update(special_types)
        simple_columns_df = df[_simple_columns]

        with self.append_lock:
            # Save simple columns to SQLite
            conn = self.connection
            if append:
                simple_columns = list(set(simple_columns_df.columns.values) & set(self.simple_columns))
                simple_columns_df = simple_columns_df[simple_columns]
                simple_columns_df.to_sql("data", conn, index=True, if_exists="append")
            else:
                simple_columns_df.to_sql("data", conn, index=True, if_exists="replace")
                for col in self.simple_columns:
                    try:
                        conn.execute(f"CREATE INDEX idx_{col} ON data ({col})")
                    except sqlite3.OperationalError as exc:
                        if "already exists" in exc.args[0]:
                            pass
                        else:
                            raise

        with self.append_lock:
            for chunk_id, chunk in df.groupby(['chunk_id'], group_keys = True, as_index = False):
                chunk_id = chunk_id[0]
                filename = f"{os.path.join(self.path, 'chunks')}/chunk_{chunk_id}.pkl"
                new_chunk = chunk[complex_columns]
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))

                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        existing_chunk = dill.load(f)
                    if append:
                        new_chunk = pd.concat([existing_chunk, new_chunk])
                        new_chunk = new_chunk.drop_duplicates(subset = ['chunk_id', 'chunking_index'])

                    #if not append:
                    #    raise RuntimeError(f'Trying to append a chunk to an already existing one! {filename}')

                new_chunk.to_pickle(filename)

    @property
    def chunk_files(self):
        chunks_dir = os.path.join(self.path, "chunks")

        if not hasattr(self, '_cached_chunk_files'):
            chunks_dir = os.path.join(self.path, "chunks")
            self._cached_chunk_files = None
            self._last_modified_time = None


        # Check if the folder's modification time has changed
        current_modified_time = os.path.getmtime(chunks_dir)
        if current_modified_time != self._last_modified_time or self._cached_chunk_files is None or not self._cached_chunk_files:
            self._cached_chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if
                                        os.path.isfile(os.path.join(chunks_dir, f)) and f.endswith("pkl")]
            self._cached_chunk_files = {int(os.path.basename(k).split("_")[1].split(".")[0]) : k for k in self._cached_chunk_files}
            self._last_modified_time = current_modified_time

        return self._cached_chunk_files


    def _restore_types(self, df):
        cache = {}
        def parallel_eval(column_data, col, special_types):
            import ast
            nonlocal cache
            def safe_eval(x):
                nonlocal  cache
                try:
                    if x == 'nan' or x == 'None':
                        if special_types[col] == list:
                            x = []
                        elif special_types[col] == dict:
                            x = {}
                        elif special_types[col] == tuple:
                            x = tuple([])

                        return

                    if isinstance(x, (torch.Size,)):
                        return tuple(x)

                    if x not in cache:
                        cache[x] = ast.literal_eval(x)
                    return cache[x]
                except (ValueError,SyntaxError) as exc:
                    cache[x] = x
                    return x

            return column_data.apply(safe_eval), col

        special_types = self.special_types
        if len(special_types) == 0 or len(df) == 0:
            return df

        special_columns = [i for i in special_types.keys()]
        if not (set(special_columns) & set(df.columns.values)):
            return df
        else:
            future_results = [self.thread_executor.submit(parallel_eval, df[col], col, special_types) for col in special_columns if col in df]
            for future in as_completed(future_results):
                col_data, col_name = future.result()
                df[col_name] = col_data

            return df

    def append(self, df, blocking=True):
        if not blocking:
            self.thread_executor.submit(self._sync_append, df)
            return
        self._sync_append(df)

    def _sync_append(self, df):
        try:
            df['chunking_index'] = df.core_index.values.astype(np.int32)
            self._store_data(df, append=True)
        except Exception as exc:
            self.logger.exception('Error in write append')
            raise

    def _fetch_chunk(self, chunk_idx, df_complex = None):

        if df_complex is None:
            df_complex = self.fetch_raw_chunk(int(chunk_idx))

        conn = self.connection

        # Step 1 & 2: Create a temporary table and insert chunking indices
        df_complex['chunking_index'] = df_complex['chunking_index'].astype(np.int32)
        start = time.time()
        df_complex[['chunking_index']].to_sql('tmp_chunking_indices', conn, if_exists='replace', index=False)
        conn.execute(f"CREATE INDEX chunking_index_idx_tmp_chunking_indices ON tmp_chunking_indices (chunking_index)")
        # Step 3: Perform a join to fetch the relevant rows
        query = """
        SELECT data.*
        FROM data
        INNER JOIN tmp_chunking_indices ON data.chunking_index = tmp_chunking_indices.chunking_index
        """
        end = time.time()
        self.logger.info(f'Joining tmp_chunking_indices with data took: {(end - start) * 1000} ms')

        df_simple = pd.read_sql(query, conn)
        df_simple = self._restore_types(df_simple)

        # Step 4: Drop the temporary table
        with self.append_lock:
            conn.execute("DROP TABLE tmp_chunking_indices")

        _, complex_columns, _ = SQLDataFrameWrapper.identify_column_types(df_complex)
        if 'index' in df_simple:
            del df_simple['index']

        if 'index' in df_complex:
            del df_complex['index']

        df_combined = df_simple.merge(df_complex[complex_columns], on=["chunk_id", "chunking_index"], suffixes=['', '_conflict'], )

        for c in df_combined.columns.values:
            if len(df_combined[c].values.shape) > 1:
                vals = df_combined[c].values
                del df_combined[c]
                df_combined[c] = vals[:, 0]

        return df_combined

    def drop_duplicates(self, subset = [], fetch_complex = True):
        subset = ",".join(subset)
        columns = self.simple_columns
        if fetch_complex:
            columns = columns + self.complex_columns
            columns = sorted(list(set(columns)))

        columns = ",".join(columns)

        query = f"""SELECT {columns}
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER(PARTITION BY {subset} ORDER BY chunking_index) AS rn
                    FROM data
                ) AS subquery
                WHERE rn = 1;
                """
        return self._inner_query(query, fetch_complex)

    def apply(self, func, axis=0, **kwargs):
        # Iterate over chunks, apply function, and then store results
        results = []
        for chunk_idx, chunk_file in self.chunk_files.items():
            df_chunk = self._fetch_chunk(chunk_idx)
            if df_chunk.empty:
                continue

            result_chunk = df_chunk.apply(func, axis = axis, **kwargs )
            results.append(result_chunk)

        result = pd.concat(results)
        return result

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        """
        Expose the same interface as pandas for merging, but operate on each chunk.
        """
        merged_chunks = []

        # Iterating over each chunk
        for df_chunk in self.chunked_dataframes:
            # Merging the current chunk with the given DataFrame
            merged_chunk = df_chunk.merge(right, how=how, on=on, left_on=left_on, right_on=right_on,
                                          left_index=left_index, right_index=right_index, sort=sort,
                                          suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)
            # Update the chunk data
            self._update_chunk(merged_chunk)

        return self

    def fetch_all_for_df(self, simple_df):
        if len(simple_df) == 0:
            return simple_df

        # Sort input simple_df by chunking_index for efficient chunk retrieval
        sorted_simple_df = simple_df.sort_values(by="chunking_index")

        # Fetch the required chunks from the pickled files and construct the complex data portion
        required_rows = []
        cached_chunk_idx = -1  # Initialize with a non-existent chunk index
        cached_chunk = None  # To store the currently loaded chunk

        for chunk_id in sorted_simple_df.chunk_id.unique():
            # If current row's chunk isn't cached, load and update cache
            cached_chunk = self.fetch_raw_chunk(chunk_id)

            if cached_chunk is None:
                continue

            # Use cached_chunk to extract the necessary rows
            existing_complex_columns = set(cached_chunk.columns.values)
            missing = set(self.complex_columns) - existing_complex_columns
            if len(missing) > 0:
                self.logger.warning(f'Cached chunk {self.chunk_files[chunk_id]} has missing columns: {list(missing)}')
                for m in missing:
                    cached_chunk[m] = ''

            cached_chunk = cached_chunk[self.complex_columns]
            required_rows.append(cached_chunk)

        if len(required_rows) > 0:
            result_df = pd.concat(required_rows).drop_duplicates(subset=['chunking_index'])
            result_df = result_df.merge(simple_df[self.simple_columns], on = ['chunking_index', 'chunk_id'])
        else:
            result_df = simple_df[self.simple_columns]

        return result_df

    def _preload_all_chunks(self):
        futures = []
        with ThreadPoolExecutor(8) as executor:
            with tqdm.tqdm(total = len(self.chunk_files), desc = 'Submitting prefetching for all files') as pbar:
                for indx, file_name in self.chunk_files.items():
                    pbar.update(1)
                    future = executor.submit(self.fetch_raw_chunk, None, file_name)
                    futures.append(future)

        with tqdm.tqdm(desc='pre-fetching full chunks to merge with query PRELOAD ALL') as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

    @property
    def index(self):
        with self.append_lock:
            conn = self.connection
            result = conn.execute("SELECT chunking_index FROM data order by chunking_index asc").fetchall()
            result = [i[0] for i in result]
            return pd.Series(list(result))

    def _update_chunk(self, df_chunk):
        # Determine the chunk file based on the 'chunking_index' of the dataframe
        simple_columns, complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df_chunk)
        # Split the columns into simple and complex
        simple_columns_df = df_chunk[simple_columns]

        complex_columns_df = None
        if len(complex_columns) > 0:
            complex_columns_df = df_chunk[complex_columns]

        conn = self.connection

        # Get existing columns in the SQLite database
        existing_columns = conn.execute("PRAGMA table_info(data)").fetchall()
        existing_column_names = [col[1] for col in existing_columns]

        first = True
        update_cols = [col for col in simple_columns_df.columns if col in existing_column_names]
        insert_cols = [col for col in simple_columns_df.columns if col not in existing_column_names]

        # Update or Add the simple columns in the SQLite DB
        for _, row in simple_columns_df.iterrows():
            # Determine columns that are being updated vs columns that are being added

            if update_cols:
                cols_str = ', '.join([f'"{col}" = ?' for col in update_cols])
                query = f'UPDATE data SET {cols_str} WHERE chunking_index = ?'
                values = [row[col] for col in update_cols] + [row.chunking_index]
                conn.execute(query, values)

            if insert_cols:
                for col in list(set(insert_cols)):
                    # Create new columns for the missing ones
                    if first:
                        conn.execute(f"ALTER TABLE data ADD COLUMN {col}")
                        first = False

                    # And then insert the data into these new columns
                    query = f'UPDATE data SET "{col}" = ? WHERE chunking_index = ?'
                    values = [row[col], row.chunking_index]
                    conn.execute(query, values)

        # Update the complex columns in the pickle chunk
        if complex_columns_df is not None:
            for chunk_id, chunk in complex_columns_df.groupby(['chunk_id'], group_keys = True, as_index = False):
                chunk_id = chunk_id[0]
                self._chunk_cache[self.chunk_files[chunk_id]] = chunk


    def commit(self):
        # wait for all the threads to finish
        self.thread_executor.shutdown(wait=True)
        self.thread_executor = ThreadPoolExecutor(8)

        self._chunk_cache.commit()

        #for key, connection in self._connection.items():
        #    connection.commit()
        #    connection.close()

        #self._.connection = {}

    def __setitem__(self, key, value):

        indexes = self.index.values

        if isinstance(value, pd.Series):
            value = value.values
        else:
            if not isinstance(value, (list, np.ndarray)):
                value = [value] * len(indexes)

        temp_df = pd.DataFrame({key: value, 'chunking_index' : indexes})
        temp_df['chunk_id'] = temp_df['chunking_index'].apply(lambda x: int(np.floor(x / self.chunk_size)))

        simple_columns, complex_columns, special_types = self.identify_column_types(temp_df)
        self._update_chunk(temp_df)
        self.logger.info('Finished setitem')


    @property
    def simple_columns(self):
        conn = self.connection

        # Get existing columns in the SQLite database
        existing_columns = conn.execute("PRAGMA table_info(data)").fetchall()
        existing_column_names = [col[1] for col in existing_columns]

        # Exclude 'chunking_index' and 'index'
        simple_columns = [col for col in existing_column_names if col not in ['index']]

        return simple_columns

    def __getattr__(self, attr):
        if attr in self.__dict__:
            super().__getattribute__(attr)

        elif attr in self.simple_columns:
            return self.__getitem__(attr)
        else:
            return False

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.simple_columns:
                # Fetch entire column from SQLite DB
                conn = self.connection
                queried_df = pd.read_sql(f"SELECT {item} FROM data", conn)
                return queried_df[item]
            else:
                # If it's a complex column, we'd have to iterate over the chunks to aggregate the data
                # (This could be resource-intensive for very large dataframes)
                all_rows = []
                for chunk_idx, chunk_file in self.chunk_files.items():
                    df_chunk = self._fetch_chunk(chunk_idx)
                    all_rows.append(df_chunk[item])
                return pd.concat(all_rows)

        elif isinstance(item, list):
            return self._loc_method(None, item)

    def fetch_raw_chunk(self, chunk_index = None, chunk_file = None):
        try:
            if chunk_file is None:
                if int(chunk_index) not in self.chunk_files:
                    self.logger.warn(f'BAD CHUNK: {chunk_index} and total chunk_files {len(self.chunk_files)}')
                    return None

                chunk_file = self.chunk_files[int(chunk_index)]

            if chunk_file not in self._chunk_cache:
                with open(chunk_file, "rb") as f:
                    cached_chunk = dill.load(f)

                had_fixes = False
                for c in cached_chunk.columns.values:
                    vals = cached_chunk[c].values
                    if len(vals.shape) > 1 and vals.shape[1] > 1:
                        vals = vals[:, 0]
                        del cached_chunk[c]
                        cached_chunk[c] = vals
                        had_fixes = True

                clean = cached_chunk.drop_duplicates(subset = ['chunking_index', 'chunk_id'])
                if len(clean) < len(cached_chunk):
                    self.logger.warning(f'WARNING: Chunk file contained duplicates! {len(clean)} vs {len(cached_chunk)})')

                self._chunk_cache[chunk_file] = clean

            return self._chunk_cache[chunk_file]
        except EOFError as exc:
            self.logger.error(f"BAD CHUNK FILE DETECTED , chunked index: {chunk_index}")
            return None

    def _inner_query(self, sql_query, fetch_complex = True):

        # Fetch the simple data portion from SQLite
        conn = self.connection

        start = time.time()
        queried_df = pd.read_sql(sql_query, conn)
        end = time.time()
        self.logger.info(f'Query for simple columns data of {len(queried_df)} took {(end - start) * 1000} ms')

        queried_df = self._restore_types(queried_df)
        # Sort queried DataFrame by chunking_index for efficient chunk retrieval

        # Fetch the required chunks from the pickled files and construct the complex data portion
        required_rows = []
        if not fetch_complex:
            result_df = queried_df
        else:
            if len(queried_df) > 0:
                done = 0
                with ThreadPoolExecutor(4) as executor:
                    results = [executor.submit(self.fetch_raw_chunk, int(chunk_idx)) for chunk_idx in list(queried_df['chunk_id'].unique())]
                    with tqdm.tqdm(desc = 'fetching full chunks to merge with query', total =  queried_df['chunk_id'].nunique()) as pbar:
                        for future in as_completed(results):
                            pbar.update(1)
                            future_result_df = future.result()
                            if future_result_df is not None and len(future_result_df) > 0:
                                future_result_df = future_result_df.drop_duplicates(subset=['chunking_index', 'chunk_id']).reset_index(drop = True)
                                required_rows.append(future_result_df)

            self.logger.info(f'Merging results in query')

            final_required_rows = []

            if len(required_rows) ==  0:
                result_df = self.iloc[0:1].head(0)
            else:
                result_df = pd.concat(required_rows, ignore_index = True).drop_duplicates(subset = ['chunking_index', 'chunk_id'])
                result_df = result_df.merge(queried_df, on = ['chunking_index', 'chunk_id'], how = 'inner', suffixes = ['_old', '_new'])

                rename = {}
                for c in result_df.columns.values:
                    if '_new' in c:
                        rename[c] = c.replace("_new", "")

                result_df = result_df.rename(columns=rename)
                result_df = result_df[[c for c in result_df.columns.values if "_old" not in c]]

            self.logger.info(f'Finished merging results in query')

        self.logger.info(f'Adding index to df')
        if 'chunking_index' in result_df:
            result_df = result_df.set_index(result_df.chunking_index.values)

        self.logger.info(f'Finished index to df')
        return result_df

    def query(self, query_str, start = None, stop = None, from_clause = "*", extra = "", with_complex = True):
        query_str = pandas_query_to_sqlite(query_str)
        if start is not None and stop is not None:
            return self._inner_query(f"SELECT {from_clause} FROM data WHERE {query_str}  {extra} LIMIT {stop} OFFSET {start}", with_complex)
        else:
            return self._inner_query(f"SELECT {from_clause} FROM data WHERE {query_str} {extra}", with_complex)

    @property
    def iloc(self):
        return self._iloc_indexer

    @property
    def loc(self):
        return self._loc_indexer

    def _iloc_method(self, row_idx, col_idx=None):
        """
        Function mimicking pandas 'iloc' functionality.

        Parameters:
        - row_idx: integer, list of integers, slice object, or boolean array.
        - col_idx: integer, list of integers, slice object, or boolean array (optional).
        """

        # Helper function to handle row selection
        def row_query(idx):
            if isinstance(idx, int):  # single row
                return f"ROWID = {idx + 1}"  # SQLite ROWID starts from 1
            elif isinstance(idx, list):  # list of rows
                rows = ', '.join(str(i + 1) for i in idx)
                return f"ROWID IN ({rows})"
            elif isinstance(idx, slice):  # slice of rows
                start, stop, step = idx.indices(len(self.chunk_files))
                if step != 1:
                    raise ValueError("Step in slice is not supported for iloc in this implementation.")
                return f"ROWID BETWEEN {start + 1} AND {stop + 1}"
            else:
                raise ValueError("Unsupported row index type for iloc.")

        # Helper function to handle column selection
        def column_selection(idx):
            conn = self.connection

            # Fetch column names from the database
            cursor = conn.execute('SELECT * FROM data LIMIT 1')
            col_names = [desc[0] for desc in cursor.description]

            if isinstance(idx, int):  # single column
                return [col_names[idx]]
            elif isinstance(idx, list):  # list of columns
                return [col_names[i] for i in idx]
            elif isinstance(idx, slice):  # slice of columns
                start, stop, step = idx.indices(len(col_names))
                if step != 1:
                    raise ValueError("Step in slice is not supported for iloc in this implementation.")
                return col_names[start:stop]
            else:
                raise ValueError("Unsupported column index type for iloc.")

        # Prepare SQL query based on row and column selection
        row_selection_query = row_query(row_idx)
        selected_columns = "*"
        if col_idx is not None:
            selected_columns = ", ".join(column_selection(col_idx))

        conn = self.connection

        simple_df = pd.read_sql(f"SELECT {selected_columns} FROM data WHERE {row_selection_query}", conn)

        simple_df = self._restore_types(simple_df)

        # Fetch the required chunks for the resulting DataFrame
        return self.fetch_all_for_df(simple_df)


def generate_dataframe(n_rows=20_000):
    """
    Generate a DataFrame with a given number of rows.
    Columns:
    - string_col: Random strings
    - dict_col: Random dictionaries
    - float_col: Random float values
    - int_col: Random integer values

    Parameters:
    - n_rows: Number of rows in the DataFrame (default is 1 million)

    Returns:
    - A Pandas DataFrame with the specified characteristics.
    """

    # Generate random strings
    string_col = np.random.choice(['A', 'B', 'C', 'D'], n_rows).tolist()

    # Generate random dictionaries
    dict_values = [{'key1': np.random.choice(['X', 'Y', 'Z']),
                    'key2': np.random.randint(0, 100)} for _ in range(n_rows)]

    # Generate random float values
    float_col = np.random.uniform(0, 1, n_rows)

    # Generate random integer values
    int_col = np.random.randint(0, 1000, n_rows)

    # Construct the DataFrame
    df = pd.DataFrame({
        'a' : string_col,
        'b': string_col,
        'd' : string_col,
        'string_col': string_col,
        'dict_col2': dict_values,
        'dict_col3': dict_values,
        'dict_col': dict_values,
        'float_col': float_col,
        'int_col': int_col,
        'list_col' : [['a','b', 'c']] * len(int_col)
    })


    return df

if __name__ == "__main__":
    # Example Usage
    import tqdm
    tqdm.tqdm.pandas()

    from pandarallel import pandarallel

    # Initialize pandarallel
    pandarallel.initialize(nb_workers=4, progress_bar=True, verbose=0)


    df = generate_dataframe()
    df = df.set_index(np.arange(0, len(df), 1))
    df['extra'] = 0
    df['core_index'] = df.index.values
    s = time.time()
    wrapper = SQLDataFrameWrapper(df)

    dfc = df.copy()
    dfc['core_index'] = np.arange(len(df), 2 * len(df), 1)

    wrapper.append(dfc)
    wrapper.append(df[wrapper.simple_columns])
    print(wrapper.query("b.str.contains('date')"))
    wrapper['new_column'] = wrapper.apply(lambda x: x['b'][0:1], axis = 1)
    wrapper.loc[10, 'new_column2'] = 0


    for d in wrapper.chunked_dataframes():
        print(len(d))

    print(wrapper['new_column'].values)

    def test(x):
        return pd.Series({'A' : len(x)})

    z = wrapper.groupby(['string_col'], group_keys = True, as_index = False).parallel_apply(test)
    with open("./pickle_test.pickle", "wb") as f:
        pickle.dump(wrapper, f)


    with open("./pickle_test.pickle", "rb") as f:
        wrapper = pickle.load(f)

    z = wrapper.loc[1]

    print(wrapper.iloc[0:10].columns.values)

    t = wrapper.to_pandas()
    t['new_column3'] = [{'a' : 5}] * len(t)
    print(wrapper['new_column'].values)

    start = time.time()
    z = wrapper.loc[[3,1000,983200]]
    end = time.time()
    print(f'Loc took {(end - start)} ms')

    print(wrapper.a.unique())
    print(wrapper.index.values)
    wrapper.update(t)
    print(wrapper['new_column3'].values)

    if True:
        vals = [1, 2, 3]
        wrapper.query("a in @vals")

    for tt in wrapper.chunked_dataframes():
        print(tt.new_column3.values)

    print(wrapper.query("b.str.contains('date')"))
    wrapper['new_column2'] = wrapper.apply(lambda x: x['a'][0:1], axis = 1)


    print(wrapper.to_pandas()['d'].values)
    wrapper.commit()
    e = time.time()

    print(f'TEST DURAITON: {(e- s) * 1000} ms')