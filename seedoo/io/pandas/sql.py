import os
import traceback

import psutil
import collections
import multiprocessing
import uuid
is_postgress = True
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

try:
    import psycopg2
    from psycopg2 import sql
except ImportError as exc:
    is_postgress = False
    pass

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
import asyncio
import threading
import tqdm
import ast
import json
import time
from seedoo.io.pandas.utils import FileCache

from concurrent.futures import ThreadPoolExecutor, as_completed
pd.options.mode.chained_assignment = None

NUM_THREADS = 4

# Store a reference to the original `sqlite3.connect`.
#_original_sqlite3_connect = sqlite3.connect

def set_pragmas_for_conn(conn):
    """Set optimized pragmas for a sqlite3 connection."""

    cur = conn.cursor()
    cur.execute("PRAGMA page_size;")
    page_size = 32768
    desired_cache_size_bytes = 5 * (1024 ** 3)


    cur.execute("PRAGMA temp_store = MEMORY;")  # Use memory for temporary tables and indices
    cur.execute("PRAGMA foreign_keys=OFF;")  # Turn off foreign key constraints (if you're sure you don't need them)
    # Calculate the number of pages required for the desired cache size
    num_pages = desired_cache_size_bytes // page_size

    # Set the cache size
    cur.execute(f"PRAGMA cache_size={num_pages};")


    conn.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA journal_mode = MEMORY;")
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


def custom_read_sql(sql, con, index_col=None, chunk_size=500000):
    logger = logging.getLogger(__name__)
    # Create a cursor from the connection
    cur = con.cursor()

    cur.execute(sql)

    # Fetch the column names from the cursor description
    columns = [col[0] for col in cur.description]

    dfs = []  # List to hold data chunks

    # Fetch the first data chunk to infer data types
    first_chunk = cur.fetchmany(chunk_size)
    if not first_chunk:
        cur.close()
        return pd.DataFrame([], columns=columns)  # Return empty DataFrame if no data

    # Infer data types from the first chunk
    df_first_chunk = pd.DataFrame(first_chunk, columns=columns).infer_objects()
    dtypes = df_first_chunk.dtypes.astype(str).to_dict()  # Convert data types to string and store in a dictionary

    dfs.append(df_first_chunk)  # Append the first chunk to dfs list

    # Fetch remaining data chunks
    while True:

        data_chunk = cur.fetchmany(chunk_size)
        print(f'Read chunk of length {len(data_chunk)}')
        if not data_chunk:
            break
        dfs.append(pd.DataFrame(data_chunk, columns=columns))

    logger.info('Finished fetching rows, building result data frame')
    cur.close()

    # Concatenate all data chunks into a single DataFrame and set data types
    df = pd.concat(dfs, ignore_index=True).astype(dtypes)
    logger.info('Finished building result DataFrame')

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
    logger = logging.getLogger(__name__)
    engine = create_engine('postgresql+psycopg2://', creator=lambda: con, echo=False, isolation_level="AUTOCOMMIT")

    try:
        # If 'replace', use original pandas to_sql for the first few rows to set up table schema
        if if_exists == 'replace':
            # We do this because for some reason executemany (later in the code) has a bug that when vals is of length 1
            # it just inserts Null. I have no idea why, but this is a hack to overcome it.
            if len(df.columns.values) < 2:
                df['extra_spare_column'] = 'extra'

            start = time.time()
            logger.debug('Started insert with to_sql')

            if "index" in df:
                del df['index']

            original_to_sql(df.iloc[:1], name, engine, if_exists=if_exists, index=False, index_label=index_label,
                            dtype=dtype)

            con.commit()
            df = df.iloc[1:]
            if_exists = 'append'  # Switch to 'append' mode for the remaining rows
            end = time.time()
            logger.debug(f'Finished insert with to_sql, it took: {(end - start) * 1000} ms')


        # Create a connection and cursor
        if if_exists == 'append':
            cur = engine.raw_connection().cursor()
            # Calculate chunk size, if not provided
            chunksize = chunksize or 150000
            column_names = ", ".join([f'"{col}"' for col in df.columns])
            # Prepare the placeholders
            num_columns = len(df.columns)
            placeholders = ", ".join(["%s"] * num_columns)

            logger = logging.getLogger(__name__)

            logger.debug(f'Starting inserting for df length {len(df)} and num columns: {len(df.columns.unique())}')
            insert_sql = sql.SQL(f"INSERT INTO {name} ({column_names}) VALUES ({placeholders})")

            for start in range(0, len(df), chunksize):
                cur.execute("BEGIN TRANSACTION;")
                end = start + chunksize
                batch = df.iloc[start:end]
                # Extract column names from the DataFrame
                data = [tuple(row) for row in batch.values]

                # Execute the in sert with explicit column names
                s = time.time()
                cur.executemany(insert_sql, data)
                e = time.time()
                cur.execute("COMMIT;")

                logger.info(f'Finished inserting: {(e -s) * 1000} ms for {len(batch)}')
                logger.debug('Committing for batch')
            logger.debug('Done committing for batch')

            cur.close()
            con.commit()
    except Exception as exc:
        logger.exception('Error in to_sql')
        raise

# Monkey patch pandas DataFrame's to_sql with our custom version
#pd.DataFrame.to_sql = custom_to_sql
#origin_read_sql = pd.read_sql
#pd.DataFrame.read_sql = custom_read_sql
#pd.read_sql = custom_read_sql

def pandas_query_to_sqlite(query_str):
    # Access calling frame's local and global variables
    frame = inspect.stack()[1]
    calling_locals = frame[0].f_locals
    calling_globals = frame[0].f_globals

    # Convert `.str.contains("value")` to `LIKE "%value%"`
    query_str = re.sub(r"\.str\.contains\('(.*?)'\)", r" LIKE '%\1%'", query_str)
    query_str = re.sub(r'\.str\.contains\("(.*?)"\)', " LIKE '%\1%'", query_str)

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

    query_str = re.sub(r'([a-zA-Z_]+[A-Z_]*[a-zA-Z_]*)\s*(=|>|<|LIKE|IN)', r'"\1" \2', query_str)
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


def optimize_sqlite(conn, desired_cache_size_gb=10):
    # Convert desired cache size from GB to bytes
    desired_cache_size_bytes = desired_cache_size_gb * 1024 ** 3

    # Connect to the SQLite database
    cur = conn.cursor()
    cur.execute(f"PRAGMA page_size=32768;")
    # Get the page size

    page_size = 32768

    # Calculate the number of pages required for the desired cache size
    num_pages = desired_cache_size_bytes // page_size

    # Set the cache size
    cur.execute(f"PRAGMA cache_size={num_pages};")

    # Set other performance enhancing pragmas
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store = MEMORY;")  # Use memory for temporary tables and indices
    cur.execute("PRAGMA journal_mode = MEMORY;")

    # Set memory-mapped I/O size
    mmap_size = desired_cache_size_bytes  # Setting it to the same size as the cache for simplicity
    cur.execute(f"PRAGMA mmap_size={mmap_size};")
    cur.execute("PRAGMA temp_store = MEMORY;")  # Use memory for temporary tables and indices
    cur.execute("PRAGMA foreign_keys=OFF;")  # Turn off foreign key constraints (if you're sure you don't need them)

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

    def __init__(self, df=None, db_name="database.db", path = os.getcwd(), chunk_size = 1000):
        self.db_name = os.path.dirname(path)
        self.table_name = os.path.basename(self.db_name)
        #os.path.join(path, db_name)
        self.complex_columns = []
        self._chunk_cache = FileCache()
        self.eval_cache = {}
        self.path = path
        self._simple_columns = None
        self.chunk_size = chunk_size
        self.always_commit = False
        self._iloc_indexer = SQLDataFrameWrapper.IlocIndexer(self)
        self.append_lock = threading.Lock()
        self._loc_indexer = SQLDataFrameWrapper.LocIndexer(self)
        self.thread_executor = ThreadPoolExecutor(NUM_THREADS)  # Initializes a thread pool executor
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
        thread_id = 0 % 1

        if thread_id not in self._connection:
            #conn = psycopg2.connect(dbname=f'seedoo',
            #                                user='seedoo',
            #                                password='kDASAEspJEdHp7',
            #                                host='localhost'
            #                            )

            connection_string = f'postgresql+psycopg2://seedoo:kDASAEspJEdHp7@localhost/seedoo'
            conn = create_engine(connection_string, isolation_level="AUTOCOMMIT")
            self._connection[thread_id] = conn

            print(f'Total connections: {len(self._connection)}')

        return self._connection[thread_id]

    def __len__(self):
        cursor = self.connection.raw_connection().cursor()

        cursor.execute(f"select count(*) from {self.table_name}")
        return cursor.fetchone()[0]

    def __del__(self):
        self.connection.raw_connection().close()

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
            columns_query = ",".join([f'{self.table_name}."{c}"' for c in columns_query])
        else:
            columns_query = f"{self.table_name}.*"

        if row_labels is None:
            # Join the temp table with the main data table on chunking_index
            query = f"""
                SELECT {columns_query}
                FROM {self.table_name}
            """
            conn = self.connection
            subset_df = pd.read_sql(query, conn)


        temp_df = pd.DataFrame({'chunking_index': row_labels})

        # Insert row_labels into a temp table
        conn = self.connection.raw_connection()
        cursor = conn.cursor()
        temp_table_name = f"temp_table_{self.table_name}_{str(uuid.uuid1()).replace('-', '')}"
        temp_table_name = temp_table_name.replace("_", "")[0:60]
        with self.append_lock:

            temp_df.to_sql(f"{temp_table_name}", self.connection, if_exists="replace", index=False, method='multi')
            cursor.execute(f"CREATE INDEX IF NOT EXISTS chunking_index_idx_temp_table ON {temp_table_name} (chunking_index)")

            # Join the temp table with the main data table on chunking_index
            query = f"""
                SELECT {columns_query}
                FROM {self.table_name}
                JOIN {temp_table_name} ON {self.table_name}.chunking_index = {temp_table_name}.chunking_index
            """
            subset_df = pd.read_sql(query, self.connection)

            cursor.execute(f"DROP TABLE {temp_table_name}")
            cursor.close()
            conn.commit()

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
            logging.getLogger(__name__).debug(f'We have overlap of complex and simple columns: {overlap} removing the overlap from the complex columns')
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
            connection.raw_connection().commit()
            connection.raw_connection().close()

        self._connection = {}

        self.thread_executor.shutdown(wait = True)
        self.thread_executor = ThreadPoolExecutor(NUM_THREADS)
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

        self._simple_columns = None

        # Step 1: Identify simple and complex columns in the provided dataframe
        provided_simple_cols, provided_complex_cols, special_types = self.identify_column_types(df)
        self.special_types.update(special_types)
        existing_column_names = self.simple_columns

        # Step 2: Identify new columns compared to existing data

        update_cols = [col for col in df.columns if col in existing_column_names if
                       col not in ['chunking_index', 'chunk_id']]
        insert_cols = [col for col in df.columns if col not in existing_column_names]

        columns_to_use = list(set(self.simple_columns) - (set(provided_simple_cols) - set(['chunking_index', 'chunk_id'])))
        common_columns = set(provided_simple_cols) & set(existing_column_names)
        columns_to_use = list(set(columns_to_use) - set(insert_cols) - set(update_cols))

        coalesce_expressions = [f'COALESCE(b."{col}", a."{col}") AS "{col}"' for col in common_columns if col != 'chunking_index' and col != 'chunk_id' and col not in insert_cols]
        if len(coalesce_expressions) > 0:
            coalesce_expressions = ",".join(coalesce_expressions)
            extra = ", "
        else:
            coalesce_expressions = ''
            extra = ""

        if len(columns_to_use) > 0:
            columns_to_use = ", ".join([f'a."{col}" as "{col}"' for col in columns_to_use if
                                     col != 'chunking_index' and col != 'chunk_id'])

            columns_to_use =  f"{extra} {columns_to_use}"
            extra = ", "
        if len(insert_cols) > 0:
            insert_cols = ", ".join([f'b."{col}" as "{col}"' for col in provided_simple_cols if col != 'chunking_index' and col != 'chunk_id' and col in insert_cols])
            insert_cols = f"{extra} {insert_cols}"
        else:
            insert_cols = ''

        conn = self.connection
        # Step 3: Insert the new DataFrame into a temporary table
        self.logger.debug(f'Inserting df of length {len(df)} into temp table for update')
        temp_table_name = f'temp_new_{self.table_name}_{str(uuid.uuid1()).replace("-", "")}'
        temp_table_name = temp_table_name.replace("_", "")[0:60]
        with self.append_lock:
            cursor = conn.raw_connection().cursor()

            cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

            df[provided_simple_cols].to_sql(temp_table_name, conn, if_exists='replace', index=False, method='multi', chunksize = 100_000)

            self.logger.info(f'Creating index for temp table on bulk insert')
            # Step 6: Re-add indices to the new data table
            cursor.execute(f"CREATE INDEX chunking_index_idx_{temp_table_name} ON {temp_table_name} (chunking_index)")
            self.logger.info(f'Done creating index for temp table on bulk insert')

        if swap:
            with self.append_lock:
                cursor.execute(f"DROP TABLE {self.table_name}")
                cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO {self.table_name}")
                conn.raw_connection().commit()
        else:
            self.logger.info(f'Joining temp table with main table for update')
            # Step 4: Join the temp table with the main data table
            cursor.execute(f"""
                CREATE TABLE temp_combined_{self.table_name} AS
                SELECT a.chunking_index as chunking_index, a.chunk_id as chunk_id {extra} {coalesce_expressions} {columns_to_use} {insert_cols}
                FROM {self.table_name} AS a
                LEFT JOIN {temp_table_name} AS b
                ON a.chunking_index = b.chunking_index
            """)

            # Step 5: Swap tables
            self.logger.info(f'Dropping {self.table_name} and renaming the temp table')

            with self.append_lock:
                cursor.execute(f"DROP TABLE {self.table_name}")
                cursor.execute(f"ALTER TABLE  temp_combined_{self.table_name} RENAME TO {self.table_name}")
                SQLDataFrameWrapper._partition_and_index_table(self.table_name, conn.raw_connection(), self.simple_columns)
                conn.raw_connection().commit()

        self.logger.info('Creating indexes')
        remaining_complex_columns = set(provided_complex_cols) - set(["chunking_index", "chunk_id"])

        if remaining_complex_columns:
            self.logger.info('Handling complex columns')
            df = df.drop_duplicates(subset = ['chunking_index'])
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

        df = self._inner_query(f"select distinct {columns_list} from {self.table_name}", fetch_complex=False)

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
                df = self.query(query, with_complex=False)
                if group_keys:
                    yield tuple([row[column] for column in by]), df
                else:
                    yield index, df
        return DataFrameGroupByWrapper(groupby_gen())

    @property
    def simple_df(self):
        conn = self.connection
        query = f"SELECT {cols} FROM {self.table_name}".format(cols=", ".join(self.simple_columns))
        df = pd.read_sql(query, conn)
        return df

    def to_pandas(self):
        """
        :return: Only the simple columns table as a simple pandas dataframe
        """
        conn = self.connection

        # Using a parameterized query to fetch records based on the chunking indices
        df_simple = pd.read_sql(f"SELECT * FROM {self.table_name}", conn)

        df_simple = self._restore_types(df_simple)
        return df_simple.set_index(df_simple.chunking_index.values)

    def update_chunked_dataframes(self, df):
        """
        Iterates through each chunk, updates with new data from the DataFrame and saves it back.

        Parameters:
        - df: DataFrame with new data.
        """
        self.thread_executor.shutdown(wait = True)
        self.thread_executor = ThreadPoolExecutor(NUM_THREADS)

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
        self.db_name = os.path.dirname(self.path)
        self.table_name = os.path.basename(self.db_name)
        self._cached_chunk_files = None
        self.special_types = state['special_types']
        self._last_modified_time = None
        self.thread_executor = ThreadPoolExecutor(NUM_THREADS)
        self._chunk_cache = FileCache()
        self.always_commit = state.get('always_commit', False)
        self._loc_indexer = SQLDataFrameWrapper.LocIndexer(self)
        self._connection = {}
        self.append_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.chunk_size = state['chunk_size']
        self._simple_columns = None
        self.eval_cache = {}

        self._iloc_indexer = SQLDataFrameWrapper.IlocIndexer(self)
        self._update_complex_columns()

    def _store_data(self, df, append=False):
        try:
            df['chunk_id'] = df['chunking_index'].apply(lambda x: int(np.floor(x / self.chunk_size)))
            _simple_columns, complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df)
            self.special_types.update(special_types)
            simple_columns_df = df[_simple_columns]

            with self.append_lock:
                # Save simple columns to SQLite
                conn = self.connection
                if append:
                    simple_columns = list(set(simple_columns_df.columns.values) & set(self.simple_columns))
                    if 'chunk_id' not in simple_columns:
                        simple_columns.append('chunk_id')

                    simple_columns_df = simple_columns_df[simple_columns]
                    simple_columns_df.to_sql(f"{self.table_name}", conn, index=False, if_exists="append", method='multi', chunksize = 1_000_000)
                else:
                    simple_columns_df.to_sql(f"{self.table_name}", conn, index=False, if_exists="replace")
                    #self.setup_partitioning()

            with self.append_lock:
                for chunk_id, chunk in df.groupby(['chunk_id'], group_keys = True, as_index = False):
                    chunk_id = chunk_id[0]
                    filename = f"{os.path.join(self.path, 'chunks')}/chunk_{chunk_id}.pkl"
                    new_chunk = chunk[complex_columns]
                    if not os.path.exists(os.path.dirname(filename)):
                        os.makedirs(os.path.dirname(filename))

                    if os.path.exists(filename):
                        existing_chunk = pd.read_pickle(filename)
                        if append:
                            new_chunk = pd.concat([existing_chunk, new_chunk])
                            new_chunk = new_chunk.drop_duplicates(subset = ['chunk_id', 'chunking_index'])

                        #if not append:
                        #    raise RuntimeError(f'Trying to append a chunk to an already existing one! {filename}')

                    new_chunk.to_pickle(filename)

        except Exception as exc:
            self.logger.exception("error in store data")


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
        temp_table_name = f"tmp_chunking_indices_{self.table_name}_{str(uuid.uuid1()).replace('-', '')}"
        temp_table_name = temp_table_name.replace("_", "")
        temp_table_name = temp_table_name[0:60]
        df_complex[['chunking_index']].to_sql(f'{temp_table_name}', conn, if_exists='replace', index=False)
        cur = conn.raw_connection().cursor()
        cur.execute(f"CREATE INDEX chunking_index_idx_{temp_table_name} ON {temp_table_name} (chunking_index)")
        cur.close()
        conn.raw_connection().commit()

        # Step 3: Perform a join to fetch the relevant rows
        query = f"""
        SELECT {self.table_name}.*
        FROM {self.table_name}
        INNER JOIN {temp_table_name} ON {self.table_name}.chunking_index = {temp_table_name}.chunking_index
        """
        end = time.time()
        self.logger.info(f'Joining tmp_chunking_indices with data took: {(end - start) * 1000} ms')

        df_simple = pd.read_sql(query, conn)
        df_simple = self._restore_types(df_simple)

        # Step 4: Drop the temporary table
        with self.append_lock:
            cur = conn.raw_connection().cursor()
            cur.execute(f"DROP TABLE {temp_table_name}")
            cur.close()
            conn.raw_connection().commit()

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
                    FROM {self.table_name}
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

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None, only_simple = True):
        """
        Expose the same interface as pandas for merging, but operate on each chunk.
        """
        merged_chunks = []
        self._simple_columns = None
        # Iterating over each chunk
        if not only_simple:
            for df_chunk in self.chunked_dataframes():
                # Merging the current chunk with the given DataFrame
                merged_chunk = df_chunk.merge(right, how=how, on=on, left_on=left_on, right_on=right_on,
                                              left_index=left_index, right_index=right_index, sort=sort,
                                              suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)
                # Update the chunk data
                self.update(merged_chunk)

        else:
            self.update(right)

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
            matching_cols = list(set(self.simple_columns) & set(simple_df.columns.values))
            result_df = pd.concat(required_rows).drop_duplicates(subset=['chunking_index'])
            result_df = result_df.merge(simple_df[matching_cols], on = ['chunking_index', 'chunk_id'])
        else:
            result_df = simple_df[self.simple_columns]

        return result_df

    def _preload_all_chunks(self):
        futures = []
        with ThreadPoolExecutor(NUM_THREADS) as executor:
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
            cursor = conn.raw_connection().cursor()
            cursor.execute(f"SELECT chunking_index FROM {self.table_name} order by chunking_index asc")
            result = cursor.fetchall()
            cursor.close()
            result = [i[0] for i in result]
            return pd.Series(list(result))



    def setup_partitioning(self):
        try:
            self.logger.info("Setting up partitioning")
            table_name = self.table_name
            chunk_size = self.chunk_size
            engine = self.connection
            # Generate the queries to create the partitions
            queries = [
                f"""
                CREATE TABLE {table_name}_chunk{i} PARTITION OF {table_name}
                FOR VALUES FROM ({i}) TO ({i + 1});
                """
                for i in range(0, 5000 + 1)
            ]
            # Execute the queries
            with engine.connect() as connection:
                for query in queries:
                    connection.execute(text(query))
            self.logger.info("Finished Setting up partitioning")
        except Exception as exc:
            self.logger.exception('Error in create partitions')



    @classmethod
    def _partition_and_index_table(cls, table_name, conn, simple_columns):
        # First, alter the table to declare it as partitioned
        # Create indexes
        cursor = conn.cursor()
        for col_name in ['chunking_index', 'chunk_id', 'core_index', 'file_name', 'type', 'cluster', 'cluster_global', 'id', 'cluster_id_size', 'cluster_id_size_global']:
            if col_name in simple_columns:
                index_name = f"{table_name}_{col_name}_idx"
                create_index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({col_name});"
                cursor.execute(create_index_query)


    def commit(self):
        # wait for all the threads to finish
        self._chunk_cache.commit()
        self.logger.info('Comitting and closing thread pool , but first waiting for it to finish')
        self.thread_executor.shutdown(wait=True)
        self.logger.info('Commited thread pool, it finished')
        self.thread_executor = ThreadPoolExecutor(NUM_THREADS)
        conn = self.connection.raw_connection()
        SQLDataFrameWrapper._partition_and_index_table(self.table_name, conn, self.simple_columns)

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
        self.update(temp_df)
        self.logger.info('Finished setitem')


    @property
    def simple_columns(self):
        if self._simple_columns is None:
            cursor = self.connection.raw_connection().cursor()

            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = '{self.table_name}'
            """)
            existing_column_names =  [row[0] for row in cursor.fetchall()]

            # Exclude 'chunking_index' and 'index'
            simple_columns = [col for col in existing_column_names if col not in ['index']]
            if 'chunk_id' not in simple_columns:
                simple_columns.append('chunk_id')

            if 'chunking_index' not in simple_columns:
                simple_columns.append('chunking_index')

            self._simple_columns = simple_columns

        return self._simple_columns

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
                queried_df = pd.read_sql(f"SELECT {item} FROM {self.table_name}", conn)
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
                cached_chunk = pd.read_pickle(chunk_file)

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
        except (EOFError, pickle.UnpicklingError):
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
                with ThreadPoolExecutor(2) as executor:
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
            return self._inner_query(f"SELECT {from_clause} FROM {self.table_name} WHERE {query_str}  {extra} LIMIT {stop} OFFSET {start}", with_complex)
        else:
            return self._inner_query(f"SELECT {from_clause} FROM {self.table_name} WHERE {query_str} {extra}", with_complex)

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
                return f"ctid = '(0, {idx})'"  # SQLite ROWID starts from 1
            elif isinstance(idx, list):  # list of rows
                idx_str = [f"'(0, {i})'" for i in idx]
                rows = ', '.join(idx_str)
                return f"ctid IN ({rows})"
            elif isinstance(idx, slice):  # slice of rows
                start, stop, step = idx.indices(len(self.chunk_files))
                if step != 1:
                    raise ValueError("Step in slice is not supported for iloc in this implementation.")
                return f"ROWID BETWEEN {start + 1} AND {stop + 1}"
            else:
                raise ValueError("Unsupported row index type for iloc.")

        # Helper function to handle column selection
        def column_selection(self, idx):
            conn = self.connection
            cursor = conn.raw_connection().cursor()

            # Fetch column names from the database
            cursor.execute(f'SELECT * FROM {self.table_name} LIMIT 1')
            col_names = [desc[0] for desc in cursor.description]

            cursor.close()  # Close the cursor

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

        simple_df = pd.read_sql(f"SELECT {selected_columns} FROM {self.table_name} WHERE {row_selection_query}", conn)

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
    with open("/seedoodata/demo/osint_v10/faiss/index_faiss/df.pkl", "rb") as f:
        wrapper = pickle.load(f)

    wrapper.commit()

    s = time.time()
    z = wrapper.loc[np.arange(0, 50000, 1), ['type', 'id']]
    e = time.time()

    z['test1'] = 1
    total = len(wrapper)
    zz = pd.DataFrame({'chunking_index' : np.arange(0, total ,1), 'cluster' : -1})

    wrapper['cluster'] = wrapper['cluster'].fillna(-1)

    wrapper.merge(z, on = ['chunking_index'], how = 'left')
    print(f"{(e-s) * 1000} as ms")


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

    def test(x):
        return pd.Series({'A' : len(x)})

    wrapper.append(dfc)
    wrapper.append(df[wrapper.simple_columns])
    z = wrapper.groupby(['string_col'], group_keys = True, as_index = False).parallel_apply(test)

    #print(wrapper.query("b.str.contains('date')"))
    wrapper['new_column'] = wrapper.apply(lambda x: x['b'][0:1], axis = 1)
    wrapper.loc[10, 'new_column2'] = 0
    zz = wrapper.loc[[10, 11], ['string_col', 'd']]


    for d in wrapper.chunked_dataframes():
        print(len(d))

    print(wrapper['new_column'].values)



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