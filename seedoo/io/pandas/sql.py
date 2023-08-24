import uuid

import seedoo.logger_helpers.logger_setup
import ast
import logging
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
import time
from concurrent.futures import ThreadPoolExecutor
pd.options.mode.chained_assignment = None

# Store a reference to the original `sqlite3.connect`.
_original_sqlite3_connect = sqlite3.connect

def set_pragmas_for_conn(conn):
    """Set optimized pragmas for a sqlite3 connection."""

    cur = conn.cursor()
    cur.execute("PRAGMA page_size;")
    page_size = cur.fetchone()[0]
    desired_cache_size_bytes = 500 * 1024 ** 2

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA wal_autocheckpoint = 5000;")

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
        df = df.iloc[1:]
        if_exists = 'append'  # Switch to 'append' mode for the remaining rows
        end = time.time()
        logger.info(f'Finished insert with to_sql, it took: {(end - start) * 1000} ms')


    # Create a connection and cursor
    if if_exists == 'append':
        with con:
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
                    con.commit()
                    logging.getLogger(__name__).info('Done committing for batch')
                    pbar.update(1)
            cur.close()

# Monkey patch pandas DataFrame's to_sql with our custom version
pd.DataFrame.to_sql = custom_to_sql


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

        for groupby_obj in self.groupby_generators:
            result_chunk = groupby_obj.parallel_apply(func, *args, **kwargs)
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


def optimize_sqlite(db_name, desired_cache_size_gb=1):
    # Convert desired cache size from GB to bytes
    desired_cache_size_bytes = desired_cache_size_gb * 1024 ** 3

    # Connect to the SQLite database
    with sqlite3.connect(db_name) as conn:
        cur = conn.cursor()

        # Get the page size
        cur.execute("PRAGMA page_size;")
        page_size = cur.fetchone()[0]

        # Calculate the number of pages required for the desired cache size
        num_pages = desired_cache_size_bytes // page_size

        # Set the cache size
        cur.execute(f"PRAGMA cache_size=-{num_pages};")

        # Set other performance enhancing pragmas
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        # Set memory-mapped I/O size
        mmap_size = desired_cache_size_bytes  # Setting it to the same size as the cache for simplicity
        cur.execute(f"PRAGMA mmap_size={mmap_size};")

        # Commit changes and close the connection
        conn.commit()



class SQLDataFrameWrapper:
    class LocIndexer:
        def __init__(self, wrapper):
            self.wrapper = wrapper

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
                if isinstance(row_labels, slice):
                    # Ensure the slice start and stop are valid labels in the DataFrame
                    # This check can be expanded further based on your specific implementation
                    assert row_labels.start in self.wrapper.chunking_index and row_labels.stop in self.wrapper.chunking_index
                return self.wrapper._loc_method(row_labels, col_labels)
            else:
                row_labels = idx
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

    def __init__(self, df=None, db_name="database.db", path = os.getcwd(), chunk_size = 100_000):
        self.db_name = os.path.join(path, db_name)
        self.complex_columns = []
        self._chunk_cache = {}
        self.path = path
        self.chunk_size = chunk_size
        self.always_commit = False
        self._iloc_indexer = SQLDataFrameWrapper.IlocIndexer(self)
        self.append_lock = threading.Lock()
        self._loc_indexer = SQLDataFrameWrapper.LocIndexer(self)
        self.thread_executor = ThreadPoolExecutor(8)  # Initializes a thread pool executor
        self.special_types = {}
        self.logger = logging.getLogger(__name__)
        optimize_sqlite(db_name)


        if df is not None:
            # If append mode, fetch the max chunking_index from the DB and adjust the new df's chunking_index accordingly
            df['chunking_index'] = df.core_index.values.astype(np.int32)

            self._simple_columns, self.complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df)
            self.special_types.update(special_types)
            self._store_data(df, append = False)


    def _loc_method(self, row_labels, col_labels=None):
        # Handle the logic for .loc indexer

        # Convert row_labels to a DataFrame for insertion to temp table
        temp_df = pd.DataFrame({'chunking_index': row_labels})

        # Insert row_labels into a temp table
        with sqlite3.connect(self.db_name) as conn:
            temp_df.to_sql("temp_table", conn, if_exists="replace", index=False, method='multi')

            # Join the temp table with the main data table on chunking_index
            query = """
                SELECT data.*
                FROM data
                JOIN temp_table ON data.chunking_index = temp_table.chunking_index
            """
            subset_df = pd.read_sql_query(query, conn)

        # Handle column selection if provided in col_labels
        if col_labels:
            subset_df = subset_df[col_labels]

        subset_df = self._restore_types(subset_df)

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
            # Check for numeric types and adjust data types if needed
            if isinstance(df[c].values[0], (str, float, int, np.float32, np.int32, np.float64, np.int64, tuple, bool, np.bool_)):
                simple_cols.append(c)
                if isinstance(df[c].values[0], (np.bool_)):
                    df[c] = df[c].apply(lambda x: bool(x))

                elif isinstance(df[c].values[0], (tuple,)):
                    df[c] = df[c].apply(lambda x: str(tuple([safely_convert(i) for i in x])) if not pd.isnull(x) else str((0,)))
                    special_types[c] = tuple
                elif isinstance(df[c].values[0], np.int64):
                    df[c] = df[c].astype(np.int32)
                elif isinstance(df[c].values[0], np.float64):
                    df[c] = df[c].astype(np.float32)
            else:
                complex_cols.append(c)

        return simple_cols, complex_cols, special_types

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
        return state

    def update(self, df):
        """
        Updates the SQLite table with new data from the provided DataFrame.

        Parameters:
        - df: DataFrame with new data.
        """

        if "index" in df:
            del df['index']


        # Step 1: Identify simple and complex columns in the provided dataframe
        provided_simple_cols, provided_complex_cols, special_types = self.identify_column_types(df)

        # Step 2: Identify new columns compared to existing data
        columns_to_use = list(set(self.simple_columns) - (set(provided_simple_cols) - set(['chunking_index'])))
        columns_to_use = ", ".join([f"a.{col} as {col}" for col in columns_to_use])
        columns_to_add_or_update = ", ".join([f"b.{col} as {col}" for col in provided_simple_cols if col != 'chunking_index'])

        with sqlite3.connect(self.db_name) as conn:
            # Step 3: Insert the new DataFrame into a temporary table
            self.logger.info(f'Inserting df of length {len(df)} into temp table for update')
            temp_table_name = f'temp_new_data_{str(uuid.uuid1()).replace("-", "")}'
            conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
            df[provided_simple_cols].to_sql(temp_table_name, conn, if_exists='replace', index=False, method='multi')

            self.logger.info(f'Creating indexes for columns')
            # Step 6: Re-add indices to the new data table
            conn.execute(f"CREATE INDEX chunking_index_idx_{temp_table_name} ON {temp_table_name} (chunking_index)")


            self.logger.info(f'Joining temp table with main table for update')
            # Step 4: Join the temp table with the main data table
            conn.execute(f"""
                CREATE TABLE temp_combined_data AS
                SELECT {columns_to_use},{columns_to_add_or_update}
                FROM data AS a
                RIGHT JOIN {temp_table_name} AS b
                ON a.chunking_index = b.chunking_index
            """)

            # Step 5: Swap tables
            self.logger.info('Dropping data and renaming the temp table')

            conn.execute("DROP TABLE data")
            conn.execute("ALTER TABLE temp_combined_data RENAME TO data")
            conn.execute(f"DROP TABLE {temp_table_name}")


            for col in provided_simple_cols:
                if col != 'index':
                    try:
                        conn.execute(f"CREATE INDEX idx_{col} ON data ({col})")
                        conn.commit()
                    except sqlite3.OperationalError as exc:
                        if "already exists" in exc.args[0]:
                            pass
                        else:
                            raise

            self.logger.info('Creating indexes')
        if provided_complex_cols:
            self.logger.info('Handling complex columns')
            self.update_chunked_dataframes(df[provided_complex_cols + ["chunking_index"]])


    def groupby(self, *args, **kwargs):
        # This will be a generator
        def groupby_gen():
            for df_chunk in self.chunked_dataframes():
                yield df_chunk.groupby(*args, **kwargs)

        return DataFrameGroupByWrapper(groupby_gen())

    @property
    def simple_df(self):
        with sqlite3.connect(self.db_name) as conn:
            query = "SELECT {cols} FROM data".format(cols=", ".join(self.simple_columns))
            df = pd.read_sql(query, conn)
        return df

    def to_pandas(self):
        """
        :return: Only the simple columns table as a simple pandas dataframe
        """
        with sqlite3.connect(self.db_name) as conn:
            # Using a parameterized query to fetch records based on the chunking indices
            df_simple = pd.read_sql_query( "SELECT * FROM data", conn)

        df_simple = self._restore_types(df_simple)
        return df_simple

    def update_chunked_dataframes(self, df):
        """
        Iterates through each chunk, updates with new data from the DataFrame and saves it back.

        Parameters:
        - df: DataFrame with new data.
        """

        # Ensure chunking_index is present in the DataFrame
        if "chunking_index" not in df.columns:
            raise ValueError("DataFrame must contain 'chunking_index' for updating.")

        with tqdm.tqdm(total=len(self.chunk_files), desc='Updating chunk files') as pbar:
            for i, chunk_file in enumerate(self.chunk_files):
                # Load the chunk using read_pickle with compression
                pbar.update(1)
                chunk_df = self.fetch_raw_chunk(chunk_file = chunk_file)

                # Update the chunk using merge on the 'chunking_index'
                updated_chunk = pd.merge(chunk_df, df, on="chunking_index", how="inner", suffixes=['_old', ''])

                # Handling columns updates
                for col in updated_chunk.columns:
                    if '_old' in col:
                        del updated_chunk[col]

                # Save the updated chunk back using to_pickle with compression
                self._chunk_cache[chunk_file] = updated_chunk
                if self.always_commit:
                    updated_chunk.to_pickle(chunk_file, compression='gzip')

            self.commit()


    def chunked_dataframes(self):
        """ Generator to produce dataframes from the stored chunks."""
        with tqdm.tqdm(total = len(self.chunk_files), desc = 'Reading chunk files') as pbar:
            for i, chunk_file in enumerate(self.chunk_files):
                chunk = self.fetch_raw_chunk(None, chunk_file=chunk_file)
                df_chunk = self._fetch_chunk(i, chunk)
                pbar.update(1)
                yield df_chunk

    def __setstate__(self, state):
        self.complex_columns = state['complex_columns']
        self.path = state['path']
        self.db_name = state['db_name']
        self.special_types = state['special_types']
        self._chunk_cache = {}
        self.always_commit = state['always_commit']
        self._loc_indexer = SQLDataFrameWrapper.LocIndexer(self)
        self.append_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.chunk_size = state['chunk_size']
        self._iloc_indexer = SQLDataFrameWrapper.IlocIndexer(self)

    def _store_data(self, df, append=False):
        _simple_columns, complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df)
        self.special_types.update(special_types)
        simple_columns_df = df.drop(columns=complex_columns)

        with self.append_lock:
            # Save simple columns to SQLite
            with sqlite3.connect(self.db_name) as conn:
                if append:
                    simple_columns_df = simple_columns_df[self.simple_columns]
                    simple_columns_df.to_sql("data", conn, index=True, if_exists="append")
                else:
                    simple_columns_df.to_sql("data", conn, index=True, if_exists="replace")
                    for col in self.simple_columns + ["chunking_index"]:
                        try:
                            conn.execute(f"CREATE INDEX idx_{col} ON data ({col})")
                        except sqlite3.OperationalError as exc:
                            if "already exists" in exc.args[0]:
                                pass
                            else:
                                raise

        num_chunks = max(1, len(df) // self.chunk_size)
        for i, chunk in enumerate(np.array_split(df[complex_columns+ ["chunking_index"]].sort_values(by=['chunking_index']),
                                                 num_chunks)):
            chunk_id = int(chunk.chunking_index.max() // self.chunk_size)
            filename = f"{os.path.join(self.path, 'chunks')}/chunk_{chunk_id}.pkl"
            new_chunk = chunk[complex_columns + ["chunking_index"]]
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            if os.path.exists(filename):
                existing_chunk = self.fetch_raw_chunk(chunk_file=filename)
                if append:
                    new_chunk = pd.concat([existing_chunk, new_chunk])

                #if not append:
                #    raise RuntimeError(f'Trying to append a chunk to an already existing one! {filename}')

            new_chunk.to_pickle(filename, compression='gzip')

    @property
    def chunk_files(self):
        chunks_dir = os.path.join(self.path, "chunks")

        if not hasattr(self, '_cached_chunk_files'):
            chunks_dir = os.path.join(self.path, "chunks")
            self._cached_chunk_files = None
            self._last_modified_time = None


        # Check if the folder's modification time has changed
        current_modified_time = os.path.getmtime(chunks_dir)
        if current_modified_time != self._last_modified_time or self._cached_chunk_files is None:
            self._cached_chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if
                                        os.path.isfile(os.path.join(chunks_dir, f))]
            self._last_modified_time = current_modified_time

        return self._cached_chunk_files

    def _restore_types(self, df):
        def save_eval(x):
            try:
                return ast.literal_eval(x)
            except Exception as exc:
                print(f'Failed to parse: {x}')
                return (0,0,0,0)


        for c,val in self.special_types.items():
            df[c] = df[c].progress_apply(lambda x: save_eval(x))

        return df

    def append(self, df, blocking=False):
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

        with sqlite3.connect(self.db_name) as conn:
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
            conn.execute("DROP TABLE tmp_chunking_indices")

        _, complex_columns, _ = SQLDataFrameWrapper.identify_column_types(df_complex)
        if 'index' in df_simple:
            del df_simple['index']

        if 'index' in df_complex:
            del df_complex['index']

        df_combined = pd.merge(df_simple, df_complex[complex_columns + ["chunking_index"]], on="chunking_index", suffixes=['', '_conflict'])
        return df_combined

    def apply(self, func, axis=0, **kwargs):
        # Check axis value
        if axis in (1, 'columns'):
            apply_along_axis = 'applymap'  # Apply to each cell in the DataFrame
        else:
            apply_along_axis = 'apply'  # Apply to each column (default behavior)

        # Iterate over chunks, apply function, and then store results
        for chunk_idx in range(len(self.chunk_files)):
            df_chunk = self._fetch_chunk(chunk_idx)
            if df_chunk.empty:
                continue

            result_chunk = df_chunk.parallel_apply(func, axis = axis, **kwargs )

            return result_chunk

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
        # Sort input simple_df by chunking_index for efficient chunk retrieval
        sorted_simple_df = simple_df.sort_values(by="chunking_index")

        # Fetch the required chunks from the pickled files and construct the complex data portion
        required_rows = []
        cached_chunk_idx = -1  # Initialize with a non-existent chunk index
        cached_chunk = None  # To store the currently loaded chunk

        for index, row in sorted_simple_df.iterrows():
            chunk_idx = max(int(row['chunking_index'] // self.chunk_size) - 1, 0)

            # If current row's chunk isn't cached, load and update cache
            if chunk_idx != cached_chunk_idx:
                cached_chunk = pd.read_pickle(self.chunk_files[chunk_idx], compression='gzip')
                cached_chunk_idx = chunk_idx

            # Use cached_chunk to extract the necessary rows
            matched_df = cached_chunk[cached_chunk["chunking_index"] == row['chunking_index']]
            required_rows.append(matched_df)

        result_df = pd.concat(required_rows).drop_duplicates(subset=['chunking_index'])
        result_df = result_df.merge(simple_df[self.simple_columns], on = ['chunking_index'])
        return result_df

    @property
    def index(self):
        with self.append_lock:
            with sqlite3.connect(self.db_name) as conn:
                result = conn.execute("SELECT distinct chunking_index FROM data order by chunking_index asc").fetchall()
            result = [i[0] for i in result]
            return pd.Series(list(result))

    def _update_chunk(self, df_chunk):
        # Determine the chunk file based on the 'chunking_index' of the dataframe
        chunk_indices = df_chunk["chunking_index"].max()
        chunk_idx = max(int(chunk_indices // self.chunk_size) - 1, 0)

        simple_columns, complex_columns, special_types = SQLDataFrameWrapper.identify_column_types(df_chunk)
        # Split the columns into simple and complex
        simple_columns_df = df_chunk.drop(columns=complex_columns)
        complex_columns_df = df_chunk[complex_columns + ['chunking_index']]

        with sqlite3.connect(self.db_name) as conn:
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
        self._chunk_cache[self.chunk_files[chunk_idx]] = complex_columns_df

        if self.always_commit:
            complex_columns_df.to_pickle(self.chunk_files[chunk_idx], compression = 'gzip')


    def commit(self):
        with tqdm.tqdm(total = len(self.chunk_files), desc = 'Committng in memory chunks') as pbar:
            for chunk_file in self.chunk_files:
                if chunk_file in self._chunk_cache:
                    complex_columns_df = self._chunk_cache[chunk_file]
                    complex_columns_df.to_pickle(chunk_file, compression='gzip')
                    pbar.update(1)

    def __setitem__(self, key, value):
        self.logger.info('Doing setitem')
        for chunk_idx in range(len(self.chunk_files)):
            df_chunk = self._fetch_chunk(chunk_idx)
            if df_chunk.empty:
                continue
            df_chunk[key] = value if isinstance(value, (int, float, str)) else value[
                                                                               df_chunk.index.min():df_chunk.index.max() + 1]

            self._update_chunk(df_chunk)

        self.logger.info('Finished setitem')
    def __len__(self):
        with sqlite3.connect(self.db_name) as conn:
            result = conn.execute("SELECT count(*) FROM data").fetchone()[0]

        return result

    @property
    def simple_columns(self):
        with sqlite3.connect(self.db_name) as conn:
            # Get existing columns in the SQLite database
            existing_columns = conn.execute("PRAGMA table_info(data)").fetchall()
            existing_column_names = [col[1] for col in existing_columns]

        # Exclude 'chunking_index' and 'index'
        return [col for col in existing_column_names if col not in ['index']]


    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.simple_columns:
                # Fetch entire column from SQLite DB
                with sqlite3.connect(self.db_name) as conn:
                    queried_df = pd.read_sql_query(f"SELECT {item} FROM data", conn)
                return queried_df[item]
            else:
                # If it's a complex column, we'd have to iterate over the chunks to aggregate the data
                # (This could be resource-intensive for very large dataframes)
                all_rows = []
                for chunk_idx in range(len(self.chunk_files)):
                    df_chunk = self._fetch_chunk(chunk_idx)
                    all_rows.append(df_chunk[item])
                return pd.concat(all_rows)


    def fetch_raw_chunk(self, chunk_index = None, chunk_file = None):
        if chunk_file is None:
            if chunk_index >= len(self.chunk_files):
                print(f'BAD CHUNK: {chunk_index} and chunk_files is: {self.chunk_files}')
            chunk_file = self.chunk_files[int(chunk_index)]

        if chunk_file not in self._chunk_cache:
            cached_chunk = pd.read_pickle(chunk_file, compression='gzip')
            self._chunk_cache[chunk_file] = cached_chunk

        return self._chunk_cache[chunk_file]

    def query(self, query_str):
        query_str = pandas_query_to_sqlite(query_str)

        # Fetch the simple data portion from SQLite
        with sqlite3.connect(self.db_name) as conn:
            start = time.time()
            queried_df = pd.read_sql_query(f"SELECT * FROM data WHERE {query_str}", conn)
            end = time.time()
            self.logger.info(f'Query for simple columns data of {len(queried_df)} took {(end - start) * 1000} ms')

        queried_df = self._restore_types(queried_df)
        # Sort queried DataFrame by chunking_index for efficient chunk retrieval

        # Fetch the required chunks from the pickled files and construct the complex data portion
        required_rows = []
        cached_chunk_idx = -1  # Initialize with a non-existent chunk index
        cached_chunk = None  # To store the currently loaded chunk
        indexes = [max(int(i // self.chunk_size) - 1, 0) for i in queried_df.chunking_index.unique()]

        handled = set([])
        with tqdm.tqdm(desc = 'fetching full chunks to merge with query') as pbar:
            for chunk_idx in indexes:
                # If current row's chunk isn't cached, load and update cache
                if chunk_idx not in handled:
                    cached_chunk = self.fetch_raw_chunk(int(chunk_idx))
                    handled.add(chunk_idx)
                    # Use cached_chunk to extract the necessary rows
                    required_rows.append(cached_chunk)
                pbar.update(1)

        self.logger.info(f'Merging results in query')

        if len(required_rows) ==  0:
            result_df = self.iloc[0:1].head(0)
        else:
            result_df = pd.concat(required_rows).drop_duplicates(subset=['chunking_index'])
            result_df = result_df.merge(queried_df, on = 'chunking_index', how = 'inner', suffixes = ['_old', ''])

            for c in result_df.columns.values:
                if '_old' in c:
                    del result_df[c]

        self.logger.info(f'Finished merging results in query')
        return result_df

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
            with sqlite3.connect(self.database_path) as conn:
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

        with sqlite3.connect(self.db_name) as conn:
            simple_df = pd.read_sql_query(f"SELECT {selected_columns} FROM data WHERE {row_selection_query}", conn)

        simple_df = self._restore_types(simple_df)

        # Fetch the required chunks for the resulting DataFrame
        return self.fetch_all_for_df(simple_df)


def generate_dataframe(n_rows=2000000):
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
        'dict_col': dict_values,
        'float_col': float_col,
        'int_col': int_col
    })

    return df

if __name__ == "__main__":
    # Example Usage
    from pandarallel import pandarallel

    # Initialize pandarallel
    pandarallel.initialize(nb_workers=6, progress_bar=True)


    df = generate_dataframe()
    df = df.set_index(np.arange(0, len(df), 1))
    df['extra'] = 0
    df['core_index'] = df.index.values
    wrapper = SQLDataFrameWrapper(df)
    print(wrapper.query("b.str.contains('date')"))
    wrapper['new_column'] = wrapper.apply(lambda x: x['b'][0:1], axis = 1)

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

    print(wrapper.iloc[0:10].columns.values)

    t = wrapper.to_pandas()
    t['new_column3'] = [{'a' : 5}] * len(t)
    print(wrapper['new_column'].values)



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
