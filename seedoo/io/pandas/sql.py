import sqlite3
import pandas as pd
import os
import pickle
import numpy as np
import re
import inspect
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


class SQLDataFrameWrapper:
    def __init__(self, df=None, db_name="database.db", append=False):
        self.db_name = db_name
        self.complex_columns = []

        if df is not None:
            # If append mode, fetch the max chunking_index from the DB and adjust the new df's chunking_index accordingly
            if append:
                with sqlite3.connect(self.db_name) as conn:
                    max_index = conn.execute("SELECT MAX(chunking_index) FROM data").fetchone()[0]
                if max_index is None:  # In case the table is empty
                    max_index = -1
                df['chunking_index'] = np.arange(max_index + 1, max_index + len(df) + 1, 1).astype(np.int32)
            else:
                df['chunking_index'] = np.arange(0, len(df), 1).astype(np.int32)

            for c in df.columns.values:
                if not isinstance(df[c].values[0], (str, float, int, np.float32, np.int32)):
                    if isinstance(df[c].values[0], np.int64):
                        df[c] = df[c].astype(np.int32)
                    elif isinstance(df[c].values[0], np.float64):
                        df[c] = df[c].astype(np.float32)
                    else:
                        self.complex_columns.append(c)

            self._store_data(df, append)

    def _store_data(self, df, append=False):
        self.simple_columns_df = df.drop(columns=self.complex_columns)
        self.complex_columns_df = df[self.complex_columns + ["chunking_index"]]

        # Save simple columns to SQLite
        with sqlite3.connect(self.db_name) as conn:
            if append:
                self.simple_columns_df.to_sql("data", conn, index=True, if_exists="append")
            else:
                self.simple_columns_df.to_sql("data", conn, index=True, if_exists="replace")
                for col in self.simple_columns_df.columns:
                    try:
                        conn.execute(f"CREATE INDEX idx_{col} ON data ({col})")
                    except sqlite3.OperationalError as exc:
                        if "already exists" in exc.args[0]:
                            pass
                        else:
                            raise

        # Save complex columns to pickled files
        self.chunk_size = max(1, len(df) // 10)  # Adjust the chunking by modifying the denominator
        os.makedirs("chunks", exist_ok=True)
        self.chunk_files = []

        if append:
            existing_chunks = len(
                [name for name in os.listdir("chunks") if os.path.isfile(os.path.join("chunks", name))])
        else:
            existing_chunks = 0

        for i, chunk in enumerate(np.array_split(self.complex_columns_df.sort_values(by=['chunking_index']),
                                                 len(df) // self.chunk_size)):
            filename = f"chunks/chunk_{existing_chunks + i}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(chunk, f)
            self.chunk_files.append(filename)

    def append(self, df):
        with sqlite3.connect(self.db_name) as conn:
            max_index = conn.execute("SELECT MAX(chunking_index) FROM data").fetchone()[0]
        if max_index is None:  # In case the table is empty
            max_index = -1
        df['chunking_index'] = np.arange(max_index + 1, max_index + len(df) + 1, 1).astype(np.int32)

        for c in df.columns.values:
            if not isinstance(df[c].values[0], (str, float, int, np.float32, np.int32)):
                if isinstance(df[c].values[0], np.float64):
                    df[c] = df[c].astype(np.int32)
                else:
                    df[c] = df[c].astype(np.float32)
                    self.complex_columns.append(c)

        self._store_data(df, append=True)

    def _fetch_chunk(self, chunk_idx):
        with open(self.chunk_files[chunk_idx], "rb") as f:
            df_complex = pickle.load(f)

        chunking_indices = df_complex["chunking_index"].tolist()

        with sqlite3.connect(self.db_name) as conn:
            # Using a parameterized query to fetch records based on the chunking indices
            query = "SELECT * FROM data WHERE \"index\" IN ({seq})".format(seq=','.join(['?'] * len(chunking_indices)))
            df_simple = pd.read_sql_query(query, conn, params=chunking_indices)

        df_combined = pd.merge(df_simple, df_complex, on="chunking_index")
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

            if apply_along_axis == 'apply':
                result_chunk = df_chunk.apply(func, **kwargs)
            else:  # applymap
                result_chunk = df_chunk.applymap(func)

            self._update_chunk(result_chunk)

    def _update_chunk(self, df_chunk):
        # Determine the chunk file based on the 'chunking_index' of the dataframe
        chunk_indices = df_chunk["chunking_index"].unique()
        chunk_idx = [i // self.chunk_size for i in
                     chunk_indices]  # Calculate which chunk it belongs to based on the chunk size
        if len(set(chunk_idx)) > 1:
            raise ValueError("The dataframe spans multiple chunks. It should belong to only one chunk.")
        chunk_idx = chunk_idx[0]

        # Split the columns into simple and complex
        simple_columns_df = df_chunk.drop(columns=self.complex_columns + ["chunking_index"])
        complex_columns_df = df_chunk[self.complex_columns + ["chunking_index"]]

        with sqlite3.connect(self.db_name) as conn:
            # Get existing columns in the SQLite database
            existing_columns = conn.execute("PRAGMA table_info(data)").fetchall()
            existing_column_names = [col[1] for col in existing_columns]

            # Update or Add the simple columns in the SQLite DB
            for _, row in simple_columns_df.iterrows():
                # Determine columns that are being updated vs columns that are being added
                update_cols = [col for col in simple_columns_df.columns if col in existing_column_names]
                insert_cols = [col for col in simple_columns_df.columns if col not in existing_column_names]

                if update_cols:
                    cols_str = ', '.join([f'"{col}" = ?' for col in update_cols])
                    query = f'UPDATE data SET {cols_str} WHERE chunking_index = ?'
                    values = [row[col] for col in update_cols] + [
                        row.name]  # row.name gives the index (chunking_index in this case)
                    conn.execute(query, values)

                if insert_cols:
                    cols_str = ', '.join(insert_cols)
                    placeholders = ', '.join(['?'] * len(insert_cols))
                    query = f'INSERT INTO data ({cols_str}) VALUES ({placeholders})'
                    values = [row[col] for col in insert_cols]
                    conn.execute(query, values)

        # Update the complex columns in the pickle chunk
        with open(self.chunk_files[chunk_idx], "wb") as f:
            pickle.dump(complex_columns_df, f)

    def __setitem__(self, key, value):
        for chunk_idx in range(len(self.chunk_files)):
            df_chunk = self._fetch_chunk(chunk_idx)
            if df_chunk.empty:
                continue
            df_chunk[key] = value if isinstance(value, (int, float, str)) else value[
                                                                               df_chunk.index.min():df_chunk.index.max() + 1]
            self._update_chunk(df_chunk)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.simple_columns_df.columns:
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


    def query(self, query_str):
        query_str = pandas_query_to_sqlite(query_str)

        # Fetch the simple data portion from SQLite
        with sqlite3.connect(self.db_name) as conn:
            queried_df = pd.read_sql_query(f"SELECT * FROM data WHERE {query_str}", conn)

        # Sort queried DataFrame by chunking_index for efficient chunk retrieval
        queried_df = queried_df.sort_values(by="chunking_index")

        # Fetch the required chunks from the pickled files and construct the complex data portion
        required_rows = []
        cached_chunk_idx = -1  # Initialize with a non-existent chunk index
        cached_chunk = None  # To store the currently loaded chunk

        for index, row in queried_df.iterrows():
            chunk_idx = min(row['index'] // self.chunk_size, len(self.chunk_files) - 1)

            # If current row's chunk isn't cached, load and update cache
            if chunk_idx != cached_chunk_idx:
                with open(self.chunk_files[chunk_idx], "rb") as f:
                    cached_chunk = pickle.load(f)
                    cached_chunk_idx = chunk_idx

            # Use cached_chunk to extract the necessary rows
            matched_df = cached_chunk[cached_chunk["chunking_index"] == row['chunking_index']]
            required_rows.append(matched_df)

        result_df = pd.concat(required_rows).drop_duplicates(subset=['chunking_index'])
        return result_df

# Example Usage
df = pd.DataFrame({
     'a': np.random.randint(0, 100, size = (1000,)),
     'b': np.random.choice(['apple', 'banana', 'cherry', 'date'], p = [0.25, 0.25, 0.49, 0.01], size = (1000,)),
     'c': [{'x': 10}] * 1000
 })
wrapper = SQLDataFrameWrapper(df)
print(wrapper.query("b.str.contains('date')"))
wrapper['new_column'] = wrapper['b'].apply(lambda x: x[0:3])
print(wrapper['new_column'].values)