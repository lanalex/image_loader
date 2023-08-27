


def foo(*args, **kwargs):
    pass

def _restore_types(df, special_types):
    from seedoo.io.pandas.worker import parallel_eval

    if len(special_types) == 0 or len(df) == 0:
        return df

    special_columns = [i for i in special_types.keys()]
    if not (set(special_columns) & set(df.columns.values)):
        return df
    else:
        df_clean = df.drop(columns=special_columns, errors="ignore")
        df = df[special_columns + ["chunking_index"]]
        df = df.drop_duplicates(subset=['chunking_index'])

        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Parallelize the ast.literal_eval over columns

        with ThreadPoolExecutor(4) as executor:
            future_results = [executor.submit(parallel_eval, df[col], col) for col in special_columns]
            for future in as_completed(future_results):
                col_data, col_name = future.result()
                df[col_name] = col_data

        df = df.merge(df_clean, on=['chunking_index'])
        return df