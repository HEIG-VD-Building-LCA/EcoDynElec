"""
Module: OFEN PDF parsing & updating
-----------------------------------
Purpose: extract tables from OFEN PDFs (pages 47/48/50 depending on the year),
reshape some columns, and aggregate them into a final DataFrame for use in EcoDynElec.
"""

import re

import pandas as pd
import tabula


def _extract_numbers(cell, page):
    """
    Extract numeric values from a cell string depending on the PDF page layout.

    Parameters
    ----------
    cell : Any
        Raw content of the PDF cell (often str, sometimes NaN).
    page : int
        Logical page number controlling the extraction strategy (47, 48, 50).

    Returns
    -------
    list

        - page 47: up to 4 values (Nuclear, Thermical, Wind, PV).
        - page 48: 2 values (Total, Conso_pompes_STEP).
        - page 50: 2 values (Conso_pompes_STEP, Prod_nette).
        - Fills with pd.NA when fewer values are found.
        - May return an empty list if nothing can be extracted.

    """
    if pd.isna(cell):
        return [pd.NA, pd.NA, pd.NA, pd.NA]

    # Normalize spaces and separators
    s = str(cell).replace("\u00A0", " ").replace("\u202F", " ").strip()
    s = re.sub(r"[;,\t]", " ", s)
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    nums = []

    if page == 47:
        # Expect 4 values (or reconstruct if split into 5+ tokens)
        if len(parts) == 4:
            nums = [pd.to_numeric(g, errors="coerce") for g in parts]
        elif len(parts) >= 5:
            # First value may be split across 2 tokens
            first = parts[0] + parts[1]
            rest = parts[2:5]
            nums = [pd.to_numeric(first, errors="coerce")] + \
                   [pd.to_numeric(x, errors="coerce") for x in rest]
        return nums

    elif page == 48:
        # Expect 2 values (Total, Conso_pompes_STEP)
        nums = []
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            first = parts[0] + parts[1]   # Handle "x xxx"
            nums.append(pd.to_numeric(first, errors="coerce"))
            nums.append(pd.to_numeric(parts[2], errors="coerce"))
        else:
            # fallback: extract first 2 numbers
            matches = re.findall(r'\d+', s)
            for t in matches[:2]:
                nums.append(pd.to_numeric(t, errors="coerce"))
        while len(nums) < 2:
            nums.append(pd.NA)
        return nums

    elif page == 50:
        # Expect 2 values (Conso_pompes_STEP, Prod_nette)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            nums.append(pd.to_numeric(parts[0], errors="coerce"))   # Conso_pompes_STEP
            prod = str(parts[1]) + str(parts[2])                    # Prod_nette = "x" + "xxx"
            nums.append(pd.to_numeric(prod, errors="coerce"))
        while len(nums) < 2:
            nums.append(pd.NA)
        return nums


def split_col(df: pd.DataFrame, page: int, names: list) -> pd.DataFrame:
    """
    Replace one source column in the DataFrame by multiple new columns,
    depending on the page and extraction logic.

    Parameters
    ----------
    df : pd.DataFrame
        Table extracted with tabula.read_pdf(...) for a given page.
    page : int
        Logical page number.
    names : list[str]
        Names of the new columns to insert.

    Returns
    -------
    pd.DataFrame
        Same DataFrame, with the source column replaced by `names`.
    """
    if page == 47:
        src_idx = 2
        arr = df.iloc[:, src_idx].apply(lambda cell: _extract_numbers(cell, page=page)).to_list()
        arr = [vals + [pd.NA] * (4 - len(vals)) for vals in arr]
        new = pd.DataFrame(arr, columns=names, index=df.index)

    elif page == 48:
        src_idx = 6
        arr = df.iloc[:, src_idx].apply(lambda cell: _extract_numbers(cell, page=page)).to_list()
        arr = [vals + [pd.NA] * (2 - len(vals)) for vals in arr]
        new = pd.DataFrame(arr, columns=names, index=df.index)

    elif page == 50:
        src_idx = 5
        arr = df.iloc[:, src_idx].apply(lambda cell: _extract_numbers(cell, page=page)).to_list()
        arr = [vals + [pd.NA] * (2 - len(vals)) for vals in arr]
        new = pd.DataFrame(arr, columns=names, index=df.index)

    # Reconstruct DataFrame with left | new | right
    left = df.iloc[:, :src_idx]
    right = df.iloc[:, src_idx+1:] if src_idx+1 <= df.shape[1]-1 else df.iloc[:, 0:0]
    fixed = pd.concat([left, new, right], axis=1)
    return fixed


def ofen_pdf_to_df(file, page):
    """
    Load a specific page from an OFEN PDF and return a pre-processed DataFrame.

    Parameters
    ----------
    file : str
        Path to the OFEN PDF (e.g. ".../2022.pdf", ".../2024.pdf").
    page : int
        Page number to extract (47, 48, 49, 50).

    Returns
    -------
    pd.DataFrame

        - Page 47: Nuclear/Thermical/Wind/PV columns extracted
        - Page 48: Total/Conso_pompes_STEP columns extracted
        - Page 50: Conso_pompes_STEP/Prod_nette columns extracted
        - Others: raw table

    """
    tables = tabula.read_pdf(file, pages=page, pandas_options={'header': None}, stream=True)

    if file.endswith('24.pdf'):
        if page == 47:
            df = pd.DataFrame(tables[0])
            if len(df) > 9:
                df = df.iloc[9:].reset_index(drop=True)  # skip header rows
                df = split_col(df, page=page, names=("Nuclear", "Thermical", "Wind", "PV"))
        if page == 48:
            df = pd.DataFrame(tables[1])
            df = split_col(df, page=page, names=("Total", "Conso_pompes_STEP"))
        return df

    else:
        if page == 50:
            df = pd.DataFrame(tables[0])
            df[7] = df[6]  # keep original logic: duplicate column 6 to 7
            df = split_col(df, page=page, names=("Conso_pompes_STEP", "Prod_nette"))
            return df
        else:
            return pd.DataFrame(tables[0])


def updating_ofen_data(file):
    """
    Full pipeline for an OFEN PDF to obtain SFOE_data dataframe fo:

    - Load expected pages (depending on year),
    - Apply column splits and cleanup,
    - Concatenate tables,
    - Rename and reorder final columns,
    - Add 'mois' column and sort by year.

    /!\ Only works for the 2024 update.

    Parameters
    ----------
    file : str
        Path to the OFEN PDF (e.g. '.../2024.pdf', '.../2022.pdf').

    Returns
    -------
    pd.DataFrame
        Final DataFrame with columns:
        ["annee","mois","Hydro","Nuclear","Thermical","Conso_pompes_STEP",
        "Prod_nette","Imports","Exports","Conso_CH","Pertes","Conso_Finale_CH"]
        Sorted by year and month.
    """
    print('Reading', file)

    if file.endswith('24.pdf'):
        page = [47, 48]

        # Page 47
        table_1a = ofen_pdf_to_df(file, page=page[0])
        table_1a.drop(table_1a.columns[[4, 5, 6, 14]], axis=1, inplace=True)
        table_1a.columns = range(table_1a.shape[1])

        # Page 48
        table_1b = ofen_pdf_to_df(file, page=page[1])
        table_1b.drop(table_1b.columns[[4, 5, 6, 9, 15]], axis=1, inplace=True)
        table_1b.columns = range(table_1b.shape[1])

        df = pd.concat([table_1a, table_1b])

    elif file.endswith('22.pdf'):
        page = [49, 50]
        table_1a, table_1b = ofen_pdf_to_df(file, page=page[0]), ofen_pdf_to_df(file, page=page[1])
        table_1a.columns = range(table_1a.shape[1])
        table_1b.columns = range(table_1b.shape[1])

        # Shift columns (year-specific logic)
        table_1b[8]  = table_1b[9]
        table_1b[9]  = table_1b[10]
        table_1b[10] = table_1b[11]
        table_1b[11] = table_1b[12]

        df = pd.concat([table_1a, table_1b])
        df.drop(df.columns[[4, 12, 13]], axis=1, inplace=True)

    # Final column names
    df.columns = ["annee", "Hydro", "Nuclear", "Thermical",
                  "Conso_pompes_STEP", "Prod_nette", "Imports", "Exports",
                  "Conso_CH", "Pertes", "Conso_Finale_CH"]

    # Normalize 'annee' to Int64
    year_series = (df['annee'].astype(str).str.extract(r'(\d{4})')[0])
    df['annee'] = pd.to_numeric(year_series, errors='coerce').astype('Int64')

    # Derive 'mois' (1..12) per year
    mois = (df.groupby(df['annee']).cumcount() % 12 + 1).astype('Int16')
    df.insert(loc=1, column='mois', value=mois)

    # Sort by year and reset index
    df_sorted = df.sort_values(by='annee', kind='mergesort').reset_index(drop=True)

    # Drop last 11 rows (incomplete year)
    df_sorted = df_sorted.iloc[:-11]

    return df_sorted