import pandas as pd

def dataframe_to_latex_table(
    df: pd.DataFrame, 
    caption: str="", 
    label: str="",
    n_decimals: int=4,
    print_instead_of_return: bool=True
    ) -> str:
    """
    Converts a pandas DataFrame to a LaTeX table.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    caption (str): The caption for the table.
    label (str): The label for the table.
    n_decimals (int): The number of decimals to display in the table. % TODO CHANGE THIS CODE

    Returns:
    str: A string containing the LaTeX code for the table.
    """

    # Make sure all column headers are strings
    df.columns = df.columns.astype(str)
    # Make sure all index values are strings
    df.index = df.index.astype(str)

    # replace _ by \_ in columns and index
    return_string = df.pipe(lambda df: df.rename(columns=lambda x: x.replace("_", "\\_"))).pipe(lambda df: df.rename(index=lambda x: x.replace("_", "\\_"))).to_latex(
                        index=True,
                        caption=caption,
                        label=label,
                        float_format="%.{}f".format(n_decimals),
                    )

    if print_instead_of_return:
        print(
            return_string
        )
        return None
    else:
        return return_string

