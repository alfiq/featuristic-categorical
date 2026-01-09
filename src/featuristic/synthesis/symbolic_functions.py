"""Functions to use in the symbolic regression"""

from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder


def safe_div(a, b) -> np.ndarray:
    """
    Perform safe division by avoiding division by zero.

    Parameters
    ----------
    a : float
        The numerator.

    b : float
        The denominator.

    Returns
    -------
    np.ndarray:
        The result of the division.
    """
    return np.select([b != 0], [a / b], default=a)


def negate(a) -> np.ndarray:
    """
    Negate the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The negated input.
    """
    return np.multiply(a, -1)


def square(a):
    """
    Square the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The squared input.
    """
    return np.multiply(a, a)


def cube(a):
    """
    Cube the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The cubed input.
    """
    return np.multiply(np.multiply(a, a), a)


def sin(a):
    """
    Compute the sine of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The sine of the input.
    """
    return np.sin(a)


def cos(a):
    """
    Compute the cosine of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The cosine of the input.
    """
    return np.cos(a)


def tan(a):
    """
    Compute the tangent of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The tangent of the input.
    """
    return np.tan(a)


def sqrt(a):
    """
    Compute the square root of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The square root of the input.
    """
    return np.sqrt(np.abs(a))


class SymbolicAdd:
    """
    The symbolic addition function.
    """
    name = "add"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = np.add
        self.arg_count = 2
        self.format_str = "({} + {})"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicSubtract:
    """
    The symbolic subtraction function.
    """
    name = "subtract"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = np.subtract
        self.arg_count = 2
        self.format_str = "({} - {})"
        self.name = "subtract"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicMultiply:
    """
    The symbolic multiplication function.
    """
    name = "mult"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = np.multiply
        self.arg_count = 2
        self.format_str = "({} * {})"
        self.name = "mult"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicDivide:
    """
    The symbolic division function. Note that is performs a safe addition
    by avoiding division by zero.
    """
    name = "div"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = safe_div
        self.arg_count = 2
        self.format_str = "({} / {})"
        self.name = "div"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicAbs:
    """
    The symbolic absolute value function.
    """
    name = "abs"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = np.abs
        self.arg_count = 1
        self.format_str = "abs({})"
        self.name = "abs"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicNegate:
    """
    The symbolic negate function. It works by multiplying the input by -1.
    """
    name = "negate"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = negate
        self.arg_count = 1
        self.format_str = "-({})"
        self.name = "negate"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicSin:
    """
    The symbolic sine function.
    """
    name = "sin"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = sin
        self.arg_count = 1
        self.format_str = "sin({})"
        self.name = "sin"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicCos:
    """
    The symbolic cosine function.
    """
    name = "cos"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = cos
        self.arg_count = 1
        self.format_str = "cos({})"
        self.name = "cos"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicTan:
    """
    The symbolic tangent function.
    """
    name = "tan"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = tan
        self.arg_count = 1
        self.format_str = "tan({})"
        self.name = "tan"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicSqrt:
    """
    The symbolic square root function.
    """
    name = "sqrt"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = sqrt
        self.arg_count = 1
        self.format_str = "sqrt({})"
        self.name = "sqrt"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicSquare:
    """
    The symbolic square function.
    """
    name = "square"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = square
        self.arg_count = 1
        self.format_str = "square({})"
        self.name = "square"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicCube:
    """
    The symbolic cube function.
    """
    name = "cube"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = cube
        self.arg_count = 1
        self.format_str = "cube({})"
        self.name = "cube"

    def __call__(self, *args):
        try:
            return self.func(*args)
        except Exception:
            return pd.Series([np.nan] * len(args[0]))

    def __str__(self):
        return self.format_str


class SymbolicAddConstant:
    """
    The symbolic addition function. It works by adding a random constant to the input
    between -1000 and 1000. Note that this function can be useful where their is an
    offset in the data. However, it can lead to overfitting.
    """
    name = "add_constant"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """

        self.random_constant = np.random.uniform(-1000, 1000)
        self.arg_count = 1
        self.format_str = f"add_constant({self.random_constant} + {{}})"
        self.name = "add_constant"

    def __call__(self, x):
        try:
            return self.random_constant + x
        except Exception:
            return pd.Series([np.nan] * len(x))

    def __str__(self):
        return self.format_str


class SymbolicMulConstant:
    """
    The symbolic multiplication function. It works by multiplying the input by a random
    constant between -1000 and 1000. Note that this function can be useful where their
    is an offset in the data. However, it can lead to overfitting.
    """
    name = "mul_constant"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """

        self.random_constant = np.random.uniform(-1000, 1000)
        self.arg_count = 1
        self.format_str = f"mul_constant({self.random_constant} * {{}})"
        self.name = "mul_constant"

    def __call__(self, x):
        try:
            return self.random_constant * x
        except Exception:
            return pd.Series([np.nan] * len(x))

    def __str__(self):
        return self.format_str

class SymbolicCatCodes:
    """
    The symbolic categorical codes function. This is a stateful transformation
    that learns the mapping from categories to codes during the first call.
    """
    name = "cat_codes"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """
        self.arg_count = 1
        self.format_str = "cat_codes({})"
        self.name = "cat_codes"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, x):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        # If not fitted, we learn the mapping from the data
        if not self.fitted_:
            # We handle categorical/object data. If numeric, we just return as is
            # or we could encode it too. For now let's be robust.
            unique_cats = x.dropna().unique()
            self.mapping_ = {cat: i for i, cat in enumerate(unique_cats)}
            self.fitted_ = True

        return x.map(self.mapping_).fillna(-1)

    def __str__(self):
        return self.format_str


class SymbolicFrequencyEncoding:
    """
    The symbolic frequency encoding function. This is a stateful transformation
    that learns the value frequencies during the first call.
    """
    name = "freq_encode"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """
        self.arg_count = 1
        self.format_str = "freq_encode({})"
        self.name = "freq_encode"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, x):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        if not self.fitted_:
            self.mapping_ = x.value_counts(normalize=True).to_dict()
            self.fitted_ = True

        # Use 0.0 for unknown categories in transform
        return x.map(self.mapping_).fillna(0.0)

    def __str__(self):
        return self.format_str


class SymbolicLength:
    """
    The symbolic length function for strings.
    """
    name = "len"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """
        self.arg_count = 1
        self.format_str = "len({})"
        self.name = "len"

    def __call__(self, x):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return x.astype(str).str.len()

    def __str__(self):
        return self.format_str


class SymbolicEquals:
    """
    The symbolic equals function that works on both numeric and categorical data.
    """
    name = "equals"

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """
        self.arg_count = 2
        self.format_str = "({} == {})"
        self.name = "equals"

    def __call__(self, a, b):
        return (a == b).astype(int)

    def __str__(self):
        return self.format_str


class SymbolicTargetEncoding:
    """
    The symbolic target encoding function. This is a stateful transformation
    that learns the average target value for each category.
    """
    name = "target_encode"
    requires_target = True

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """
        self.arg_count = 1
        self.format_str = "target_encode({})"
        self.name = "target_encode"
        # We force continuous target type because symbolic trees expect a single column output.
        # For classification problems, this will still compute a meaningful average if y is numeric.
        self.encoder_ = TargetEncoder(smooth="auto", target_type="continuous")
        self.fitted_ = False

    def __call__(self, x, y=None):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        X_2d = x.to_frame()

        if not self.fitted_ and y is not None:
            # We need to handle potential issues with y (e.g. NaNs)
            # but TargetEncoder usually handles it or we should have preprocessed.
            self.encoder_.fit(X_2d, y)
            self.fitted_ = True

        if not self.fitted_:
            # transform-only mode but not fitted
            return pd.Series([np.nan] * len(x))

        encoded = self.encoder_.transform(X_2d)
        if len(encoded.shape) > 1 and encoded.shape[1] > 0:
            return pd.Series(encoded[:, 0])
        return pd.Series(encoded.flatten())

    def __str__(self):
        return self.format_str


class SymbolicCombine:
    """
    Combines two categorical features into one joint category.
    """
    name = "combine"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "combine({}, {})"
        self.name = "combine"

    def __call__(self, a, b):
        return pd.Series(a).astype(str) + "_" + pd.Series(b).astype(str)

    def __str__(self):
        return self.format_str


class SymbolicPairFrequency:
    """
    Count encoding of pairs (feature crossing frequency).
    """
    name = "pair_freq"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "pair_freq({}, {})"
        self.name = "pair_freq"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, a, b):
        pair = pd.Series(a).astype(str) + "_" + pd.Series(b).astype(str)
        if not self.fitted_:
            # We learn the frequencies of the joint categories
            self.mapping_ = pair.value_counts(normalize=True).to_dict()
            self.fitted_ = True

        return pair.map(self.mapping_).fillna(0.0)

    def __str__(self):
        return self.format_str


class SymbolicPairTargetEncoding:
    """
    Target encoding of pairs of categories.
    """
    name = "pair_target_encode"
    requires_target = True

    def __init__(self):
        self.arg_count = 2
        self.format_str = "pair_target_encode({}, {})"
        self.name = "pair_target_encode"
        self.encoder_ = TargetEncoder(smooth="auto", target_type="continuous")
        self.fitted_ = False

    def __call__(self, a, b, y=None):
        pair = (pd.Series(a).astype(str) + "_" + pd.Series(b).astype(str)).to_frame()

        if not self.fitted_ and y is not None:
            self.encoder_.fit(pair, y)
            self.fitted_ = True

        if not self.fitted_:
            return pd.Series([np.nan] * len(a))

        encoded = self.encoder_.transform(pair)
        if len(encoded.shape) > 1 and encoded.shape[1] > 0:
            return pd.Series(encoded[:, 0])
        return pd.Series(encoded.flatten())

    def __str__(self):
        return self.format_str


class SymbolicHashingInteraction:
    """
    Hashes combinations of two categories. Returns a numeric hash.
    """
    name = "hash_interact"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "hash_interact({}, {})"
        self.name = "hash_interact"

    def __call__(self, a, b):
        pair = pd.Series(a).astype(str) + "_" + pd.Series(b).astype(str)
        # Using a stable hash from pandas
        return pd.Series(pd.util.hash_pandas_object(pair, index=False) % 10**8)

    def __str__(self):
        return self.format_str


class SymbolicGroupMean:
    """
    Compute the mean of the numerical feature grouped by categorical values.
    """
    name = "group_mean"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_mean({}, {})"
        self.name = "group_mean"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].mean().to_dict()
            self.fitted_ = True

        return cat_s.map(self.mapping_).fillna(np.nan)

    def __str__(self):
        return self.format_str


class SymbolicGroupSum:
    """
    Compute the sum of the numerical feature grouped by categorical values.
    """
    name = "group_sum"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_sum({}, {})"
        self.name = "group_sum"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].sum().to_dict()
            self.fitted_ = True

        return cat_s.map(self.mapping_).fillna(np.nan)

    def __str__(self):
        return self.format_str


class SymbolicGroupMax:
    """
    Compute the max of the numerical feature grouped by categorical values.
    """
    name = "group_max"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_max({}, {})"
        self.name = "group_max"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].max().to_dict()
            self.fitted_ = True

        return cat_s.map(self.mapping_).fillna(np.nan)

    def __str__(self):
        return self.format_str


class SymbolicGroupMin:
    """
    Compute the min of the numerical feature grouped by categorical values.
    """
    name = "group_min"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_min({}, {})"
        self.name = "group_min"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].min().to_dict()
            self.fitted_ = True

        return cat_s.map(self.mapping_).fillna(np.nan)

    def __str__(self):
        return self.format_str


class SymbolicGroupStd:
    """
    Compute the standard deviation of the numerical feature grouped by categorical values.
    """
    name = "group_std"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_std({}, {})"
        self.name = "group_std"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].std().to_dict()
            self.fitted_ = True

        return cat_s.map(self.mapping_).fillna(np.nan)

    def __str__(self):
        return self.format_str


class SymbolicGroupDifference:
    """
    Difference between numerical feature and its group mean.
    """
    name = "group_diff"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_diff({}, {})"
        self.name = "group_diff"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].mean().to_dict()
            self.fitted_ = True

        return num_s - cat_s.map(self.mapping_).fillna(np.nan)

    def __str__(self):
        return self.format_str


class SymbolicGroupRatio:
    """
    Ratio between numerical feature and its group mean.
    """
    name = "group_ratio"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "group_ratio({}, {})"
        self.name = "group_ratio"
        self.mapping_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            df = pd.DataFrame({"cat": cat_s, "num": num_s})
            self.mapping_ = df.groupby("cat")["num"].mean().to_dict()
            self.fitted_ = True

        means = cat_s.map(self.mapping_).fillna(np.nan)
        return safe_div(num_s, means)

    def __str__(self):
        return self.format_str


class SymbolicCatIndicatorMultiply:
    """
    Multiplies a specific category indicator by a numerical feature.
    """
    name = "cat_indicator_mult"

    def __init__(self):
        self.arg_count = 2
        self.format_str = "cat_indicator_mult({}, {})"
        self.name = "cat_indicator_mult"
        self.category_ = None
        self.fitted_ = False

    def __call__(self, cat, num):
        cat_s = pd.Series(cat).astype(str)
        num_s = pd.to_numeric(pd.Series(num), errors="coerce")

        if not self.fitted_:
            unique_cats = cat_s.dropna().unique()
            if len(unique_cats) > 0:
                self.category_ = np.random.choice(unique_cats)
            self.fitted_ = True

        indicator = (cat_s == self.category_).astype(int)
        return indicator * num_s

    def __str__(self):
        return self.format_str


operations = [
    SymbolicAdd,
    SymbolicSubtract,
    SymbolicMultiply,
    SymbolicDivide,
    # SymbolicSqrt,
    SymbolicSquare,
    SymbolicCube,
    SymbolicAbs,
    SymbolicNegate,
    SymbolicSin,
    SymbolicCos,
    SymbolicTan,
    SymbolicMulConstant,
    SymbolicAddConstant,
    SymbolicCatCodes,
    SymbolicFrequencyEncoding,
    SymbolicLength,
    SymbolicEquals,
    SymbolicTargetEncoding,
    SymbolicCombine,
    SymbolicPairFrequency,
    SymbolicPairTargetEncoding,
    SymbolicHashingInteraction,
    SymbolicGroupMean,
    SymbolicGroupSum,
    SymbolicGroupMax,
    SymbolicGroupMin,
    SymbolicGroupStd,
    SymbolicGroupDifference,
    SymbolicGroupRatio,
    SymbolicCatIndicatorMultiply,
]


class CustomSymbolicFunction:
    """
    The base class for creating custom symbolic functions.
    """

    def __init__(self, func: Callable, arg_count: int, name: str, format_str: str):
        """
        Initialize the CustomSymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function. Must be a valid python format string,
            for example, "({} + {})" or "abs({})" and needs to have the same number of
            placeholders as the number of arguments the function takes.
        """
        self.func = func
        self.arg_count = arg_count
        self.format_str = format_str
        self.name = name

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


def list_symbolic_functions() -> List[str]:
    """
    List all the available built-in symbolic functions.

    Returns
    -------
    list
        The list of built-in operations.
    """
    return [op.name for op in operations]


def list_functions() -> List[str]:
    """
    Alias for list_symbolic_functions.
    """
    return list_symbolic_functions()
