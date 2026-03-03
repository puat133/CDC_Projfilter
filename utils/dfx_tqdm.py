from tqdm import tqdm_notebook as tqdm
import diffrax as dfx

class WebTqdmProgressMeter(dfx.TqdmProgressMeter):
    """
    A custom progress meter for Diffrax that uses TQDM's notebook progress bar.

    This class extends the TqdmProgressMeter from Diffrax to provide a progress bar
    suitable for Jupyter notebooks using TQDM's tqdm_notebook.

    Methods
    -------
    _init_bar() -> tqdm.tqdm_notebook
        Initializes and returns a TQDM notebook progress bar.
    """

    @staticmethod
    def _init_bar() -> "tqdm.tqdm_notebook":  # pyright: ignore  # noqa: F821
        import tqdm  # pyright: ignore

        bar_format = (
            "{percentage:.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        return tqdm.tqdm_notebook(
            total=100,
            unit="%",
            bar_format=bar_format,
        )
