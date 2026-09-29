from typer.testing import CliRunner

from faissight import __version__
from faissight.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"faissight {__version__}"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_info_ivf_pq(synthetic) -> None:
    result = runner.invoke(app, ["info", str(synthetic["ivf_pq"])])
    assert result.exit_code == 0, result.output
    for text in ("IVF_PQ", "IndexIVFPQ", "nlist", "PQ nbits", "L2", "2,000"):
        assert text in result.output


def test_info_pretransform(synthetic) -> None:
    result = runner.invoke(app, ["info", str(synthetic["pca_ivf_flat"])])
    assert result.exit_code == 0, result.output
    assert "PCAMatrix" in result.output
    assert "core index: 16" in result.output


def test_info_unsupported(binary_index_path) -> None:
    result = runner.invoke(app, ["info", str(binary_index_path)])
    assert result.exit_code == 0, result.output
    assert "UNSUPPORTED" in result.output
    assert "Binary indexes are not supported" in result.output


def test_info_missing_file(tmp_path) -> None:
    result = runner.invoke(app, ["info", str(tmp_path / "nope.index")])
    assert result.exit_code == 1
    assert "not found" in result.output
