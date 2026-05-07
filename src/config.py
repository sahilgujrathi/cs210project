from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "archive"
DATABASE_DIR = ROOT_DIR / "database"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

DEFAULT_DB_PATH = DATABASE_DIR / "retailrocket.db"
DEFAULT_LATENT_MODEL_PATH = MODELS_DIR / "latent_factors.npz"

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def ensure_project_dirs() -> None:
    DATABASE_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def raw_file(name: str) -> Path:
    return RAW_DATA_DIR / name
