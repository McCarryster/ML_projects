import logging
import os
from config import *

# Define the log directory and file path
log_dir = os.path.join(stuff_and_paths['training_results_dir'], 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'prints.log')

# Configure the logger
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

# Just for visuals
def log_separator(length=100, char='#'):
    logger.info(char * length)