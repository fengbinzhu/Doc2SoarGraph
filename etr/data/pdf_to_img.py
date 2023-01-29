import argparse
import json
from pathlib import Path

from pdf2image import convert_from_path
from tqdm import tqdm

from etr.model.sys_util import create_logger
logger = create_logger('PDF 2 Image')
parser = argparse.ArgumentParser("TAT Doc Data Parsing task.")
parser.add_argument("--data_dir", type=str, default="./dataset_tatdqa")
parser.add_argument("--doc_dir", type=str, default='tat_docs')
parser.add_argument("--mode", type=str, default='train')

args = parser.parse_args()

base_folder = Path(args.data_dir)
tat_doc_folder = base_folder / args.doc_dir / args.mode
tat_ds = base_folder / f"tat_dqa_dataset_{args.mode}.json"

save_path = tat_doc_folder
if not save_path.exists():
    save_path.mkdir()

tat_data_list = json.load(open(tat_ds, 'r'))

for one in tqdm(tat_data_list):
    uid = one['table']['uid']
    logger.info(f'processing {uid}')
    # table_page_id = one['table']['page']
    tat_pdf_path = tat_doc_folder / f'{uid}.pdf'

    logger.info(f'processing {tat_pdf_path}')

    images = convert_from_path(tat_pdf_path, size=(224, 224), strict=True)
    logger.info(images)
    for idx, img in enumerate(images):
        image = img.convert("RGB")
        image.save(save_path / f'{uid}_{idx+1}.png', 'PNG')