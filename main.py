import copy
import logging
import os
import time
import uuid

import cv2
import numpy as np
from ocr.model import STROCR

from layout.model import CRAFTLayout
from utils import cut_polygon_deskew, pprint, save_json

logging.getLogger().setLevel(logging.INFO)


class SceneTextExtractor():
    def __init__(self,
                 layout_model_path='./models/layout/clova.zip',
                 ocr_model_path='./models/ocr/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth',
                 chars_set_path='./models/ocr/chars_set.txt',
                 debug=False):
        self.layout_model = CRAFTLayout(
            layout_model_path, debug=debug, canvas_size=576, text_threshold=0.1, link_threshold=0.33)
        self.ocr_model = STROCR(ocr_model_path, chars_set_path)
        if debug:
            logging.info('Debug mode is on')
        self.debug = debug
        self.debug_dir = './debugs/'
        logging.info("Loaded all models")

    def process(self, image, debug_id=None):
        debug_dir = None
        if self.debug:
            debug_id = uuid.uuid4() if debug_id is None else debug_id
            debug_dir = './debugs/{}/'.format(debug_id)
            logging.info('Log stored at {}'.format(debug_dir))
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, 'raw.jpg'), image)

        # Layout analysis
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        doc = self.layout_model.process(image, debug_dir)

        # OCR
        start_time = time.time()
        new_doc = []
        for index, each_text_line in enumerate(doc):
            text_line_img = cut_polygon_deskew(
                image, each_text_line['location'], is_gray=True)
            if self.debug:
                cv2.imwrite(os.path.join(debug_dir, '{}.jpg'.format(
                    each_text_line['id'])), text_line_img)
            text, confidents = self.ocr_model.predict([text_line_img])
            each_text_line['text'], each_text_line['confidents'] = text[0], confidents[0]
            new_doc.append(each_text_line)
        doc = new_doc
        if self.debug:
            logging.info('OCR took {} secs'.format(time.time() - start_time))
            save_json(doc, os.path.join(
                debug_dir, 'layout_ocr_{}.json'.format(index)))
            logging.info('Result: {}'.format(pprint(doc)))
        return doc


if __name__ == '__main__':
    model = SceneTextExtractor(debug=True)
    import glob
    from tqdm import tqdm
    data_path = './data/'
    data_test = glob.glob(os.path.join(data_path, "*.jpg")) + \
        glob.glob(os.path.join(data_path, "*.png"))
    for x in tqdm(data_test):
        logging.info('Processing: {}'.format(x))
        img = cv2.imread(x)
        output = model.process(img, os.path.basename(x).split('.')[0])